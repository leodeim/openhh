"""Web dashboard for the Developer/QA pipeline — a simplified kanban board.

Each created task is one pipeline run: To Do -> In Dev -> In QA (looping back
to In Dev on NEEDS_WORK) -> Done. Tasks run in parallel up to --max-parallel;
the rest queue in To Do. Every event is tagged with its task_id and streamed
over one SSE connection; refreshing the page replays the full board history.

Usage:
    uv run ui.py [--host 127.0.0.1] [--port 7799] [--max-parallel 2]
"""

import argparse
import itertools
import json
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from openhands.sdk.event import (
    ActionEvent,
    AgentErrorEvent,
    MessageEvent,
    ObservationEvent,
)
from config import LLMConfig, PLACEHOLDER_MODEL, load_profiles, save_profiles
from pipeline import STAGE_ORDER, run_pipeline

app = FastAPI(title="openhh")

_lock = threading.Condition()
_events: list[dict] = []
_task_ids = itertools.count(1)
_slots = threading.Semaphore(2)  # reassigned from --max-parallel in __main__
_busy_workspaces: set[Path] = set()
_tasks: dict[int, dict] = {}  # task_id -> run parameters + running flag, for reopen

MAX_TEXT = 4000


def _content_text(parts) -> str:
    return "\n".join(p.text for p in parts if hasattr(p, "text"))


def _summarize(role: str, event) -> list[dict]:
    """Convert an SDK event into small JSON-safe dicts for the browser."""
    if isinstance(event, ActionEvent):
        args = event.action.model_dump(mode="json", exclude_none=True)
        kind = args.pop("kind", type(event.action).__name__)
        return [{
            "type": "action",
            "role": role,
            "title": f"{event.tool_name} · {kind}",
            "thought": _content_text(event.thought)[:MAX_TEXT],
            "text": json.dumps(args, indent=1)[:MAX_TEXT],
        }]
    if isinstance(event, ObservationEvent):
        obs = event.observation
        items = []
        # task tracker "plan" observations carry the agent's full current task
        # list; stream it separately so the plan panel stays live
        if getattr(obs, "command", None) == "plan" and hasattr(obs, "task_list"):
            items.append({
                "type": "plan",
                "role": role,
                "tasks": [t.model_dump(mode="json") for t in obs.task_list],
            })
        try:
            text = obs.text
        except Exception:
            text = str(obs)
        items.append({"type": "observation", "role": role, "title": event.tool_name, "text": text[:MAX_TEXT]})
        return items
    if isinstance(event, MessageEvent):
        message = event.llm_message
        return [{
            "type": "message",
            "role": role,
            "title": f"message ({message.role})",
            "text": _content_text(message.content)[:MAX_TEXT],
        }]
    if isinstance(event, AgentErrorEvent):
        return [{"type": "error", "role": role, "title": f"error · {event.tool_name}", "text": event.error[:MAX_TEXT]}]
    return []


def _emit(item: dict) -> None:
    with _lock:
        _events.append(item)
        _lock.notify_all()


class TaskRequest(BaseModel):
    title: str = ""
    workspace: str
    spec_path: str = ""
    spec_text: str = ""
    stages: list[str] = ["developer", "qa"]
    max_steps: int = 12
    docker: bool = False
    llm: str = "default"  # profile name from the LLM profiles modal


def _run_task(task_id: int, req: TaskRequest, spec: str, llm_config: LLMConfig,
              workspace: Path, feedback: str | None = None) -> None:
    def on_event(item: dict) -> None:
        if item["type"] == "agent_event":
            for summary in _summarize(item["role"], item["event"]):
                _emit({**summary, "task_id": task_id})
        else:
            _emit({**item, "task_id": task_id})

    with _slots:
        _emit({"type": "task_status", "task_id": task_id, "status": "starting"})
        try:
            run_pipeline(workspace, spec, req.max_steps, on_event,
                         stages=req.stages, docker=req.docker, llm_config=llm_config,
                         feedback=feedback)
        except Exception as exc:  # surfaced to the UI, not just the server log
            _emit({"type": "error", "role": "pipeline", "task_id": task_id,
                   "title": "pipeline crashed", "text": str(exc)})
            _emit({"type": "done", "task_id": task_id, "approved": False, "rounds": 0})
        finally:
            with _lock:
                _busy_workspaces.discard(workspace)
                if task_id in _tasks:
                    _tasks[task_id]["running"] = False


class ProfileRequest(BaseModel):
    name: str
    model: str
    base_url: str
    api_key: str = ""  # empty = keep the stored key
    temperature: float = 0.3


_profiles_lock = threading.Lock()


def _public(name: str, cfg: LLMConfig) -> dict:
    """Profile view for the browser — the api_key never leaves the server."""
    return {"name": name, "model": cfg.model, "base_url": cfg.base_url,
            "temperature": cfg.temperature,
            "has_key": bool(cfg.api_key and cfg.api_key != "not-needed")}


@app.get("/api/dirs")
def list_dirs(q: str = ""):
    """Directory autocomplete for the workspace field (~ and relative paths ok)."""
    raw = q.strip()
    base = Path(raw).expanduser() if raw else Path.cwd()
    if not raw or raw.endswith(("/", "~")) and base.is_dir():
        parent, prefix = (base if base.is_dir() else base.parent), ""
    else:
        parent, prefix = base.parent, base.name.lower()
    if not parent.is_dir():
        return []
    home = str(Path.home())
    out = []
    try:
        for d in sorted(parent.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            if prefix and not d.name.lower().startswith(prefix):
                continue
            path = str(d)
            # echo the user's own notation back: ~-style in, ~-style out
            if raw.startswith("~") and path.startswith(home):
                path = "~" + path[len(home):]
            elif not raw or not raw.startswith(("/", "~")):
                path = str(d.relative_to(Path.cwd())) if d.is_relative_to(Path.cwd()) else path
            out.append(path)
            if len(out) >= 30:
                break
    except PermissionError:
        return []
    return out


@app.get("/api/llms")
def list_llms():
    with _profiles_lock:
        return [_public(n, c) for n, c in load_profiles().items()]


@app.post("/api/llms")
def save_llm(req: ProfileRequest):
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "Profile name is required")
    with _profiles_lock:
        profiles = load_profiles()
        current = profiles.get(name)
        api_key = req.api_key or (current.api_key if current else "not-needed")
        try:
            profiles[name] = LLMConfig(model=req.model, base_url=req.base_url,
                                       api_key=api_key, temperature=req.temperature)
        except Exception as exc:
            raise HTTPException(400, f"Invalid profile: {exc}")
        save_profiles(profiles)
    return {"ok": True}


@app.delete("/api/llms/{name}")
def delete_llm(name: str):
    with _profiles_lock:
        profiles = load_profiles()
        if name not in profiles:
            raise HTTPException(404, f"No profile named {name}")
        if len(profiles) == 1:
            raise HTTPException(400, "Cannot delete the last profile")
        del profiles[name]
        save_profiles(profiles)
    return {"ok": True}


def _overlapping(workspace: Path) -> Path | None:
    """A busy workspace that is the same as, inside, or containing this one."""
    for busy in _busy_workspaces:
        if workspace == busy or workspace.is_relative_to(busy) or busy.is_relative_to(workspace):
            return busy
    return None


@app.post("/tasks")
def create_task(req: TaskRequest):
    spec = req.spec_text.strip() or (
        Path(req.spec_path).expanduser().read_text() if req.spec_path else "")
    if not spec:
        raise HTTPException(400, "Provide spec_text or spec_path")
    unknown = [s for s in req.stages if s not in STAGE_ORDER]
    if unknown:
        raise HTTPException(400, f"Unknown stages: {', '.join(unknown)}")
    req.stages = [s for s in STAGE_ORDER if s in req.stages]
    if not req.stages:
        raise HTTPException(400, "Select at least one stage")
    with _profiles_lock:
        profiles = load_profiles()
    if req.llm not in profiles:
        raise HTTPException(400, f"Unknown LLM profile: {req.llm}")
    llm_config = profiles[req.llm]
    if llm_config.model == PLACEHOLDER_MODEL:
        raise HTTPException(400, f"LLM profile '{req.llm}' is a placeholder — "
                                 "set a real model and base URL in LLM profiles first")
    # nested workspaces count too: a task in ~/work and one in ~/work/repos/foo
    # would trample each other's files just like an exact match
    workspace = Path(req.workspace).expanduser().resolve()
    with _lock:
        busy = _overlapping(workspace)
        if busy:
            raise HTTPException(409, f"A task is already running in {busy}, which overlaps {workspace}")
        _busy_workspaces.add(workspace)
    task_id = next(_task_ids)
    title = req.title.strip() or f"{workspace.name} #{task_id}"
    with _lock:
        _tasks[task_id] = {"req": req, "spec": spec, "llm_config": llm_config,
                           "workspace": workspace, "running": True}
    _emit({"type": "task_created", "task_id": task_id, "title": title,
           "workspace": req.workspace, "stages": req.stages, "max_steps": req.max_steps,
           "llm": req.llm, "model": llm_config.model})
    threading.Thread(target=_run_task, args=(task_id, req, spec, llm_config, workspace),
                     daemon=True).start()
    return {"ok": True, "task_id": task_id}


class ReopenRequest(BaseModel):
    comment: str


@app.post("/tasks/{task_id}/reopen")
def reopen_task(task_id: int, req: ReopenRequest):
    """Re-run a finished task in its workspace, seeding the pipeline with the
    user's comment as rework notes (same spec, stages, and LLM profile)."""
    comment = req.comment.strip()
    if not comment:
        raise HTTPException(400, "A comment is required — tell the agents what to fix")
    with _lock:
        info = _tasks.get(task_id)
        if info is None:
            raise HTTPException(404, f"No task #{task_id} (tasks live in server memory; "
                                     "a restarted server cannot reopen older tasks)")
        if info["running"]:
            raise HTTPException(409, f"Task #{task_id} is still running")
        busy = _overlapping(info["workspace"])
        if busy:
            raise HTTPException(409, f"A task is already running in {busy}, "
                                     f"which overlaps {info['workspace']}")
        _busy_workspaces.add(info["workspace"])
        info["running"] = True
    _emit({"type": "task_reopened", "task_id": task_id, "comment": comment})
    threading.Thread(target=_run_task,
                     args=(task_id, info["req"], info["spec"], info["llm_config"],
                           info["workspace"], comment),
                     daemon=True).start()
    return {"ok": True, "task_id": task_id}


@app.get("/events")
def events():
    def stream():
        index = 0
        while True:
            with _lock:
                while index >= len(_events):
                    if not _lock.wait(timeout=15):
                        break  # heartbeat so proxies don't kill the connection
                batch = _events[index:]
                index = len(_events)
            if not batch:
                yield ": keepalive\n\n"
                continue
            for item in batch:
                yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # the UI has no auth and tasks execute code — never expose it beyond
    # localhost unless you understand that anyone who reaches it owns the host
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (0.0.0.0 exposes the unauthenticated UI to your network)")
    parser.add_argument("--port", type=int, default=7799)
    parser.add_argument("--max-parallel", type=int, default=2,
                        help="Concurrent pipeline runs; extra tasks queue in To Do")
    args = parser.parse_args()
    _slots = threading.Semaphore(args.max_parallel)
    # printed directly: imported libs hijack logging and can garble uvicorn's own line
    shown_host = "localhost" if args.host in ("0.0.0.0", "127.0.0.1") else args.host
    print(f"openhh: http://{shown_host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)
