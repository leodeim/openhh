"""Multi-agent pipeline (Architect/Developer/QA) on the OpenHands SDK.

A task runs through a configurable subset of stages — architect (architecture
+ requirements), developer (implementation), qa (testing). Every agent ends
its turn with an explicit handoff to any other enabled stage; the final
enabled stage alone may declare the task DONE. The loop follows the handoffs
until DONE or max_steps agent turns.

CLI usage:
    uv run pipeline.py --workspace path/to/project --spec spec.md \
        [--stages architect,developer,qa] [--max-steps 12] [--llm profile]

For the web dashboard, see ui.py.
"""

import argparse
import re
import sys
from collections.abc import Callable
from pathlib import Path

from openhands.sdk import Agent, AgentContext, Conversation
from openhands.sdk.event import ActionEvent, MessageEvent
from openhands.sdk.tool.builtins import FinishAction
from openhands.tools.preset.default import get_default_tools

from config import LLMConfig, default_profile, load_profiles

STAGE_ORDER = ["architect", "developer", "qa"]

ARCH_ROLE = """
You are the ARCHITECT in a multi-agent pipeline working in a shared workspace.

Rules:
- Read the spec (and the existing code, if any), then write ARCHITECTURE.md in
  the workspace root: the chosen architecture, key technical decisions, module
  breakdown, and precise, testable requirements for the developer.
- Keep it proportionate to the task — no enterprise ceremony for small tools.
- Do NOT implement the software; code is the developer's job.
- When work comes back to you with questions or issues, answer concretely and
  refine ARCHITECTURE.md so the answer is recorded there.
"""

DEV_ROLE = """
You are the DEVELOPER in a multi-agent pipeline working in a shared workspace.

Rules:
- On your first task, read the spec (and ARCHITECTURE.md, if present — follow
  it), write PLAN.md in the workspace root (short implementation plan with
  milestones), then implement the full spec.
- Keep README.md up to date with exact instructions for running and using the
  project — QA relies on it.
- Verify your own work runs before finishing.
- Fix every issue reported in handoffs you receive. If the requirements or
  architecture are ambiguous or contradictory, hand off to the ARCHITECT
  (when that stage exists) with concrete questions instead of guessing.
"""

QA_ROLE = """
You are the QA ENGINEER in a multi-agent pipeline working in a shared
workspace where other agents produced work against a spec.

Rules:
- Read the spec and README, then actually run and use the software like a real
  user. Run the test suite if present. Probe edge cases from the spec.
- If the project has a web UI, you MUST test it with your browser tool at its
  canonical entry URL (e.g. http://localhost:PORT/): load the page, check for
  failed subresource requests and console errors (a 404 on a script or
  stylesheet is a bug even when every API endpoint works), drive the main
  user flows by clicking and typing like a user, and reload to confirm data
  persisted. curl/API-level checks NEVER count as testing a UI.
- Your handoff must list what you verified and HOW (browser, test suite,
  curl). If part of the spec could not be verified — e.g. the browser tool is
  unavailable or broken — name it as unverified and hand off to the DEVELOPER
  instead of declaring DONE; DONE on unverified claims is the one failure
  mode you must never have.
- Do NOT modify the project's source code. If you need throwaway scripts or
  fixtures, keep them inside a .qa/ directory.
- Route your handoff by root cause: implementation bugs go to the DEVELOPER
  with numbered issues (reproduction steps, expected vs. actual); flawed or
  contradictory requirements/architecture go to the ARCHITECT (when that
  stage exists). Declare DONE only if everything demonstrably works.
"""

ROLES = {"architect": ARCH_ROLE, "developer": DEV_ROLE, "qa": QA_ROLE}

INSTALL_NOTE = """
The project directory is /project — create and edit ALL project files there.
Ignore /workspace; it holds internal agent state, not the project.

You are working inside a disposable Linux container as a NON-ROOT user, so
apt-get will fail with permission errors — do not waste turns on it. If a
tool, compiler, or runtime you need is missing, install it into a
user-writable location from the official tarball and call it by absolute
path. Example for Go (check `uname -m` for the architecture first):
  mkdir -p $HOME/tools && curl -fsSL https://go.dev/dl/go1.26.0.linux-arm64.tar.gz | tar -C $HOME/tools -xz
  $HOME/tools/go/bin/go version
"""

OnEvent = Callable[[dict], None]


def handoff_protocol(stages: list[str], role: str) -> str:
    """Per-role instructions for the HANDOFF line that must end every turn."""
    targets = [s.upper() for s in stages if s != role]
    is_last = role == stages[-1]
    if is_last:
        targets.append("DONE")
    lines = [
        "",
        "Handoff protocol:",
        f"- This task's pipeline stages, in order: {' -> '.join(stages)}. You are the {role.upper()}.",
        "- End EVERY turn by calling finish with a report whose FIRST line is exactly:",
        "    HANDOFF: <target>",
        f"- Your valid targets: {', '.join(targets)}.",
        "- After the HANDOFF line, write concrete notes for the receiving agent:",
        "  numbered issues or instructions, with the context they need to act.",
    ]
    if is_last:
        lines.append(
            "- You are the final stage: use HANDOFF: DONE only when the spec "
            "demonstrably works end to end; otherwise hand off to the stage "
            "that must fix things."
        )
    else:
        after = stages[stages.index(role) + 1].upper()
        lines.append(f"- When your part is complete and you need nothing from the others, hand off to {after}.")
    return "\n".join(lines)


def build_agent(role: str, stages: list[str], docker: bool, llm_config: LLMConfig) -> Agent:
    suffix = ROLES[role] + handoff_protocol(stages, role)
    # only containerized agents may install tools; on the host that's off-limits
    if docker:
        suffix += INSTALL_NOTE
    return Agent(
        llm=llm_config.build(role),
        # QA gets the browser: web UIs must be verified as a user sees them,
        # not just via curl (which happily passes while the frontend is broken)
        tools=get_default_tools(enable_browser=(role == "qa")),
        agent_context=AgentContext(system_message_suffix=suffix),
        system_prompt_kwargs={"cli_mode": True},
    )


def make_workspace(workspace_dir: Path, docker: bool, docker_platform: str):
    """Local dir path, or a fresh per-agent container mounting that dir."""
    if not docker:
        return str(workspace_dir)
    from openhands.workspace import DockerWorkspace  # needs a running Docker daemon

    # mount at /project, NOT /workspace: the agent server writes its own state
    # (conversations/, .git) into /workspace, which would pollute the project dir
    return DockerWorkspace(
        volumes=[f"{workspace_dir}:/project:rw"],
        working_dir="/project",
        platform=docker_platform,
    )


def final_message(conversation: Conversation) -> str:
    """Return the agent's finish message, or its last plain message as fallback."""
    for event in reversed(list(conversation.state.events)):
        if isinstance(event, ActionEvent) and isinstance(event.action, FinishAction):
            return event.action.message
        if isinstance(event, MessageEvent) and event.llm_message.role == "assistant":
            return "\n".join(
                part.text
                for part in event.llm_message.content
                if hasattr(part, "text")
            )
    return ""


# tolerant of markdown/emoji decoration: "**HANDOFF: DONE**", "### Handoff: developer"
HANDOFF_RE = re.compile(r"HANDOFF\W{0,5}(DONE|ARCHITECT|DEVELOPER|QA)\b", re.IGNORECASE)


def parse_handoff(report: str) -> str | None:
    """The declared handoff target ('done' or a role), or None if missing."""
    match = HANDOFF_RE.search(report)
    return match.group(1).lower() if match else None


# The local model sometimes emits raw chat-format tokens instead of tool calls;
# such turns do no work and their text must never be fed back into the loop.
GARBLED_MARKERS = ("<|start|>", "<|message|>", "<atem:")


def looks_garbled(text: str) -> bool:
    return any(marker in text for marker in GARBLED_MARKERS)


def compose_task(sender: str | None, spec: str, report: str, first_visit: bool) -> str:
    parts = []
    if first_visit:
        parts.append(f"Here is the full task specification.\n\n<spec>\n{spec}\n</spec>")
    if sender:
        parts.append(
            f"The {sender.upper()} handed this task to you with these notes:\n\n"
            f"<handoff>\n{report}\n</handoff>\n\nAddress them."
        )
    parts.append("Do your stage's work, then finish with your HANDOFF report.")
    return "\n\n".join(parts)


def run_pipeline(
    workspace: Path,
    spec: str,
    max_steps: int,
    on_event: OnEvent,
    stages: list[str] | tuple[str, ...] = ("developer", "qa"),
    docker: bool = False,
    docker_platform: str = "linux/arm64",
    llm_config: LLMConfig | None = None,
    feedback: str | None = None,
) -> bool:
    """Run the staged handoff loop; emits progress dicts via on_event.

    feedback: user notes from reopening a finished task; delivered to the
    first stage as a handoff from the USER alongside the spec.

    on_event receives:
      {"type": "stage", "step": n, "max": m, "role": role}
      {"type": "agent_event", "role": ..., "event": <sdk Event>}   (live stream)
      {"type": "handoff", "step": n, "from": role, "to": role|"done",
       "report": str, "backward": bool}
      {"type": "done", "approved": bool, "steps": n}

    Returns True iff the final enabled stage declared HANDOFF: DONE.
    """
    stages = [s for s in STAGE_ORDER if s in set(stages)]
    if not stages:
        raise ValueError("At least one stage is required")
    last = stages[-1]

    workspace = workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    llm_config = llm_config or default_profile()

    action_counts = {role: 0 for role in stages}

    def forward(role: str):
        def callback(event):
            if isinstance(event, ActionEvent) and event.tool_name != "finish":
                action_counts[role] += 1
            on_event({"type": "agent_event", "role": role, "event": event})

        return callback

    if docker:
        on_event({"type": "status", "text": "starting per-agent containers (first run pulls the image — can take minutes)"})
    workspaces = {role: make_workspace(workspace, docker, docker_platform) for role in stages}

    def new_conversation(role: str) -> Conversation:
        return Conversation(
            agent=build_agent(role, stages, docker, llm_config),
            workspace=workspaces[role],
            callbacks=[forward(role)],
        )

    conversations = {role: new_conversation(role) for role in stages}

    def run_turn(role: str, task: str, needs_handoff: bool) -> str:
        """Run one agent turn; on an invalid turn (no tool work, garbled output,
        missing handoff) restart the agent with a FRESH conversation — a garbled
        assistant message left in history gets imitated on every retry."""
        message = ""
        for attempt in (1, 2, 3):
            before = action_counts[role]
            conversation = conversations[role]
            conversation.send_message(task)
            conversation.run()
            message = final_message(conversation)
            worked = action_counts[role] > before
            valid = worked and not looks_garbled(message)
            if valid and needs_handoff:
                valid = parse_handoff(message) is not None
            if valid:
                return message
            on_event({"type": "retry", "role": role, "attempt": attempt,
                      "title": "invalid turn, restarting agent",
                      "text": message[:500] or "(no tool calls made)"})
            conversation.close()
            conversations[role] = new_conversation(role)
            # vary the opening move so the model takes a different token path
            task += (
                "\n\nNote: an earlier session on this task was aborted, so the "
                "workspace may already contain partial work. Begin with a "
                "`terminal` tool call running `ls -la` to inspect it, then continue."
            )
        return message

    approved = False
    steps_used = 0
    visited: set[str] = set()
    current = stages[0]
    task = compose_task(sender="user" if feedback else None, spec=spec,
                        report=feedback or "", first_visit=True)
    try:
        for step in range(1, max_steps + 1):
            steps_used = step
            visited.add(current)
            on_event({"type": "stage", "step": step, "max": max_steps, "role": current})
            report = run_turn(current, task, needs_handoff=(current == last))
            if looks_garbled(report):
                report = ""
            target = parse_handoff(report)

            idx = stages.index(current)
            fwd = stages[idx + 1] if idx + 1 < len(stages) else None
            back = stages[idx - 1] if idx > 0 else None

            if target == "done" and current == last:
                on_event({"type": "handoff", "step": step, "from": current, "to": "done",
                          "report": report, "backward": False})
                approved = True
                break
            if target in stages and target != current:
                next_role = target
            elif current != last:
                # missing/invalid directive from a middle stage: move forward
                next_role = fwd
            else:
                # the gatekeeper produced no usable handoff even after retries
                next_role = back or current
                report = (
                    f"The {current} could not produce a valid handoff this step. "
                    "Re-verify the work against the spec yourself, fix any gaps, "
                    "and hand off again."
                )
            backward = stages.index(next_role) < idx
            on_event({"type": "handoff", "step": step, "from": current, "to": next_role,
                      "report": report, "backward": backward})
            task = compose_task(sender=current, spec=spec, report=report,
                                first_visit=next_role not in visited)
            current = next_role
    finally:
        for conversation in conversations.values():
            conversation.close()
        if docker:
            for ws in workspaces.values():
                ws.cleanup()
        on_event({"type": "done", "approved": approved, "steps": steps_used})

    return approved


def _print_progress(item: dict) -> None:
    line = "=" * 72
    if item["type"] == "stage":
        print(f"\n{line}\n  Step {item['step']}/{item['max']} — {item['role'].upper()}\n{line}\n")
    elif item["type"] == "handoff":
        arrow = "DONE" if item["to"] == "done" else f"-> {item['to'].upper()}" + (" (back)" if item["backward"] else "")
        print(f"\n{line}\n  Step {item['step']} handoff: {item['from'].upper()} {arrow}\n{line}\n{item['report']}")
    elif item["type"] == "status":
        print(f"\n[pipeline] {item['text']}")
    elif item["type"] == "retry":
        print(f"\n[pipeline] {item['role']} turn invalid (attempt {item['attempt']}), retrying")
    elif item["type"] == "done":
        outcome = "DONE — spec implemented" if item["approved"] else "STOPPED without approval"
        print(f"\n{line}\n  {outcome} (steps used: {item['steps']})\n{line}")
    # agent_event: the SDK's default visualizer already prints agent activity


def parse_stages(text: str) -> list[str]:
    names = [s.strip().lower() for s in text.split(",") if s.strip()]
    unknown = [s for s in names if s not in STAGE_ORDER]
    if unknown:
        raise ValueError(f"Unknown stages: {', '.join(unknown)} (valid: {', '.join(STAGE_ORDER)})")
    stages = [s for s in STAGE_ORDER if s in names]
    if not stages:
        raise ValueError("At least one stage is required")
    return stages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, help="Project directory (created if missing)")
    parser.add_argument("--spec", required=True, help="Path to the spec file")
    parser.add_argument("--stages", default="developer,qa",
                        help=f"Comma-separated subset of: {','.join(STAGE_ORDER)}")
    parser.add_argument("--max-steps", type=int, default=12, help="Max agent turns before giving up")
    parser.add_argument("--docker", action="store_true", help="Run each agent in its own container")
    parser.add_argument("--docker-platform", default="linux/arm64", help="Container platform (match your host)")
    parser.add_argument("--llm", default="default", help="LLM profile name from llms.json")
    args = parser.parse_args()

    profiles = load_profiles()
    if args.llm not in profiles:
        parser.error(f"unknown LLM profile {args.llm!r} (available: {', '.join(profiles)})")
    llm_config = profiles[args.llm]
    approved = run_pipeline(
        workspace=Path(args.workspace),
        spec=Path(args.spec).expanduser().read_text(),
        max_steps=args.max_steps,
        on_event=_print_progress,
        stages=parse_stages(args.stages),
        docker=args.docker,
        docker_platform=args.docker_platform,
        llm_config=llm_config,
    )
    return 0 if approved else 1


if __name__ == "__main__":
    sys.exit(main())
