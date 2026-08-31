"""LLM connection profiles, stored in llms.json (managed via the web UI)."""

import json
from pathlib import Path

from pydantic import BaseModel, SecretStr

from openhands.sdk import LLM

PROFILES_PATH = Path(__file__).parent / "llms.json"

# written on first run; task creation rejects profiles still carrying it
PLACEHOLDER_MODEL = "openai/CHANGE-ME"


class LLMConfig(BaseModel):
    model: str
    base_url: str
    api_key: str = "not-needed"
    temperature: float = 0.3

    def build(self, usage_id: str) -> LLM:
        return LLM(
            model=self.model,
            base_url=self.base_url,
            api_key=SecretStr(self.api_key),
            usage_id=usage_id,
            temperature=self.temperature,
        )


def load_profiles() -> dict[str, LLMConfig]:
    """Named LLM profiles; first run seeds llms.json with a placeholder default."""
    if not PROFILES_PATH.exists():
        save_profiles({"default": LLMConfig(
            model=PLACEHOLDER_MODEL, base_url="http://localhost:8000/v1")})
    data = json.loads(PROFILES_PATH.read_text())
    return {name: LLMConfig.model_validate(cfg) for name, cfg in data.items()}


def save_profiles(profiles: dict[str, LLMConfig]) -> None:
    PROFILES_PATH.write_text(json.dumps(
        {name: cfg.model_dump() for name, cfg in profiles.items()}, indent=2) + "\n")


def default_profile() -> LLMConfig:
    """The 'default' profile, or the first one if it was renamed."""
    profiles = load_profiles()
    return profiles.get("default") or next(iter(profiles.values()))
