from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from agentcli.llm.pricing import OPENROUTER_CATALOG
from agentcli.paths import agentcli_home


@dataclass(frozen=True, slots=True)
class ModelProfile:
    id: str
    name: str
    provider: str
    model: str
    base_url: str
    context_window: int
    description: str = ""
    api_key: str = ""
    api_key_env: str = "AGENTCLI_API_KEY"
    custom: bool = False

    @classmethod
    def custom_profile(
        cls,
        *,
        name: str,
        provider: str,
        model: str,
        base_url: str,
        context_window: int,
        api_key: str = "",
        api_key_env: str = "AGENTCLI_API_KEY",
    ) -> ModelProfile:
        name = name.strip()
        provider = provider.strip().lower()
        model = model.strip()
        base_url = base_url.strip().rstrip("/")
        api_key_env = api_key_env.strip()
        if not name or not provider or not model:
            raise ValueError("name, provider, and model are required")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base URL must start with http:// or https://")
        if context_window <= 0:
            raise ValueError("context window must be greater than zero")
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "model"
        return cls(
            id=f"custom-{slug}-{uuid4().hex[:8]}",
            name=name,
            provider=provider,
            model=model,
            base_url=base_url,
            context_window=context_window,
            api_key=api_key,
            api_key_env=api_key_env,
            custom=True,
        )

    def resolve_api_key(
        self,
        *,
        current_provider: str = "",
        current_api_key: str = "",
        env: Mapping[str, str] | None = None,
    ) -> str:
        if self.api_key:
            return self.api_key
        env_map = env if env is not None else os.environ
        candidates = [self.api_key_env, "AGENTCLI_API_KEY"]
        candidates.extend(PROVIDER_API_KEY_ENVS.get(self.provider, ()))
        for key in candidates:
            if key and env_map.get(key):
                return str(env_map[key])
        if self.provider == current_provider.lower():
            return current_api_key
        return ""


# Name and one-line pitch for each OpenRouter model; ids, context windows and prices come from
# OPENROUTER_CATALOG so there is one source of truth.
_OPENROUTER_ABOUT: dict[str, tuple[str, str]] = {
    "openai/gpt-6-astra": ("GPT-6 Astra", "OpenAI flagship for the hardest long tasks"),
    "openai/gpt-6-sol": ("GPT-6 Sol", "OpenAI high-end at a fifth of Astra's price"),
    "openai/gpt-6-luna": ("GPT-6 Luna", "OpenAI fast tier, very cheap"),
    "anthropic/claude-opus-5.5": ("Claude Opus 5.5", "Anthropic flagship for large codebases"),
    "deepseek/deepseek-v4-flash": ("DeepSeek V4 Flash", "cheapest capable agent model"),
    "deepseek/deepseek-v4.1-flash": ("DeepSeek V4.1 Flash", "newer Flash, fast, near-free cache"),
    "deepseek/deepseek-v4-pro-0813": ("DeepSeek V4 Pro 0813", "stronger DeepSeek, August build"),
    "z-ai/glm-5.3": ("GLM-5.3", "Zhipu flagship for long agent tasks"),
    "z-ai/glm-5.3-flash": ("GLM-5.3 Flash", "Zhipu budget model"),
    "qwen/qwen3.8-flash": ("Qwen3.8 Flash", "Alibaba multimodal, cheap"),
    "google/gemini-3.8-flash": ("Gemini 3.8 Flash", "Google's strongest Flash"),
}


def _price(value: float) -> str:
    return f"${value:.3g}"


_OPENROUTER_PROFILES = tuple(
    ModelProfile(
        id=f"openrouter-{model.split('/', 1)[1]}",
        name=f"{_OPENROUTER_ABOUT[model][0]} (OpenRouter)",
        provider="openrouter",
        model=model,
        base_url="https://openrouter.ai/api/v1",
        context_window=context,
        description=(
            f"{_OPENROUTER_ABOUT[model][1]} · {_price(uncached)} in / {_price(output)} out "
            f"per 1M tokens"
        ),
        api_key_env="OPENROUTER_API_KEY",
    )
    for model, context, uncached, _cached, output in OPENROUTER_CATALOG
)

DEFAULT_MODEL_PROFILES: tuple[ModelProfile, ...] = (
    # DeepSeek's own API: needs DEEPSEEK_API_KEY (its cache hits cost 1/50 of a miss).
    ModelProfile(
        id="deepseek-v4-flash",
        name="DeepSeek V4 Flash",
        provider="deepseek",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        context_window=1_000_000,
        description="Fast, cost-efficient Agent model with thinking support",
        api_key_env="DEEPSEEK_API_KEY",
    ),
    ModelProfile(
        id="deepseek-v4-pro",
        name="DeepSeek V4 Pro",
        provider="deepseek",
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com",
        context_window=1_000_000,
        description="Higher-quality DeepSeek model for difficult coding tasks",
        api_key_env="DEEPSEEK_API_KEY",
    ),
    # One OpenRouter key reaches all of these.
    *_OPENROUTER_PROFILES,
)


PROVIDER_DEFAULTS: dict[str, tuple[str, str, int]] = {
    "deepseek": ("DeepSeek", "https://api.deepseek.com", 1_000_000),
    "glm": ("GLM / Zhipu", "https://open.bigmodel.cn/api/paas/v4", 200_000),
    "openai-compatible": ("OpenAI-compatible", "https://api.openai.com/v1", 128_000),
    "openrouter": ("OpenRouter", "https://openrouter.ai/api/v1", 1_048_576),
}

PROVIDER_API_KEY_ENVS: dict[str, tuple[str, ...]] = {
    "deepseek": ("DEEPSEEK_API_KEY",),
    "glm": ("ZAI_API_KEY", "GLM_API_KEY"),
    "zhipu": ("ZAI_API_KEY", "GLM_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "openai-compatible": ("OPENAI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
}


class CustomModelStore:
    """Persist BYOK model profiles in a user-only JSON file."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or agentcli_home() / "models.json").expanduser()

    def list(self) -> list[ModelProfile]:
        data = self._read()
        profiles: list[ModelProfile] = []
        for item in data.get("models", []):
            if not isinstance(item, dict):
                continue
            try:
                profile = ModelProfile(**item)
            except (TypeError, ValueError):
                continue
            if profile.custom:
                profiles.append(profile)
        return profiles

    def add(self, profile: ModelProfile) -> None:
        if not profile.custom:
            raise ValueError("only custom profiles can be persisted")
        profiles = [item for item in self.list() if item.id != profile.id]
        profiles.append(profile)
        self._write({"version": 1, "models": [asdict(item) for item in profiles]})

    def delete(self, profile_id: str) -> bool:
        profiles = self.list()
        remaining = [item for item in profiles if item.id != profile_id]
        if len(remaining) == len(profiles):
            return False
        self._write({"version": 1, "models": [asdict(item) for item in remaining]})
        return True

    def _read(self) -> dict:
        if not self.path.is_file():
            return {"version": 1, "models": []}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "models": []}
        return data if isinstance(data, dict) else {"version": 1, "models": []}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        fd, temp_name = tempfile.mkstemp(prefix="models-", suffix=".json", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_name, 0o600)
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
