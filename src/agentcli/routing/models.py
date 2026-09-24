from __future__ import annotations

import os
from dataclasses import replace
from typing import Literal

from agentcli.config import AgentCliConfig
from agentcli.llm.base import LlmClient

Tier = Literal["fast", "strong", "top"]
TIERS: tuple[Tier, ...] = ("fast", "strong", "top")
KNOWN_PROVIDERS = {
    "deepseek", "openrouter", "glm", "zhipu", "kimi", "moonshot", "step",
    "openai", "openai-compatible", "compatible",
}  # fmt: skip


def parse_model_spec(spec: str, default_provider: str) -> tuple[str, str]:
    """'openrouter:openai/gpt-6-sol' -> ('openrouter', 'openai/gpt-6-sol').

    Only a known provider name counts as a prefix, because model ids may contain ':' too
    ('openai/gpt-6-luna:batch'); without one, the spec is a model on the session's provider.
    """

    provider, sep, model = spec.strip().partition(":")
    if sep and provider.lower() in KNOWN_PROVIDERS and model:
        return provider.lower(), model
    return default_provider, spec.strip()


def next_tier(tier: Tier) -> Tier:
    return TIERS[min(TIERS.index(tier) + 1, len(TIERS) - 1)]


class ModelTiers:
    """Which model plays which role in /plan and /team.

    Roles: planner, fast, strong, top, and classifier. A role left unset, or whose provider
    key cannot be found, uses the session model, so configuring tiers is optional and a
    missing key degrades to "one model does everything" instead of failing the run.
    """

    def __init__(self, config: AgentCliConfig, session_client: LlmClient):
        self.config = config
        self.session_client = session_client
        self.warnings: list[str] = []
        self._clients: dict[str, LlmClient] = {}

    @property
    def configured(self) -> bool:
        routing = self.config.routing
        return any(
            (routing.planner_model, routing.fast_model, routing.strong_model, routing.top_model)
        )

    def planner(self) -> LlmClient:
        return self._client(self.config.routing.planner_model or self.config.routing.top_model)

    def worker(self, tier: Tier) -> LlmClient:
        return self._client(self._spec(tier))

    def reviewer(self, worker_tier: Tier) -> LlmClient:
        """One tier above the worker, so no model reviews its own output (except at the top)."""

        return self.worker(next_tier(worker_tier))

    def classifier(self) -> LlmClient:
        return self._client(self.config.routing.classifier_model or self.config.routing.fast_model)

    def _spec(self, tier: Tier) -> str:
        routing = self.config.routing
        return {
            "fast": routing.fast_model,
            "strong": routing.strong_model,
            "top": routing.top_model,
        }[tier]

    def _client(self, spec: str) -> LlmClient:
        if not spec.strip():
            return self.session_client
        if spec in self._clients:
            return self._clients[spec]
        from agentcli.llm.factory import create_llm_client
        from agentcli.llm.model_profiles import PROVIDER_API_KEY_ENVS

        session = self.config.llm
        provider, model = parse_model_spec(spec, session.provider.lower())
        if provider == session.provider.lower():
            api_key = session.api_key
            base_url = session.base_url
        else:
            api_key = next(
                (
                    os.environ[name]
                    for name in PROVIDER_API_KEY_ENVS.get(provider, ())
                    if os.environ.get(name)
                ),
                "",
            )
            base_url = None
        if not api_key:
            self.warnings.append(
                f"no API key for {provider}; {spec} falls back to the session model"
            )
            self._clients[spec] = self.session_client
            return self.session_client
        client = create_llm_client(
            replace(
                session,
                provider=provider,
                model=model,
                api_key=api_key,
                base_url=base_url,
                context_window=None,
                prices={},
            )
        )
        self._clients[spec] = client
        return client
