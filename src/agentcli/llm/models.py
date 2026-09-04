"""models.py — Central registry of known models for display and selection.

Provides a curated built-in list of models aggregated from across the codebase
(factory context windows, pricing profiles, known providers) plus a function
to retrieve the full list.  Users can always configure arbitrary models via
LlmConfig; this module only surfaces the ones AgentCLI knows about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agentcli.llm.model_profiles import DEFAULT_MODEL_PROFILES


@dataclass(frozen=True, slots=True)
class KnownModel:
    """A single known model entry for display in ``/model list``."""

    provider: str
    """Provider slug, e.g. ``"deepseek"``, ``"openai"``."""

    model: str
    """Full model identifier as passed to the API, e.g. ``"deepseek-v4-flash"``."""

    context_window: int
    """Maximum context window in tokens."""

    pricing: Literal["builtin", "unknown"] = "unknown"
    """Whether built-in price data is available for this model."""


# ---------------------------------------------------------------------------
# Curated built-in list
# ---------------------------------------------------------------------------
# Kept in sync with factory.MODEL_CONTEXT_WINDOWS, factory.PROVIDER_BASE_URLS,
# and pricing.DEEPSEEK_V4_PRICE_PROFILES.

_BUILTIN_MODELS: list[KnownModel] = [
    *(
        KnownModel(
            provider=profile.provider,
            model=profile.model,
            context_window=profile.context_window,
            pricing="builtin" if profile.provider == "deepseek" else "unknown",
        )
        for profile in DEFAULT_MODEL_PROFILES
    ),
    # ---- OpenAI-compatible (generic) ----
    KnownModel(
        provider="openai",
        model="gpt-4o",
        context_window=128_000,
        pricing="unknown",
    ),
    KnownModel(
        provider="openai",
        model="gpt-4o-mini",
        context_window=128_000,
        pricing="unknown",
    ),
    # ---- Kimi / Moonshot ----
    KnownModel(
        provider="kimi",
        model="kimi-think",
        context_window=128_000,
        pricing="unknown",
    ),
    KnownModel(
        provider="moonshot",
        model="moonshot-v1-8k",
        context_window=8_000,
        pricing="unknown",
    ),
    KnownModel(
        provider="moonshot",
        model="moonshot-v1-32k",
        context_window=32_000,
        pricing="unknown",
    ),
    KnownModel(
        provider="moonshot",
        model="moonshot-v1-128k",
        context_window=128_000,
        pricing="unknown",
    ),
    # ---- Step ----
    KnownModel(
        provider="step",
        model="step-2-16k",
        context_window=16_000,
        pricing="unknown",
    ),
]


def get_available_models() -> list[KnownModel]:
    """Return the curated list of known models.

    This list is the single source of truth for ``/model list`` display.
    It is safe to call without any API key or network access.
    """
    return list(_BUILTIN_MODELS)


def get_models_by_provider(provider: str) -> list[KnownModel]:
    """Filter :func:`get_available_models` by provider slug (case-insensitive)."""
    provider_lower = provider.lower()
    return [m for m in _BUILTIN_MODELS if m.provider.lower() == provider_lower]
