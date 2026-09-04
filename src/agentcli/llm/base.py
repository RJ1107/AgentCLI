from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

from agentcli.llm.pricing import CostBreakdown, ModelPriceProfile
from agentcli.types import Message, Usage


class LlmClient(Protocol):
    model_name: str
    provider_name: str
    max_context_window: int
    price_profile: ModelPriceProfile | None

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        system_prompt: str,
    ) -> AsyncIterator[dict[str, Any]]: ...

    def calculate_cost(
        self,
        usage: Usage | dict[str, Any],
        *,
        currency: str = "usd",
    ) -> CostBreakdown: ...
