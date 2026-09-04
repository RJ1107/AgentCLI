from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from agentcli.llm.pricing import CostBreakdown, ModelPriceProfile, calculate_cost
from agentcli.types import Message, Usage


@dataclass(slots=True)
class OpenAICompatibleClient:
    provider_name: str
    model: str
    api_key: str
    base_url: str
    max_tokens: int = 8192
    temperature: float = 0.7
    timeout: float = 120.0
    max_context_window: int = 128_000
    prompt_cache: bool = False
    price_profile: ModelPriceProfile | None = None

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def supports_images(self) -> bool:
        model = self.model.lower()
        provider = self.provider_name.lower()
        return any(marker in model for marker in ("vision", "image", "5v", "vl")) or (
            provider in {"glm", "zhipu"} and "5v" in model
        )

    def calculate_cost(
        self,
        usage: Usage | dict[str, Any],
        *,
        currency: str = "usd",
    ) -> CostBreakdown:
        if self.price_profile is None:
            raise ValueError(
                f'No price profile is configured for model "{self.model}". '
                "Set llm.prices in AgentCLI config."
            )
        return calculate_cost(usage, self.price_profile, currency=currency)

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        system_prompt: str,
    ) -> AsyncIterator[dict[str, Any]]:
        if not self.api_key:
            yield {
                "type": "error",
                "error": RuntimeError(
                    "AGENTCLI_API_KEY is not configured. Set it in env, ~/.agentcli/config.json, "
                    "or project .agentcli/config.json."
                ),
            }
            return

        payload = self._build_payload(messages, tools, system_prompt=system_prompt)

        headers = {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
            "user-agent": "AgentCLI-Python/0.1.0",
        }
        url = self.base_url.rstrip("/") + "/chat/completions"

        yield {"type": "message_start", "model": self.model}
        try:
            async with (
                httpx.AsyncClient(timeout=self.timeout, http2=False) as client,
                client.stream("POST", url, headers=headers, json=payload) as response,
            ):
                response.raise_for_status()
                async for event in _iter_sse(response):
                    if event == "[DONE]":
                        break
                    try:
                        chunk = json.loads(event)
                    except json.JSONDecodeError:
                        continue
                    async for parsed in self._parse_chunk(chunk):
                        yield parsed
        except httpx.TimeoutException:
            yield {
                "type": "error",
                "error": RuntimeError(
                    f"{self.provider_name} request timed out after {self.timeout:g}s. "
                    "Check the network and retry."
                ),
            }
        except httpx.HTTPStatusError as exc:
            yield {
                "type": "error",
                "error": RuntimeError(
                    f"{self.provider_name} API returned HTTP {exc.response.status_code}. "
                    "Check the API key, model access, account balance, and provider status."
                ),
            }
        except httpx.RequestError:
            yield {
                "type": "error",
                "error": RuntimeError(
                    f"Could not connect to {self.provider_name} at {self.base_url}. "
                    "Check the network, VPN/proxy, and provider status, then retry."
                ),
            }

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        system_prompt: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._format_messages(messages, system_prompt),
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    def _format_messages(self, messages: list[Message], system_prompt: str) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for message in messages:
            if message.role == "tool":
                formatted.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id or "",
                        "content": str(message.content),
                    }
                )
            elif message.role == "assistant":
                item: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
                if message.tool_calls:
                    item["tool_calls"] = message.tool_calls
                formatted.append(item)
            else:
                formatted.append(
                    {"role": message.role, "content": self._format_content(message.content)}
                )
        return formatted

    def _format_content(self, content: str | list[dict[str, Any]]) -> str | list[dict[str, Any]]:
        if isinstance(content, str):
            return content
        if self.supports_images:
            cleaned = []
            for part in content:
                item = {key: value for key, value in part.items() if key != "metadata"}
                cleaned.append(item)
            return cleaned
        text_parts = []
        for part in content:
            if part.get("type") == "text":
                text_parts.append(str(part.get("text") or ""))
            elif part.get("type") == "image_url":
                metadata = part.get("metadata") or {}
                source = metadata.get("source", "remote image")
                width = metadata.get("width", "?")
                height = metadata.get("height", "?")
                text_parts.append(f"[Image omitted: {source}, {width}x{height}]")
        return "\n".join(text_parts)

    async def list_models(self) -> list[str]:
        """Fetch available model IDs from the API's /v1/models endpoint.

        Returns an empty list if the API key is missing or the request fails.
        """
        if not self.api_key:
            return []
        headers = {
            "authorization": f"Bearer {self.api_key}",
            "user-agent": "AgentCLI-Python/0.1.0",
        }
        url = self.base_url.rstrip("/") + "/models"
        async with httpx.AsyncClient(timeout=15.0, http2=False) as client:
            try:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                return sorted(item["id"] for item in data.get("data", []))
            except Exception:
                return []

    async def _parse_chunk(self, chunk: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        choices = chunk.get("choices") or []
        if choices:
            choice = choices[0]
            delta = choice.get("delta") or {}

            reasoning = delta.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                yield {"type": "thinking_delta", "thinking": reasoning}

            content = delta.get("content")
            if isinstance(content, str) and content:
                yield {"type": "text_delta", "text": content}

            tool_calls = delta.get("tool_calls") or []
            for tool_call in tool_calls:
                yield {"type": "tool_call_delta", "tool_call": tool_call}

            finish_reason = choice.get("finish_reason")
            if finish_reason:
                yield {
                    "type": "message_end",
                    "stop_reason": _map_finish_reason(str(finish_reason)),
                }

        usage = chunk.get("usage")
        if isinstance(usage, dict):
            yield {"type": "usage", "usage": Usage.from_mapping(usage).to_dict()}


async def _iter_sse(response: httpx.Response) -> AsyncIterator[str]:
    buffer = ""
    async for text in response.aiter_text():
        buffer += text
        while "\n\n" in buffer:
            event, buffer = buffer.split("\n\n", 1)
            data_lines = []
            for line in event.splitlines():
                line = line.strip()
                if line.startswith("data:"):
                    data_lines.append(line[5:].strip())
            if data_lines:
                yield "\n".join(data_lines)
    if buffer.strip():
        data_lines = []
        for line in buffer.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if data_lines:
            yield "\n".join(data_lines)


def _map_finish_reason(reason: str) -> str:
    if reason in {"tool_calls", "tool_use"}:
        return "tool_use"
    if reason == "length":
        return "max_tokens"
    if reason == "content_filter":
        return "stop_sequence"
    return "end_turn"
