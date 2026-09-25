from __future__ import annotations

import asyncio

import httpx

from agentcli.llm import openai_compatible
from agentcli.llm.openai_compatible import OpenAICompatibleClient


def _client() -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        provider_name="test",
        model="m",
        api_key="k",
        base_url="https://example.invalid/v1",
        max_context_window=8_000,
    )


def _run(client, responses, monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return responses[min(len(calls) - 1, len(responses) - 1)]

    transport = httpx.MockTransport(handler)
    real = httpx.AsyncClient
    monkeypatch.setattr(
        openai_compatible.httpx,
        "AsyncClient",
        lambda **kwargs: real(transport=transport, **kwargs),
    )

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(openai_compatible.asyncio, "sleep", no_sleep)

    async def collect():
        return [e async for e in client.chat([], [], system_prompt="s")]

    return asyncio.run(collect()), calls


SSE = 'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'


def test_rate_limit_is_retried_then_succeeds(monkeypatch):
    events, calls = _run(
        _client(),
        [httpx.Response(429, headers={"retry-after": "1"}), httpx.Response(200, text=SSE)],
        monkeypatch,
    )
    assert len(calls) == 2
    assert any(e.get("type") == "text_delta" and e.get("text") == "hi" for e in events)
    assert not any(e.get("type") == "error" for e in events)


def test_gives_up_after_the_last_attempt(monkeypatch):
    events, calls = _run(_client(), [httpx.Response(503)], monkeypatch)
    assert len(calls) == openai_compatible._MAX_ATTEMPTS
    assert [e for e in events if e.get("type") == "error"]


def test_client_errors_are_not_retried(monkeypatch):
    _events, calls = _run(_client(), [httpx.Response(401)], monkeypatch)
    assert len(calls) == 1
