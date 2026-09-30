"""A small client for Jev, TypeSafe AI's decision model (https://docs.typesafe.ai/api.md).

Jev does not write text: it takes a state and typed questions (noul: probability of yes;
choice: one of up to 255 options) and returns probabilities in about a quarter of a second.
AgentCLI uses it where a quick judgement helps and a wrong or missing answer costs nothing:
every caller has its own fallback, so Jev being off, slow, or down never stops a request.

The key comes from TYPESAFE_API_KEY. The HTTP client is kept per event loop, so repeated
calls reuse one connection instead of paying a TLS handshake each time.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

JEV_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"

_clients: dict[int, httpx.AsyncClient] = {}


def api_key() -> str:
    return os.environ.get("TYPESAFE_API_KEY", "").strip()


def available() -> bool:
    return bool(api_key())


def _client(transport: httpx.AsyncBaseTransport | None) -> httpx.AsyncClient:
    if transport is not None:
        return httpx.AsyncClient(transport=transport, timeout=10)
    loop = id(asyncio.get_running_loop())
    client = _clients.get(loop)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(timeout=10)
        _clients.clear()  # clients of finished loops cannot be reused
        _clients[loop] = client
    return client


async def ask(
    state: Any,
    questions: dict[str, dict[str, Any]],
    *,
    model: str = MODEL,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, dict[str, Any]]:
    """Answers keyed like *questions*. Raises on any failure; callers fall back."""

    key = api_key()
    if not key:
        raise RuntimeError("no TYPESAFE_API_KEY")
    client = _client(transport)
    try:
        response = await client.post(
            JEV_URL,
            headers={"authorization": f"Bearer {key}"},
            json={"model": model, "state": state, "questions": questions},
        )
        response.raise_for_status()
        return response.json()["answers"]
    finally:
        if transport is not None:
            await client.aclose()


__all__ = ["JEV_URL", "MODEL", "api_key", "ask", "available"]
