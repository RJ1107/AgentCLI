from __future__ import annotations

import asyncio
import json

import httpx

from agentcli.tools.base import Tool, ToolResult, object_schema
from agentcli.tools.preload import preload_tools
from agentcli.tools.registry import ToolRegistry


async def _noop(_payload, _context):
    return ToolResult("ok")


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(Tool("read_file", "Read a file.", object_schema({}), _noop, is_read_only=True))
    for server in ("chrome-devtools", "chrome-visible"):
        for short in ("navigate_page", "take_snapshot", "click", "list_console_messages"):
            registry.register(
                Tool(f"mcp__{server}__{short}", f"Browser {short}.", object_schema({}), _noop, deferred=True)
            )
    registry.server_descriptions = {"chrome-devtools": "background browser", "chrome-visible": "visible browser"}
    return registry


def _jev(server: dict[str, float], tool: dict[str, float], *, status: int = 200):
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if status != 200:
            return httpx.Response(status)
        options = body["questions"]["server"]["criteria"]
        full = {option: server.get(option, 0.0) for option in options}
        answers = {"server": {"type": "choice", "choice": max(full, key=full.get), "probabilities": full}}
        for key in body["questions"]:
            if key.startswith("tool:"):
                answers[key] = {"type": "noul", "noul": tool.get(key[5:], 0.0)}
        return httpx.Response(200, json={"answers": answers})

    return httpx.MockTransport(handler), seen


def _run(registry, transport, threshold=0.6, min_tools=1):
    return asyncio.run(
        preload_tools("打开 localhost:5173 看看控制台报错", registry, threshold=threshold,
                      min_tools=min_tools, timeout=2, transport=transport)
    )


def test_the_likeliest_tools_of_the_chosen_server_are_loaded(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    registry = _registry()
    transport, seen = _jev(
        {"chrome-devtools": 0.9}, {"list_console_messages": 0.9, "navigate_page": 0.7, "click": 0.2}
    )

    loaded = _run(registry, transport)

    # Every tool the task likely needs (yes >= 0.6), from the chosen server only.
    assert loaded == ["mcp__chrome-devtools__list_console_messages", "mcp__chrome-devtools__navigate_page"]
    # None likely: still the min_tools likeliest, so the first step has something to call.
    assert _run(_registry(), _jev({"chrome-devtools": 0.9}, {"click": 0.3})[0], min_tools=2) == [
        "mcp__chrome-devtools__click",
        "mcp__chrome-devtools__list_console_messages",
    ]
    assert not registry.is_active("mcp__chrome-visible__click")
    # One request answers both questions; "none" is always an option.
    assert len(seen) == 1 and "none" in seen[0]["questions"]["server"]["criteria"]


def test_nothing_is_loaded_when_built_in_tools_suffice_or_jev_is_unsure(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    registry = _registry()

    assert _run(registry, _jev({"none": 0.8, "chrome-devtools": 0.2}, {})[0]) == []
    assert _run(registry, _jev({"chrome-devtools": 0.5, "none": 0.45}, {"click": 1.0})[0]) == []
    assert not any(registry.is_active(tool.name) for tool in registry.deferred_tools())


def test_a_jev_failure_raises_for_the_caller_to_ignore(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    try:
        _run(_registry(), _jev({}, {}, status=503)[0])
    except httpx.HTTPStatusError:
        pass
    else:
        raise AssertionError("expected the failure to propagate")
