"""Load the deferred tools a request will need before the model asks for them.

MCP tools are deferred: their definitions stay out of the prompt until the model calls
load_tools, which costs a model round trip on the first step of every browser task. Jev can
tell from the request alone which tool it calls for (evals/jev_tools.py: the right tool
first for 54 of 54 requests over 48 options, in about a quarter of a second), so one Jev
request here picks a server (a Choice) and the tools the task will need (a yes/no per
tool), and they are loaded up front.

The guess can only add: a wrong one costs a few tool definitions in the prompt, a missed
one leaves load_tools where it was. Jev off, slow, or down means nothing is preloaded.
"""

from __future__ import annotations

import asyncio
import re

import httpx

from agentcli import jev
from agentcli.tools.registry import ToolRegistry

NONE = "none"
_NONE_MEANS = (
    "Built-in tools are enough (read, edit and search files, run shell commands, web search "
    "and fetch, memory), or the request needs no tool at all"
)


def _split(name: str) -> tuple[str, str]:
    parts = name.split("__")
    if len(parts) >= 3 and parts[0] == "mcp":
        return parts[1], "__".join(parts[2:])
    return "", name


def _first_sentence(text: str) -> str:
    text = " ".join((text or "").split())
    return re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0][:160]


async def preload_tools(
    message: str,
    registry: ToolRegistry,
    *,
    threshold: float,
    timeout: float,
    tool_threshold: float = 0.6,
    min_tools: int = 3,
    max_tools: int = 10,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[str]:
    """Activate the deferred tools Jev expects this request to need; returns their names."""

    servers: dict[str, dict[str, str]] = {}
    for tool in registry.deferred_tools():
        server, short = _split(tool.name)
        if server and not registry.is_active(tool.name):
            servers.setdefault(server, {})[short] = _first_sentence(tool.description)
    if not servers or not message.strip():
        return []

    server_options = {
        server: f"{registry.server_descriptions.get(server, '').strip() or server}. "
        f"Tools: {', '.join(sorted(tools))}"[:600]
        for server, tools in servers.items()
    }
    tool_options: dict[str, str] = {}
    for tools in servers.values():
        for short, description in tools.items():
            tool_options.setdefault(short, description)
    questions: dict[str, dict] = {
        "server": {
            "type": "choice",
            "instructions": "A coding agent received `request`. Which of these tool servers "
            "does it need for it?",
            "criteria": {**server_options, NONE: _NONE_MEANS},
        }
    }
    # A Choice is good at the one best tool but ranks the rest poorly; the set of tools a
    # task needs is a yes/no question per tool, all asked in the same request.
    for short, description in list(tool_options.items())[:200]:
        questions[f"tool:{short}"] = {
            "type": "noul",
            "instructions": {
                "tool": {"name": short, "does": description},
                "question": "Will an agent doing `request` need to call `tool` at some point?",
            },
        }
    answers = await asyncio.wait_for(
        jev.ask({"request": message[:4000]}, questions, transport=transport), timeout
    )
    server_probs = answers["server"]["probabilities"]
    server = max(server_probs, key=server_probs.get)
    if server == NONE or server_probs[server] < threshold:
        return []
    need = {
        short: float(answers.get(f"tool:{short}", {}).get("noul", 0.0)) for short in servers[server]
    }
    ranked = sorted(need, key=lambda short: -need[short])
    chosen = [short for short in ranked if need[short] >= tool_threshold][:max_tools]
    chosen = chosen if len(chosen) >= min_tools else ranked[:min_tools]
    return registry.activate([f"mcp__{server}__{short}" for short in chosen])


__all__ = ["preload_tools"]
