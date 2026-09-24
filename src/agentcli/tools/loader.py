from __future__ import annotations

from typing import Any

from agentcli.tools.base import Tool, ToolContext, ToolResult, object_schema
from agentcli.tools.registry import ToolRegistry

LOAD_TOOLS = "load_tools"
_MAX_QUERY_MATCHES = 10


def build_load_tools_tool(
    registry: ToolRegistry,
    server_descriptions: dict[str, str] | None = None,
) -> Tool:
    """A tool that loads deferred tools' full definitions on demand.

    Its description is the index of what can be loaded: one line per MCP server, with the
    server's purpose and its tool names. That index lives in the tool definitions, which are
    sent with every request and stay byte-identical until something is loaded, so it is
    cached like the rest of the prefix.
    """

    async def handler(payload: dict[str, Any], _context: ToolContext) -> ToolResult:
        return _load(registry, payload)

    return Tool(
        name=LOAD_TOOLS,
        description=_description(registry, server_descriptions or {}),
        parameters=object_schema(
            {
                "names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tool names to load: full (mcp__server__tool) or short "
                    "(tool) when only one server has it",
                },
                "server": {
                    "type": "string",
                    "description": "Limit short names or the query to one server",
                },
                "query": {
                    "type": "string",
                    "description": "Keywords to search tool names and descriptions; "
                    f"loads up to {_MAX_QUERY_MATCHES} matches",
                },
            }
        ),
        handler=handler,
        # Changes which schemas the model sees, not the workspace: no approval or snapshot.
        is_read_only=True,
        is_concurrency_safe=False,
    )


def _description(registry: ToolRegistry, server_descriptions: dict[str, str]) -> str:
    groups: dict[str, list[str]] = {}
    for tool in registry.deferred_tools():
        server, short = _split(tool.name)
        groups.setdefault(server, []).append(short)
    lines = [
        "Load the full definitions of deferred tools so you can call them. The tools below "
        "exist but their parameters are not shown until loaded. Load everything a step needs "
        "in one call, naming the server when several servers share tool names, e.g. "
        'server="chrome-devtools", names=["navigate_page", "take_snapshot"]. Loaded tools '
        "stay available for the rest of the session.",
    ]
    for server in sorted(groups):
        purpose = server_descriptions.get(server, "").strip()
        head = f"- {server}" + (f" ({purpose})" if purpose else "")
        lines.append(f"{head}: {', '.join(groups[server])}")
    return "\n".join(lines)


def _load(registry: ToolRegistry, payload: dict[str, Any]) -> ToolResult:
    server = str(payload.get("server") or "").strip()
    names = [str(name).strip() for name in payload.get("names") or [] if str(name).strip()]
    query = str(payload.get("query") or "").strip().lower()
    deferred = [
        tool for tool in registry.deferred_tools() if not server or _split(tool.name)[0] == server
    ]
    if not names and not query and not server:
        return ToolResult("Pass names, a query, or a server.", is_error=True)

    chosen: dict[str, Tool] = {}
    problems: list[str] = []
    for name in names:
        matches = [tool for tool in deferred if name in (tool.name, _split(tool.name)[1])]
        if len(matches) == 1:
            chosen[matches[0].name] = matches[0]
        elif not matches:
            problems.append(f"no deferred tool named {name}")
        else:
            servers = ", ".join(_split(tool.name)[0] for tool in matches)
            problems.append(f"{name} exists on several servers ({servers}); pass server")
    if query:
        terms = query.split()
        hits = [
            tool
            for tool in deferred
            if all(term in f"{tool.name} {tool.description}".lower() for term in terms)
        ]
        for tool in hits[:_MAX_QUERY_MATCHES]:
            chosen[tool.name] = tool
        if not hits:
            problems.append(f"no deferred tool matches '{query}'")
    if server and not names and not query:
        chosen.update({tool.name: tool for tool in deferred})

    registry.activate(list(chosen))
    lines = [f"- {tool.name}: {_first_line(tool.description)}" for tool in chosen.values()]
    if lines:
        lines.insert(0, f"Loaded {len(lines)} tool(s); call them from your next step:")
    lines.extend(f"! {problem}" for problem in problems)
    return ToolResult("\n".join(lines), is_error=not chosen)


def _split(name: str) -> tuple[str, str]:
    parts = name.split("__")
    if len(parts) >= 3 and parts[0] == "mcp":
        return parts[1], "__".join(parts[2:])
    return "other", name


def _first_line(text: str) -> str:
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    return line if len(line) <= 160 else line[:157] + "..."
