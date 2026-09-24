from __future__ import annotations

import asyncio
import json

from agentcli.agent import QueryEngine
from agentcli.config import load_config
from agentcli.mcp.client import McpClientManager
from agentcli.mcp.config import McpServerSpec
from agentcli.tools import ToolRegistry
from agentcli.tools.base import Tool, ToolContext, ToolResult, object_schema
from agentcli.tools.loader import build_load_tools_tool


async def _ok(_payload, _context):
    return ToolResult("ok")


def _tool(name: str, description: str = "", deferred: bool = True) -> Tool:
    return Tool(name, description or name, object_schema({}), _ok, deferred=deferred)


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_tool("read_file", deferred=False))
    for server in ("chrome-devtools", "chrome-visible"):
        registry.register(_tool(f"mcp__{server}__navigate_page", "Navigate the page to a URL"))
        registry.register(_tool(f"mcp__{server}__take_snapshot", "Text snapshot of the page"))
    registry.register(_tool("mcp__github__create_issue", "Create a GitHub issue"))
    registry.register(build_load_tools_tool(registry, {"chrome-devtools": "background browser"}))
    return registry


def _sent_names(registry: ToolRegistry) -> set[str]:
    return {definition["function"]["name"] for definition in registry.definitions()}


def _load(registry: ToolRegistry, tmp_path, **payload) -> ToolResult:
    context = ToolContext(cwd=str(tmp_path), config=load_config(project_root=tmp_path))
    return asyncio.run(registry.get("load_tools").execute(payload, context))


def test_deferred_tools_cost_only_their_names_until_loaded(tmp_path):
    registry = _registry()

    assert _sent_names(registry) == {"read_file", "load_tools"}
    index = registry.get("load_tools").description
    assert "- chrome-devtools (background browser): navigate_page, take_snapshot" in index
    assert "- github: create_issue" in index

    result = _load(registry, tmp_path, names=["create_issue"])

    assert not result.is_error
    assert "mcp__github__create_issue" in _sent_names(registry)
    assert "mcp__chrome-devtools__navigate_page" not in _sent_names(registry)


def test_short_names_that_exist_on_two_servers_need_a_server(tmp_path):
    registry = _registry()

    ambiguous = _load(registry, tmp_path, names=["navigate_page"])
    assert ambiguous.is_error and "several servers" in ambiguous.content

    scoped = _load(registry, tmp_path, server="chrome-visible", names=["navigate_page"])
    assert not scoped.is_error
    assert "mcp__chrome-visible__navigate_page" in _sent_names(registry)
    assert "mcp__chrome-devtools__navigate_page" not in _sent_names(registry)


def test_query_and_whole_server_loading(tmp_path):
    registry = _registry()

    _load(registry, tmp_path, query="snapshot page", server="chrome-devtools")
    assert "mcp__chrome-devtools__take_snapshot" in _sent_names(registry)

    _load(registry, tmp_path, server="chrome-visible")
    assert {"mcp__chrome-visible__navigate_page", "mcp__chrome-visible__take_snapshot"} <= (
        _sent_names(registry)
    )

    missing = _load(registry, tmp_path, query="teleport")
    assert missing.is_error and "no deferred tool matches" in missing.content


# ---------------------------------------------------------------------------
# Tool-list cache: startup without starting servers
# ---------------------------------------------------------------------------


def _write_server_config(tmp_path, args: list[str]) -> None:
    (tmp_path / ".agentcli").mkdir(exist_ok=True)
    (tmp_path / ".agentcli" / "mcp.json").write_text(
        json.dumps({"mcpServers": {"fake": {"command": "fake-server", "args": args}}}),
        encoding="utf-8",
    )


def _counting_manager(tmp_path, calls: list[str]) -> McpClientManager:
    manager = McpClientManager(tmp_path)

    async def discover(spec: McpServerSpec):
        calls.append(spec.name)
        return [{"name": "echo", "description": "Echo text", "inputSchema": None}]

    manager._discover = discover  # type: ignore[method-assign]
    return manager


def test_second_start_uses_the_cache_and_launches_nothing(tmp_path):
    _write_server_config(tmp_path, ["--v1"])
    calls: list[str] = []

    first = asyncio.run(_counting_manager(tmp_path, calls).load_tools())
    second_manager = _counting_manager(tmp_path, calls)
    second = asyncio.run(second_manager.load_tools())

    assert calls == ["fake"]  # asked once, then remembered
    assert "mcp__fake__echo" in {tool.name for tool in first}
    assert "mcp__fake__echo" in {tool.name for tool in second}
    assert all(tool.deferred for tool in second if tool.name.startswith("mcp__"))
    assert second_manager._sessions == {}  # no server is running


def test_changing_the_server_config_refreshes_the_list(tmp_path):
    calls: list[str] = []
    _write_server_config(tmp_path, ["--v1"])
    asyncio.run(_counting_manager(tmp_path, calls).load_tools())

    _write_server_config(tmp_path, ["--v2"])  # e.g. a new pinned package version
    asyncio.run(_counting_manager(tmp_path, calls).load_tools())

    assert calls == ["fake", "fake"]


def test_discovery_on_cache_miss_does_not_leave_the_server_running(tmp_path):
    _write_server_config(tmp_path, [])
    manager = McpClientManager(tmp_path)
    closed: list[str] = []

    class _Listed:
        name = "echo"
        description = "Echo"
        inputSchema = None
        annotations = None

    async def list_server_tools(spec):
        manager._sessions[spec.name] = object()  # pretend a connection was opened
        return [_Listed()]

    async def close_server(name):
        closed.append(name)
        manager._sessions.pop(name, None)

    manager.list_server_tools = list_server_tools  # type: ignore[method-assign]
    manager._close_server = close_server  # type: ignore[method-assign]

    asyncio.run(manager.load_tools())

    assert closed == ["fake"]
    assert manager._sessions == {}


# ---------------------------------------------------------------------------
# The agent loop picks up tools loaded mid-request
# ---------------------------------------------------------------------------


class _LoadingClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 200_000

    def __init__(self):
        self.tools_per_call: list[set[str]] = []

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        names = {tool["function"]["name"] for tool in tools}
        self.tools_per_call.append(names)
        if len(self.tools_per_call) == 1:
            yield {
                "type": "tool_call_delta",
                "tool_call": {
                    "index": 0,
                    "id": "call_1",
                    "function": {
                        "name": "load_tools",
                        "arguments": json.dumps({"names": ["create_issue"]}),
                    },
                },
            }
            yield {"type": "message_end", "stop_reason": "tool_use"}
            return
        yield {"type": "text_delta", "text": "ready"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


def test_a_tool_loaded_in_one_turn_is_offered_in_the_next(tmp_path):
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    client = _LoadingClient()
    engine = QueryEngine(
        llm_client=client, tool_registry=_registry(), config=config, cwd=str(tmp_path)
    )

    result = asyncio.run(engine.ask_complete_async("open an issue"))

    assert result.text == "ready"
    assert "mcp__github__create_issue" not in client.tools_per_call[0]
    assert "mcp__github__create_issue" in client.tools_per_call[1]
