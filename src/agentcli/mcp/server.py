"""Expose AgentCLI's tools to other MCP clients (Claude Code, IDEs, other agents).

Built on the official MCP SDK so the handshake, notifications, and transports follow the
protocol; the earlier hand-rolled JSON-RPC loop could not complete a real client's handshake.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import mcp.types as types
from mcp.server.lowlevel import Server

from agentcli import __version__
from agentcli.config import load_config
from agentcli.tools import ToolRegistry, get_builtin_tools
from agentcli.tools.base import Tool, ToolContext

# Reach out to the internet: clients use this hint to warn about data leaving the machine.
_OPEN_WORLD = {"web_search", "web_fetch"}


def build_server(cwd: str | None = None, *, read_only: bool = False) -> Server:
    """An MCP server offering AgentCLI's built-in tools, rooted at *cwd*.

    Approval belongs to the client that connects (Claude Code, for example, asks before each
    tool call), so AgentCLI's own prompt is off here; its path and command guards still apply.
    Each tool carries annotations so the client can tell reads from writes. With read_only,
    only tools that cannot change anything are offered at all.
    """

    root = str(Path(cwd or ".").resolve())
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    tools = {
        name: tool
        for name in registry.list_names()
        if (tool := registry.get(name)) and (tool.is_read_only or not read_only)
    }
    # Shared across calls, so a client that reads a file and then edits it is not refused as
    # "not read in this session".
    file_state: dict[str, int] = {}
    server: Server = Server("agentcli", version=__version__)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [_describe(tool) for tool in tools.values()]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        tool = tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")
        config = load_config(project_root=root)
        config.policy.hitl_mode = "never"
        context = ToolContext(cwd=root, config=config, file_state=file_state)
        try:
            result = await tool.execute(arguments or {}, context)
            text, is_error = result.content, result.is_error
        except Exception as exc:  # noqa: BLE001 - a failed tool is a result, not a crash
            text, is_error = f"{type(exc).__name__}: {exc}", True
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=text)], isError=is_error
        )

    return server


def _describe(tool: Tool) -> types.Tool:
    return types.Tool(
        name=tool.name,
        description=tool.description,
        inputSchema=tool.parameters,
        annotations=types.ToolAnnotations(
            readOnlyHint=tool.is_read_only,
            destructiveHint=not tool.is_read_only,
            openWorldHint=tool.name in _OPEN_WORLD,
        ),
    )


async def serve_stdio(cwd: str | None = None, *, read_only: bool = False) -> None:
    from mcp.server.stdio import stdio_server

    server = build_server(cwd, read_only=read_only)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def serve_http(port: int = 3000, cwd: str | None = None, *, read_only: bool = False) -> None:
    """Streamable HTTP on 127.0.0.1 at /mcp.

    Only this machine can connect, and Host/Origin checking (DNS-rebinding protection) stops a
    web page in the user's browser from sending requests to it.
    """

    import uvicorn
    from mcp.server.fastmcp.server import StreamableHTTPASGIApp
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from mcp.server.transport_security import TransportSecuritySettings
    from starlette.applications import Starlette
    from starlette.routing import Route

    local = [f"127.0.0.1:{port}", f"localhost:{port}"]
    manager = StreamableHTTPSessionManager(
        app=build_server(cwd, read_only=read_only),
        security_settings=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=local,
            allowed_origins=[f"http://{host}" for host in local],
        ),
    )

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        async with manager.run():
            yield

    app = Starlette(
        routes=[Route("/mcp", endpoint=StreamableHTTPASGIApp(manager))], lifespan=lifespan
    )
    print(f"AgentCLI MCP server listening on http://127.0.0.1:{port}/mcp", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
