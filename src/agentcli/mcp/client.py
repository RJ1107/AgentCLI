from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from pydantic import AnyUrl

from agentcli.mcp.cache import ToolListCache
from agentcli.mcp.config import McpServerSpec, load_mcp_server_specs
from agentcli.tools.base import Tool, ToolContext, ToolResult, object_schema


class _PersistentSession:
    """One long-lived connection to an MCP server, shared by every call in a session.

    MCP servers are stateful: a browser server keeps the page that navigate_page opened, so
    take_snapshot on a fresh connection would see a blank browser. The transport context must
    be entered and exited by the same task, so a dedicated task owns it for its lifetime and
    callers only borrow the ClientSession.
    """

    def __init__(self, opener):
        self._opener = opener
        self._loop = asyncio.get_running_loop()
        self._lock = asyncio.Lock()
        self._session: ClientSession | None = None
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def usable_from_current_loop(self) -> bool:
        return self._loop is asyncio.get_running_loop()

    async def get(self) -> ClientSession:
        async with self._lock:
            if self._session is not None and self._task and not self._task.done():
                return self._session
            ready: asyncio.Future[ClientSession] = self._loop.create_future()
            self._stop = asyncio.Event()
            self._task = asyncio.create_task(self._run(ready))
            return await ready

    async def _run(self, ready: asyncio.Future) -> None:
        try:
            async with self._opener() as session:
                self._session = session
                ready.set_result(session)
                await self._stop.wait()
        except BaseException as exc:  # noqa: BLE001 - surfaced to the waiting caller
            if not ready.done():
                ready.set_exception(exc if isinstance(exc, Exception) else RuntimeError(exc))
            if isinstance(exc, asyncio.CancelledError):
                raise
        finally:
            self._session = None

    async def close(self) -> None:
        if self._task is None or self._task.done():
            return
        self._stop.set()
        try:
            await asyncio.wait_for(self._task, timeout=10)
        except (TimeoutError, Exception):  # noqa: BLE001 - shutting down regardless
            self._task.cancel()


class McpClientManager:
    def __init__(self, project_root: str | Path):
        self.project_root = str(Path(project_root).resolve())
        self.specs = load_mcp_server_specs(self.project_root)
        self.last_errors: dict[str, str] = {}
        self._sessions: dict[str, _PersistentSession] = {}
        self.cache = ToolListCache()

    async def aclose(self) -> None:
        """Stop every server this manager started (their browsers, subprocesses, streams)."""

        sessions, self._sessions = list(self._sessions.values()), {}
        for session in sessions:
            if session.usable_from_current_loop():
                await session.close()

    async def load_tools(self) -> list[Tool]:
        """Tool wrappers for every enabled server, normally without starting any of them.

        The tool list comes from the on-disk cache. Only a server with no cached list is
        started, asked for its tools, and stopped again; every server is then started for real
        on the first call to one of its tools.
        """

        tools: list[Tool] = []
        self.last_errors.clear()
        for spec in self.specs.values():
            if not spec.enabled:
                continue
            try:
                infos = self.cache.get(spec)
                if infos is None:
                    infos = await self._discover(spec)
                    self.cache.put(spec, infos)
                tools.extend(self._wrap(spec, infos))
                tools.extend(self._virtual_resource_tools(spec))
                tools.extend(self._virtual_prompt_tools(spec))
            except Exception as exc:  # noqa: BLE001 - keep broken MCP servers isolated
                self.last_errors[spec.name] = str(exc)
        return tools

    async def _discover(self, spec: McpServerSpec) -> list[dict[str, Any]]:
        try:
            remote = await self.list_server_tools(spec)
        finally:
            # Asked only for the list; nothing may need this server this session.
            await self._close_server(spec.name)
        return [_tool_info(tool) for tool in remote]

    async def _close_server(self, name: str) -> None:
        session = self._sessions.pop(name, None)
        if session and session.usable_from_current_loop():
            await session.close()

    async def list_server_tools(self, spec: McpServerSpec) -> list[Any]:
        async with self._session(spec) as session:
            result = await session.list_tools()
            return list(result.tools)

    async def call_server_tool(
        self,
        spec: McpServerSpec,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        async with self._session(spec) as session:
            result = await session.call_tool(
                tool_name,
                arguments,
                read_timeout_seconds=timedelta(seconds=spec.timeout),
            )
        return ToolResult(content=_content_to_text(result.content), is_error=bool(result.isError))

    async def list_resources(self, spec: McpServerSpec) -> ToolResult:
        async with self._session(spec) as session:
            result = await session.list_resources()
        lines = [
            f"{resource.uri} {resource.name or ''} {resource.description or ''}".strip()
            for resource in result.resources
        ]
        return ToolResult("\n".join(lines) or "(no resources)")

    async def read_resource(self, spec: McpServerSpec, uri: str) -> ToolResult:
        async with self._session(spec) as session:
            result = await session.read_resource(AnyUrl(uri))
        return ToolResult(_content_to_text(result.contents))

    async def list_prompts(self, spec: McpServerSpec) -> ToolResult:
        async with self._session(spec) as session:
            result = await session.list_prompts()
        lines = [f"{prompt.name} {prompt.description or ''}".strip() for prompt in result.prompts]
        return ToolResult("\n".join(lines) or "(no prompts)")

    async def get_prompt(
        self,
        spec: McpServerSpec,
        name: str,
        arguments: dict[str, str] | None = None,
    ) -> ToolResult:
        async with self._session(spec) as session:
            result = await session.get_prompt(name, arguments or {})
        return ToolResult(_content_to_text(result.messages))

    async def _tools_for_server(self, spec: McpServerSpec) -> list[Tool]:
        """Ask the server itself (no cache) and wrap what it reports."""

        remote = await self.list_server_tools(spec)
        return self._wrap(spec, [_tool_info(tool) for tool in remote])

    def _wrap(self, spec: McpServerSpec, infos: list[dict[str, Any]]) -> list[Tool]:
        wrapped: list[Tool] = []
        for info in infos:
            tool_name = str(info["name"])
            local_name = f"mcp__{spec.name}__{tool_name}"
            schema = info.get("inputSchema") or object_schema({})
            # A server describes itself; its read-only claim only counts if the user trusts it.
            read_only = spec.trusted and bool(info.get("readOnlyHint"))

            async def handler(
                payload: dict[str, Any],
                context: ToolContext,
                *,
                server_spec: McpServerSpec = spec,
                remote_name: str = tool_name,
            ) -> ToolResult:
                _ = context
                return await self.call_server_tool(server_spec, remote_name, payload)

            wrapped.append(
                Tool(
                    name=local_name,
                    description=info.get("description") or f"MCP tool {tool_name}",
                    parameters=schema,
                    handler=handler,
                    is_read_only=read_only,
                    is_concurrency_safe=False,
                    danger_level="safe" if read_only else "medium",
                    requires_approval=not read_only,
                    deferred=spec.defer,
                )
            )
        return wrapped

    def _virtual_resource_tools(self, spec: McpServerSpec) -> list[Tool]:
        async def list_handler(payload: dict[str, Any], context: ToolContext) -> ToolResult:
            _ = payload, context
            return await self.list_resources(spec)

        async def read_handler(payload: dict[str, Any], context: ToolContext) -> ToolResult:
            _ = context
            return await self.read_resource(spec, str(payload["uri"]))

        return [
            Tool(
                name=f"mcp__{spec.name}__list_resources",
                description=f"List MCP resources from {spec.name}.",
                parameters=object_schema({}),
                handler=list_handler,
                is_read_only=True,
                deferred=spec.defer,
            ),
            Tool(
                name=f"mcp__{spec.name}__read_resource",
                description=f"Read an MCP resource from {spec.name}.",
                parameters=object_schema(
                    {"uri": {"type": "string", "description": "Resource URI"}},
                    ["uri"],
                ),
                required_keys=["uri"],
                handler=read_handler,
                is_read_only=True,
                deferred=spec.defer,
            ),
        ]

    def _virtual_prompt_tools(self, spec: McpServerSpec) -> list[Tool]:
        async def list_handler(payload: dict[str, Any], context: ToolContext) -> ToolResult:
            _ = payload, context
            return await self.list_prompts(spec)

        async def get_handler(payload: dict[str, Any], context: ToolContext) -> ToolResult:
            _ = context
            arguments = payload.get("arguments")
            if arguments is not None and not isinstance(arguments, dict):
                return ToolResult("arguments must be an object", is_error=True)
            return await self.get_prompt(
                spec,
                str(payload["name"]),
                {str(k): str(v) for k, v in (arguments or {}).items()},
            )

        return [
            Tool(
                name=f"mcp__{spec.name}__list_prompts",
                description=f"List MCP prompts from {spec.name}.",
                parameters=object_schema({}),
                handler=list_handler,
                is_read_only=True,
                deferred=spec.defer,
            ),
            Tool(
                name=f"mcp__{spec.name}__get_prompt",
                description=f"Get an MCP prompt from {spec.name}.",
                parameters=object_schema(
                    {
                        "name": {"type": "string", "description": "Prompt name"},
                        "arguments": {"type": "object", "description": "Prompt arguments"},
                    },
                    ["name"],
                ),
                required_keys=["name"],
                handler=get_handler,
                is_read_only=True,
                deferred=spec.defer,
            ),
        ]

    @asynccontextmanager
    async def _session(self, spec: McpServerSpec):
        """Borrow the server's shared session, connecting on first use."""

        persistent = self._sessions.get(spec.name)
        # A session is bound to the event loop that opened it; a later asyncio.run (Runtime API
        # workers) needs its own.
        if persistent is None or not persistent.usable_from_current_loop():
            persistent = _PersistentSession(lambda: self._connect(spec))
            self._sessions[spec.name] = persistent
        yield await persistent.get()

    @asynccontextmanager
    async def _connect(self, spec: McpServerSpec):
        if spec.type in {"stdio", "local"}:
            if not spec.command:
                raise ValueError(f"MCP server {spec.name} is missing command")
            params = StdioServerParameters(
                command=spec.command,
                args=spec.args,
                env={**os.environ, **spec.env},
                cwd=spec.cwd or self.project_root,
            )
            with open(os.devnull, "w", encoding="utf-8") as errlog:
                async with (
                    stdio_client(params, errlog=errlog) as (read, write),
                    ClientSession(read, write) as session,
                ):
                    await session.initialize()
                    yield session
            return
        if spec.type in {"http", "streamable_http", "streamable-http"}:
            if not spec.url:
                raise ValueError(f"MCP server {spec.name} is missing url")
            async with (
                streamablehttp_client(
                    spec.url,
                    headers=spec.headers or None,
                    timeout=spec.timeout,
                ) as (read, write, _session_id),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                yield session
            return
        raise ValueError(f"Unsupported MCP transport: {spec.type}")


def _tool_info(remote_tool: Any) -> dict[str, Any]:
    """The parts of a server's tool listing that AgentCLI needs, in a cacheable shape."""

    annotations = getattr(remote_tool, "annotations", None)
    return {
        "name": str(remote_tool.name),
        "description": remote_tool.description or "",
        "inputSchema": remote_tool.inputSchema or None,
        "readOnlyHint": bool(getattr(annotations, "readOnlyHint", False)),
    }


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(filter(None, (_content_to_text(item) for item in content)))
    if hasattr(content, "text"):
        return str(content.text)
    if hasattr(content, "data") and hasattr(content, "mimeType"):
        data = str(content.data)
        return f"[image {content.mimeType} base64 chars={len(data)}]"
    if hasattr(content, "resource"):
        return _content_to_text(content.resource)
    if hasattr(content, "model_dump"):
        return json.dumps(content.model_dump(mode="json"), ensure_ascii=False)
    return str(content)
