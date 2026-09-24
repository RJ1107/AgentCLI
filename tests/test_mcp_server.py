"""AgentCLI as an MCP server, exercised through the official MCP client.

The previous server was only ever tested by calling its own request handler, which is how it
shipped unable to finish a real client's handshake. These tests speak the protocol.
"""

from __future__ import annotations

import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.memory import create_connected_server_and_client_session

from agentcli.mcp.server import build_server


def _with_client(tmp_path, check, *, read_only: bool = False):
    async def run():
        server = build_server(str(tmp_path), read_only=read_only)
        async with create_connected_server_and_client_session(server) as client:
            return await check(client)

    return asyncio.run(run())


def test_client_lists_tools_with_read_write_annotations(tmp_path):
    async def check(client):
        return {tool.name: tool for tool in (await client.list_tools()).tools}

    tools = _with_client(tmp_path, check)

    assert tools["read_file"].annotations.readOnlyHint is True
    assert tools["bash"].annotations.destructiveHint is True
    assert tools["web_fetch"].annotations.openWorldHint is True
    assert "execute_command" not in tools  # aliases are gone


def test_client_reads_then_edits_across_calls(tmp_path):
    (tmp_path / "notes.txt").write_text("alpha\n", encoding="utf-8")

    async def check(client):
        read = await client.call_tool("read_file", {"path": "notes.txt"})
        edit = await client.call_tool(
            "edit_file", {"path": "notes.txt", "old_text": "alpha", "new_text": "beta"}
        )
        missing = await client.call_tool("read_file", {"path": "nope.txt"})
        return read, edit, missing

    read, edit, missing = _with_client(tmp_path, check)

    assert not read.isError and "alpha" in read.content[0].text
    # The read in the first call counts for the edit in the second.
    assert not edit.isError, edit.content[0].text
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "beta\n"
    assert missing.isError


def test_read_only_mode_offers_no_way_to_change_anything(tmp_path):
    async def check(client):
        names = {tool.name for tool in (await client.list_tools()).tools}
        refused = await client.call_tool("bash", {"command": "echo hi"})
        return names, refused

    names, refused = _with_client(tmp_path, check, read_only=True)

    assert "read_file" in names and "grep" in names
    assert not names & {"bash", "write_file", "edit_file", "revert_turn"}
    assert refused.isError


def test_real_stdio_process_completes_the_handshake(tmp_path):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agentcli", "mcp", "serve", "--transport", "stdio", "--cwd", str(tmp_path)],
    )

    async def run():
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            info = await asyncio.wait_for(session.initialize(), 30)
            tools = await session.list_tools()
            return info.serverInfo.name, len(tools.tools)

    name, count = asyncio.run(run())

    assert name == "agentcli"
    assert count == 17
