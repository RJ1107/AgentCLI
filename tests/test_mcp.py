from __future__ import annotations

import asyncio
import json

from agentcli.config import load_config
from agentcli.mcp import McpClientManager
from agentcli.tools.base import ToolContext


def test_mcp_client_registers_and_calls_stdio_tool(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    server = tmp_path / "fake_mcp_server.py"
    server.write_text(
        """
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("fake")

@mcp.tool()
def echo(text: str) -> str:
    return "echo:" + text

if __name__ == "__main__":
    mcp.run(transport="stdio")
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / ".agentcli").mkdir()
    (tmp_path / ".agentcli" / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "fake": {
                        "type": "stdio",
                        "command": "python",
                        "args": [str(server)],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    async def run():
        manager = McpClientManager(tmp_path)
        tools = await manager.load_tools()
        names = [tool.name for tool in tools]
        tool = next(item for item in tools if item.name == "mcp__fake__echo")
        config = load_config(project_root=tmp_path)
        config.policy.hitl_mode = "never"
        result = await tool.execute({"text": "ok"}, ToolContext(cwd=str(tmp_path), config=config))
        return names, result

    names, result = asyncio.run(run())
    assert "mcp__fake__echo" in names
    assert result.content == "echo:ok"


def test_mcp_client_suppresses_stdio_server_stderr(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    server = tmp_path / "noisy_mcp_server.py"
    server.write_text(
        """
import sys
from mcp.server.fastmcp import FastMCP

sys.stderr.write("NOISY_MCP_STARTUP\\n")
sys.stderr.flush()

mcp = FastMCP("noisy")

@mcp.tool()
def echo(text: str) -> str:
    return text

if __name__ == "__main__":
    mcp.run(transport="stdio")
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / ".agentcli").mkdir()
    (tmp_path / ".agentcli" / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "noisy": {
                        "type": "stdio",
                        "command": "python",
                        "args": [str(server)],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    async def run():
        manager = McpClientManager(tmp_path)
        return await manager.load_tools()

    tools = asyncio.run(run())

    assert any(tool.name == "mcp__noisy__echo" for tool in tools)
    captured = capsys.readouterr()
    assert "NOISY_MCP_STARTUP" not in captured.err


def test_servers_are_told_the_project_folder_as_their_root(tmp_path, monkeypatch):
    # Servers that write files (a browser saving a snapshot) only write inside the roots
    # the client names; with none they refuse every path.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = tmp_path / "项目"
    project.mkdir()
    server = tmp_path / "roots_server.py"
    server.write_text(
        """
from mcp.server.fastmcp import Context, FastMCP

mcp = FastMCP("roots")

@mcp.tool()
async def roots(ctx: Context) -> str:
    result = await ctx.session.list_roots()
    return "|".join(str(root.uri) for root in result.roots)

if __name__ == "__main__":
    mcp.run(transport="stdio")
""".lstrip(),
        encoding="utf-8",
    )
    (project / ".agentcli").mkdir()
    (project / ".agentcli" / "mcp.json").write_text(
        json.dumps(
            {"mcpServers": {"roots": {"command": "python", "args": [str(server)], "defer": False}}}
        ),
        encoding="utf-8",
    )

    async def run():
        manager = McpClientManager(project)
        try:
            tools = {tool.name: tool for tool in await manager.load_tools()}
            context = ToolContext(cwd=str(project), config=load_config(project_root=project))
            return await tools["mcp__roots__roots"].execute({}, context)
        finally:
            await manager.aclose()

    result = asyncio.run(run())

    assert not result.is_error, result.content
    assert result.content.strip() == project.resolve().as_uri()
