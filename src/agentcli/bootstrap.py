from __future__ import annotations

from agentcli.config import AgentCliConfig
from agentcli.mcp import McpClientManager
from agentcli.tools import ToolRegistry, get_builtin_tools
from agentcli.tools.loader import build_load_tools_tool


async def build_tool_registry(
    *,
    config: AgentCliConfig,
    cwd: str,
) -> tuple[ToolRegistry, McpClientManager | None]:
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    manager: McpClientManager | None = None
    if config.features.mcp:
        manager = McpClientManager(cwd)
        registry.register_all(await manager.load_tools())
        if registry.deferred_tools():
            descriptions = {name: spec.description for name, spec in manager.specs.items()}
            registry.register(build_load_tools_tool(registry, descriptions))
    return registry, manager
