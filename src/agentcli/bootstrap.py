from __future__ import annotations

from dataclasses import replace

from agentcli.config import AgentCliConfig
from agentcli.mcp import McpClientManager
from agentcli.skill import SkillRegistry
from agentcli.tools import ToolRegistry, get_builtin_tools
from agentcli.tools.loader import build_load_tools_tool


async def build_tool_registry(
    *,
    config: AgentCliConfig,
    cwd: str,
) -> tuple[ToolRegistry, McpClientManager | None]:
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    if config.features.skill:
        _describe_skills(registry, cwd)
    manager: McpClientManager | None = None
    if config.features.mcp:
        manager = McpClientManager(cwd)
        registry.register_all(await manager.load_tools())
        if registry.deferred_tools():
            descriptions = {name: spec.description for name, spec in manager.specs.items()}
            registry.register(build_load_tools_tool(registry, descriptions))
    return registry, manager


def _describe_skills(registry: ToolRegistry, cwd: str) -> None:
    """List every enabled skill in load_skill's description, the way load_tools lists tools.

    The model then knows each skill exists even when the request shares no words with it
    ("reply like a stand-up comedian" still finds a comedy skill). The list is fixed for the
    session, so it stays inside the cached prefix.
    """

    tool = registry.get("load_skill")
    index = SkillRegistry(cwd).index_text()
    if tool and index:
        registry.register(replace(tool, description=f"{tool.description}\n\n{index}"))
