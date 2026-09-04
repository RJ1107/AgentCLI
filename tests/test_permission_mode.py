from __future__ import annotations

import asyncio

from agentcli.config import load_config
from agentcli.entrypoints.repl import PermissionModeController
from agentcli.tools.base import Tool, ToolContext, ToolResult, object_schema
from agentcli.tools.executor import ToolExecutor
from agentcli.tools.registry import ToolRegistry


def test_auto_mode_runs_approval_required_tool_without_callback(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    controller = PermissionModeController(config)
    executions: list[str] = []

    async def mutate(payload, _context):
        executions.append(str(payload["value"]))
        return ToolResult("done")

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="mutate",
            description="Mutate test state",
            parameters=object_schema({"value": {"type": "string"}}, ["value"]),
            required_keys=["value"],
            handler=mutate,
            is_read_only=False,
            requires_approval=True,
        )
    )
    executor = ToolExecutor(registry)
    call = {"id": "call-1", "name": "mutate", "arguments": {"value": "ok"}}
    context = ToolContext(cwd=str(tmp_path), config=config)

    denied = asyncio.run(executor.execute_all([call], context))[0]
    assert denied.is_error
    assert executions == []

    controller.set("auto")
    approved = asyncio.run(executor.execute_all([call], context))[0]
    assert not approved.is_error
    assert executions == ["ok"]
