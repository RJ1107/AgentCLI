from __future__ import annotations

import asyncio
import os

from agentcli.agent.orchestrator import AgentOrchestrator, ExecutionStep, find_cycle
from agentcli.config import load_config
from agentcli.tools import ToolRegistry, get_builtin_tools
from agentcli.tools.base import Tool, ToolContext, ToolResult, object_schema
from agentcli.tools.executor import ToolExecutor
from agentcli.tools.file_ops import edit_file


def _context(tmp_path, monkeypatch, hitl: str = "never") -> ToolContext:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.policy.hitl_mode = hitl
    return ToolContext(cwd=str(tmp_path), config=config)


def _call(call_id: str, name: str, **arguments) -> dict:
    return {"id": call_id, "name": name, "arguments": arguments}


def _builtins() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    return registry


def test_read_after_write_in_one_batch_sees_the_write(tmp_path, monkeypatch):
    context = _context(tmp_path, monkeypatch)
    state = {"value": "old"}

    async def read(_payload, _context):
        return ToolResult(state["value"])

    async def write(payload, _context):
        state["value"] = payload["value"]
        return ToolResult("ok")

    registry = ToolRegistry()
    registry.register(Tool("peek", "read", object_schema({}), read))
    registry.register(
        Tool(
            "put", "write", object_schema({}), write, is_read_only=False, is_concurrency_safe=False
        )
    )
    calls = [
        _call("1", "peek"),
        _call("2", "put", value="new"),
        _call("3", "peek"),
        _call("4", "peek"),
    ]

    results = asyncio.run(ToolExecutor(registry).execute_all(calls, context))

    assert [r.tool_use_id for r in results] == ["1", "2", "3", "4"]
    assert [r.content for r in results] == ["old", "ok", "new", "new"]


def test_file_writes_need_approval_by_default(tmp_path, monkeypatch):
    context = _context(tmp_path, monkeypatch, hitl="auto")
    call = _call("1", "write_file", path="a.txt", content="x")

    result = asyncio.run(ToolExecutor(_builtins()).execute_all([call], context))[0]

    assert result.is_error
    assert not (tmp_path / "a.txt").exists()


def test_edit_rejects_ambiguous_match_unless_replace_all(tmp_path):
    target = tmp_path / "code.py"
    target.write_text("x = 1\ny = 1\n", encoding="utf-8")

    ambiguous = edit_file(str(tmp_path), "code.py", "= 1", "= 2")
    assert ambiguous.is_error
    assert "matches 2 places" in ambiguous.content
    assert target.read_text(encoding="utf-8") == "x = 1\ny = 1\n"

    everywhere = edit_file(str(tmp_path), "code.py", "= 1", "= 2", replace_all=True)
    assert not everywhere.is_error
    assert target.read_text(encoding="utf-8") == "x = 2\ny = 2\n"


def test_changes_require_a_current_read(tmp_path, monkeypatch):
    context = _context(tmp_path, monkeypatch)
    executor = ToolExecutor(_builtins())
    target = tmp_path / "notes.txt"
    target.write_text("alpha\n", encoding="utf-8")

    def run(call):
        return asyncio.run(executor.execute_all([call], context))[0]

    unread = run(_call("1", "edit_file", path="notes.txt", old_text="alpha", new_text="beta"))
    assert unread.is_error and "has not been read" in unread.content

    run(_call("2", "read_file", path="notes.txt"))
    edited = run(_call("3", "edit_file", path="notes.txt", old_text="alpha", new_text="beta"))
    assert not edited.is_error
    # Our own edit refreshes the record, so a follow-up edit needs no extra read.
    again = run(_call("4", "edit_file", path="notes.txt", old_text="beta", new_text="gamma"))
    assert not again.is_error

    # Someone else changes the file: the next change must re-read first.
    target.write_text("delta\n", encoding="utf-8")
    stat = target.stat()
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    stale = run(_call("5", "write_file", path="notes.txt", content="overwrite"))
    assert stale.is_error and "modified after you last read it" in stale.content
    assert target.read_text(encoding="utf-8") == "delta\n"

    created = run(_call("6", "write_file", path="new.txt", content="fresh"))
    assert not created.is_error


def test_team_plan_cycles_are_detected_and_unknown_deps_dropped(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    orchestrator = AgentOrchestrator(
        llm_client=None,  # parse_plan never calls the model
        tool_registry=ToolRegistry(),
        config=load_config(project_root=tmp_path),
        cwd=str(tmp_path),
    )
    steps = orchestrator.parse_plan(
        '{"steps": ['
        '{"id": "a", "description": "A", "dependencies": ["c"]},'
        '{"id": "b", "description": "B", "dependencies": ["a", "ghost"]},'
        '{"id": "c", "description": "C", "dependencies": ["b"]}]}'
    )

    assert steps[1].dependencies == ["step_1"]
    assert find_cycle(steps) == ["step_1", "step_3", "step_2", "step_1"]
    assert find_cycle([ExecutionStep("s1", "x", "COMMAND", [])]) == []
