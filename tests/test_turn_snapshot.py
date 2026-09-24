from __future__ import annotations

import asyncio
from types import SimpleNamespace

from agentcli.agent.agent import Agent
from agentcli.config import load_config
from agentcli.mcp.client import McpClientManager
from agentcli.mcp.config import McpServerSpec, _spec_from_raw
from agentcli.snapshot import SnapshotService, TurnSnapshot
from agentcli.tools import ToolRegistry, get_builtin_tools
from agentcli.tools.base import ToolContext
from agentcli.tools.executor import ToolExecutor


def _isolate_home(tmp_path, monkeypatch):
    # Path.home() reads USERPROFILE on Windows and HOME elsewhere.
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))


def _call(call_id: str, name: str, **arguments) -> dict:
    return {"id": call_id, "name": name, "arguments": arguments}


def _run(tmp_path, calls, *, approve: str = "approve") -> TurnSnapshot:
    config = load_config(project_root=tmp_path)
    config.policy.hitl_mode = "auto"
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    snapshot = TurnSnapshot(tmp_path)
    context = ToolContext(
        cwd=str(tmp_path),
        config=config,
        approval_callback=lambda _request: approve,
        turn_snapshot=snapshot,
    )
    asyncio.run(ToolExecutor(registry).execute_all(calls, context))
    return snapshot


def test_read_only_request_takes_no_snapshot(tmp_path, monkeypatch):
    _isolate_home(tmp_path, monkeypatch)
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")

    snapshot = _run(
        tmp_path, [_call("1", "read_file", path="a.txt"), _call("2", "list_dir", path=".")]
    )

    assert snapshot.record is None
    assert SnapshotService(tmp_path).list() == []


def test_first_write_takes_one_snapshot_of_the_original_state(tmp_path, monkeypatch):
    _isolate_home(tmp_path, monkeypatch)
    (tmp_path / "a.txt").write_text("original\n", encoding="utf-8")

    snapshot = _run(
        tmp_path,
        [
            _call("1", "read_file", path="a.txt"),
            _call("2", "edit_file", path="a.txt", old_text="original", new_text="changed"),
            _call("3", "write_file", path="b.txt", content="new"),
        ],
    )

    records = SnapshotService(tmp_path).list()
    assert len(records) == 1 and records[0].phase == "pre-write"
    assert (snapshot.record.path / "a.txt").read_text(encoding="utf-8") == "original\n"
    assert not (snapshot.record.path / "b.txt").exists()
    # The fake home (holding the snapshot store) sits inside this project: it must be skipped,
    # or the snapshot would try to copy itself into itself.
    assert not (snapshot.record.path / "home").exists()


def test_denied_write_takes_no_snapshot(tmp_path, monkeypatch):
    _isolate_home(tmp_path, monkeypatch)

    _run(tmp_path, [_call("1", "write_file", path="b.txt", content="x")], approve="deny")

    assert SnapshotService(tmp_path).list() == []


class _AnswerOnlyClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 100_000

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        yield {"type": "text_delta", "text": "just an answer"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


def test_agent_answer_without_tools_leaves_no_snapshots(tmp_path, monkeypatch):
    _isolate_home(tmp_path, monkeypatch)
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    agent = Agent(
        llm_client=_AnswerOnlyClient(),
        tool_registry=ToolRegistry(),
        config=config,
        cwd=str(tmp_path),
    )

    result = asyncio.run(agent.run_complete("what does this project do?"))

    assert result.text == "just an answer"
    assert SnapshotService(tmp_path).list() == []


def test_mcp_read_only_hint_is_trusted_only_for_trusted_servers(tmp_path, monkeypatch):
    _isolate_home(tmp_path, monkeypatch)
    assert _spec_from_raw("x", {"command": "srv", "trusted": True}, tmp_path).trusted
    assert not _spec_from_raw("x", {"command": "srv"}, tmp_path).trusted

    manager = McpClientManager(tmp_path)
    remote = SimpleNamespace(
        name="delete_everything",
        description="claims to be harmless",
        inputSchema=None,
        annotations=SimpleNamespace(readOnlyHint=True),
    )

    async def fake_list(_spec):
        return [remote]

    monkeypatch.setattr(manager, "list_server_tools", fake_list)

    untrusted = asyncio.run(manager._tools_for_server(McpServerSpec(name="s")))[0]
    trusted = asyncio.run(manager._tools_for_server(McpServerSpec(name="s", trusted=True)))[0]

    assert not untrusted.is_read_only and untrusted.requires_approval
    assert trusted.is_read_only and not trusted.requires_approval
