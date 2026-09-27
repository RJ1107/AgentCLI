from __future__ import annotations

import asyncio
import io
import os
import time

from rich.console import Console

from agentcli.agent.agent import Agent
from agentcli.config import load_config
from agentcli.context import spill
from agentcli.context.spill import ToolResultStore, use_session_folder
from agentcli.entrypoints.repl import _ChatSession
from agentcli.session import SessionStore
from agentcli.tools import ToolRegistry
from agentcli.types import Message


class _Client:
    model_name = "fake-model"
    provider_name = "fake"
    max_context_window = 200_000

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        yield {"type": "text_delta", "text": "ok"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


def _agent(tmp_path) -> Agent:
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test"
    return Agent(
        llm_client=_Client(), tool_registry=ToolRegistry(), config=config, cwd=str(tmp_path)
    )


def _history() -> list[Message]:
    return [
        Message(role="user", content="读一下 a.py"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        ),
        Message(role="tool", content="1: print('hi')", tool_call_id="c1"),
        Message(role="assistant", content="它打印 hi。"),
    ]


def test_history_round_trips_with_tool_calls(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    session = store.create(str(tmp_path / "project"), "m")

    session.record_turn("读一下 a.py", "它打印 hi。", _history(), "m")
    reopened = store.open(session.id)

    loaded = reopened.load_history()
    assert [(m.role, m.content, m.tool_call_id) for m in loaded] == [
        (m.role, m.content, m.tool_call_id) for m in _history()
    ]
    assert loaded[1].tool_calls[0]["id"] == "c1"
    assert reopened.meta.title == "读一下 a.py"
    assert reopened.meta.turns == 1
    transcript = (session.folder / "transcript.jsonl").read_text(encoding="utf-8")
    assert "它打印 hi。" in transcript


def test_nothing_is_written_before_the_first_message(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    session = store.create(str(tmp_path), "m")
    session.save_history([])
    assert not session.folder.exists()
    assert store.list() == []


def test_listing_is_per_project_newest_first_and_ids_resolve_by_prefix(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    a = store.create(str(tmp_path / "a"), "m")
    a.record_turn("first", "x", [], "m")
    time.sleep(0.01)
    b = store.create(str(tmp_path / "a"), "m")
    b.record_turn("second", "y", [], "m")
    other = store.create(str(tmp_path / "b"), "m")
    other.record_turn("elsewhere", "z", [], "m")

    assert [m.title for m in store.list(str(tmp_path / "a"))] == ["second", "first"]
    assert store.latest(str(tmp_path / "a")).id == b.id
    assert store.find(a.id[:-2], str(tmp_path / "a")).id == a.id


def test_rename_sticks_and_old_sessions_expire(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    session = store.create(str(tmp_path), "m")
    session.record_turn("hello", "hi", [], "m")
    session.rename("数据库迁移")
    session.record_turn("next", "ok", [], "m")
    assert store.open(session.id).meta.title == "数据库迁移"

    long_ago = time.time() - 40 * 86_400
    os.utime(session.folder / "meta.json", (long_ago, long_ago))
    assert store.cleanup(30) == 1
    assert store.list() == []


def test_resume_restores_history_and_when_it_last_ran(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    first = _ChatSession(store, str(tmp_path), _agent(tmp_path))
    first.agent.history = _history()
    first.record("读一下 a.py")
    saved_at = first.current.meta.updated_at

    later = _ChatSession(store, str(tmp_path), _agent(tmp_path))
    assert asyncio.run(later.resume("last", Console(file=io.StringIO())))

    assert [m.content for m in later.agent.history] == [m.content for m in _history()]
    assert later.agent.last_active_at == saved_at
    assert later.current.id == first.current.id


def test_new_session_keeps_the_old_one(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    chat = _ChatSession(store, str(tmp_path), _agent(tmp_path))
    chat.agent.history = _history()
    chat.record("读一下 a.py")
    old_id = chat.current.id

    chat.start_new()

    assert chat.agent.history == []
    assert chat.current.id != old_id
    assert store.open(old_id).load_history()


def test_tool_result_copies_go_to_the_session_and_expire_on_their_own(tmp_path):
    folder = tmp_path / "sessions" / "s1"
    use_session_folder(folder)
    try:
        saved = ToolResultStore(retention_days=7).save("call_1", "output")
    finally:
        use_session_folder(None)
    assert saved == folder / "tool-results" / "call_1.txt"

    old = tmp_path / "sessions" / "s0"
    (old / "tool-results").mkdir(parents=True)
    (old / "meta.json").write_text("{}", encoding="utf-8")
    long_ago = time.time() - 10 * 86_400
    os.utime(old / "tool-results", (long_ago, long_ago))
    spill._cleaned = False
    ToolResultStore(tmp_path / "sessions", retention_days=7).save("call_2", "new")

    assert not (old / "tool-results").exists()
    assert (old / "meta.json").exists()  # the session itself is kept for its own 30 days
