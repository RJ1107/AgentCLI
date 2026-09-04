from __future__ import annotations

import asyncio
from typing import Any

from agentcli.agent import QueryEngine
from agentcli.config import load_config
from agentcli.skill import SkillRegistry
from agentcli.tools import ToolRegistry, get_builtin_tools
from agentcli.types import Message


class FakeClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 1000

    def __init__(self):
        self.calls = 0

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        self.calls += 1
        if self.calls == 1:
            yield {
                "type": "tool_call_delta",
                "tool_call": {
                    "index": 0,
                    "id": "call_1",
                    "function": {"name": "read_file", "arguments": '{"path":"note.txt"}'},
                },
            }
            yield {"type": "message_end", "stop_reason": "tool_use"}
        else:
            tool_messages = [message for message in messages if message.role == "tool"]
            assert tool_messages
            assert "1: hello" in tool_messages[-1].content
            yield {"type": "text_delta", "text": "done"}
            yield {"type": "message_end", "stop_reason": "end_turn"}


class SkillLoadingClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 1000

    def __init__(self):
        self.calls = 0

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        self.calls += 1
        if self.calls == 1:
            user_content = str(messages[-1].content)
            assert "Relevant skill candidates" in user_content
            assert "demo-skill" in user_content
            assert any(tool["function"]["name"] == "load_skill" for tool in tools)
            yield {
                "type": "tool_call_delta",
                "tool_call": {
                    "index": 0,
                    "id": "load_1",
                    "function": {"name": "load_skill", "arguments": '{"name":"demo-skill"}'},
                },
            }
            yield {"type": "message_end", "stop_reason": "tool_use"}
            return

        tool_messages = [message for message in messages if message.role == "tool"]
        assert tool_messages
        assert "## Loaded Skill: demo-skill" in str(tool_messages[-1].content)
        assert "Apply the demo workflow now." in str(tool_messages[-1].content)
        yield {"type": "text_delta", "text": "skill applied"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


class UsageAndCompressionClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 800

    def __init__(self):
        self.saw_summary = False

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        self.saw_summary = any(
            "conversation-summary" in str(message.content) for message in messages
        )
        yield {"type": "text_delta", "text": "compressed"}
        yield {"type": "message_end", "stop_reason": "end_turn"}
        yield {
            "type": "usage",
            "usage": {
                "input_tokens": 120,
                "output_tokens": 8,
                "prompt_cache_hit_tokens": 20,
                "prompt_cache_miss_tokens": 100,
            },
        }


def test_query_engine_executes_tool_and_replays_result(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "note.txt").write_text("hello\n", encoding="utf-8")
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    engine = QueryEngine(
        llm_client=FakeClient(),
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
    )

    async def run() -> Any:
        return await engine.ask_complete_async("read note")

    result = asyncio.run(run())
    assert result.text == "done"
    assert result.turns == 2


def test_load_skill_is_injected_in_the_same_query_next_model_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    SkillRegistry(tmp_path).create(
        "demo-skill",
        description="Use for the demo workflow",
        body="Apply the demo workflow now.",
        tags=["demo"],
    )
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    client = SkillLoadingClient()
    engine = QueryEngine(
        llm_client=client,
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
    )

    result = asyncio.run(engine.ask_complete_async("请使用 demo-skill 完成这个任务"))

    assert result.text == "skill applied"
    assert result.turns == 2
    assert client.calls == 2


def test_query_integrates_context_compression_and_detailed_usage(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    config.llm.max_tokens = 100
    config.memory.compression_reserve_tokens = 20
    config.memory.compression_threshold = 0.6
    client = UsageAndCompressionClient()
    engine = QueryEngine(
        llm_client=client,
        tool_registry=ToolRegistry(),
        config=config,
        cwd=str(tmp_path),
    )
    history = []
    for index in range(8):
        history.extend(
            [
                Message(role="user", content=f"old request {index} " + "x" * 220),
                Message(role="assistant", content=f"old answer {index} " + "y" * 220),
            ]
        )

    result = asyncio.run(engine.ask_complete_async("new request", history=history))

    assert result.text == "compressed"
    assert client.saw_summary
    assert result.total_tokens == 128
    assert result.usage.cache_hit_tokens == 20
    assert result.usage.cache_miss_tokens == 100
