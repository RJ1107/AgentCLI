from __future__ import annotations

import asyncio

from agentcli.context import ContextBudget, ContextWindowManager
from agentcli.types import Message


def _tool_turn(index: int, payload: str) -> list[Message]:
    call_id = f"call_{index}"
    return [
        Message(role="user", content=f"step {index}"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        ),
        Message(role="tool", content=payload, tool_call_id=call_id),
        Message(role="assistant", content=f"read {index}"),
    ]


def _chat(start: int, stop: int, size: int = 300) -> list[Message]:
    messages = []
    for index in range(start, stop):
        messages.extend(
            [
                Message(role="user", content=f"request {index} " + "x" * size),
                Message(role="assistant", content=f"answer {index} " + "y" * size),
            ]
        )
    return messages


def test_microcompact_clears_old_tool_results_before_summarizing():
    messages = []
    for index in range(4):
        messages.extend(_tool_turn(index, "z" * 3000))
    manager = ContextWindowManager(
        ContextBudget(4_000, 200, 0.8, 0.55, 100), keep_recent_tool_results=1
    )

    result = manager.prepare(messages)

    assert result.method == "microcompact"
    assert result.cleared_tool_results == 3
    assert result.summarized_messages == 0
    # Every user request and assistant reply survives verbatim; only old tool output is stubbed.
    assert [m.content for m in result.messages if m.role == "user"] == [
        f"step {i}" for i in range(4)
    ]
    tool_contents = [str(m.content) for m in result.messages if m.role == "tool"]
    assert all("read_file result cleared" in text for text in tool_contents[:3])
    assert tool_contents[3] == "z" * 3000


def test_llm_summary_is_used_and_rolled_forward():
    calls: list[str] = []

    async def summarizer(_older, previous):
        calls.append(previous)
        return f"summary v{len(calls)}"

    manager = ContextWindowManager(
        ContextBudget(2_000, 100, 0.6, 0.4, 50),
        min_recent_messages=2,
        min_llm_summary_tokens=0,
    )

    first = asyncio.run(manager.prepare_async(_chat(0, 10), summarizer=summarizer))
    assert first.method == "llm"
    assert "summary v1" in str(first.messages[0].content)

    grown = [*first.messages, *_chat(10, 16)]
    second = asyncio.run(manager.prepare_async(grown, summarizer=summarizer))

    assert second.method == "llm"
    # The previous summary is handed over as-is instead of being re-summarized as a line.
    assert calls[1] == "summary v1"
    assert "summary v2" in str(second.messages[0].content)
    assert sum("conversation-summary" in str(m.content) for m in second.messages) == 1


def test_llm_summary_failure_falls_back_to_extractive():
    async def broken(_older, _previous):
        raise RuntimeError("summarizer down")

    manager = ContextWindowManager(
        ContextBudget(900, 150, 0.6, 0.4, 50),
        min_recent_messages=4,
        min_llm_summary_tokens=0,
    )

    result = asyncio.run(manager.prepare_async(_chat(0, 8, 240), summarizer=broken))

    assert result.method == "extractive"
    assert "request 0" in str(result.messages[0].content)


def test_small_older_part_skips_the_llm_call():
    async def summarizer(_older, _previous):
        raise AssertionError("a tiny older part is not worth a model call")

    manager = ContextWindowManager(ContextBudget(900, 150, 0.6, 0.4, 50), min_recent_messages=4)

    result = asyncio.run(manager.prepare_async(_chat(0, 8, 240), summarizer=summarizer))

    assert result.method == "extractive"
