from __future__ import annotations

import asyncio

from agentcli.context import ContextBudget, ContextWindowManager, estimate_text_tokens
from agentcli.types import Message


def _budget(**overrides) -> ContextBudget:
    # A 10k-token workspace: protected 2k, clear at 5k (if 1k is clearable), summarize at 8k.
    values = {"context_window": 12_500, "max_output_tokens": 0, "reserve_tokens": 0}
    values.update(overrides)
    return ContextBudget(**values)


def _chat(start: int, stop: int, size: int = 600) -> list[Message]:
    messages = []
    for index in range(start, stop):
        messages.extend(
            [
                Message(role="user", content=f"request {index} " + "x" * size),
                Message(role="assistant", content=f"answer {index} " + "y" * size),
            ]
        )
    return messages


def test_budget_scales_with_the_model_but_stops_at_the_workspace_cap():
    big = ContextBudget(context_window=1_050_000, max_output_tokens=8_192)
    assert big.workspace == 200_000
    assert (big.protect_tokens, big.clear_limit, big.clear_min, big.summarize_limit) == (
        40_000,
        100_000,
        20_000,
        160_000,
    )
    small = ContextBudget(context_window=128_000, max_output_tokens=8_192)
    assert small.workspace == int((128_000 - 8_192 - 1_024) * 0.8)
    assert small.protect_tokens == int(small.workspace * 0.2)


def test_under_the_first_threshold_nothing_changes():
    manager = ContextWindowManager(_budget())
    messages = _chat(0, 5)

    result = manager.prepare(messages)

    assert not result.compressed
    assert result.messages == messages


def test_summary_keeps_the_protected_part_and_the_latest_request_verbatim():
    manager = ContextWindowManager(_budget())
    messages = _chat(0, 30)  # ~12k tokens of plain conversation, nothing to clear

    result = manager.prepare(messages)

    assert result.method == "extractive"
    assert "conversation-summary" in str(result.messages[0].content)
    kept = [m.content for m in result.messages if str(m.content).startswith("request")]
    assert kept[-1].startswith("request 29")
    assert not any(k.startswith("request 0 ") for k in kept)
    kept_tokens = sum(estimate_text_tokens(str(m.content)) for m in result.messages[2:])
    assert kept_tokens <= _budget().protect_tokens + 400  # about the protected part
    assert result.estimated_tokens_after < _budget().clear_limit


def test_split_never_separates_a_tool_call_from_its_result():
    manager = ContextWindowManager(_budget())
    messages = _chat(0, 20)
    for index in range(8):
        call_id = f"call_{index}"
        messages += [
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{}"},
                    }
                ],
            ),
            Message(role="tool", content="ok " * 30, tool_call_id=call_id),
        ]

    result = manager.prepare(messages)

    ids = {c["id"] for m in result.messages for c in m.tool_calls}
    for message in result.messages:
        if message.role == "tool":
            assert message.tool_call_id in ids


def test_rolling_summary_gets_the_previous_summary():
    manager = ContextWindowManager(_budget())
    seen: list[str] = []

    async def summarizer(older, previous):
        seen.append(previous)
        return f"summary {len(seen)}"

    first = asyncio.run(manager.prepare_async(_chat(0, 30), summarizer=summarizer))
    assert first.method == "llm"
    second = asyncio.run(
        manager.prepare_async(first.messages + _chat(30, 60), summarizer=summarizer)
    )

    assert second.method == "llm"
    assert seen == ["", "summary 1"]
    assert "summary 2" in str(second.messages[0].content)


def test_summarizer_failure_falls_back_to_extractive():
    manager = ContextWindowManager(_budget())

    async def broken(older, previous):
        raise RuntimeError("down")

    result = asyncio.run(manager.prepare_async(_chat(0, 30), summarizer=broken))

    assert result.method == "extractive"
    assert "request 0" in str(result.messages[0].content)


def test_manual_compact_keeps_only_the_last_few_messages():
    manager = ContextWindowManager(ContextBudget(context_window=200_000, max_output_tokens=1_000))

    result = asyncio.run(manager.prepare_async(_chat(0, 10), force=True))

    kept = [m.content[:9] for m in result.messages if str(m.content).startswith("request")]
    assert kept == ["request 7", "request 8", "request 9"]


def test_truncate_strategy_drops_without_summarizing():
    manager = ContextWindowManager(_budget(), strategy="truncate")

    result = asyncio.run(manager.prepare_async(_chat(0, 30)))

    assert result.method == "truncate"
    assert not any("conversation-summary" in str(m.content) for m in result.messages)
    assert str(result.messages[-1].content).startswith("answer 29")


def test_message_count_is_only_a_safety_valve():
    manager = ContextWindowManager(
        ContextBudget(context_window=1_000_000, max_output_tokens=1_000), max_history_messages=40
    )
    result = manager.prepare(_chat(0, 30, size=5))  # 60 tiny messages

    assert result.compressed
    assert len(result.messages) <= 40
