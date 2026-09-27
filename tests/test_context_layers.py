from __future__ import annotations

import os
import time

from agentcli.context import ContextBudget, ContextWindowManager, spill
from agentcli.context.spill import ToolResultStore
from agentcli.types import Message


def _budget() -> ContextBudget:
    # A 10k-token workspace: protected 2k, clear at 5k (if 1k is clearable), summarize at 8k.
    return ContextBudget(context_window=12_500, max_output_tokens=0, reserve_tokens=0)


def _tool_turn(index: int, payload: str, tool: str = "read_file") -> list[Message]:
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
                    "function": {"name": tool, "arguments": f'{{"path": "f{index}.py"}}'},
                }
            ],
        ),
        Message(role="tool", content=payload, tool_call_id=call_id),
        Message(role="assistant", content=f"read {index}"),
    ]


def test_layer_one_clears_only_old_tool_results_and_keeps_every_word_of_conversation(tmp_path):
    manager = ContextWindowManager(_budget(), store=ToolResultStore(tmp_path))
    messages = [m for i in range(12) for m in _tool_turn(i, "x" * 1_500)]  # ~6.5k tokens

    result = manager.prepare(messages)

    assert result.method == "clear"
    assert result.cleared_tool_results > 0
    assert result.estimated_tokens_after < _budget().clear_limit
    tools = [m for m in result.messages if m.role == "tool"]
    assert tools[0].content.startswith("[cleared read_file")
    assert "Run it again" in tools[0].content
    assert tools[-1].content == "x" * 1_500  # the protected part is untouched
    texts = [m.content for m in result.messages if m.role != "tool"]
    assert texts == [m.content for m in messages if m.role != "tool"]


def test_layer_one_waits_until_clearing_frees_enough():
    manager = ContextWindowManager(_budget())
    # Over the clear threshold, but the old tool results are tiny: clearing would break the
    # prompt cache for almost nothing, so nothing happens.
    messages = [m for i in range(3) for m in _tool_turn(i, "y" * 210)]
    messages += [Message(role="user", content="z" * 15_000)]

    result = manager.prepare(messages)

    assert not result.compressed


def test_cleared_command_output_is_saved_and_the_stub_points_to_it(tmp_path):
    manager = ContextWindowManager(_budget(), store=ToolResultStore(tmp_path))
    messages = [m for i in range(12) for m in _tool_turn(i, f"output {i} " + "o" * 1_500, "bash")]

    result = manager.prepare(messages)

    stub = next(m.content for m in result.messages if m.role == "tool")
    assert stub.startswith("[cleared bash")
    saved = stub.split("saved at ")[1].split("; read it")[0]
    assert os.path.exists(saved)
    with open(saved, encoding="utf-8") as handle:
        assert handle.read().startswith("output 0 ")


def test_a_new_oversized_result_is_cut_to_head_and_tail_before_the_model_sees_it(tmp_path):
    manager = ContextWindowManager(_budget(), store=ToolResultStore(tmp_path))
    output = "HEAD " + "m" * 30_000 + " TAIL"
    messages = _tool_turn(0, "small")[:2] + [
        Message(role="tool", content=output, tool_call_id="call_0")
    ]

    result = manager.prepare(messages)

    cut = result.messages[-1].content
    assert cut.startswith("HEAD ") and cut.endswith(" TAIL")
    assert "omitted from the middle" in cut
    assert len(cut) < len(output) // 5
    assert "saved at" in cut


def test_long_session_never_compacts_on_consecutive_calls():
    # Hysteresis: each layer frees a lot at once, so the next one is far away and a call
    # that just compacted is never followed by another compaction.
    manager = ContextWindowManager(_budget())
    messages: list[Message] = []
    flags: list[bool] = []
    methods: list[str] = []
    for index in range(200):
        messages.extend(_tool_turn(index, "r" * 900))
        messages.append(Message(role="user", content=f"note {index} " + "n" * 300))
        result = manager.prepare(messages)
        flags.append(result.compressed)
        methods.append(result.method)
        messages = result.messages
    assert not any(a and b for a, b in zip(flags, flags[1:], strict=False))
    assert methods.count("clear") > 0
    assert methods.count("extractive") > 0
    assert sum(flags) < 40


def test_saved_results_are_deleted_after_the_retention_period(tmp_path):
    old = tmp_path / "20260101-000000-1-abc" / "tool-results"
    old.mkdir(parents=True)
    (old / "x.txt").write_text("old", encoding="utf-8")
    ten_days_ago = time.time() - 10 * 86_400
    os.utime(old, (ten_days_ago, ten_days_ago))
    spill._cleaned = False

    ToolResultStore(tmp_path, retention_days=7).save("call_1", "new")

    assert not old.exists()
    assert old.parent.exists()  # the session folder itself has its own retention
    assert any(tmp_path.rglob("call_1.txt"))
