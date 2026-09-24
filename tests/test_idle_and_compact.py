from __future__ import annotations

import asyncio

from agentcli.agent.agent import Agent
from agentcli.config import LlmConfig, load_config
from agentcli.entrypoints.repl import cache_reminder
from agentcli.llm.factory import create_llm_client
from agentcli.tools import ToolRegistry
from agentcli.types import Message

HOUR = 3600


def _config(tmp_path):
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    return config


def test_reminder_needs_both_a_long_pause_and_a_large_context(tmp_path):
    config = _config(tmp_path)  # defaults: 60 min, 30k tokens, stale after 8 h

    assert cache_reminder(idle_seconds=None, tokens=200_000, config=config) is None
    assert cache_reminder(idle_seconds=50 * 60, tokens=200_000, config=config) is None
    assert cache_reminder(idle_seconds=3 * HOUR, tokens=20_000, config=config) is None

    cold = cache_reminder(idle_seconds=90 * 60, tokens=150_000, config=config)
    stale = cache_reminder(idle_seconds=9 * HOUR, tokens=150_000, config=config)
    assert cold.level == "cold" and stale.level == "stale"


def test_reminder_prices_the_re_read_when_the_model_has_a_price_table(tmp_path):
    config = _config(tmp_path)
    deepseek = create_llm_client(
        LlmConfig(provider="deepseek", model="deepseek-v4-flash", api_key="x")
    )
    openrouter = create_llm_client(
        LlmConfig(provider="openrouter", model="openai/gpt-6-sol", api_key="x")
    )
    unknown = create_llm_client(
        LlmConfig(provider="openrouter", model="someone/unlisted-model", api_key="x")
    )

    def extra(client) -> str:
        return cache_reminder(
            idle_seconds=2 * HOUR, tokens=150_000, config=config, llm_client=client
        ).extra_cost

    # 150k tokens. DeepSeek direct: ¥1.00/M uncached vs ¥0.02/M cached.
    assert extra(deepseek) == "¥0.15"
    # GPT-6 Sol via OpenRouter's catalog: $2.00/M vs $0.20/M.
    assert extra(openrouter) == "$0.27"
    assert extra(unknown) == ""


class _SummarizingClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 1_000_000

    def __init__(self):
        self.prompts: list[str] = []

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        self.prompts.append(str(messages[-1].content))
        yield {"type": "text_delta", "text": "## User requests\n- migrate the database"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


def test_manual_compact_summarizes_even_a_small_conversation(tmp_path):
    client = _SummarizingClient()
    agent = Agent(
        llm_client=client, tool_registry=ToolRegistry(), config=_config(tmp_path), cwd=str(tmp_path)
    )
    for index in range(6):
        agent.history.extend(
            [
                Message(role="user", content=f"request {index}"),
                Message(role="assistant", content=f"answer {index}"),
            ]
        )

    result = asyncio.run(agent.compact("数据库迁移的决策"))

    assert result.method == "llm"  # far under every threshold, but the user asked
    assert "conversation-summary" in str(agent.history[0].content)
    assert "migrate the database" in str(agent.history[0].content)
    assert "数据库迁移的决策" in client.prompts[-1]  # the focus reached the summarizer
    assert agent.history[-1].content == "answer 5"  # recent turns stay verbatim
    assert agent.last_active_at is not None


def test_compact_with_no_conversation_does_nothing(tmp_path):
    agent = Agent(
        llm_client=_SummarizingClient(),
        tool_registry=ToolRegistry(),
        config=_config(tmp_path),
        cwd=str(tmp_path),
    )

    assert asyncio.run(agent.compact()) is None
