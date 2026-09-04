from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from agentcli.agent import Agent
from agentcli.config import AgentCliConfig, LlmConfig, load_config
from agentcli.llm import calculate_cost, create_llm_client, get_builtin_price_profile
from agentcli.llm.openai_compatible import OpenAICompatibleClient
from agentcli.tools import ToolRegistry
from agentcli.types import Message, Usage


def test_streaming_payload_requests_usage() -> None:
    client = _client()

    payload = client._build_payload(
        [Message(role="user", content="hello")],
        [],
        system_prompt="system",
    )

    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}


def test_connection_error_becomes_recoverable_error_event(monkeypatch) -> None:
    def fail_stream(*args, **kwargs):
        request = httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions")
        raise httpx.ConnectError("temporary connection failure", request=request)

    monkeypatch.setattr(httpx.AsyncClient, "stream", fail_stream)

    events = asyncio.run(_collect_chat_events(_client()))

    assert [event["type"] for event in events] == ["message_start", "error"]
    assert "Could not connect to deepseek" in str(events[-1]["error"])
    assert "VPN/proxy" in str(events[-1]["error"])


def test_timeout_becomes_recoverable_error_event(monkeypatch) -> None:
    def fail_stream(*args, **kwargs):
        request = httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions")
        raise httpx.ReadTimeout("temporary timeout", request=request)

    monkeypatch.setattr(httpx.AsyncClient, "stream", fail_stream)

    events = asyncio.run(_collect_chat_events(_client()))

    assert [event["type"] for event in events] == ["message_start", "error"]
    assert "timed out after 120s" in str(events[-1]["error"])


def test_agent_propagates_connection_error_without_raising(tmp_path, monkeypatch) -> None:
    class NoopSnapshotService:
        def __init__(self, cwd):
            self.cwd = cwd

        def create(self, label):
            return None

    def fail_stream(*args, **kwargs):
        request = httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions")
        raise httpx.ConnectError("temporary connection failure", request=request)

    monkeypatch.setattr("agentcli.agent.agent.SnapshotService", NoopSnapshotService)
    monkeypatch.setattr(httpx.AsyncClient, "stream", fail_stream)
    config = AgentCliConfig()
    config.llm.api_key = "key"
    config.memory.long_term_db_path = str(tmp_path / "memory.db")
    agent = Agent(
        llm_client=_client(),
        tool_registry=ToolRegistry(),
        system_prompt="system",
        cwd=str(tmp_path),
        config=config,
    )

    events = asyncio.run(_collect_agent_events(agent))

    assert [event["type"] for event in events] == ["error"]
    assert "Could not connect to deepseek" in str(events[0]["error"])


def test_usage_only_chunk_without_choices_is_parsed() -> None:
    chunk = {
        "choices": [],
        "usage": {
            "prompt_tokens": 1_000,
            "completion_tokens": 200,
            "prompt_cache_hit_tokens": 600,
            "prompt_cache_miss_tokens": 400,
            "completion_tokens_details": {"reasoning_tokens": 50},
            "total_tokens": 1_200,
        },
    }

    events = asyncio.run(_collect_events(_client(), chunk))

    assert events == [
        {
            "type": "usage",
            "usage": {
                "input_tokens": 1_000,
                "output_tokens": 200,
                "cache_hit_tokens": 600,
                "cache_miss_tokens": 400,
                "reasoning_tokens": 50,
                "total_tokens": 1_200,
            },
        }
    ]


def test_usage_normalizes_old_and_provider_fields() -> None:
    legacy = Usage.from_mapping({"input_tokens": 10, "output_tokens": 5})
    provider = Usage.from_mapping(
        {
            "prompt_cache_hit_tokens": 7,
            "prompt_cache_miss_tokens": 3,
            "completion_tokens": 4,
            "reasoning_tokens": 2,
        }
    )

    assert legacy.total_tokens == 15
    assert legacy.to_dict()["input_tokens"] == 10
    assert provider.input_tokens == 10
    assert provider.output_tokens == 4
    assert provider.reasoning_tokens == 2
    assert provider.total_tokens == 14


def test_deepseek_v4_profiles_and_cost_formula() -> None:
    flash = get_builtin_price_profile("deepseek-v4-flash")
    pro = get_builtin_price_profile("deepseek-v4-pro")
    assert flash is not None
    assert pro is not None
    assert flash.context_window == 1_000_000
    assert pro.context_window == 1_000_000
    assert flash.for_currency("usd").input_cache_hit == 0.0028
    assert flash.for_currency("cny").input_cache_miss == 1.0
    assert pro.for_currency("usd").output == 0.87
    assert pro.for_currency("cny").output == 6.0

    usage = Usage(
        input_tokens=1_000,
        output_tokens=200,
        cache_hit_tokens=600,
        cache_miss_tokens=400,
        reasoning_tokens=50,
    )
    usd = calculate_cost(usage, flash, currency="usd")
    cny = calculate_cost(usage, flash, currency="cny")

    assert usd.input_cache_hit_cost == pytest.approx(0.00000168)
    assert usd.input_cache_miss_cost == pytest.approx(0.000056)
    assert usd.output_cost == pytest.approx(0.000056)
    assert usd.total_cost == pytest.approx(0.00011368)
    assert cny.total_cost == pytest.approx(0.000812)
    assert usd.reasoning_tokens == 50


def test_uncategorized_legacy_input_uses_cache_miss_price() -> None:
    profile = get_builtin_price_profile("deepseek-v4-flash")
    assert profile is not None

    cost = calculate_cost({"input_tokens": 1_000, "output_tokens": 0}, profile)

    assert cost.input_cache_hit_tokens == 0
    assert cost.input_cache_miss_tokens == 0
    assert cost.input_uncategorized_tokens == 1_000
    assert cost.input_uncategorized_cost == pytest.approx(0.00014)
    assert cost.total_cost == pytest.approx(0.00014)


def test_unknown_model_context_and_prices_can_be_configured(tmp_path) -> None:
    project_config = tmp_path / ".agentcli" / "config.json"
    project_config.parent.mkdir(parents=True)
    project_config.write_text(
        json.dumps(
            {
                "llm": {
                    "provider": "openai-compatible",
                    "model": "private-coder",
                    "base_url": "https://llm.example/v1",
                    "context_window": 321_000,
                    "prices": {
                        "usd": {
                            "input_cache_hit": 0.1,
                            "input_cache_miss": 0.5,
                            "output": 1.5,
                        },
                        "cny": {
                            "input_cache_hit": 0.7,
                            "input_cache_miss": 3.5,
                            "output": 10.5,
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(project_root=tmp_path, env={})
    client = create_llm_client(config.llm)

    assert client.max_context_window == 321_000
    assert client.price_profile is not None
    assert client.price_profile.as_of == "config"
    assert client.price_profile.for_currency("usd").output == 1.5
    assert client.calculate_cost(Usage(input_tokens=1_000), currency="usd").total_cost == (
        pytest.approx(0.0005)
    )


def test_deepseek_builtin_prices_allow_partial_config_override() -> None:
    client = create_llm_client(
        LlmConfig(
            provider="deepseek",
            model="deepseek-v4-flash",
            context_window=900_000,
            prices={"usd": {"output": 9.0}},
        )
    )

    assert client.max_context_window == 900_000
    assert client.price_profile is not None
    assert client.price_profile.context_window == 900_000
    assert client.price_profile.for_currency("usd").input_cache_miss == 0.14
    assert client.price_profile.for_currency("usd").output == 9.0


async def _collect_events(
    client: OpenAICompatibleClient,
    chunk: dict,
) -> list[dict]:
    return [event async for event in client._parse_chunk(chunk)]


async def _collect_chat_events(client: OpenAICompatibleClient) -> list[dict]:
    return [
        event
        async for event in client.chat(
            [Message(role="user", content="hello")],
            [],
            system_prompt="system",
        )
    ]


async def _collect_agent_events(agent: Agent) -> list[dict]:
    return [event async for event in agent.run("hello")]


def _client() -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        provider_name="deepseek",
        model="deepseek-v4-flash",
        api_key="key",
        base_url="https://api.deepseek.com/v1",
    )
