"""Shared pieces for the experiments in this folder.

Every run is isolated: its own copy of the fixture, its own AGENTCLI_HOME (so no user
config, memory, skills, or MCP servers leak in), approval off (the copy is disposable),
and memory, skills, MCP, and the code index disabled so only the variable under test
changes between runs.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
RUNS = EVALS / "_runs"
RESULTS = EVALS / "results"

# Isolate before agentcli reads any paths.
os.environ["AGENTCLI_HOME"] = str(EVALS / "_home")

from agentcli.agent.agent import Agent  # noqa: E402
from agentcli.bootstrap import build_tool_registry  # noqa: E402
from agentcli.config import AgentCliConfig, load_config  # noqa: E402
from agentcli.llm import (
    create_llm_client,  # noqa: E402
    openai_compatible,  # noqa: E402
)
from agentcli.types import Usage  # noqa: E402

# Usage per model for the run in progress. Runs execute concurrently in one process, and a
# ContextVar follows each run into the tasks it spawns (team workers, reviewers).
_TRACKER: ContextVar[dict[str, list] | None] = ContextVar("usage_by_model", default=None)
_original_chat = openai_compatible.OpenAICompatibleClient.chat


async def _tracked_chat(self, *args, **kwargs):
    async for event in _original_chat(self, *args, **kwargs):
        tracker = _TRACKER.get()
        if tracker is not None and event.get("type") == "usage":
            usage, calls = tracker.get(self.model, [Usage(), 0])
            tracker[self.model] = [usage + Usage.from_mapping(event.get("usage") or {}), calls + 1]
        yield event


openai_compatible.OpenAICompatibleClient.chat = _tracked_chat


def start_tracking() -> dict[str, list]:
    tracker: dict[str, list] = {}
    _TRACKER.set(tracker)
    return tracker


_PRICES: dict[str, tuple[float, float, float]] | None = None


def live_prices() -> dict[str, tuple[float, float, float]]:
    """OpenRouter's current (input, cached input, output) USD per token, fetched once."""

    global _PRICES
    if _PRICES is None:
        import httpx

        cache = EVALS / "_home" / "openrouter_prices.json"
        if cache.exists() and time.time() - cache.stat().st_mtime < 12 * 3600:
            raw = json.loads(cache.read_text(encoding="utf-8"))
        else:
            raw = httpx.get("https://openrouter.ai/api/v1/models", timeout=30).json()["data"]
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(raw), encoding="utf-8")
        _PRICES = {}
        for model in raw:
            pricing = model.get("pricing") or {}
            prompt = float(pricing.get("prompt") or 0)
            cached = float(pricing.get("input_cache_read") or prompt)
            _PRICES[model["id"]] = (prompt, cached, float(pricing.get("completion") or 0))
    return _PRICES


def live_cost(tracker: dict[str, list]) -> float:
    """Cost of a run at today's OpenRouter prices, model by model."""

    prices = live_prices()
    total = 0.0
    for model, (usage, _calls) in tracker.items():
        prompt, cached, output = prices.get(model, (0.0, 0.0, 0.0))
        hit = usage.cache_hit_tokens
        miss = max(0, usage.input_tokens - hit)
        total += miss * prompt + hit * cached + usage.output_tokens * output
    return total


def by_model(tracker: dict[str, list]) -> dict[str, dict[str, int]]:
    return {
        model: {"calls": calls, **usage_dict(usage)} for model, (usage, calls) in tracker.items()
    }


def make_config(model: str, workspace: Path, overrides: dict[str, Any] | None = None):
    base: dict[str, Any] = {
        "llm": {"provider": "openrouter", "model": model, "temperature": 0.2},
        "features": {"mcp": False, "skill": False, "memory": False, "code_index": False},
        "policy": {"hitl_mode": "never"},
        "routing": {"suggest_modes": False},
    }
    config = load_config(
        project_root=workspace,
        overrides=_merge(base, overrides or {}),
        env={"OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY", "")},
    )
    return config


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for key, value in b.items():
        out[key] = _merge(out[key], value) if isinstance(out.get(key), dict) else value
    return out


def fresh_workspace(fixture: Path, label: str) -> Path:
    target = RUNS / f"{label}-{uuid.uuid4().hex[:8]}"
    shutil.copytree(fixture, target, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    return target


async def build_agent(config: AgentCliConfig, workspace: Path, mode: str = "react") -> Agent:
    registry, _ = await build_tool_registry(config=config, cwd=str(workspace))
    client = create_llm_client(config.llm)
    return Agent(
        llm_client=client,
        tool_registry=registry,
        config=config,
        cwd=str(workspace),
        mode=mode,  # type: ignore[arg-type]
        max_turns=30,
    )


@dataclass
class TurnStats:
    text: str = ""
    usage: Usage = field(default_factory=Usage)
    model_calls: int = 0
    tool_calls: int = 0
    compressions: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    seconds: float = 0.0


async def run_turn(agent: Agent, message: str) -> TurnStats:
    """Send one message; tally usage from every model call, summaries included."""

    stats = TurnStats()
    started = time.monotonic()
    done_usage: Usage | None = None
    async for event in agent.run(message):
        kind = event.get("type")
        if kind == "text_delta":
            stats.text += str(event.get("text") or "")
        elif kind == "usage":
            stats.usage = stats.usage + Usage.from_mapping(event.get("usage") or {})
            if event.get("phase") != "summary":
                stats.model_calls += 1
        elif kind == "tool_call":
            stats.tool_calls += 1
        elif kind == "context_compressed":
            stats.compressions.append(str(event.get("method") or "?"))
        elif kind == "error":
            stats.errors.append(repr(event.get("error"))[:300])
        elif kind == "done":
            done_usage = Usage.from_mapping(event.get("usage") or {})
            stats.model_calls = max(stats.model_calls, int(event.get("total_turns") or 0))
    # Team and plan runs do not forward per-call usage events; their done event has the total.
    if done_usage is not None and done_usage.total_tokens > stats.usage.total_tokens:
        stats.usage = done_usage
    stats.seconds = time.monotonic() - started
    return stats


def cost_usd(agent: Agent, usage: Usage) -> float:
    try:
        return float(agent.llm_client.calculate_cost(usage, currency="usd").total_cost)
    except (AttributeError, KeyError, TypeError, ValueError):
        return 0.0


def record(experiment: str, row: dict[str, Any]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **row}
    with (RESULTS / f"{experiment}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def usage_dict(usage: Usage) -> dict[str, int]:
    data = asdict(usage) if hasattr(usage, "__dataclass_fields__") else usage.to_dict()
    return {k: int(v) for k, v in data.items() if isinstance(v, int)}
