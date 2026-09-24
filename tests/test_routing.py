from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from agentcli.agent.orchestrator import (
    AgentMessage,
    AgentOrchestrator,
    AgentRole,
    ExecutionStep,
    StepStatus,
)
from agentcli.agent.plan_execute import PlanExecuteAgent
from agentcli.config import load_config
from agentcli.llm.factory import create_llm_client
from agentcli.plan import Planner
from agentcli.routing import IntentRouter, ModelTiers, parse_model_spec, rules_decision
from agentcli.tools import ToolRegistry

MIGRATION = (
    "帮我把整个项目从 Flask 重构迁移到 FastAPI：首先梳理所有路由，然后逐个迁移，"
    "接着改测试，最后更新部署脚本和文档"
)
REVIEW_ALL = "分别审查 agent、context、tools 三个模块的代码质量，每个模块给出问题清单，然后汇总成报告，涉及整个项目的架构"  # noqa: E501
# Rules score this 3 ("moderate", confidence 0.5): unsure, so a classifier is consulted.
UNCLEAR = "把登录模块改一下，加上验证码，然后注册页面也要同步修改，接着跑一遍测试"


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "level", "mode"),
    [
        ("这个函数是什么意思？", "simple", "react"),
        ("帮我修复 cart.py 里 subtotal 的 bug", "simple", "react"),
        (MIGRATION, "complex", "plan"),  # "逐个" is sequential, not parallel
        (REVIEW_ALL, "complex", "team"),
    ],
)
def test_rules_classify_clear_cases(message, level, mode):
    decision = rules_decision(message)
    assert (decision.level, decision.mode) == (level, mode)


# ---------------------------------------------------------------------------
# Classifier chain: Jev, then a model, then the rules; any may be down
# ---------------------------------------------------------------------------


def _config(tmp_path, **routing):
    config = load_config(project_root=tmp_path)
    config.llm.provider, config.llm.model, config.llm.api_key = "openrouter", "x/session", "k"
    for key, value in routing.items():
        setattr(config.routing, key, value)
    return config


def _jev(noul: dict[str, float] | None = None, status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if status != 200:
            return httpx.Response(status, text="down")
        body = json.loads(request.content)
        assert body["model"] == "jev-latest" and set(body["questions"]) == set(noul)
        answers = {name: {"type": "noul", "noul": value} for name, value in noul.items()}
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": answers})

    return httpx.MockTransport(handler)


class _Classifier:
    model_name = "classifier"
    provider_name = "fake"
    max_context_window = 100_000

    def __init__(self, reply: str | None):
        self.reply, self.calls = reply, 0

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        self.calls += 1
        if self.reply is None:
            yield {"type": "error", "error": RuntimeError("model down")}
            return
        yield {"type": "text_delta", "text": self.reply}


class _Tiers:
    def __init__(self, client):
        self.client = client

    def classifier(self):
        return self.client


def test_clear_requests_never_wait_for_a_classifier(tmp_path):
    model = _Classifier('{"level":"complex","mode":"plan"}')
    router = IntentRouter(_config(tmp_path), _Tiers(model))

    decision = asyncio.run(router.route(MIGRATION))

    assert decision.source == "rules" and model.calls == 0


def test_unclear_request_goes_to_jev_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "jev-key")
    router = IntentRouter(
        _config(tmp_path, jev_enabled=True),
        _Tiers(_Classifier(None)),
        jev_transport=_jev({"trivial": 0.1, "needs_plan": 0.9, "parallel": 0.2}),
    )

    decision = asyncio.run(router.route(UNCLEAR))

    assert (decision.source, decision.level, decision.mode) == ("jev", "complex", "plan")


def test_jev_outage_falls_back_to_the_model_then_the_rules(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "jev-key")
    down = _jev(status=503)

    model = _Classifier('{"level":"complex","mode":"team","reason":"三处改动可以并行"}')
    via_model = asyncio.run(
        IntentRouter(_config(tmp_path, jev_enabled=True), _Tiers(model), jev_transport=down).route(
            UNCLEAR
        )
    )
    assert via_model.source == "model" and via_model.mode == "team"
    assert via_model.fallbacks == ["jev: HTTPStatusError"]

    both_down = asyncio.run(
        IntentRouter(
            _config(tmp_path, jev_enabled=True), _Tiers(_Classifier(None)), jev_transport=down
        ).route(UNCLEAR)
    )
    assert both_down.source == "rules"
    assert both_down.fallbacks == ["jev: HTTPStatusError", "llm: RuntimeError"]


def test_jev_is_skipped_until_enabled_and_keyed(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    router = IntentRouter(_config(tmp_path, classifiers=["jev"]), None)
    assert asyncio.run(router.route(UNCLEAR)).fallbacks == ["jev: disabled"]

    router = IntentRouter(_config(tmp_path, classifiers=["jev"], jev_enabled=True), None)
    assert asyncio.run(router.route(UNCLEAR)).fallbacks == ["jev: no TYPESAFE_API_KEY"]


# ---------------------------------------------------------------------------
# Model tiers
# ---------------------------------------------------------------------------


def test_model_specs_keep_colons_inside_model_ids():
    assert parse_model_spec("openrouter:openai/gpt-6-sol", "deepseek") == (
        "openrouter",
        "openai/gpt-6-sol",
    )
    assert parse_model_spec("openai/gpt-6-luna:batch", "openrouter") == (
        "openrouter",
        "openai/gpt-6-luna:batch",
    )


def test_tiers_resolve_models_and_degrade_without_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config = _config(
        tmp_path,
        fast_model="openrouter:openai/gpt-6-luna",
        strong_model="openai/gpt-6-sol",
        top_model="deepseek:deepseek-v4-pro",
    )
    session = create_llm_client(config.llm)
    tiers = ModelTiers(config, session)

    assert tiers.worker("fast").model_name == "openai/gpt-6-luna"
    assert tiers.reviewer("fast").model_name == "openai/gpt-6-sol"  # one tier up
    assert tiers.planner().model_name == session.model_name  # top has no key: falls back
    assert tiers.warnings == [
        "no API key for deepseek; deepseek:deepseek-v4-pro falls back to the session model"
    ]  # noqa: E501
    assert ModelTiers(_config(tmp_path), session).worker("strong") is session  # unset: session


# ---------------------------------------------------------------------------
# Team escalation ladder
# ---------------------------------------------------------------------------


def _tiered_orchestrator(tmp_path, **routing):
    config = _config(
        tmp_path,
        fast_model="openrouter:m/fast",
        strong_model="openrouter:m/strong",
        top_model="openrouter:m/top",
        **routing,
    )
    session = create_llm_client(config.llm)
    return AgentOrchestrator(
        llm_client=session,
        tool_registry=ToolRegistry(),
        config=config,
        cwd=str(tmp_path),
        tiers=ModelTiers(config, session),
    )


class _Worker:
    """Produces work that only the models in `good` get right."""

    def __init__(self, good: set[str]):
        self.good, self.runs = good, []

    async def execute(self, task, context, *, mode):  # noqa: ARG002
        model = self.llm_client.model_name
        self.runs.append(model)
        verdict = "good" if model in self.good else "bad"
        return AgentMessage.result("worker", AgentRole.WORKER, f"{verdict} work by {model}")


class _Reviewer:
    def __init__(self):
        self.reviewed_by = []

    async def review(self, original_task, execution_result):  # noqa: ARG002
        self.reviewed_by.append(self.llm_client.model_name)
        approved = execution_result.startswith("good")
        body = {"approved": approved, "issues": [] if approved else ["tests still fail"]}
        return AgentMessage.result("reviewer", AgentRole.REVIEWER, json.dumps(body))

    def clear_history(self):
        return None


def test_easy_step_escalates_from_fast_to_strong_with_a_higher_reviewer(tmp_path):
    orchestrator = _tiered_orchestrator(tmp_path)
    step = ExecutionStep("step_1", "fix the bug", "ANALYSIS", [])
    steps = [step]
    worker, reviewer = _Worker(good={"m/strong"}), _Reviewer()

    asyncio.run(orchestrator._run_step(step, steps, {}, worker, reviewer))

    assert steps[0].status == StepStatus.COMPLETED and steps[0].model == "m/strong"
    assert worker.runs == ["m/fast", "m/fast", "m/strong"]
    # Nobody grades their own work: fast is reviewed by strong, strong by top.
    assert reviewer.reviewed_by == ["m/strong", "m/strong", "m/top"]


def test_escalation_is_bounded_and_hands_the_issues_back(tmp_path):
    orchestrator = _tiered_orchestrator(tmp_path)
    step = ExecutionStep("step_1", "design the migration", "ANALYSIS", [], difficulty="hard")
    steps = [step]
    worker = _Worker(good=set())

    asyncio.run(orchestrator._run_step(step, steps, {}, worker, _Reviewer()))

    # hard: strong x2, then one escalation to top x2, then stop.
    assert worker.runs == ["m/strong", "m/strong", "m/top", "m/top"]
    assert steps[0].status == StepStatus.FAILED
    assert "4 attempts (m/strong → m/top)" in steps[0].result
    assert "tests still fail" in steps[0].result


def test_plan_tasks_pick_their_model_by_difficulty(tmp_path):
    config = _config(tmp_path, fast_model="m/fast", strong_model="m/strong", planner_model="m/plan")
    session = create_llm_client(config.llm)
    agent = PlanExecuteAgent(
        llm_client=session,
        tool_registry=ToolRegistry(),
        config=config,
        cwd=str(tmp_path),
        tiers=ModelTiers(config, session),
    )
    plan = Planner(session).parse_plan(
        "goal",
        json.dumps(
            {
                "tasks": [
                    {"id": "a", "description": "read files", "difficulty": "easy"},
                    {"id": "b", "description": "design schema", "difficulty": "hard"},
                    {"id": "c", "description": "no label"},
                ]
            }
        ),
    )

    picked = {task.id: agent._task_client(task).model_name for task in plan.all_tasks()}

    assert picked == {"task_1": "m/fast", "task_2": "m/strong", "task_3": "m/fast"}
    assert agent.planner.llm_client.model_name == "m/plan"
    assert "困难任务" in plan.summarize()
