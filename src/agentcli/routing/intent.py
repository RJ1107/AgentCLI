"""Decide, before a request runs, whether it looks like a job for /plan or /team.

Layered so it always answers: built-in rules decide instantly when the signal is clear; only
unclear requests go to a smarter classifier (Jev, then a cheap model), each bounded by a
timeout, and any classifier that is disabled, slow, or down is skipped in favor of the rules'
own verdict. Swapping Jev in or out never changes whether a request can run.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Literal

import httpx

from agentcli.config import AgentCliConfig
from agentcli.types import Message

Level = Literal["simple", "moderate", "complex"]
Mode = Literal["react", "plan", "team"]
CONFIDENT = 0.75
JEV_URL = "https://api.typesafe.ai/v1/systemone"


@dataclass(slots=True)
class RouteDecision:
    level: Level
    mode: Mode
    confidence: float
    reasons: list[str] = field(default_factory=list)
    source: str = "rules"
    # Classifiers that were tried and skipped, with why; kept for /context and debugging.
    fallbacks: list[str] = field(default_factory=list)


_STEP_CUES = (
    "然后",
    "接着",
    "之后",
    "最后",
    "首先",
    "第一步",
    "第二步",
    "并且",
    "同时",
    "first",
    "then",
    "after that",
    "finally",
    "next,",
)
_SCOPE_CUES = (
    "整个",
    "全部",
    "所有",
    "项目",
    "架构",
    "重构",
    "迁移",
    "从零",
    "实现一个",
    "设计",
    "系统",
    "模块",
    "框架",
    "端到端",
    "refactor",
    "migrate",
    "architecture",
    "from scratch",
    "end-to-end",
    "implement a",
    "build a",
    "redesign",
    "entire",
    "whole codebase",
    "across",
)
_PARALLEL_CUES = (
    "分别",
    "并行",
    "各个",
    "每个模块",
    "每个文件",
    "parallel",
    "each of",
    "separately",
    "every module",
    "every file",
)
_QUESTION_CUES = (
    "是什么",
    "为什么",
    "怎么",
    "解释",
    "什么意思",
    "what is",
    "why ",
    "explain",
    "how do",
)


def rules_decision(message: str) -> RouteDecision:
    text = message.strip()
    lowered = text.lower()
    score, reasons = 0, []

    if len(text) > 400:
        score, reasons = score + 2, [*reasons, f"很长的需求（{len(text)} 字）"]
    elif len(text) > 150:
        score += 1

    steps = [cue for cue in _STEP_CUES if cue in lowered]
    if len(steps) >= 2:
        score, reasons = score + 2, [*reasons, "包含多个先后步骤"]
    elif steps:
        score += 1

    listed = len(re.findall(r"(?m)^\s*(?:\d+[.、)]|[-*•])\s+\S", text))
    if listed >= 3:
        score, reasons = score + 2, [*reasons, f"列了 {listed} 项要求"]

    scope = [cue for cue in _SCOPE_CUES if cue in lowered]
    if scope:
        score += min(len(scope), 3)
        reasons.append(f"范围较大（{'、'.join(scope[:3])}）")

    if len(text) < 80 and any(cue in lowered for cue in _QUESTION_CUES):
        score -= 2

    # "逐个/one by one" is sequential on purpose: only real fan-out cues count here.
    parallel = any(cue in lowered for cue in _PARALLEL_CUES)
    if parallel:
        score += 1
    if score >= 5:
        confidence = 0.85 if score >= 7 else 0.75
        mode: Mode = "team" if parallel else "plan"
        if parallel:
            reasons.append("可以拆成互相独立、能并行的部分")
        return RouteDecision("complex", mode, confidence, reasons)
    if score >= 3:
        return RouteDecision("moderate", "react", 0.5, reasons)
    return RouteDecision("simple", "react", 0.9 if score <= 0 else 0.8, reasons)


_CLASSIFIER_PROMPT = """You triage requests to a coding agent. Reply with JSON only:
{"level": "simple|moderate|complex", "mode": "react|plan|team", "reason": "<one short clause>"}
simple: a question or a change of one or two steps. moderate: several steps, one area.
complex: many steps or files, design work, or a large change; use "plan" for it, or "team"
when it splits into independent parts that can run in parallel. Otherwise mode is "react".
Write the reason in the user's language."""


async def llm_decision(client, message: str) -> RouteDecision:
    text = ""
    async for event in client.chat(
        [Message(role="user", content=message[:4000])], [], system_prompt=_CLASSIFIER_PROMPT
    ):
        if event.get("type") == "text_delta":
            text += str(event.get("text") or "")
        elif event.get("type") == "error":
            raise event["error"]
    match = re.search(r"\{.*\}", text, re.S)
    data = json.loads(match.group(0) if match else text)
    level = data.get("level") if data.get("level") in ("simple", "moderate", "complex") else None
    mode = data.get("mode") if data.get("mode") in ("react", "plan", "team") else "react"
    if level is None:
        raise ValueError(f"classifier gave no level: {text[:80]!r}")
    if level != "complex":
        mode = "react"
    reason = str(data.get("reason") or "").strip()
    return RouteDecision(level, mode, 0.8, [reason] if reason else [], source="model")


async def jev_decision(
    message: str,
    *,
    api_key: str,
    model: str = "jev-latest",
    transport: httpx.AsyncBaseTransport | None = None,
) -> RouteDecision:
    """Ask Jev three yes/no questions; probabilities are unambiguous, unlike score scales."""

    questions = {
        "trivial": {
            "type": "noul",
            "instructions": "The request is a question or a change that takes one or two steps",
        },
        "needs_plan": {
            "type": "noul",
            "instructions": "The request needs several steps or files, or design work, and "
            "is best planned before doing it",
        },
        "parallel": {
            "type": "noul",
            "instructions": "The work splits into independent parts that could be done in parallel",
        },
    }
    async with httpx.AsyncClient(transport=transport, timeout=10) as client:
        response = await client.post(
            JEV_URL,
            headers={"authorization": f"Bearer {api_key}"},
            json={"model": model, "state": message[:8000], "questions": questions},
        )
        response.raise_for_status()
        answers = response.json()["answers"]
    trivial = float(answers["trivial"]["noul"])
    needs_plan = float(answers["needs_plan"]["noul"])
    parallel = float(answers["parallel"]["noul"])
    if needs_plan >= 0.6 and needs_plan > trivial:
        mode: Mode = "team" if parallel >= 0.6 else "plan"
        reasons = [f"需要规划的概率 {needs_plan:.0%}"]
        if mode == "team":
            reasons.append(f"可并行的概率 {parallel:.0%}")
        return RouteDecision("complex", mode, needs_plan, reasons, source="jev")
    if trivial >= 0.6:
        return RouteDecision("simple", "react", trivial, source="jev")
    return RouteDecision("moderate", "react", 1 - abs(trivial - needs_plan), source="jev")


class IntentRouter:
    """Rules first; unclear requests go to the configured classifiers, rules as fallback."""

    def __init__(self, config: AgentCliConfig, tiers=None, *, jev_transport=None):
        self.config = config
        self.tiers = tiers
        self.jev_transport = jev_transport

    async def route(self, message: str) -> RouteDecision:
        decision = rules_decision(message)
        if decision.confidence >= CONFIDENT:
            return decision
        routing = self.config.routing
        for name in routing.classifiers:
            attempt, timeout = self._classifier(name)
            if attempt is None:
                decision.fallbacks.append(f"{name}: {timeout}")
                continue
            try:
                better = await asyncio.wait_for(attempt(message), timeout)
            except Exception as exc:  # noqa: BLE001 - any classifier failure falls through
                decision.fallbacks.append(f"{name}: {type(exc).__name__}")
                continue
            better.fallbacks = decision.fallbacks
            return better
        return decision

    def _classifier(self, name: str):
        routing = self.config.routing
        if name == "jev":
            api_key = os.environ.get("TYPESAFE_API_KEY", "")
            if not routing.jev_enabled:
                return None, "disabled"
            if not api_key:
                return None, "no TYPESAFE_API_KEY"
            return (
                lambda message: jev_decision(
                    message, api_key=api_key, model=routing.jev_model, transport=self.jev_transport
                ),
                routing.jev_timeout,
            )
        if name == "llm":
            if self.tiers is None:
                return None, "no model"
            client = self.tiers.classifier()
            return (lambda message: llm_decision(client, message)), routing.classifier_timeout
        return None, "unknown classifier"
