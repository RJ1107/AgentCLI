"""Experiment 6: which classifier should settle the requests the routing rules are unsure of?

Data: evals/intent_dataset.py (labelled requests). For every request we record the rules'
verdict and each classifier's own verdict, then score two things:

  classifier alone   its verdict on every request, to compare the classifiers directly
  chain              what the router really does: the rules when they are confident
                     (>= CONFIDENT), otherwise the classifier

Classifiers: Jev (the three nouls in routing/intent.py), and the LLM classifier prompt on
GPT-6 Luna and DeepSeek V4.1 Flash. Answers are cached in fixtures/jev_intent_cache.json.

    python evals/jev_intent.py          # needs TYPESAFE_API_KEY and OPENROUTER_API_KEY
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from intent_dataset import INTENTS

from agentcli.routing.intent import _CLASSIFIER_PROMPT, CONFIDENT, jev_decision, rules_decision

EVALS = Path(__file__).resolve().parent
CACHE = EVALS / "fixtures" / "jev_intent_cache.json"
RESULTS = EVALS / "results" / "jev_intent.json"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LLMS = {"luna": "openai/gpt-6-luna", "deepseek": "deepseek/deepseek-v4.1-flash"}
LLM_BUDGET_USD = 0.5


async def ask_llm(client: httpx.AsyncClient, model: str, text: str) -> dict:
    started = time.perf_counter()
    response = await client.post(
        OPENROUTER_URL,
        json={
            "model": model,
            "temperature": 0,
            "max_tokens": 1500,
            "usage": {"include": True},
            "messages": [
                {"role": "system", "content": _CLASSIFIER_PROMPT},
                {"role": "user", "content": text[:4000]},
            ],
        },
    )
    response.raise_for_status()
    body = response.json()
    seconds = time.perf_counter() - started
    content = body["choices"][0]["message"]["content"] or ""
    cost = float((body.get("usage") or {}).get("cost") or 0)
    try:
        data = json.loads(re.search(r"\{.*\}", content, re.S).group(0))
        level = data.get("level") if data.get("level") in ("simple", "moderate", "complex") else None
        mode = data.get("mode") if data.get("mode") in ("react", "plan", "team") else "react"
    except (AttributeError, json.JSONDecodeError):
        level, mode = None, "react"
    if level != "complex":
        mode = "react"
    return {"level": level, "mode": mode, "seconds": seconds, "cost": cost}


async def ask_jev(text: str) -> dict:
    started = time.perf_counter()
    decision = await jev_decision(text, api_key=os.environ["TYPESAFE_API_KEY"])
    return {
        "level": decision.level,
        "mode": decision.mode,
        "confidence": decision.confidence,
        "seconds": time.perf_counter() - started,
        "cost": 0.0,
    }


async def collect() -> dict:
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    spent = sum(v.get("cost", 0) for v in cache.values())
    client = httpx.AsyncClient(
        timeout=60, headers={"authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    )
    try:
        for index, (text, *_rest) in enumerate(INTENTS):
            for name in ("jev", *LLMS):
                key = f"{name}|{text}"
                if key in cache:
                    continue
                if name != "jev" and spent >= LLM_BUDGET_USD:
                    raise SystemExit(f"OpenRouter budget ${LLM_BUDGET_USD} reached")
                for attempt in range(3):
                    try:
                        result = await (ask_jev(text) if name == "jev" else ask_llm(client, LLMS[name], text))
                        break
                    except httpx.HTTPError:
                        if attempt == 2:
                            raise
                        await asyncio.sleep(2 * (attempt + 1))
                cache[key] = result
                spent += result["cost"]
            if index % 10 == 9:
                CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
                print(f"  {index + 1}/{len(INTENTS)}  OpenRouter so far ${spent:.4f}", flush=True)
    finally:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
        await client.aclose()
    return cache


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * len(ordered)) - 1))] if ordered else 0.0


def score(verdicts: list[tuple[str | None, str]]) -> dict:
    """verdicts[i] = (level, mode) for INTENTS[i]."""

    n = len(INTENTS)
    mode_right = level_right = 0
    confusion: dict[str, dict[str, int]] = {}
    complex_missed = complex_total = simple_escalated = simple_total = 0
    for (level, mode), (_, want_level, want_mode, _) in zip(verdicts, INTENTS):
        mode_right += mode == want_mode
        level_right += level == want_level
        confusion.setdefault(want_mode, {}).setdefault(mode, 0)
        confusion[want_mode][mode] += 1
        if want_level == "complex":
            complex_total += 1
            complex_missed += mode == "react"
        if want_level == "simple":
            simple_total += 1
            simple_escalated += mode != "react"
    return {
        "mode_accuracy": mode_right / n,
        "level_accuracy": level_right / n,
        "complex_sent_to_react": complex_missed / complex_total,
        "simple_sent_to_plan_or_team": simple_escalated / simple_total,
        "confusion": confusion,
    }


def main() -> None:
    cache = asyncio.run(collect())
    rules = [rules_decision(text) for text, *_ in INTENTS]
    confident = [d.confidence >= CONFIDENT for d in rules]
    report: dict = {
        "requests": len(INTENTS),
        "rules_confident_share": sum(confident) / len(INTENTS),
        "rules_confident_accuracy": (
            sum(d.mode == want for d, ok, (_, _, want, _) in zip(rules, confident, INTENTS) if ok)
            / max(1, sum(confident))
        ),
        "alone": {"rules": score([(d.level, d.mode) for d in rules])},
        "chain": {"rules only": score([(d.level, d.mode) for d in rules])},
        "latency": {},
        "cost_usd": {},
    }
    for name in ("jev", *LLMS):
        answers = [cache[f"{name}|{text}"] for text, *_ in INTENTS]
        report["alone"][name] = score([(a["level"], a["mode"]) for a in answers])
        chain = [
            (d.level, d.mode) if ok else (a["level"], a["mode"])
            for d, ok, a in zip(rules, confident, answers)
        ]
        report["chain"][f"rules + {name}"] = score(chain)
        seconds = [a["seconds"] for a in answers]
        report["latency"][name] = {"p50_s": percentile(seconds, 0.5), "p95_s": percentile(seconds, 0.95)}
        report["cost_usd"][name] = sum(a["cost"] for a in answers)
    report["disagreements"] = [
        {
            "request": text[:80],
            "want": f"{want_level}/{want_mode}",
            "rules": f"{d.level}/{d.mode} ({d.confidence:.2f})",
            **{name: f"{cache[f'{name}|{text}']['level']}/{cache[f'{name}|{text}']['mode']}" for name in ("jev", *LLMS)},
        }
        for d, (text, want_level, want_mode, _) in zip(rules, INTENTS)
        if any(cache[f"{name}|{text}"]["mode"] != want_mode for name in ("jev", *LLMS)) or d.mode != want_mode
    ]
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{report['requests']} requests; rules confident on {report['rules_confident_share']:.0%} "
          f"(right {report['rules_confident_accuracy']:.0%} of those)\n")
    for section in ("alone", "chain"):
        print(f"{section}:")
        for name, row in report[section].items():
            print(f"  {name:18} mode {row['mode_accuracy']:.3f}  level {row['level_accuracy']:.3f}  "
                  f"complex->react {row['complex_sent_to_react']:.2f}  "
                  f"simple->plan/team {row['simple_sent_to_plan_or_team']:.2f}")
    for name in ("jev", *LLMS):
        lat = report["latency"][name]
        print(f"  {name:10} p50 {lat['p50_s']:.2f}s  p95 {lat['p95_s']:.2f}s  cost ${report['cost_usd'][name]:.4f}")


if __name__ == "__main__":
    main()
