"""Experiment 6b: can Jev's routing get better by tuning its thresholds, and by fixing the rules?

Jev's three answers for each request (trivial / needs_plan / parallel, the questions in
routing/intent.py) are fetched once and cached raw. The thresholds that turn them into a
mode are then searched on one half of the requests and scored on the other half (two
folds, alternating within each label), so the reported accuracy is on unseen requests.

Also scores a narrower rules layer: the rules only settle requests that are plainly a
question or a small edit, everything else goes to the classifier.

    python evals/jev_intent_tune.py
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from intent_dataset import INTENTS

from agentcli.routing.intent import _QUESTION_CUES, JEV_URL, rules_decision

EVALS = Path(__file__).resolve().parent
RAW = EVALS / "fixtures" / "jev_intent_raw.json"
CACHE = EVALS / "fixtures" / "jev_intent_cache.json"
RESULTS = EVALS / "results" / "jev_intent_tune.json"

QUESTIONS = {
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


async def fetch_raw() -> dict:
    raw = json.loads(RAW.read_text(encoding="utf-8")) if RAW.exists() else {}
    headers = {"authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}"}
    async with httpx.AsyncClient(timeout=30, headers=headers) as client:
        for text, *_ in INTENTS:
            if text in raw:
                continue
            response = await client.post(
                JEV_URL, json={"model": "jev-latest", "state": text[:8000], "questions": QUESTIONS}
            )
            response.raise_for_status()
            raw[text] = {k: float(v["noul"]) for k, v in response.json()["answers"].items()}
    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_text(json.dumps(raw, ensure_ascii=False, indent=0), encoding="utf-8")
    return raw


def jev_mode(p: dict, plan_at: float, team_at: float) -> str:
    if p["needs_plan"] >= plan_at and p["needs_plan"] > p["trivial"]:
        return "team" if p["parallel"] >= team_at else "plan"
    return "react"


def folds() -> tuple[list[int], list[int]]:
    seen: dict[str, int] = {}
    halves: tuple[list[int], list[int]] = ([], [])
    for index, (_, _, mode, _) in enumerate(INTENTS):
        position = seen.get(mode, 0)
        seen[mode] = position + 1
        halves[position % 2].append(index)
    return halves


def accuracy(modes: list[str], indices) -> float:
    indices = list(indices)
    return sum(modes[i] == INTENTS[i][2] for i in indices) / len(indices)


def narrow_rules(text: str) -> str | None:
    """Settle only what the rules can see for sure: a short question."""

    lowered = text.lower().strip()
    if len(lowered) < 60 and (
        any(cue in lowered for cue in _QUESTION_CUES) or lowered.endswith(("?", "？", "吗"))
    ):
        return "react"
    return None


def main() -> None:
    raw = asyncio.run(fetch_raw())
    probs = [raw[text] for text, *_ in INTENTS]
    grid = [round(0.05 * i, 2) for i in range(6, 20)]
    pairs = list(itertools.product(grid, grid))

    def modes_for(plan_at, team_at):
        return [jev_mode(p, plan_at, team_at) for p in probs]

    report: dict = {"requests": len(INTENTS)}
    report["jev_current_thresholds"] = accuracy(modes_for(0.6, 0.6), range(len(INTENTS)))

    halves = folds()
    right = 0
    picked = []
    for train, test in ((halves[0], halves[1]), (halves[1], halves[0])):
        best = max(pairs, key=lambda pt: (accuracy(modes_for(*pt), train), -abs(pt[0] - 0.6)))
        picked.append(best)
        right += accuracy(modes_for(*best), test) * len(test)
    report["jev_tuned_cross_validated"] = right / len(INTENTS)
    report["jev_tuned_thresholds"] = picked

    # Confusion with the fold-chosen thresholds, pooled over the two test halves.
    tuned = [None] * len(INTENTS)
    for (train, test), (plan_at, team_at) in zip(
        ((halves[0], halves[1]), (halves[1], halves[0])), picked
    ):
        for i in test:
            tuned[i] = jev_mode(probs[i], plan_at, team_at)
    confusion: dict[str, dict[str, int]] = {}
    for mode, (_, _, want, _) in zip(tuned, INTENTS):
        confusion.setdefault(want, {}).setdefault(mode, 0)
        confusion[want][mode] += 1
    report["jev_tuned_confusion"] = confusion

    # Chains with the narrow rules layer in front.
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    settled = [narrow_rules(text) for text, *_ in INTENTS]
    report["narrow_rules_share"] = sum(s is not None for s in settled) / len(INTENTS)
    report["narrow_rules_accuracy"] = sum(
        s == want for s, (_, _, want, _) in zip(settled, INTENTS) if s is not None
    ) / max(1, sum(s is not None for s in settled))
    chains = {"old rules + luna": None}
    for name in ("luna", "deepseek"):
        chains[f"narrow rules + {name}"] = [
            s or cache[f"{name}|{text}"]["mode"] for s, (text, *_) in zip(settled, INTENTS)
        ]
    chains["narrow rules + jev (tuned, cross-validated)"] = [s or m for s, m in zip(settled, tuned)]
    chains.pop("old rules + luna")
    report["chains"] = {name: accuracy(modes, range(len(INTENTS))) for name, modes in chains.items()}
    report["old_rules_alone"] = accuracy(
        [rules_decision(text).mode for text, *_ in INTENTS], range(len(INTENTS))
    )

    RESULTS.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
