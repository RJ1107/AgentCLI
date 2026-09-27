"""Experiment 4: how well long-term memory is recalled, and where to set its knobs.

Dataset: evals/memory_dataset.py (80 memories of a made-up shop backend, 119 labelled
questions). Keywords for each memory are written once by a model with the same instruction
save_memory gives the agent, without seeing the questions, and cached in
fixtures/memory_keywords.json; everything else runs offline, no model calls.

Compared:
  legacy        the old SQLite scorer (word/character overlap + weighted importance etc.)
  bm25          BM25 on title and content only
  bm25+kw       BM25 with the saved keywords
  then a grid over the importance and recency nudges, and over the coverage gates for
  tier 2 (automatic recall: precision first) and tier 3 (search_memory: recall first).

    python evals/memory_recall.py --generate-keywords   # once, ~$0.01 with GPT-6 Luna
    python evals/memory_recall.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memory_dataset import MEMORIES, QUERIES

from agentcli.memory import MemoryRecord, rank

EVALS = Path(__file__).resolve().parent
KEYWORDS = EVALS / "fixtures" / "memory_keywords.json"
RESULTS = EVALS / "results" / "memory_recall.json"
NOW = datetime(2026, 9, 27, tzinfo=UTC)

KEYWORD_PROMPT = """For each memory below, write the 5-10 search terms someone might use to look it
up later: synonyms, abbreviations, and both Chinese and English names (for a database port:
数据库, DB, database, 端口, port). Reply with only a JSON object mapping each id to a list
of strings.

{memories}"""


async def generate_keywords() -> None:
    import os

    os.environ.setdefault("AGENTCLI_HOME", str(EVALS / "_home"))
    from agentcli.config import load_config
    from agentcli.llm import create_llm_client
    from agentcli.types import Message

    config = load_config(
        overrides={
            "llm": {"provider": "openrouter", "model": "openai/gpt-6-luna", "temperature": 0}
        },
        env={"OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY", "")},
    )
    client = create_llm_client(config.llm)
    listing = "\n".join(f"{mid}: {title} — {content}" for mid, _, _, _, title, content in MEMORIES)
    text = ""
    async for event in client.chat(
        [Message(role="user", content=KEYWORD_PROMPT.format(memories=listing))],
        [],
        system_prompt="You write search keywords. Output JSON only.",
    ):
        if event.get("type") == "text_delta":
            text += event.get("text") or ""
        elif event.get("type") == "usage":
            print("usage", event.get("usage"))
        elif event.get("type") == "error":
            raise event["error"]
    data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    KEYWORDS.parent.mkdir(parents=True, exist_ok=True)
    KEYWORDS.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote keywords for {len(data)} memories to {KEYWORDS}")


def records(with_keywords: bool) -> list[MemoryRecord]:
    keywords = json.loads(KEYWORDS.read_text(encoding="utf-8")) if with_keywords else {}
    return [
        MemoryRecord(
            name=mid,
            title=title,
            content=content,
            kind=kind,
            importance=importance,
            keywords=keywords.get(mid, []),
            updated=(NOW - timedelta(days=days)).isoformat(),
        )
        for mid, kind, importance, days, title, content in MEMORIES
    ]


# ---------------------------------------------------------------- the old scorer, for comparison

_WORD = re.compile(r"[a-z0-9]+")
_CJK = re.compile(r"[一-鿿]+")


def _legacy_features(text: str) -> set[str]:
    text = " ".join(text.lower().split())
    features = set(_WORD.findall(text))
    for run in _CJK.findall(text):
        features.update(run)
        features.update(run[i : i + 2] for i in range(len(run) - 1))
        if len(run) > 1:
            features.add(run)
    return features


def legacy_rank(query: str, items: list[MemoryRecord]) -> list[tuple[MemoryRecord, float]]:
    query_features = _legacy_features(query)
    ranked = []
    for item in items:
        content = f"{item.title} {item.content}"
        overlap = query_features & _legacy_features(content)
        substring = " ".join(query.lower().split()) in content.lower()
        if not overlap and not substring:
            continue
        lexical = min(1.0, 0.8 * len(overlap) / len(query_features) + (0.2 if substring else 0))
        days = (NOW - datetime.fromisoformat(item.updated)).days
        score = 0.72 * lexical + 0.12 * item.importance + 0.08 + 0.06 / (1 + days / 30)
        if score >= 0.05:
            ranked.append((item, score))
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked


# ---------------------------------------------------------------- metrics


def evaluate(rank_fn, *, k: int = 3) -> dict[str, float]:
    """Ranking quality on the answerable questions: recall@k, precision@k, MRR."""

    recalls, precisions, reciprocal = [], [], []
    for text, relevant, _kind in QUERIES:
        if not relevant:
            continue
        ranked = [name for name, _ in rank_fn(text)]
        top = ranked[:k]
        hits = len(set(top) & set(relevant))
        recalls.append(hits / len(relevant))
        precisions.append(hits / max(1, len(top)) if top else 0.0)
        first = next((i for i, name in enumerate(ranked) if name in relevant), None)
        reciprocal.append(1 / (first + 1) if first is not None else 0.0)
    return {
        f"recall@{k}": _mean(recalls),
        f"precision@{k}": _mean(precisions),
        "mrr": _mean(reciprocal),
    }


def gated(rank_fn, gate: float, k: int) -> dict[str, float]:
    """What an injection policy (top k, coverage >= gate) puts in front of the model."""

    injected = relevant_injected = relevant_total = false_on_none = none_total = 0
    for text, relevant, _kind in QUERIES:
        chosen = [name for name, coverage in rank_fn(text) if coverage >= gate][:k]
        injected += len(chosen)
        relevant_injected += len(set(chosen) & set(relevant))
        relevant_total += len(relevant)
        if not relevant:
            none_total += 1
            false_on_none += bool(chosen)
    precision = relevant_injected / injected if injected else 1.0
    recall = relevant_total and relevant_injected / relevant_total
    return {
        "gate": gate,
        "precision": precision,
        "recall": recall,
        "f0.5": _f(precision, recall, 0.5),
        "f2": _f(precision, recall, 2.0),
        "injected_per_query": injected / len(QUERIES),
        "false_recall_on_unrelated": false_on_none / none_total,
    }


def by_type(rank_fn, k: int = 3) -> dict[str, float]:
    out = {}
    for kind in ("direct", "paraphrase", "english", "multi"):
        subset = [q for q in QUERIES if q[2] == kind]
        hits = total = 0
        for text, relevant, _ in subset:
            top = [name for name, _ in rank_fn(text)][:k]
            hits += len(set(top) & set(relevant))
            total += len(relevant)
        out[kind] = hits / total if total else 0.0
    return out


def _f(precision: float, recall: float, beta: float) -> float:
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-keywords", action="store_true")
    args = parser.parse_args()
    if args.generate_keywords:
        asyncio.run(generate_keywords())
        return

    plain, rich = records(False), records(True)

    def bm25(items, iw=0.0, rw=0.0):
        return lambda q: [
            (hit.record.name, hit.coverage)
            for hit in rank(q, items, importance_weight=iw, recency_weight=rw, now=NOW)
        ]

    def legacy(q):
        return [(item.name, score) for item, score in legacy_rank(q, plain)]

    report: dict = {"queries": len(QUERIES), "memories": len(MEMORIES)}
    report["methods"] = {
        "legacy": {**evaluate(legacy), **{"by_type": by_type(legacy)}},
        "bm25": {**evaluate(bm25(plain)), **{"by_type": by_type(bm25(plain))}},
        "bm25+kw": {**evaluate(bm25(rich)), **{"by_type": by_type(bm25(rich))}},
    }

    grid = []
    for iw in (0.0, 0.02, 0.05, 0.1, 0.2, 0.4):
        for rw in (0.0, 0.02, 0.05, 0.1, 0.2):
            grid.append(
                {"importance_weight": iw, "recency_weight": rw, **evaluate(bm25(rich, iw, rw))}
            )
    grid.sort(
        key=lambda row: (row["mrr"], row["recall@3"], -row["importance_weight"]), reverse=True
    )
    report["weight_grid_top"] = grid[:5]
    report["weight_grid_worst"] = grid[-3:]
    best = grid[0]

    ranker = bm25(rich, best["importance_weight"], best["recency_weight"])
    gates = [round(0.05 * i, 2) for i in range(0, 17)]
    tier2 = [gated(ranker, g, 3) for g in gates]
    tier3 = [gated(ranker, g, 8) for g in gates]
    report["tier2_sweep"] = tier2
    report["tier3_sweep"] = tier3
    report["tier2_choice"] = max(tier2, key=lambda row: (row["f0.5"], row["gate"]))
    report["tier3_choice"] = max(tier3, key=lambda row: (row["f2"], row["gate"]))
    # The old policy: top 6 of anything with an overlap (its min score let almost all through).
    report["legacy_policy"] = gated(lambda q: [(n, 1.0) for n, _ in legacy(q)], 0.0, 6)

    misses = []
    for text, relevant, kind in QUERIES:
        top = [name for name, _ in ranker(text)][:3]
        if relevant and not set(top) & set(relevant):
            misses.append({"query": text, "type": kind, "expected": relevant, "got": top})
    report["tier3_misses_at_3"] = misses

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    _print(report)


def _print(report: dict) -> None:
    print(f"{report['memories']} memories, {report['queries']} questions\n")
    print(f"{'method':10} {'recall@3':>9} {'prec@3':>7} {'MRR':>6}   by type (recall@3)")
    for name, row in report["methods"].items():
        types = "  ".join(f"{k} {v:.2f}" for k, v in row["by_type"].items())
        print(
            f"{name:10} {row['recall@3']:9.3f} {row['precision@3']:7.3f} {row['mrr']:6.3f}   {types}"
        )
    print("\nweight grid, best:")
    for row in report["weight_grid_top"]:
        print(
            f"  importance {row['importance_weight']:<4} recency {row['recency_weight']:<4} MRR {row['mrr']:.3f} recall@3 {row['recall@3']:.3f}"
        )
    print("weight grid, worst:")
    for row in report["weight_grid_worst"]:
        print(
            f"  importance {row['importance_weight']:<4} recency {row['recency_weight']:<4} MRR {row['mrr']:.3f} recall@3 {row['recall@3']:.3f}"
        )
    for tier in ("tier2", "tier3"):
        choice = report[f"{tier}_choice"]
        print(
            f"\n{tier} choice: gate {choice['gate']}  precision {choice['precision']:.3f}  recall {choice['recall']:.3f}  injected/query {choice['injected_per_query']:.2f}  false recall on unrelated {choice['false_recall_on_unrelated']:.2f}"
        )
    legacy = report["legacy_policy"]
    print(
        f"legacy policy (top 6, any overlap): precision {legacy['precision']:.3f}  recall {legacy['recall']:.3f}  injected/query {legacy['injected_per_query']:.2f}  false recall on unrelated {legacy['false_recall_on_unrelated']:.2f}"
    )
    print(f"\nmisses at 3: {len(report['tier3_misses_at_3'])}")
    for miss in report["tier3_misses_at_3"]:
        print(f"  [{miss['type']}] {miss['query']}  expected {miss['expected']} got {miss['got']}")


if __name__ == "__main__":
    main()
