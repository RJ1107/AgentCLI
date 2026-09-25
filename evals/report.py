"""Summarize the experiment results in evals/results/ as tables.

    python evals/report.py

Runs that failed for infrastructure reasons (rate limits, upstream 5xx, timeouts before the
agent could work) are listed separately and left out of the success rates.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def load(name: str) -> list[dict]:
    path = RESULTS / f"{name}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def infra_failure(row: dict) -> bool:
    text = " ".join(row.get("errors") or [])
    return any(code in text for code in ("HTTP 429", "HTTP 5", "timed out", "Could not connect"))


def mean(values) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def cached_share(rows: list[dict]) -> float:
    hit = sum(r["usage"].get("cache_hit_tokens", 0) for r in rows)
    total = sum(r["usage"].get("input_tokens", 0) for r in rows)
    return hit / total if total else 0.0


def table(headers: list[str], rows: list[list]) -> str:
    cells = [[str(c) for c in row] for row in rows]
    widths = [
        max(len(h), *(len(r[i]) for r in cells)) if cells else len(h) for i, h in enumerate(headers)
    ]
    line = lambda r: "| " + " | ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) + " |"  # noqa: E731
    return "\n".join(
        [line(headers), "|" + "|".join("-" * (w + 2) for w in widths) + "|", *map(line, cells)]
    )


def context_report(name: str) -> None:
    rows = [r for r in load(name) if not infra_failure(r)]
    if not rows:
        return
    print(f"\n## {name}\n")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["strategy"]].append(row)
    out = []
    for strategy in ("truncate", "summary", "layered_nohyst", "layered"):
        runs = groups.get(strategy)
        if not runs:
            continue
        of = runs[0]["of"]
        methods: dict[str, float] = defaultdict(float)
        for run in runs:
            for method, count in run["compression_methods"].items():
                methods[method] += count / len(runs)
        out.append(
            [
                strategy,
                len(runs),
                f"{mean(r['retained'] for r in runs):.1f}/{of}",
                f"{mean(r.get('updated_correct', 0) for r in runs):.1f}"
                if "updated_correct" in runs[0]
                else "-",
                f"{mean(len(r.get('stale', [])) for r in runs):.1f}" if "stale" in runs[0] else "-",
                f"{mean(r['compressions'] for r in runs):.1f}",
                f"{methods.get('llm', 0):.1f}",
                f"{cached_share(runs):.1%}",
                f"${mean(r['cost_usd'] for r in runs):.4f}",
            ]
        )
    print(
        table(
            [
                "strategy",
                "runs",
                "retained",
                "updated ok",
                "stale",
                "compressions",
                "LLM summaries",
                "cached",
                "cost/run",
            ],
            out,
        )
    )


def coding_report(name: str, key: str) -> None:
    rows = load(name)
    if not rows:
        return
    print(f"\n## {name}\n")
    infra = [r for r in rows if infra_failure(r)]
    rows = [r for r in rows if not infra_failure(r)]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[key]) + (f" [{row['tiers']}]" if row.get("tiers") else "")].append(row)
    baseline = {}
    if key == "workers":
        baseline = {(r["task"], r["repeat"]): r["seconds"] for r in rows if r["workers"] == 1}
    out = []
    for group, runs in sorted(groups.items(), key=lambda kv: (len(kv[0]), kv[0])):
        solved = [r for r in runs if r["success"]]
        cost = sum(r["cost_usd"] for r in runs)
        row = [
            group,
            len(runs),
            f"{len(solved) / len(runs):.0%}",
            f"{mean(r['passed'] for r in runs):.2f}/6",
            f"{mean(r['seconds'] for r in runs):.0f}s",
            f"{mean(r['model_calls'] for r in runs):.0f}",
            f"${mean(r['cost_usd'] for r in runs):.4f}",
            f"${cost / len(solved):.4f}" if solved else "-",
        ]
        if key == "workers":
            ratios = [
                r["seconds"] / baseline[(r["task"], r["repeat"])]
                for r in runs
                if (r["task"], r["repeat"]) in baseline
            ]
            row.append(f"{1 - mean(ratios):.0%}" if ratios else "-")
        out.append(row)
    headers = [key, "runs", "success", "modules", "time", "calls", "cost/run", "cost/solved"]
    if key == "workers":
        headers.append("time saved vs 1")
    print(table(headers, out))
    if infra:
        print(
            f"\nleft out (rate limits or upstream errors): {len(infra)} runs: "
            + ", ".join(sorted({f"{r[key]} {r['task']}" for r in infra}))
        )


if __name__ == "__main__":
    context_report("context_retention")
    context_report("context_retention_v2")
    context_report("context_retention_v3")
    coding_report("workers", "workers")
    coding_report("models", "model")
    coding_report("models_fixed", "model")
