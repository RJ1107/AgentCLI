"""Experiment 5: does a Jev relevance check improve automatic memory recall?

Same data as experiment 4 (80 memories, 119 labelled questions). BM25 with the saved
keywords makes a shortlist of 15; then each policy decides what reaches the model:

  A  bm25 gate    today's tier 2: top 3 with coverage >= gate
  B  jev pairs    one Jev noul per (question, memory) pair, 15 requests in parallel
  C  jev packed   one Jev request per question, 15 nouls over the shortlist in the state
  D  luna         GPT-6 Luna reads the shortlist and scores each memory 0-1

B, C and D keep the top 3 whose score clears a threshold. Every threshold, including A's
coverage gate, is chosen on one half of the questions and reported on the other half
(two folds, stratified by question type), so no policy is scored on the data that tuned it.

Answers are cached in fixtures/jev_recall_cache.json, so a rerun costs nothing.

    python evals/jev_recall.py          # needs TYPESAFE_API_KEY and OPENROUTER_API_KEY
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

from memory_dataset import QUERIES
from memory_recall import NOW, records

from agentcli.memory import rank

EVALS = Path(__file__).resolve().parent
CACHE = EVALS / "fixtures" / "jev_recall_cache.json"
RESULTS = EVALS / "results" / "jev_recall.json"
JEV_URL = "https://api.typesafe.ai/v1/systemone"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LUNA = "openai/gpt-6-luna"
SHORTLIST = 15
TOP = 3
LLM_BUDGET_USD = 1.0

QUESTION = {
    "type": "noul",
    "instructions": "Does `memory` contain information that helps answer `question`?",
    "criteria": {
        "true": "The memory states a fact, rule, or preference that answers the question "
        "or that someone answering it would need",
        "false": "The memory is about something else, even if it shares some words",
    },
}

LUNA_PROMPT = """A coding agent is about to answer the question below. Its long-term memory
search found these candidate notes. For each note, give the probability (0 to 1) that it
contains information that helps answer the question. A note that only shares words with
the question but is about something else scores low.

Question: {question}

Notes:
{notes}

Reply with only a JSON object mapping each note id to a number."""


def describe(record) -> str:
    return f"{record.title}: {record.content}"


# ---------------------------------------------------------------- cache


class Cache:
    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def get(self, key: str):
        return self.data.get(key)

    def put(self, key: str, value) -> None:
        self.data[key] = value

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=0), encoding="utf-8")


# ---------------------------------------------------------------- scorers


async def jev(client: httpx.AsyncClient, state, questions) -> tuple[dict, int]:
    for attempt in range(5):
        response = await client.post(
            JEV_URL, json={"model": "jev-latest", "state": state, "questions": questions}
        )
        if response.status_code in (429, 529, 500, 502, 503):
            await asyncio.sleep(float(response.headers.get("retry-after") or 2 * (attempt + 1)))
            continue
        response.raise_for_status()
        body = response.json()
        return body["answers"], int(body["usage"]["input_tokens"])
    raise RuntimeError("Jev kept refusing")


async def jev_pairs(client, cache, query, shortlist) -> dict:
    key = f"pairs|{query}"
    if (hit := cache.get(key)) is not None:
        return hit

    async def one(record):
        answers, tokens = await jev(
            client, {"question": query, "memory": describe(record)}, {"relevant": QUESTION}
        )
        return record.name, float(answers["relevant"]["noul"]), tokens

    started = time.perf_counter()
    rows = await asyncio.gather(*(one(r) for r in shortlist))
    result = {
        "scores": {name: score for name, score, _ in rows},
        "seconds": time.perf_counter() - started,
        "tokens": sum(tokens for _, _, tokens in rows),
    }
    cache.put(key, result)
    return result


async def jev_packed(client, cache, query, shortlist) -> dict:
    key = f"packed|{query}"
    if (hit := cache.get(key)) is not None:
        return hit
    state = {"question": query, "memories": {r.name: describe(r) for r in shortlist}}
    questions = {
        r.name: {
            **QUESTION,
            "instructions": f"Does `memories.{r.name}` contain information that helps answer "
            "`question`?",
        }
        for r in shortlist
    }
    started = time.perf_counter()
    answers, tokens = await jev(client, state, questions)
    result = {
        "scores": {name: float(a["noul"]) for name, a in answers.items()},
        "seconds": time.perf_counter() - started,
        "tokens": tokens,
    }
    cache.put(key, result)
    return result


async def luna(client, cache, query, shortlist, spent: list[float]) -> dict | None:
    key = f"luna|{query}"
    if (hit := cache.get(key)) is not None:
        return hit
    if sum(spent) >= LLM_BUDGET_USD:
        return None
    notes = "\n".join(f"{r.name}: {describe(r)}" for r in shortlist)
    started = time.perf_counter()
    response = await client.post(
        OPENROUTER_URL,
        json={
            "model": LUNA,
            "temperature": 0,
            "max_tokens": 1500,
            "usage": {"include": True},
            "messages": [
                {"role": "user", "content": LUNA_PROMPT.format(question=query, notes=notes)}
            ],
        },
    )
    response.raise_for_status()
    body = response.json()
    text = body["choices"][0]["message"]["content"] or ""
    match = re.search(r"\{.*\}", text, re.S)
    scores = {}
    if match:
        for name, value in json.loads(match.group(0)).items():
            try:
                scores[str(name)] = float(value)
            except (TypeError, ValueError):
                pass
    cost = float((body.get("usage") or {}).get("cost") or 0)
    spent.append(cost)
    result = {"scores": scores, "seconds": time.perf_counter() - started, "cost": cost}
    cache.put(key, result)
    return result


# ---------------------------------------------------------------- policies and metrics


def folds():
    """Two halves, alternating within each question type."""

    seen: dict[str, int] = {}
    halves = ([], [])
    for index, (_, _, kind) in enumerate(QUERIES):
        position = seen.get(kind, 0)
        seen[kind] = position + 1
        halves[position % 2].append(index)
    return halves


def choose(scored: dict[int, list[tuple[str, float]]], index: int, threshold: float) -> list[str]:
    return [name for name, score in scored[index] if score >= threshold][:TOP]


def measure(scored, indices, threshold) -> dict:
    injected = relevant_injected = relevant_total = false_on_none = none_total = 0
    for index in indices:
        _, relevant, _ = QUERIES[index]
        chosen = choose(scored, index, threshold)
        injected += len(chosen)
        relevant_injected += len(set(chosen) & set(relevant))
        relevant_total += len(relevant)
        if not relevant:
            none_total += 1
            false_on_none += bool(chosen)
    precision = relevant_injected / injected if injected else 1.0
    recall = relevant_injected / relevant_total if relevant_total else 0.0
    return {
        "threshold": threshold,
        "injected": injected,
        "relevant_injected": relevant_injected,
        "relevant_total": relevant_total,
        "false_on_none": false_on_none,
        "none_total": none_total,
        "precision": precision,
        "recall": recall,
        "f0.5": f_beta(precision, recall, 0.5),
    }


def f_beta(precision: float, recall: float, beta: float) -> float:
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def cross_validated(scored, grid) -> dict:
    """Pick the threshold on one half by F0.5, apply it to the other; pool both test halves."""

    halves = folds()
    pooled = {"injected": 0, "relevant_injected": 0, "relevant_total": 0, "false_on_none": 0,
              "none_total": 0}
    picked = []
    for train, test in ((halves[0], halves[1]), (halves[1], halves[0])):
        best = max((measure(scored, train, t) for t in grid), key=lambda r: (r["f0.5"], r["threshold"]))
        picked.append(best["threshold"])
        result = measure(scored, test, best["threshold"])
        for field in pooled:
            pooled[field] += result[field]
    precision = pooled["relevant_injected"] / pooled["injected"] if pooled["injected"] else 1.0
    recall = pooled["relevant_injected"] / pooled["relevant_total"]
    return {
        "thresholds": picked,
        "precision": precision,
        "recall": recall,
        "f0.5": f_beta(precision, recall, 0.5),
        "injected_per_query": pooled["injected"] / len(QUERIES),
        "false_recall_on_unrelated": pooled["false_on_none"] / pooled["none_total"],
    }


def ranking(scored) -> dict:
    """recall@3 of the order alone, no threshold, by question type."""

    by_kind: dict[str, list[int]] = {}
    hits = total = 0
    for index, (_, relevant, kind) in enumerate(QUERIES):
        if not relevant:
            continue
        top = [name for name, _ in scored[index]][:TOP]
        found = len(set(top) & set(relevant))
        hits += found
        total += len(relevant)
        pair = by_kind.setdefault(kind, [0, 0])
        pair[0] += found
        pair[1] += len(relevant)
    return {"recall@3": hits / total, "by_type": {k: h / t for k, (h, t) in by_kind.items()}}


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * len(ordered)) - 1))] if ordered else 0.0


# ---------------------------------------------------------------- run


async def collect() -> dict:
    items = records(True)
    cache = Cache(CACHE)
    shortlists = []
    for text, _, _ in QUERIES:
        hits = rank(text, items, now=NOW)[:SHORTLIST]
        shortlists.append(hits)

    jev_client = httpx.AsyncClient(
        timeout=30, headers={"authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}"}
    )
    llm_client = httpx.AsyncClient(
        timeout=90, headers={"authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    )
    spent: list[float] = [v["cost"] for k, v in cache.data.items() if k.startswith("luna|")]
    raw = {"pairs": [], "packed": [], "luna": []}
    try:
        for index, ((text, _, _), hits) in enumerate(zip(QUERIES, shortlists)):
            shortlist = [hit.record for hit in hits]
            if not shortlist:
                empty = {"scores": {}, "seconds": 0.0, "tokens": 0, "cost": 0.0}
                for name in raw:
                    raw[name].append(empty)
                continue
            raw["pairs"].append(await jev_pairs(jev_client, cache, text, shortlist))
            raw["packed"].append(await jev_packed(jev_client, cache, text, shortlist))
            raw["luna"].append(await luna(llm_client, cache, text, shortlist, spent))
            if index % 10 == 9:
                cache.save()
                print(f"  {index + 1}/{len(QUERIES)}  OpenRouter so far ${sum(spent):.3f}")
    finally:
        cache.save()
        await jev_client.aclose()
        await llm_client.aclose()
    return {"shortlists": shortlists, "raw": raw, "spent": sum(spent)}


def main() -> None:
    data = asyncio.run(collect())
    shortlists, raw = data["shortlists"], data["raw"]

    def sorted_scores(scores: dict[str, float], hits) -> list[tuple[str, float]]:
        # Ties keep BM25 order; a memory the scorer skipped counts as 0.
        order = {hit.record.name: i for i, hit in enumerate(hits)}
        return sorted(
            ((hit.record.name, scores.get(hit.record.name, 0.0)) for hit in hits),
            key=lambda pair: (-pair[1], order[pair[0]]),
        )

    scored = {
        "A bm25 gate": {i: [(h.record.name, h.coverage) for h in hits] for i, hits in enumerate(shortlists)},
    }
    for label, name in (("B jev pairs", "pairs"), ("C jev packed", "packed"), ("D luna", "luna")):
        if any(entry is None for entry in raw[name]):
            print(f"{label}: incomplete (budget reached), skipped")
            continue
        scored[label] = {
            i: sorted_scores(entry["scores"], shortlists[i]) for i, entry in enumerate(raw[name])
        }

    grid = [round(0.05 * i, 2) for i in range(1, 20)]
    shortlist_recall = sum(
        len({h.record.name for h in hits} & set(rel)) for hits, (_, rel, _) in zip(shortlists, QUERIES)
    ) / sum(len(rel) for _, rel, _ in QUERIES)

    report = {"queries": len(QUERIES), "shortlist": SHORTLIST, "shortlist_recall": shortlist_recall,
              "policies": {}}
    for label, table in scored.items():
        report["policies"][label] = {
            **ranking(table),
            "cross_validated": cross_validated(table, grid),
            "in_sample_sweep": [measure(table, range(len(QUERIES)), t) for t in grid],
        }
    for label, name in (("B jev pairs", "pairs"), ("C jev packed", "packed"), ("D luna", "luna")):
        entries = [e for e in raw[name] if e and e.get("seconds")]
        if label in report["policies"] and entries:
            seconds = [e["seconds"] for e in entries]
            report["policies"][label]["latency"] = {
                "p50_s": percentile(seconds, 0.5), "p95_s": percentile(seconds, 0.95)
            }
            if name == "luna":
                report["policies"][label]["cost_usd"] = sum(e.get("cost", 0) for e in entries)
            else:
                tokens = sum(e.get("tokens", 0) for e in entries)
                report["policies"][label]["input_tokens"] = tokens
                report["policies"][label]["cost_usd"] = tokens * 0.042 / 1e6
    report["openrouter_spent"] = data["spent"]

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    _print(report)


def _print(report: dict) -> None:
    print(f"\n{report['queries']} questions, shortlist {report['shortlist']} "
          f"(contains {report['shortlist_recall']:.1%} of the relevant memories)\n")
    print(f"{'policy':14} {'recall@3':>8} {'direct':>7} {'para':>6} {'eng':>6} {'multi':>6} | "
          f"{'thresh':>10} {'prec':>6} {'recall':>6} {'F0.5':>6} {'inj/q':>6} {'false':>6} | "
          f"{'p50':>6} {'p95':>6} {'$':>8}")
    for label, row in report["policies"].items():
        cv = row["cross_validated"]
        types = row["by_type"]
        latency = row.get("latency", {})
        print(
            f"{label:14} {row['recall@3']:8.3f} {types.get('direct', 0):7.2f} "
            f"{types.get('paraphrase', 0):6.2f} {types.get('english', 0):6.2f} "
            f"{types.get('multi', 0):6.2f} | {str(cv['thresholds']):>10} {cv['precision']:6.3f} "
            f"{cv['recall']:6.3f} {cv['f0.5']:6.3f} {cv['injected_per_query']:6.2f} "
            f"{cv['false_recall_on_unrelated']:6.2f} | {latency.get('p50_s', 0):6.2f} "
            f"{latency.get('p95_s', 0):6.2f} {row.get('cost_usd', 0):8.4f}"
        )
    print(f"\nOpenRouter spent: ${report['openrouter_spent']:.3f}")


if __name__ == "__main__":
    main()
