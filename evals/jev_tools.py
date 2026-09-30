"""Experiment 7: can Jev tell, from the user's request, which tool the agent needs first?

Tools: AgentCLI's 17 built-ins, the 30 Chrome DevTools MCP tools (deferred until loaded),
and "none" for a request that needs no tool: 48 options.

  jev choice   one Choice over the 48 options, each described by its tool description
  jev nouls    one request, one Noul per option ("is this the tool to call first?")
  luna         GPT-6 Luna reads the same list and names its top three

Scored: top-1 (the best guess is an acceptable tool), hit@3 (one of the top three is),
group accuracy (right kind of tool: file / search / edit / shell / web / memory / skill /
browser / none), and browser detection (the signal that decides loading the deferred
browser tools). Answers are cached in fixtures/jev_tools_cache.json.

    python evals/jev_tools.py
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

from tool_dataset import TOOLS_OK

from agentcli.tools import get_builtin_tools

EVALS = Path(__file__).resolve().parent
CACHE = EVALS / "fixtures" / "jev_tools_cache.json"
RESULTS = EVALS / "results" / "jev_tools.json"
CHROME = Path(os.environ.get("AGENTCLI_HOME", r"D:\agentcli-data")) / "mcp-cache"
JEV_URL = "https://api.typesafe.ai/v1/systemone"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LUNA = "openai/gpt-6-luna"

GROUPS = {
    "read_file": "file", "list_dir": "file", "glob": "file", "directory_tree": "file",
    "get_file_info": "file", "grep": "search", "search_code": "search",
    "write_file": "edit", "edit_file": "edit", "bash": "shell",
    "web_search": "web", "web_fetch": "web", "save_memory": "memory",
    "search_memory": "memory", "load_skill": "skill", "save_skill": "skill",
    "revert_turn": "skill", "none": "none",
}


def first_sentence(text: str) -> str:
    text = " ".join((text or "").split())
    return re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0][:200]


def catalogue() -> dict[str, str]:
    tools = {t.name: first_sentence(t.description) for t in get_builtin_tools()}
    chrome = next(CHROME.glob("chrome-devtools-*.json"))
    for tool in json.loads(chrome.read_text(encoding="utf-8"))["tools"]:
        tools[tool["name"]] = "Browser: " + first_sentence(tool.get("description", ""))
    tools["none"] = "No tool: answer from knowledge, or just reply to the user"
    return tools


def group(name: str) -> str:
    return GROUPS.get(name, "browser")


async def jev(client, state, questions):
    for attempt in range(5):
        response = await client.post(
            JEV_URL, json={"model": "jev-latest", "state": state, "questions": questions}
        )
        if response.status_code in (429, 529, 500, 502, 503):
            await asyncio.sleep(2 * (attempt + 1))
            continue
        response.raise_for_status()
        body = response.json()
        return body["answers"], int(body["usage"]["input_tokens"])
    raise RuntimeError("Jev kept refusing")


async def jev_choice(client, text, tools) -> dict:
    started = time.perf_counter()
    answers, tokens = await jev(
        client,
        {"request": text},
        {
            "tool": {
                "type": "choice",
                "instructions": "A coding agent received `request`. Which tool should it call "
                "first to act on it?",
                "criteria": tools,
            }
        },
    )
    probabilities = answers["tool"]["probabilities"]
    ranked = sorted(probabilities, key=probabilities.get, reverse=True)
    return {"ranked": ranked, "seconds": time.perf_counter() - started, "tokens": tokens,
            "confidence": answers["tool"].get("confidence")}


async def jev_nouls(client, text, tools) -> dict:
    started = time.perf_counter()
    questions = {
        name: {
            "type": "noul",
            "instructions": {
                "tool": {"name": name, "does": description},
                "question": "Should a coding agent that received `request` call `tool` first?",
            },
        }
        for name, description in tools.items()
    }
    answers, tokens = await jev(client, {"request": text}, questions)
    scores = {name: float(a["noul"]) for name, a in answers.items()}
    ranked = sorted(scores, key=scores.get, reverse=True)
    return {"ranked": ranked, "seconds": time.perf_counter() - started, "tokens": tokens}


async def luna(client, text, tools) -> dict:
    listing = "\n".join(f"{name}: {description}" for name, description in tools.items())
    prompt = (
        f"A coding agent received this request:\n\n{text}\n\nIts tools:\n{listing}\n\n"
        "Which tool should it call first? Reply with only a JSON array of the three most "
        'likely tool names, best first, e.g. ["grep", "read_file", "none"].'
    )
    started = time.perf_counter()
    response = await client.post(
        OPENROUTER_URL,
        json={"model": LUNA, "temperature": 0, "max_tokens": 1500, "usage": {"include": True},
              "messages": [{"role": "user", "content": prompt}]},
    )
    response.raise_for_status()
    body = response.json()
    content = body["choices"][0]["message"]["content"] or ""
    try:
        ranked = [str(n) for n in json.loads(re.search(r"\[.*\]", content, re.S).group(0))]
    except (AttributeError, json.JSONDecodeError):
        ranked = []
    return {"ranked": ranked, "seconds": time.perf_counter() - started,
            "cost": float((body.get("usage") or {}).get("cost") or 0)}


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * len(ordered)) - 1))] if ordered else 0.0


def score(runs: list[dict]) -> dict:
    top1 = hit3 = group_ok = 0
    browser_tp = browser_fp = browser_fn = 0
    for run, (_, ok) in zip(runs, TOOLS_OK):
        ranked = run["ranked"]
        best = ranked[0] if ranked else "none"
        top1 += best in ok
        hit3 += bool(set(ranked[:3]) & set(ok))
        group_ok += group(best) in {group(name) for name in ok}
        want_browser = group(ok[0]) == "browser"
        got_browser = group(best) == "browser"
        browser_tp += want_browser and got_browser
        browser_fp += got_browser and not want_browser
        browser_fn += want_browser and not got_browser
    n = len(TOOLS_OK)
    seconds = [run["seconds"] for run in runs]
    return {
        "top1": top1 / n,
        "hit@3": hit3 / n,
        "group_accuracy": group_ok / n,
        "browser_precision": browser_tp / max(1, browser_tp + browser_fp),
        "browser_recall": browser_tp / max(1, browser_tp + browser_fn),
        "p50_s": percentile(seconds, 0.5),
        "p95_s": percentile(seconds, 0.95),
        "cost_usd": sum(run.get("cost", 0) for run in runs)
        + sum(run.get("tokens", 0) for run in runs) * 0.042 / 1e6,
    }


async def collect(tools) -> dict:
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    jev_client = httpx.AsyncClient(
        timeout=30, headers={"authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}"})
    llm_client = httpx.AsyncClient(
        timeout=60, headers={"authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
    methods = {
        "jev choice": lambda t: jev_choice(jev_client, t, tools),
        "jev nouls": lambda t: jev_nouls(jev_client, t, tools),
        "luna": lambda t: luna(llm_client, t, tools),
    }
    try:
        for text, _ in TOOLS_OK:
            for name, method in methods.items():
                key = f"{name}|{text}"
                if key not in cache:
                    cache[key] = await method(text)
    finally:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
        await jev_client.aclose()
        await llm_client.aclose()
    return {name: [cache[f"{name}|{text}"] for text, _ in TOOLS_OK] for name in methods}


def main() -> None:
    tools = catalogue()
    runs = asyncio.run(collect(tools))
    report = {"requests": len(TOOLS_OK), "options": len(tools),
              "methods": {name: score(r) for name, r in runs.items()}}
    report["misses"] = [
        {"request": text[:60], "want": ok, **{name: runs[name][i]["ranked"][:3] for name in runs}}
        for i, (text, ok) in enumerate(TOOLS_OK)
        if any((runs[name][i]["ranked"] or ["none"])[0] not in ok for name in runs)
    ]
    RESULTS.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{report['requests']} requests, {report['options']} options")
    for name, row in report["methods"].items():
        print(f"  {name:11} top1 {row['top1']:.3f}  hit@3 {row['hit@3']:.3f}  group {row['group_accuracy']:.3f}  "
              f"browser P/R {row['browser_precision']:.2f}/{row['browser_recall']:.2f}  "
              f"p50 {row['p50_s']:.2f}s p95 {row['p95_s']:.2f}s  ${row['cost_usd']:.4f}")
    for miss in report["misses"]:
        print("  miss:", json.dumps(miss, ensure_ascii=False))


if __name__ == "__main__":
    main()
