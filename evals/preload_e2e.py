"""Experiment 8: does preloading deferred tools with Jev make browser tasks cheaper or faster?

A real agent (GPT-6 Luna) with the Chrome DevTools MCP server (headless, deferred) runs
browser tasks against a local test site, and a few tasks that need no browser, with Jev
preloading on and off. Per run: success (the answer contains the expected text), model
calls, input tokens, cost, wall time, and whether load_tools was still called.

    python evals/preload_e2e.py --reps 2
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import http.server
import json
import shutil
import statistics
import threading
from pathlib import Path

from harness import EVALS, RUNS, cost_usd, make_config, record, run_turn

from agentcli.agent.agent import Agent
from agentcli.bootstrap import build_tool_registry
from agentcli.llm import create_llm_client

SITE = EVALS / "fixtures" / "preload_site"
PORT = 8765
URL = f"http://127.0.0.1:{PORT}"
MODEL = "openai/gpt-6-luna"

BROWSER = [
    (f"打开 {URL}/ 告诉我页面标题是什么", "FinMate Lab"),
    (f"{URL}/ 这个页面的浏览器控制台里有什么报错？", "quota"),
    (f"打开 {URL}/ ，点一下 Load 按钮，告诉我按钮下方出现了什么文字", "loaded-7731"),
    (f"{URL}/ 这个页面加载时请求了哪个 json 文件？", "data.json"),
    (f"在 {URL}/form.html 的表单里 name 填 RJ 并提交，页面显示了什么？", "hello RJ"),
    (f"用 JavaScript 取一下 {URL}/ 页面上 id 为 secret 的元素的 data-code 属性值", "k9x"),
]
LOCAL = [
    ("读一下 notes.txt，里面写的测试数据库端口是多少？", "5433"),
    ("当前目录下有几个 .txt 文件？只回答数字", "2"),
    ("一句话解释什么是 RRF（倒数排名融合）", "排名"),
]


def serve_site() -> http.server.ThreadingHTTPServer:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(SITE))
    handler.log_message = lambda *args: None  # type: ignore[assignment]
    server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def workspace(label: str) -> Path:
    target = RUNS / f"preload-{label}"
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    (target / "notes.txt").write_text("测试数据库端口：5433\n", encoding="utf-8")
    (target / "todo.txt").write_text("- ship\n", encoding="utf-8")
    return target


def mcp_config() -> None:
    home = EVALS / "_home"
    home.mkdir(parents=True, exist_ok=True)
    user = json.loads(Path(r"D:\agentcli-data\mcp.json").read_text(encoding="utf-8"))
    only = {"chrome-devtools": user["mcpServers"]["chrome-devtools"]}
    (home / "mcp.json").write_text(json.dumps({"mcpServers": only}, indent=1), encoding="utf-8")


async def one(task: str, expect: str, preload: bool, label: str) -> dict:
    root = workspace(label)
    config = make_config(
        MODEL,
        root,
        {"features": {"mcp": True}, "routing": {"preload_tools": preload}},
    )
    registry, manager = await build_tool_registry(config=config, cwd=str(root))
    agent = Agent(
        llm_client=create_llm_client(config.llm),
        tool_registry=registry,
        config=config,
        cwd=str(root),
        mode="react",
        max_turns=20,
    )
    tools_called: list[str] = []
    preloaded: list[str] = []
    loads: list = []
    original_run = agent.run

    async def watched(message):
        async for event in original_run(message):
            if event.get("type") == "tool_call":
                tools_called.append(str(event.get("name") or ""))
                if "load_tools" in str(event.get("name")):
                    loads.append(event.get("input"))
            elif event.get("type") == "tools_preloaded":
                preloaded.extend(event.get("names") or [])
            yield event

    agent.run = watched  # type: ignore[method-assign]
    try:
        stats = await run_turn(agent, task)
    finally:
        if manager is not None:
            await manager.aclose()
    return {
        "task": task[:50],
        "preload": preload,
        "ok": expect.lower() in stats.text.lower(),
        "model_calls": stats.model_calls,
        "input_tokens": stats.usage.input_tokens,
        "cost": cost_usd(agent, stats.usage),
        "seconds": round(stats.seconds, 2),
        "load_tools_called": sum("load_tools" in name for name in tools_called),
        "preloaded": [name.split("__")[-1] for name in preloaded],
        "tools": [name.split("__")[-1] for name in tools_called],
        "loads": loads,
        "errors": stats.errors[:2],
    }


def summarize(rows: list[dict]) -> dict:
    out = {}
    for kind, tasks in (("browser", BROWSER), ("local", LOCAL)):
        names = {task[:50] for task, _ in tasks}
        for preload in (False, True):
            group = [r for r in rows if r["task"] in names and r["preload"] is preload]
            if not group:
                continue
            out[f"{kind} preload={'on' if preload else 'off'}"] = {
                "runs": len(group),
                "success": sum(r["ok"] for r in group) / len(group),
                "model_calls": statistics.mean(r["model_calls"] for r in group),
                "input_tokens": statistics.mean(r["input_tokens"] for r in group),
                "cost": statistics.mean(r["cost"] for r in group),
                "seconds_median": statistics.median(r["seconds"] for r in group),
                "seconds_mean": statistics.mean(r["seconds"] for r in group),
                "load_tools_calls": statistics.mean(r["load_tools_called"] for r in group),
                "preloaded_runs": sum(bool(r["preloaded"]) for r in group) / len(group),
            }
    return out


async def main(reps: int) -> None:
    mcp_config()
    server = serve_site()
    rows: list[dict] = []
    try:
        for rep in range(reps):
            for index, (task, expect) in enumerate(BROWSER + LOCAL):
                # Alternate which condition goes first, so drift hits both equally.
                order = (False, True) if (rep + index) % 2 == 0 else (True, False)
                for preload in order:
                    row = await one(task, expect, preload, f"{rep}-{index}-{int(preload)}")
                    row["rep"] = rep
                    rows.append(row)
                    record("preload_e2e", row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        server.shutdown()
    summary = summarize(rows)
    summary["total_cost"] = sum(r["cost"] for r in rows)
    (EVALS / "results" / "preload_e2e_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=2)
    asyncio.run(main(parser.parse_args().reps))
