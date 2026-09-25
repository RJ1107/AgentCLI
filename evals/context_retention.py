"""Experiment 1: what each compression strategy keeps, and what it costs.

A 40-turn session under a small context window, so compression fires many times. The user
states 15 project facts in three batches (turns 1, 14, 27); every turn also has the agent
read a source file, which fills the context the way real work does. At the end the agent
is quizzed on the facts without tools. The facts appear nowhere in the files, so the only
way to answer is to have kept them through compression.

Strategies:
  truncate        drop the oldest turns, no summary
  summary         summarize whenever over budget, no clearing of tool results first
  layered_nohyst  clear old tool results, then summarize; compact only just below the limit
  layered         the default: clear, then summarize, compacting down to 55% (hysteresis)

    python evals/context_retention.py --repeats 3 --parallel 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import (  # noqa: E402
    build_agent,
    by_model,
    fresh_workspace,
    live_cost,
    make_config,
    record,
    run_turn,
    start_tracking,
    usage_dict,
)

CORPUS = Path(__file__).resolve().parents[1] / "src" / "agentcli"
EXPERIMENT = "context_retention_v3"

STRATEGIES = {
    "truncate": {"memory": {"compression_strategy": "truncate"}},
    "summary": {"memory": {"compression_strategy": "summary"}},
    "layered_nohyst": {"memory": {"compression_strategy": "layered", "compression_target": 0.78}},
    "layered": {"memory": {"compression_strategy": "layered", "compression_target": 0.55}},
}

# id -> (statement, question, keywords that must all appear in a correct answer)
FACTS = {
    1: ("测试数据库端口是 5433", "测试数据库的端口是多少？", ["5433"]),
    2: ("发布分支统一叫 release/kestrel", "发布分支叫什么？", ["kestrel"]),
    3: ("代码评审负责人是 Mira Okafor", "代码评审负责人是谁？", ["mira"]),
    4: ("日志保留 17 天", "日志保留多少天？", ["17"]),
    5: ("内部 API 前缀是 /v3/core", "内部 API 前缀是什么？", ["v3/core"]),
    6: ("金额字段一律用 Decimal，禁止 float", "金额字段必须用什么类型？", ["decimal"]),
    7: ("任何人都不许改 migrations 目录", "哪个目录不许改？", ["migrations"]),
    8: ("预发环境域名是 staging.lumen-test.io", "预发环境的域名是什么？", ["staging.lumen"]),
    9: ("功能开关服务叫 Flagpole", "功能开关服务叫什么？", ["flagpole"]),
    10: ("每周部署窗口是周四 14:00", "每周部署窗口是什么时候？", ["四", "14"]),
    11: ("最大客户的代号是 Bluefin", "最大客户的代号是什么？", ["bluefin"]),
    12: ("缓存 TTL 默认 90 秒", "缓存 TTL 默认多少秒？", ["90"]),
    13: ("错误码统一以 AC- 开头", "错误码以什么开头？", ["ac-"]),
    14: ("值班群叫 #ops-heron", "值班群叫什么？", ["heron"]),
    15: ("文档站点用 Docusaurus 搭建", "文档站点用什么搭建？", ["docusaurus"]),
    16: ("HTTP 请求超时是 2300 毫秒", "HTTP 请求超时是多少毫秒？", ["2300"]),
    17: ("重试退避基数是 1.8 秒", "重试退避基数是多少秒？", ["1.8"]),
    18: ("CPU 告警阈值是 83%", "CPU 告警阈值是多少？", ["83"]),
    19: ("上传文件最大 250 MB", "上传文件最大多少？", ["250"]),
    20: ("灰度发布比例是 12%", "灰度发布比例是多少？", ["12"]),
    21: ("主数据库是 Postgres 16.4", "主数据库是什么版本？", ["16.4"]),
    22: (
        "订单创建的消息 topic 叫 orders.v2.created",
        "订单创建的消息 topic 叫什么？",
        ["orders.v2.created"],
    ),
    23: ("CI 超时是 38 分钟", "CI 超时是多少分钟？", ["38"]),
    24: ("代码覆盖率门槛是 81%", "代码覆盖率门槛是多少？", ["81"]),
    25: ("前端组件库用 Mantine", "前端组件库用什么？", ["mantine"]),
    26: ("服务器时区统一用 Asia/Singapore", "服务器时区统一用什么？", ["singapore"]),
    27: ("密钥轮换周期是 62 天", "密钥轮换周期是多少天？", ["62"]),
    28: ("监控看板叫 Lighthouse-7", "监控看板叫什么？", ["lighthouse-7"]),
    29: ("合规审计联系人是 Priya Nair", "合规审计联系人是谁？", ["priya"]),
    30: ("最大并发任务数是 24", "最大并发任务数是多少？", ["24"]),
}
# id -> (update statement, keywords of the new value). Graded against the new value; an
# answer with the old value instead counts as stale.
UPDATES = {
    1: ("测试数据库端口从 5433 改成 6544", ["6544"]),
    3: ("代码评审负责人从 Mira Okafor 换成 Tomas Reyes", ["tomas"]),
    8: ("预发环境域名从 staging.lumen-test.io 改成 preprod.lumen-test.io", ["preprod"]),
    10: ("每周部署窗口从周四 14:00 改成周二 10:30", ["二", "10:30"]),
    12: ("缓存 TTL 默认值从 90 秒改成 45 秒", ["45"]),
    16: ("HTTP 请求超时从 2300 毫秒改成 3100 毫秒", ["3100"]),
    20: ("灰度发布比例从 12% 改成 35%", ["35"]),
    30: ("最大并发任务数从 24 改成 40", ["40"]),
}
FACT_TURNS = {turn: list(range(1 + 5 * n, 6 + 5 * n)) for n, turn in enumerate(range(0, 60, 10))}
UPDATE_TURNS = {60: [1, 3, 8, 10], 65: [12, 16, 20, 30]}
TURNS = 72
CONTEXT_WINDOW = 64_000


def pick_files(count: int) -> list[str]:
    files = sorted(
        path
        for path in CORPUS.rglob("*.py")
        if "__pycache__" not in path.parts and 5_000 <= path.stat().st_size <= 20_000
    )
    step = max(1, len(files) // count)
    return [str(path.relative_to(CORPUS)).replace("\\", "/") for path in files[::step][:count]]


def turn_message(index: int, path: str) -> str:
    ask = f"请用 read_file 读 {path}，用一两句话说明它负责什么。"
    if index in FACT_TURNS:
        listed = "\n".join(f"- {FACTS[i][0]}" for i in FACT_TURNS[index])
        return f"先记住几条项目约定，之后会用到：\n{listed}\n\n然后，{ask}"
    if index in UPDATE_TURNS:
        listed = "\n".join(f"- {UPDATES[i][0]}" for i in UPDATE_TURNS[index])
        return f"有几条约定变了，以新的为准：\n{listed}\n\n然后，{ask}"
    return ask


QUIZ = (
    "不要调用任何工具，只根据我们之前的对话回答下面的问题，约定改过的以最新为准；"
    "不记得就写“不知道”。"
    '只输出一个 JSON 对象，形如 {"1": "...", "2": "..."}。\n'
    + "\n".join(f"{i}. {FACTS[i][1]}" for i in FACTS)
)


def grade(text: str) -> tuple[list[bool], list[int]]:
    """Per fact: answered with the current value. Also the updated facts answered stale."""

    match = re.search(r"\{.*\}", text, re.S)
    answers: dict[str, str] = {}
    if match:
        try:
            answers = {str(k): str(v).lower() for k, v in json.loads(match.group(0)).items()}
        except json.JSONDecodeError:
            answers = {}
    correct, stale = [], []
    for i, (_, _, keywords) in FACTS.items():
        answer = answers.get(str(i), "")
        current = UPDATES[i][1] if i in UPDATES else keywords
        ok = all(k.lower() in answer for k in current)
        correct.append(ok)
        if i in UPDATES and not ok and all(k.lower() in answer for k in keywords):
            stale.append(i)
    return correct, stale


async def run_once(strategy: str, repeat: int, model: str, turns: int) -> dict:
    workspace = fresh_workspace(CORPUS, f"ctx-{strategy}")
    config = make_config(
        model,
        workspace,
        {"llm": {"context_window": CONTEXT_WINDOW, "max_tokens": 4096}, **STRATEGIES[strategy]},
    )
    agent = await build_agent(config, workspace)
    tracker = start_tracking()
    files = pick_files(turns)
    totals = None
    calls = tools = 0
    compressions: list[str] = []
    errors: list[str] = []
    seconds = 0.0
    for index in range(turns):
        stats = await run_turn(agent, turn_message(index, files[index % len(files)]))
        totals = stats.usage if totals is None else totals + stats.usage
        calls += stats.model_calls
        tools += stats.tool_calls
        compressions += stats.compressions
        errors += stats.errors
        seconds += stats.seconds
    quiz = await run_turn(agent, QUIZ)
    totals = totals + quiz.usage
    compressions += quiz.compressions
    graded, stale = grade(quiz.text)
    row = {
        "strategy": strategy,
        "repeat": repeat,
        "model": model,
        "turns": turns,
        "context_window": CONTEXT_WINDOW,
        "retained": sum(graded),
        "of": len(graded),
        "per_fact": graded,
        "stale": stale,
        "updated_correct": sum(graded[i - 1] for i in UPDATES),
        "compressions": len(compressions),
        "compression_methods": {m: compressions.count(m) for m in sorted(set(compressions))},
        "model_calls": calls + quiz.model_calls,
        "tool_calls": tools,
        "usage": usage_dict(totals),
        "cost_usd": round(live_cost(tracker), 5),
        "by_model": by_model(tracker),
        "seconds": round(seconds + quiz.seconds, 1),
        "errors": errors[:5],
        "quiz_answer": quiz.text[-1500:],
    }
    record(EXPERIMENT, row)
    print(
        f"{strategy:15} #{repeat}: retained {row['retained']}/{row['of']}, "
        f"updated {row['updated_correct']}/{len(UPDATES)}, stale {stale}, "
        f"{row['compressions']} compressions {row['compression_methods']}, "
        f"${row['cost_usd']:.4f}, cached {totals.cache_hit_tokens}/{totals.input_tokens}",
        flush=True,
    )
    return row


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-6-luna")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--turns", type=int, default=TURNS)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    args = parser.parse_args()

    gate = asyncio.Semaphore(args.parallel)

    async def guarded(strategy: str, repeat: int):
        async with gate:
            try:
                return await run_once(strategy, repeat, args.model, args.turns)
            except Exception as exc:  # noqa: BLE001 - one failed run must not stop the rest
                print(f"{strategy} #{repeat} failed: {exc!r}", flush=True)
                return None

    jobs = [
        guarded(strategy, repeat)
        for repeat in range(1, args.repeats + 1)
        for strategy in args.strategies.split(",")
    ]
    await asyncio.gather(*jobs)


if __name__ == "__main__":
    asyncio.run(main())
