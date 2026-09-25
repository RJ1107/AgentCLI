"""Experiments 2 and 3: coding tasks graded by their tests.

Each task (see make_task_fixtures.py) asks for six independent modules. A run copies the
task, lets the agent work, restores the original tests (so editing them cannot help), and
runs pytest. A task passes when all six modules pass.

Worker sweep (experiment 2), one model, team mode:
    python evals/coding_tasks.py --label workers --mode team --workers 1,2,4,6

Model comparison (experiment 3), single agent:
    python evals/coding_tasks.py --label models --mode react --models openai/gpt-6-luna,...

Tiered team versus one strong model:
    python evals/coding_tasks.py --label tiers --mode team --tiers fast=...,strong=...
"""

from __future__ import annotations

import argparse
import asyncio
import re
import shutil
import subprocess
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

TASKS = Path(__file__).resolve().parent / "fixtures" / "tasks"
PYTHON = sys.executable


def grade(workspace: Path, task: str) -> tuple[int, int, str]:
    """Restore the original tests, run them, and count passing modules."""

    shutil.rmtree(workspace / "tests", ignore_errors=True)
    shutil.copytree(TASKS / task / "tests", workspace / "tests")
    try:
        completed = subprocess.run(
            [PYTHON, "-m", "pytest", "-q", "--tb=no", "-rA", "-p", "no:cacheprovider"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=180,
        )
        output = completed.stdout + completed.stderr
    except subprocess.TimeoutExpired:
        return 0, 6, "pytest timed out"
    # A module passes when every test in its file passes (some files have more than one).
    files = sorted(path.name for path in (TASKS / task / "tests").glob("test_*.py"))
    outcomes: dict[str, set[str]] = {name: set() for name in files}
    for status, name in re.findall(
        r"^(PASSED|FAILED|ERROR) tests[\\/](test_\w+\.py)", output, re.M
    ):
        outcomes.setdefault(name, set()).add(status)
    passed = sum(1 for name in files if outcomes[name] == {"PASSED"})
    return passed, len(files), output[-600:]


def parse_tiers(text: str) -> dict[str, str]:
    tiers: dict[str, str] = {}
    for part in filter(None, text.split(",")):
        role, _, model = part.partition("=")
        tiers[f"{role.strip()}_model"] = f"openrouter:{model.strip()}"
    return tiers


async def run_once(
    *, label: str, mode: str, model: str, workers: int, task: str, repeat: int, tiers: str
) -> dict:
    workspace = fresh_workspace(TASKS / task, f"{label}-{task}")
    routing: dict = {"team_workers": workers}
    routing.update(parse_tiers(tiers))
    config = make_config(model, workspace, {"routing": routing})
    agent = await build_agent(config, workspace, mode=mode)
    message = (workspace / "TASK.md").read_text(encoding="utf-8")
    tracker = start_tracking()
    stats = await run_turn(agent, message)
    passed, total, tail = grade(workspace, task)
    row = {
        "label": label,
        "mode": mode,
        "model": model,
        "tiers": tiers,
        "workers": workers if mode == "team" else 1,
        "task": task,
        "repeat": repeat,
        "passed": passed,
        "of": total,
        "success": passed == total,
        "model_calls": stats.model_calls,
        "tool_calls": stats.tool_calls,
        "usage": usage_dict(stats.usage),
        "cost_usd": round(live_cost(tracker), 5),
        "by_model": by_model(tracker),
        "seconds": round(stats.seconds, 1),
        "errors": stats.errors[:3],
        "pytest_tail": tail if passed != total else "",
    }
    record(label, row)
    print(
        f"{label} {mode} {model} w={row['workers']} {task} #{repeat}: {passed}/{total} "
        f"in {row['seconds']:.0f}s, {row['model_calls']} calls, ${row['cost_usd']:.4f}",
        flush=True,
    )
    return row


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--mode", default="react", choices=["react", "team", "plan"])
    parser.add_argument("--models", default="openai/gpt-6-luna")
    parser.add_argument("--workers", default="2")
    parser.add_argument("--tasks", default="textkit,numkit,collkit,parsekit")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument(
        "--tiers", default="", help="fast=model,strong=model,top=model,planner=model"
    )
    args = parser.parse_args()

    gate = asyncio.Semaphore(args.parallel)

    async def guarded(**kwargs):
        async with gate:
            try:
                return await run_once(**kwargs)
            except Exception as exc:  # noqa: BLE001 - one failed run must not stop the rest
                print(f"failed {kwargs}: {exc!r}", flush=True)
                return None

    jobs = [
        guarded(
            label=args.label,
            mode=args.mode,
            model=model,
            workers=int(workers),
            task=task,
            repeat=repeat,
            tiers=args.tiers,
        )
        for repeat in range(1, args.repeats + 1)
        for task in args.tasks.split(",")
        for model in args.models.split(",")
        for workers in args.workers.split(",")
    ]
    await asyncio.gather(*jobs)


if __name__ == "__main__":
    asyncio.run(main())
