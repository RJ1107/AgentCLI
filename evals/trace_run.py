import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import build_agent, fresh_workspace, make_config

TASKS = Path(__file__).resolve().parent / "fixtures" / "tasks"


async def main(model: str, task: str) -> None:
    ws = fresh_workspace(TASKS / task, f"trace-{task}")
    config = make_config(model, ws)
    agent = await build_agent(config, ws)
    text = ""
    async for e in agent.run((ws / "TASK.md").read_text(encoding="utf-8")):
        t = e.get("type")
        if t == "tool_call":
            print("CALL", e["name"], str(e.get("input"))[:160].replace("\n", " "))
        elif t == "tool_result":
            print(
                "  ->",
                "ERR" if e.get("is_error") else "ok",
                str(e.get("result"))[:200].replace("\n", " "),
            )
        elif t == "text_delta":
            text += e.get("text") or ""
        elif t == "turn_complete":
            print("TURN", e.get("turn"), e.get("stop_reason"))
        elif t == "usage":
            print("  usage out", (e.get("usage") or {}).get("output_tokens"))
        elif t == "error":
            print("ERROR", e.get("error"))
    print("FINAL TEXT:", text[-600:])


asyncio.run(main(sys.argv[1], sys.argv[2]))
