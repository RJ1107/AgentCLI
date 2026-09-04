from __future__ import annotations

import asyncio
from typing import Any

from agentcli.agent import PlanExecuteAgent
from agentcli.config import load_config
from agentcli.plan import ExecutionPlan, Planner, Task, TaskType
from agentcli.tools import ToolRegistry, get_builtin_tools


def test_execution_plan_exposes_dag_batches():
    plan = ExecutionPlan(id="plan_1", goal="demo")
    task_1 = Task("task_1", "read a", TaskType.FILE_READ)
    task_2 = Task("task_2", "read b", TaskType.FILE_READ)
    task_3 = Task("task_3", "summarize", TaskType.ANALYSIS, ["task_1", "task_2"])

    plan.add_task(task_1)
    plan.add_task(task_2)
    plan.add_task(task_3)

    assert plan.execution_order() == ["task_1", "task_2", "task_3"]
    assert plan.execution_batches() == [[task_1, task_2], [task_3]]
    assert plan.executable_tasks() == [task_1, task_2]
    task_1.mark_completed("done")
    assert plan.executable_tasks() == [task_2]


def test_execution_plan_summary_uses_chinese_labels():
    plan = ExecutionPlan(id="plan_1", goal="检查项目")
    plan.summary = "先检查再汇总"
    plan.add_task(Task("task_1", "检查文件", TaskType.FILE_READ))

    summary = plan.summarize()

    assert "计划 plan_1：先检查再汇总" in summary
    assert "任务数：1" in summary
    assert "当前可执行：1" in summary


def test_planner_parses_tasks_and_dependencies():
    planner = Planner(FakeClient())

    plan = planner.parse_plan(
        "demo",
        """
        ```json
        {
          "summary": "demo plan",
          "tasks": [
            {"id": "a", "description": "A", "type": "COMMAND", "dependencies": []},
            {"id": "b", "description": "B", "type": "VERIFICATION", "dependencies": ["a"]}
          ]
        }
        ```
        """,
    )

    assert plan.summary == "demo plan"
    assert plan.get_task("task_2").dependencies == ["task_1"]
    assert plan.get_task("task_2").type == TaskType.VERIFICATION


def test_plan_execute_runs_independent_tasks_in_parallel(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    client = ParallelPlanClient()
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    config = load_config(project_root=tmp_path)
    config.policy.hitl_mode = "never"
    agent = PlanExecuteAgent(
        llm_client=client,
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
    )

    async def run():
        text = ""
        events = []
        async for event in agent.run("先做 A 和 B，然后汇总"):
            events.append(event)
            if event.get("type") == "text_delta":
                text += str(event.get("text") or "")
            elif event.get("type") == "error":
                raise event["error"]
        return text, events

    result, events = asyncio.run(run())

    assert "正在规划任务：" in result
    assert "开始执行计划" in result
    assert "已完成 [task_1]" in result
    assert "已完成 [task_2]" in result
    assert "计划执行完成" in result
    assert "Planning task" not in result
    assert "Completed [" not in result
    assert client.task_system_prompts
    assert all(
        "所有进度说明、分析和最终结果都必须使用中文" in prompt
        for prompt in client.task_system_prompts
    )
    assert client.peak_concurrency == 2
    assert any(
        event.get("type") == "thinking_delta" and event.get("phase") == "planning"
        for event in events
    )
    assert {
        event.get("task_id")
        for event in events
        if event.get("type") == "thinking_delta" and event.get("phase") == "execution"
    } == {"task_1", "task_2"}
    assert {
        event.get("task_id") for event in events if event.get("type") == "plan_task_started"
    } == {"task_1", "task_2"}


class FakeClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 1000

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        yield {"type": "text_delta", "text": "{}"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


class ParallelPlanClient(FakeClient):
    def __init__(self):
        self.current_concurrency = 0
        self.peak_concurrency = 0
        self.ready = asyncio.Event()
        self.task_system_prompts: list[str] = []

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        body = _message_text(messages[-1].content)
        if "请为以下目标创建执行计划" in body:
            yield {"type": "thinking_delta", "thinking": "先拆分可并行任务"}
            yield {
                "type": "text_delta",
                "text": (
                    '{"summary":"parallel","tasks":['
                    '{"id":"a","description":"任务 A","type":"ANALYSIS","dependencies":[]},'
                    '{"id":"b","description":"任务 B","type":"ANALYSIS","dependencies":[]}'
                    "]}"
                ),
            }
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return

        if "任务 A" in body or "任务 B" in body:
            self.task_system_prompts.append(system_prompt)
            self.current_concurrency += 1
            self.peak_concurrency = max(self.peak_concurrency, self.current_concurrency)
            if self.current_concurrency == 2:
                self.ready.set()
            await asyncio.wait_for(self.ready.wait(), timeout=2)
            self.current_concurrency -= 1
            text = "A 的结果" if "任务 A" in body else "B 的结果"
            yield {"type": "thinking_delta", "thinking": f"分析{text}"}
            yield {"type": "text_delta", "text": text}
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return

        yield {"type": "text_delta", "text": "fallback"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return str(content)
