from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from agentcli.llm.base import LlmClient
from agentcli.plan.models import ExecutionPlan, Task, TaskType
from agentcli.types import Message, Usage

PLANNER_PROMPT = """你是 AgentCLI 的任务规划器。
请为用户任务创建一个简洁、可执行的 DAG，并仅返回以下结构的 JSON：
{
  "summary": "short summary",
  "tasks": [
    {
      "id": "stable_source_id",
      "description": "concrete executable step",
      "type": "FILE_READ|FILE_WRITE|COMMAND|ANALYSIS|VERIFICATION",
      "dependencies": ["stable_source_id"]
    }
  ]
}
可以并行的独立任务应放在同一执行批次中。
summary 和 description 必须使用与用户目标相同的语言；用户目标包含中文时，必须使用中文。
JSON 字段名、任务 id 和 type 枚举值保持上述英文格式。
"""


class Planner:
    def __init__(self, llm_client: LlmClient):
        self.llm_client = llm_client
        self.last_usage = Usage()

    async def create_plan(self, goal: str) -> ExecutionPlan:
        plan: ExecutionPlan | None = None
        async for event in self.stream_plan(goal):
            if event.get("type") == "plan_created":
                plan = event["plan"]
        if plan is None:
            raise ValueError("planner did not produce an execution plan")
        return plan

    async def stream_plan(self, goal: str) -> AsyncIterator[dict[str, Any]]:
        """Create a plan while preserving provider reasoning and usage events."""
        self.last_usage = Usage()
        if _is_simple_goal(goal):
            yield {"type": "plan_created", "plan": _minimal_plan(goal)}
            return

        text = ""
        messages = [Message(role="user", content=f"请为以下目标创建执行计划：\n{goal}")]
        async for event in self.llm_client.chat(messages, [], system_prompt=PLANNER_PROMPT):
            event_type = event.get("type")
            if event_type == "text_delta":
                # Planner text is machine-readable JSON. Keep it out of the user-facing
                # stream and expose the parsed plan below instead.
                text += str(event.get("text") or "")
            elif event_type == "thinking_delta":
                yield {
                    "type": "thinking_delta",
                    "thinking": str(event.get("thinking") or ""),
                    "phase": "planning",
                }
            elif event_type == "usage":
                usage = Usage.from_mapping(event.get("usage") or {})
                self.last_usage = self.last_usage + usage
                yield {"type": "usage", "usage": usage.to_dict(), "phase": "planning"}
            elif event_type == "error":
                raise event["error"]

        yield {"type": "plan_created", "plan": self.parse_plan(goal, text)}

    async def replan(self, failed_plan: ExecutionPlan, failure_reason: str) -> ExecutionPlan:
        completed = "\n".join(
            f"- {task.id}: {task.description}"
            for task in failed_plan.all_tasks()
            if task.result and not task.error
        )
        return await self.create_plan(
            f"{failed_plan.goal}\n失败原因：{failure_reason}\n已完成任务：\n{completed}"
        )

    def parse_plan(self, goal: str, plan_json: str) -> ExecutionPlan:
        data = _parse_json_object(plan_json)
        task_nodes = data.get("tasks") or data.get("steps") or []
        if not isinstance(task_nodes, list) or not task_nodes:
            raise ValueError("planner output did not contain a non-empty tasks/steps array")

        plan = ExecutionPlan(id=f"plan_{int(time.time() * 1000)}", goal=goal)
        plan.summary = str(data.get("summary") or "")
        id_mapping: dict[str, str] = {}

        for index, node in enumerate(task_nodes, start=1):
            if not isinstance(node, dict):
                continue
            original_id = str(node.get("id") or f"task_{index}")
            new_id = f"task_{index}"
            id_mapping[original_id] = new_id
            plan.add_task(
                Task(
                    id=new_id,
                    description=str(node.get("description") or original_id),
                    type=_parse_task_type(str(node.get("type") or "ANALYSIS")),
                )
            )

        for index, node in enumerate(task_nodes, start=1):
            if not isinstance(node, dict):
                continue
            task = plan.get_task(f"task_{index}")
            if not task:
                continue
            dependencies = node.get("dependencies") or []
            if not isinstance(dependencies, list):
                continue
            for raw_dep in dependencies:
                dep_id = id_mapping.get(str(raw_dep), str(raw_dep))
                if dep_id in plan.tasks:
                    task.add_dependency(dep_id)
                    plan.tasks[dep_id].add_dependent(task.id)

        if not plan.compute_execution_order():
            raise ValueError("plan contains a cyclic dependency")
        return plan


async def _collect_text(
    llm_client: LlmClient,
    messages: list[Message],
    *,
    system_prompt: str,
) -> str:
    text, _usage = await _collect_text_and_usage(llm_client, messages, system_prompt=system_prompt)
    return text


async def _collect_text_and_usage(
    llm_client: LlmClient,
    messages: list[Message],
    *,
    system_prompt: str,
) -> tuple[str, Usage]:
    text = ""
    usage = Usage()
    async for event in llm_client.chat(messages, [], system_prompt=system_prompt):
        event_type = event.get("type")
        if event_type == "text_delta":
            text += str(event.get("text") or "")
        elif event_type == "usage":
            usage = usage + Usage.from_mapping(event.get("usage") or {})
        elif event_type == "error":
            raise event["error"]
    return text, usage


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?\s*", "", text or "").replace("```", "").strip()
    if not cleaned:
        raise ValueError("empty planner output")
    return json.loads(cleaned)


def _parse_task_type(value: str) -> TaskType:
    normalized = value.upper()
    try:
        return TaskType(normalized)
    except ValueError:
        return TaskType.ANALYSIS


def _is_simple_goal(goal: str | None) -> bool:
    normalized = (goal or "").strip()
    if not normalized or len(normalized) > 30:
        return False
    multi_step_cues = ["然后", "并且", "再", "最后", "同时", "先", "之后", "接着", "以及"]
    if any(cue in normalized for cue in multi_step_cues):
        return False
    simple_cues = ["列出", "查看", "读取", "显示", "执行", "运行", "搜索", "当前目录", "文件"]
    return any(cue in normalized for cue in simple_cues)


def _minimal_plan(goal: str) -> ExecutionPlan:
    normalized = goal.strip()
    plan = ExecutionPlan(id=f"plan_{int(time.time() * 1000)}", goal=normalized)
    plan.summary = f"直接执行简单任务：{normalized}"
    plan.add_task(Task(id="task_1", description=normalized, type=_infer_simple_type(normalized)))
    plan.compute_execution_order()
    return plan


def _infer_simple_type(goal: str) -> TaskType:
    if any(token in goal for token in ["读取", "打开", "查看"]) and "文件" in goal:
        return TaskType.FILE_READ
    if any(token in goal for token in ["写入", "修改", "创建文件"]):
        return TaskType.FILE_WRITE
    if any(token in goal for token in ["分析", "总结", "解释"]):
        return TaskType.ANALYSIS
    if any(token in goal for token in ["验证", "检查"]):
        return TaskType.VERIFICATION
    return TaskType.COMMAND
