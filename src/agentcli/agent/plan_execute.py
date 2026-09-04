from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from agentcli.agent.query import query
from agentcli.config import AgentCliConfig
from agentcli.llm.base import LlmClient
from agentcli.plan import ExecutionPlan, Planner, Task, TaskStatus
from agentcli.prompt import PromptAssembler
from agentcli.skill import SkillContextBuffer
from agentcli.snapshot import SnapshotService
from agentcli.tools.registry import ToolRegistry
from agentcli.types import Message, Usage


@dataclass(slots=True)
class TaskRunResult:
    task: Task
    text: str
    usage: Usage
    turns: int
    error: Exception | None = None

    @property
    def tokens(self) -> int:
        return self.usage.total_tokens


class PlanExecuteAgent:
    def __init__(
        self,
        *,
        llm_client: LlmClient,
        tool_registry: ToolRegistry,
        config: AgentCliConfig,
        cwd: str,
        approval_callback=None,
        planner: Planner | None = None,
        max_task_turns: int = 8,
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.config = config
        self.cwd = cwd
        self.approval_callback = approval_callback
        self.planner = planner or Planner(llm_client)
        self.max_task_turns = max_task_turns
        self.history: list[Message] = []

    async def run(self, message: str) -> AsyncIterator[dict[str, Any]]:
        snapshot = SnapshotService(self.cwd)
        with suppress(Exception):
            snapshot.create("pre-turn")
        total_usage = Usage()
        total_turns = 0
        final_text = ""
        try:
            yield {"type": "text_delta", "text": f"正在规划任务：{message}\n\n"}
            yield {"type": "plan_status", "phase": "planning"}
            plan: ExecutionPlan | None = None
            async for event in self.planner.stream_plan(message):
                if event.get("type") == "plan_created":
                    plan = event["plan"]
                    continue
                if event.get("type") == "usage":
                    total_usage = total_usage + Usage.from_mapping(event.get("usage") or {})
                yield event
            if plan is None:
                raise ValueError("planner did not produce an execution plan")
            yield {"type": "text_delta", "text": plan.summarize() + "\n\n"}
            async for event in self._execute_plan(plan):
                if event.get("type") == "usage":
                    total_usage = total_usage + Usage.from_mapping(event.get("usage") or {})
                elif event.get("type") == "plan_task_done":
                    total_turns += int(event.get("turns") or 0)
                    continue
                elif event.get("type") == "text_delta":
                    final_text += str(event.get("text") or "")
                yield event
            self.history = [
                Message(role="user", content=message),
                Message(role="assistant", content=final_text),
            ]
        except Exception as exc:  # noqa: BLE001
            yield {"type": "error", "error": exc}
            return
        finally:
            with suppress(Exception):
                snapshot.create("post-turn")
        done: dict[str, Any] = {
            "type": "done",
            "total_turns": total_turns,
            "total_tokens": total_usage.total_tokens,
            "usage": total_usage.to_dict(),
            "messages": self.history,
        }
        costs = _calculate_costs(self.llm_client, total_usage)
        if costs:
            done["cost"] = costs
        yield done

    async def _execute_plan(self, plan: ExecutionPlan) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "text_delta", "text": "开始执行计划……\n\n"}
        yield {"type": "plan_status", "phase": "execution"}
        plan.mark_started()
        while True:
            executable = _executable_tasks_in_order(plan)
            if not executable:
                break
            if len(executable) == 1:
                result: TaskRunResult | None = None
                async for event in self._execute_task_stream(plan, executable[0]):
                    if event.get("type") == "plan_task_result":
                        result = event["result"]
                    else:
                        yield event
                if result is None:
                    raise RuntimeError(f"task {executable[0].id} did not produce a result")
                async for event in self._apply_task_result(result):
                    yield event
                continue
            yield {
                "type": "text_delta",
                "text": (f"正在并行执行：{', '.join(task.id for task in executable)}\n\n"),
            }
            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
            runners = [
                asyncio.create_task(self._pump_task_events(plan, task, queue))
                for task in executable
            ]
            results: dict[str, TaskRunResult] = {}
            finished = 0
            try:
                while finished < len(runners):
                    event = await queue.get()
                    event_type = event.get("type")
                    if event_type == "plan_task_stream_done":
                        finished += 1
                    elif event_type == "plan_task_result":
                        result = event["result"]
                        results[result.task.id] = result
                    else:
                        yield event
            finally:
                if finished < len(runners):
                    for runner in runners:
                        if not runner.done():
                            runner.cancel()
                await asyncio.gather(*runners, return_exceptions=True)

            for task in executable:
                result = results.get(task.id)
                if result is None:
                    result = TaskRunResult(
                        task=task,
                        text="",
                        usage=Usage(),
                        turns=0,
                        error=RuntimeError(f"task {task.id} did not produce a result"),
                    )
                async for event in self._apply_task_result(result):
                    yield event

        if plan.has_failed():
            plan.mark_failed()
            yield {"type": "text_delta", "text": "计划部分完成，但有任务执行失败。\n\n"}
        elif plan.is_all_completed():
            plan.mark_completed()
            yield {"type": "text_delta", "text": _build_plan_result(plan)}
        else:
            plan.mark_failed()
            yield {
                "type": "text_delta",
                "text": "计划已停滞，因为任务依赖未满足。\n\n",
            }

    async def _apply_task_result(self, result: TaskRunResult) -> AsyncIterator[dict[str, Any]]:
        if result.error:
            result.task.mark_failed(str(result.error))
            yield {"type": "text_delta", "text": f"失败 [{result.task.id}]：{result.error}\n\n"}
            return
        result.task.mark_completed(result.text)
        yield {
            "type": "text_delta",
            "text": f"已完成 [{result.task.id}]：{_preview(result.text)}\n\n",
        }
        yield {
            "type": "usage",
            "usage": result.usage.to_dict(),
        }
        yield {"type": "plan_task_done", "turns": result.turns, "tokens": result.tokens}

    async def _pump_task_events(
        self,
        plan: ExecutionPlan,
        task: Task,
        queue: asyncio.Queue[dict[str, Any]],
    ) -> None:
        try:
            async for event in self._execute_task_stream(plan, task):
                await queue.put(event)
        finally:
            await queue.put({"type": "plan_task_stream_done", "task_id": task.id})

    async def _execute_task_stream(
        self,
        plan: ExecutionPlan,
        task: Task,
    ) -> AsyncIterator[dict[str, Any]]:
        task.mark_started()
        yield {
            "type": "plan_task_started",
            "task_id": task.id,
            "task_description": task.description,
        }
        text = ""
        tool_results: list[str] = []
        usage = Usage()
        turns = 0
        try:
            async for event in query(
                llm_client=self.llm_client,
                tool_registry=self.tool_registry,
                system_prompt=self._task_system_prompt(plan, task),
                user_message=_task_context(plan, task),
                history=[],
                cwd=self.cwd,
                config=self.config,
                approval_callback=self.approval_callback,
                # Parallel plan tasks must never share one-shot Skill context.
                skill_context_buffer=SkillContextBuffer(),
                max_turns=self.max_task_turns,
            ):
                if event.get("type") == "text_delta":
                    text += str(event.get("text") or "")
                elif event.get("type") == "tool_result":
                    content = str(event.get("result") or "")
                    if content:
                        tool_results.append(content)
                    yield _with_task_context(event, task)
                elif event.get("type") in {
                    "thinking_delta",
                    "tool_call",
                    "context_compressed",
                }:
                    yield _with_task_context(event, task)
                elif event.get("type") == "done":
                    turns += int(event.get("total_turns") or 0)
                    usage = usage + Usage.from_mapping(event.get("usage") or {})
                elif event.get("type") == "error":
                    raise event["error"]
            result_text = text.strip() or "\n".join(tool_results).strip()
            yield {
                "type": "plan_task_result",
                "result": TaskRunResult(task=task, text=result_text, usage=usage, turns=turns),
            }
        except Exception as exc:  # noqa: BLE001
            yield {
                "type": "plan_task_result",
                "result": TaskRunResult(
                    task=task,
                    text="",
                    usage=usage,
                    turns=turns,
                    error=exc,
                ),
            }

    def _task_system_prompt(self, plan: ExecutionPlan, task: Task) -> str:
        base = PromptAssembler(
            config=self.config,
            cwd=self.cwd,
            tool_names=self.tool_registry.list_names(),
            model=self.llm_client.model_name,
            provider=self.llm_client.provider_name,
        ).build_static()
        return (
            base
            + "\n\n你正在执行 Plan-and-Execute DAG 中的一个任务。\n"
            + f"任务 id：{task.id}\n任务类型：{task.type.value}\n"
            + "请具体完成任务，并在需要时使用工具。"
            + _task_language_instruction(plan.goal)
        )


def _with_task_context(event: dict[str, Any], task: Task) -> dict[str, Any]:
    return {
        **event,
        "phase": "execution",
        "task_id": task.id,
        "task_description": task.description,
    }


def _executable_tasks_in_order(plan: ExecutionPlan) -> list[Task]:
    executable_ids = {task.id for task in plan.executable_tasks()}
    return [plan.tasks[task_id] for task_id in plan.execution_order() if task_id in executable_ids]


def _task_context(plan: ExecutionPlan, task: Task) -> str:
    lines = [
        f"目标：{plan.goal}",
        f"当前任务 [{task.id}]：{task.description}",
        "",
        "已完成的依赖任务结果：",
    ]
    for dep_id in task.dependencies:
        dep = plan.get_task(dep_id)
        if dep and dep.status == TaskStatus.COMPLETED:
            lines.append(f"- [{dep.id}] {dep.description}: {_preview(dep.result, 800)}")
    return "\n".join(lines)


def _build_plan_result(plan: ExecutionPlan) -> str:
    status_labels = {
        TaskStatus.PENDING: "待执行",
        TaskStatus.RUNNING: "执行中",
        TaskStatus.COMPLETED: "已完成",
        TaskStatus.FAILED: "失败",
        TaskStatus.SKIPPED: "已跳过",
    }
    lines = ["计划执行完成。", "", "任务摘要："]
    for task in plan.all_tasks():
        lines.append(f"- [{task.id}] {status_labels[task.status]}：{task.description}")
        if task.result:
            lines.append(f"  结果：{_preview(task.result)}")
    return "\n".join(lines) + "\n"


def _task_language_instruction(goal: str) -> str:
    if any("\u4e00" <= char <= "\u9fff" for char in goal):
        return "\n用户目标包含中文；所有进度说明、分析和最终结果都必须使用中文。"
    return "\nUse the same language as the user's goal for all progress and results."


def _preview(text: str, max_len: int = 160) -> str:
    value = (text or "").replace("\r\n", "\n").strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def _calculate_costs(llm_client: LlmClient, usage: Usage) -> dict[str, Any]:
    calculator = getattr(llm_client, "calculate_cost", None)
    if not callable(calculator):
        return {}
    result: dict[str, Any] = {}
    for currency in ("usd", "cny"):
        try:
            result[currency] = calculator(usage, currency=currency).to_dict()
        except (KeyError, TypeError, ValueError):
            continue
    return result
