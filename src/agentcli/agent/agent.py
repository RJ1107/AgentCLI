"""agent.py — Core terminal Agent inspired by Claude Code.

Integrates LLM API, parses natural-language instructions, schedules command
and file operations via the tool system, and generates streaming responses.

Supports three execution modes:
    - react  : Standard ReAct loop (thought → tool → observation → answer)
    - plan   : Plan-then-execute with a DAG of sub-tasks
    - team   : Multi-agent orchestrator (planner → workers → reviewer)

All modes share the same streaming event protocol so callers can render
progress incrementally.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from typing import Any, Literal

from agentcli.agent.orchestrator import AgentOrchestrator
from agentcli.agent.plan_execute import PlanExecuteAgent
from agentcli.config import AgentCliConfig
from agentcli.context import ContextBudget, ContextWindowManager
from agentcli.image import parse_image_references
from agentcli.llm.base import LlmClient
from agentcli.prompt import PromptAssembler
from agentcli.skill import SkillContextBuffer, SkillRegistry
from agentcli.snapshot import SnapshotService
from agentcli.tools.base import ToolContext
from agentcli.tools.executor import ToolExecutor
from agentcli.tools.registry import ToolRegistry
from agentcli.types import Message, QueryResult, Usage

AgentMode = Literal["react", "plan", "team"]


class Agent:
    """A terminal AI agent that connects an LLM to tools for task execution.

    The Agent owns the conversation history, manages context compression, and
    delegates the actual LLM-tool interaction loop to mode-specific runners.
    All ``run()`` methods yield the same streaming event protocol so that UI
    layers (REPL, CLI, or programmatic) can render progress uniformly.

    Typical usage::

        agent = Agent(
            llm_client=my_client,
            tool_registry=my_registry,
            config=my_config,
            cwd="/workspace",
        )
        async for event in agent.run("list all Python files"):
            if event["type"] == "text_delta":
                print(event["text"], end="")
    """

    def __init__(
        self,
        *,
        llm_client: LlmClient,
        tool_registry: ToolRegistry,
        config: AgentCliConfig,
        cwd: str,
        approval_callback: Callable | None = None,
        mode: AgentMode = "react",
        system_prompt: str | None = None,
        max_turns: int = 20,
        max_plan_depth: int = 1,
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.config = config
        self.cwd = cwd
        self.approval_callback = approval_callback
        self.mode = mode
        self.max_turns = max_turns
        self.max_plan_depth = max_plan_depth

        # Build the base system prompt from personality profile and config.
        self.system_prompt = (
            system_prompt
            or PromptAssembler(
                config=config,
                cwd=cwd,
                tool_names=tool_registry.list_names(),
                model=llm_client.model_name,
                provider=llm_client.provider_name,
            ).build_static()
        )

        # Conversation state.
        self.history: list[Message] = []
        self.skill_context_buffer = SkillContextBuffer()

        # Accumulated usage / cost across all turns of the session.
        self.last_usage = Usage()
        self.last_cost: dict[str, Any] = {}

        # Validate configuration.
        self._validate_config()

    # ------------------------------------------------------------------
    # Public API — run the agent
    # ------------------------------------------------------------------

    async def run(self, message: str) -> AsyncIterator[dict[str, Any]]:
        """Execute *message* and yield streaming events.

        Events follow this protocol (``type`` field determines the shape):

        ``text_delta``
            {"type": "text_delta", "text": "..."}
        ``thinking_delta``
            {"type": "thinking_delta", "thinking": "..."}
        ``tool_call``
            {"type": "tool_call", "name": "...", "input": {...}}
        ``tool_result``
            {"type": "tool_result", "name": "...", "result": "...", "is_error": bool}
        ``usage``
            {"type": "usage", "usage": {...}}
        ``turn_complete``
            {"type": "turn_complete", "turn": int, "stop_reason": str}
        ``context_compressed``
            {"type": "context_compressed", "before_tokens": ..., "after_tokens": ...,
             "summarized_messages": int}
        ``error``
            {"type": "error", "error": Exception}
        ``done``
            {"type": "done", "total_turns": int, "total_tokens": int,
             "usage": {...}, "cost": {...}, "messages": [Message, ...]}
        """
        snapshot = SnapshotService(self.cwd)
        with suppress(Exception):
            snapshot.create("pre-turn")

        try:
            if self.mode == "plan":
                async for event in self._run_plan(message):
                    yield event
            elif self.mode == "team":
                async for event in self._run_team(message):
                    yield event
            else:
                async for event in self._run_react(message):
                    yield event
        finally:
            with suppress(Exception):
                snapshot.create("post-turn")

    async def run_complete(self, message: str) -> QueryResult:
        """Run the agent synchronously (collect all events) and return a result."""
        text = ""
        tokens = 0
        turns = 0
        usage = Usage()
        cost: dict[str, Any] = {}
        async for event in self.run(message):
            event_type = event.get("type")
            if event_type == "text_delta":
                text += str(event.get("text") or "")
            elif event_type == "error":
                raise event["error"]  # type: ignore[arg-type]
            elif event_type == "done":
                tokens = int(event.get("total_tokens") or 0)
                turns = int(event.get("total_turns") or 0)
                usage = Usage.from_mapping(event.get("usage") or {})
                cost = dict(event.get("cost") or {})
        return QueryResult(text=text, total_tokens=tokens, turns=turns, usage=usage, cost=cost)

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def clear_history(self) -> None:
        """Reset conversation history and skill context buffer."""
        self.history = []
        self.skill_context_buffer.clear()
        self.last_usage = Usage()
        self.last_cost = {}

    # ------------------------------------------------------------------
    # Mode runners
    # ------------------------------------------------------------------

    async def _run_react(self, message: str) -> AsyncIterator[dict[str, Any]]:
        """Standard ReAct loop (the default and most common mode)."""
        # Prepend skill candidates that match the user message.
        original = message
        message = _prepend_skill_candidates(message, self.cwd, self.config)
        message = _prepend_skill_context(message, self.skill_context_buffer)

        messages = [
            *(self.history or []),
            Message(role="user", content=parse_image_references(message, self.cwd)),
        ]

        tool_defs = self.tool_registry.definitions()
        executor = ToolExecutor(self.tool_registry)
        context = ToolContext(
            cwd=self.cwd,
            config=self.config,
            approval_callback=self.approval_callback,
            skill_context_buffer=self.skill_context_buffer,
        )

        # Build the dynamic part of the system prompt (tool list, cwd, etc.).
        dynamic_prompt = PromptAssembler(
            config=self.config,
            cwd=self.cwd,
            tool_names=self.tool_registry.list_names(),
            model=self.llm_client.model_name,
            provider=self.llm_client.provider_name,
        ).build_dynamic(original)

        effective_system = f"{self.system_prompt}\n\n{dynamic_prompt}".strip()

        # Context-window manager for history compression.
        context_manager = ContextWindowManager(
            ContextBudget(
                context_window=self.llm_client.max_context_window,
                max_output_tokens=self.config.llm.max_tokens,
                compression_threshold=self.config.memory.compression_threshold,
                compression_target=self.config.memory.compression_target,
                reserve_tokens=self.config.memory.compression_reserve_tokens,
            ),
            max_history_messages=self.config.memory.max_conversation_history,
            min_recent_messages=self.config.memory.min_recent_messages,
            summary_max_chars=self.config.memory.summary_max_chars,
        )

        total_usage = Usage()
        turn = 0

        while turn < self.max_turns:
            turn += 1
            text = ""
            thinking = ""
            stop_reason = "end_turn"
            turn_usage = Usage()
            tool_states: dict[int, dict[str, Any]] = {}

            # Compress if needed.
            if self.config.features.context_compression:
                compression = context_manager.prepare(
                    messages,
                    system_prompt=effective_system,
                    tool_definitions=tool_defs,
                )
                messages = compression.messages
                if compression.compressed:
                    yield {
                        "type": "context_compressed",
                        "before_tokens": compression.estimated_tokens_before,
                        "after_tokens": compression.estimated_tokens_after,
                        "summarized_messages": compression.summarized_messages,
                    }

            # Call the LLM.
            async for event in self.llm_client.chat(
                messages,
                tool_defs,
                system_prompt=effective_system,
            ):
                event_type = event.get("type")
                if event_type == "text_delta":
                    delta = str(event.get("text") or "")
                    text += delta
                    yield {"type": "text_delta", "text": delta}
                elif event_type == "thinking_delta":
                    delta = str(event.get("thinking") or "")
                    thinking += delta
                    yield {"type": "thinking_delta", "thinking": delta}
                elif event_type == "tool_call_delta":
                    _merge_tool_delta(tool_states, event["tool_call"])
                elif event_type == "message_end":
                    stop_reason = str(event.get("stop_reason") or "end_turn")
                elif event_type == "usage":
                    usage = Usage.from_mapping(event.get("usage") or {})
                    turn_usage = turn_usage + usage
                    yield {"type": "usage", "usage": usage.to_dict()}
                elif event_type == "error":
                    yield {"type": "error", "error": event["error"]}
                    return

            total_usage = total_usage + turn_usage
            tool_calls = _finalize_tool_calls(tool_states)
            assistant_msg = Message(
                role="assistant",
                content=text or "",
                tool_calls=tool_calls,
            )
            if thinking and not text:
                assistant_msg.content = ""
            messages.append(assistant_msg)
            yield {"type": "turn_complete", "turn": turn, "stop_reason": stop_reason}

            # If the model didn't request any tools, we're done.
            if stop_reason != "tool_use" and not tool_calls:
                break

            # Run tools.
            for call in tool_calls:
                name = call.get("function", {}).get("name", "unknown")
                yield {"type": "tool_call", "name": name, "input": _tool_input(call)}

            tool_results = await executor.execute_all(tool_calls, context)
            loaded_skill_context = _drain_skill_context(self.skill_context_buffer)
            load_skill_ids = [
                str(call.get("id") or "")
                for call in tool_calls
                if str(call.get("function", {}).get("name") or "") == "load_skill"
            ]
            injection_target = load_skill_ids[-1] if load_skill_ids else ""
            injected = False

            for result in tool_results:
                yield {
                    "type": "tool_result",
                    "name": _tool_name_by_id(tool_calls, result.tool_use_id or ""),
                    "result": result.content,
                    "is_error": result.is_error,
                }
                model_content = result.content
                if loaded_skill_context and result.tool_use_id == injection_target:
                    model_content = f"{model_content}\n\n{loaded_skill_context}"
                    injected = True
                messages.append(
                    Message(
                        role="tool",
                        content=model_content,
                        tool_call_id=result.tool_use_id,
                    )
                )

            if loaded_skill_context and not injected:
                messages.append(
                    Message(
                        role="user",
                        content=(
                            f"{loaded_skill_context}\n\n"
                            "Use these loaded instructions to continue the current request."
                        ),
                    )
                )

        # Persist history for next user message and report final usage.
        self.history = list(messages)
        self.last_usage = total_usage

        done_event: dict[str, Any] = {
            "type": "done",
            "total_turns": turn,
            "total_tokens": total_usage.total_tokens,
            "usage": total_usage.to_dict(),
            "messages": self.history,
        }
        costs = _calculate_costs(self.llm_client, total_usage)
        if costs:
            done_event["cost"] = costs
            self.last_cost = costs
        yield done_event

    async def _run_plan(self, message: str) -> AsyncIterator[dict[str, Any]]:
        """Plan-then-execute mode: the planner creates a DAG, then workers run it."""
        agent = PlanExecuteAgent(
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            config=self.config,
            cwd=self.cwd,
            approval_callback=self.approval_callback,
            max_task_turns=self.max_turns,
        )
        agent.history = list(self.history)

        async for event in agent.run(message):
            if event.get("type") == "done":
                self.history = list(event.get("messages") or [])
                self.last_usage = Usage.from_mapping(event.get("usage") or {})
                self.last_cost = dict(event.get("cost") or {})
            yield event

    async def _run_team(self, message: str) -> AsyncIterator[dict[str, Any]]:
        """Multi-agent team mode: planner → parallel workers → reviewer."""
        orchestrator = AgentOrchestrator(
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            config=self.config,
            cwd=self.cwd,
            approval_callback=self.approval_callback,
            default_worker_mode="react",
        )
        async for event in orchestrator.run(message):
            if event.get("type") == "done":
                self.history = list(event.get("messages") or [])
                self.last_usage = Usage.from_mapping(event.get("usage") or {})
                self.last_cost = dict(event.get("cost") or {})
            yield event

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        """Raise early if the configuration is obviously wrong."""
        if not self.config.llm.api_key:
            raise ValueError(
                "LLM API key is not configured. "
                "Set AGENTCLI_API_KEY (or DEEPSEEK_API_KEY / GLM_API_KEY / etc.) "
                "in the environment or in ~/.agentcli/config.json."
            )
        if not self.llm_client.max_context_window:
            raise ValueError(
                "LLM context window is not configured. "
                "Set an AGENTCLI_CONTEXT_WINDOW environment variable or configure "
                "it in ~/.agentcli/config.json."
            )


# ------------------------------------------------------------------
# Shared helpers (also used by query.py)
# ------------------------------------------------------------------


def _merge_tool_delta(tool_states: dict[int, dict[str, Any]], delta: dict[str, Any]) -> None:
    index = int(delta.get("index") or 0)
    state = tool_states.setdefault(
        index,
        {
            "id": delta.get("id") or f"tool_{index}",
            "type": "function",
            "function": {"name": "", "arguments": ""},
        },
    )
    if delta.get("id"):
        state["id"] = delta["id"]
    function = delta.get("function") or {}
    if function.get("name"):
        state["function"]["name"] = function["name"]
    if function.get("arguments"):
        state["function"]["arguments"] += function["arguments"]


def _finalize_tool_calls(tool_states: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    calls = []
    for index in sorted(tool_states):
        state = tool_states[index]
        if state["function"]["name"]:
            calls.append(state)
    return calls


def _tool_input(call: dict[str, Any]) -> dict[str, Any]:
    import json

    raw = call.get("function", {}).get("arguments") or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _tool_name_by_id(calls: list[dict[str, Any]], tool_call_id: str) -> str:
    for call in calls:
        if call.get("id") == tool_call_id:
            return str(call.get("function", {}).get("name") or "unknown")
    return "unknown"


def _prepend_skill_candidates(user_message: str, cwd: str, config: AgentCliConfig) -> str:
    if not config.features.skill:
        return user_message
    candidates = SkillRegistry(cwd).match(user_message, top_k=5)
    if not candidates:
        return user_message
    lines = [
        "Relevant skill candidates for this request:",
        "Call load_skill(name) before proceeding when a candidate applies.",
    ]
    for skill in candidates:
        description = " ".join(skill.description.split())
        if len(description) > 300:
            description = description[:297] + "..."
        tags = f" [tags: {', '.join(skill.tags)}]" if skill.tags else ""
        lines.append(f"- {skill.name}: {description}{tags}")
    candidate_text = "\n".join(lines)
    return f"{candidate_text}\n\n---\nUser request:\n{user_message}"


def _prepend_skill_context(user_message: str, skill_context_buffer: SkillContextBuffer) -> str:
    if not skill_context_buffer or skill_context_buffer.is_empty():
        return user_message
    drained = skill_context_buffer.drain()
    if not drained:
        return user_message
    return f"{drained}\n\n---\nUser request:\n{user_message}"


def _drain_skill_context(skill_context_buffer: SkillContextBuffer) -> str:
    if not skill_context_buffer or skill_context_buffer.is_empty():
        return ""
    return skill_context_buffer.drain()


def _calculate_costs(llm_client: LlmClient, usage: Usage) -> dict[str, Any]:
    calculator = getattr(llm_client, "calculate_cost", None)
    if not callable(calculator):
        return {}
    result: dict[str, Any] = {}
    for currency in ("usd", "cny"):
        try:
            breakdown = calculator(usage, currency=currency)
        except (KeyError, TypeError, ValueError):
            continue
        result[currency] = breakdown.to_dict()
    return result
