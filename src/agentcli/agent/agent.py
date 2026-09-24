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

import time
from collections.abc import AsyncIterator, Callable
from typing import Any, Literal

from agentcli.agent.orchestrator import AgentOrchestrator
from agentcli.agent.plan_execute import PlanExecuteAgent
from agentcli.agent.query import build_context_manager, query
from agentcli.config import AgentCliConfig
from agentcli.context import CompressionResult, build_summarizer, estimate_request_tokens
from agentcli.llm.base import LlmClient
from agentcli.prompt import PromptAssembler
from agentcli.routing import ModelTiers
from agentcli.skill import SkillContextBuffer
from agentcli.snapshot import TurnSnapshot
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
        # Path -> mtime_ns at last read/write. Edits to files changed since then are refused.
        self.file_state: dict[str, int] = {}
        # When the conversation last went to the model (a request or a compaction). Frontends
        # use it to warn that a long pause has probably let the provider's prompt cache expire.
        self.last_active_at: float | None = None

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
        turn_snapshot = TurnSnapshot(self.cwd)
        if self.mode == "plan":
            runner = self._run_plan(message, turn_snapshot)
        elif self.mode == "team":
            runner = self._run_team(message, turn_snapshot)
        else:
            runner = self._run_react(message, turn_snapshot)
        try:
            async for event in runner:
                yield event
        finally:
            self.last_active_at = time.time()

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
        self.file_state.clear()
        self.last_active_at = None
        self.last_usage = Usage()
        self.last_cost = {}

    def context_tokens(self) -> int:
        """Estimated size of the next request's input before any new message is added."""

        return estimate_request_tokens(
            self.history, self.system_prompt, self.tool_registry.definitions()
        )

    async def compact(self, focus: str = "") -> CompressionResult | None:
        """Summarize the conversation now (/compact), whatever its size.

        Recent turns stay verbatim and older ones become one summary, written by the
        configured summarizer model when there is one. focus names what to keep in detail.
        Returns None when there is nothing to compact.
        """

        if not self.history:
            return None
        summarizer = build_summarizer(self.llm_client, self.config)
        result = await build_context_manager(self.llm_client, self.config).prepare_async(
            self.history,
            system_prompt=self.system_prompt,
            tool_definitions=self.tool_registry.definitions(),
            summarizer=summarizer,
            force=True,
            focus=focus,
        )
        self.history = result.messages
        # The next request re-reads this new, shorter history anyway; no reminder for it.
        self.last_active_at = time.time()
        if summarizer:
            self.last_usage = summarizer.take_usage()
        return result

    # ------------------------------------------------------------------
    # Mode runners
    # ------------------------------------------------------------------

    async def _run_react(
        self, message: str, turn_snapshot: TurnSnapshot | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Standard ReAct loop (the default and most common mode)."""
        async for event in query(
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            system_prompt=self.system_prompt,
            user_message=message,
            history=self.history,
            cwd=self.cwd,
            config=self.config,
            approval_callback=self.approval_callback,
            skill_context_buffer=self.skill_context_buffer,
            max_turns=self.max_turns,
            file_state=self.file_state,
            turn_snapshot=turn_snapshot,
        ):
            if event.get("type") == "done":
                # Persist the (possibly compacted) history for the next user message.
                self.history = list(event.get("messages") or [])
                self.last_usage = Usage.from_mapping(event.get("usage") or {})
                self.last_cost = dict(event.get("cost") or {})
            yield event

    async def _run_plan(
        self, message: str, turn_snapshot: TurnSnapshot | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Plan-then-execute mode: the planner creates a DAG, then workers run it."""
        agent = PlanExecuteAgent(
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            config=self.config,
            cwd=self.cwd,
            approval_callback=self.approval_callback,
            max_task_turns=self.max_turns,
            turn_snapshot=turn_snapshot,
            tiers=ModelTiers(self.config, self.llm_client),
        )
        agent.history = list(self.history)

        async for event in agent.run(message):
            if event.get("type") == "done":
                self.history = list(event.get("messages") or [])
                self.last_usage = Usage.from_mapping(event.get("usage") or {})
                self.last_cost = dict(event.get("cost") or {})
            yield event

    async def _run_team(
        self, message: str, turn_snapshot: TurnSnapshot | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Multi-agent team mode: planner → parallel workers → reviewer."""
        orchestrator = AgentOrchestrator(
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            config=self.config,
            cwd=self.cwd,
            approval_callback=self.approval_callback,
            default_worker_mode="react",
            turn_snapshot=turn_snapshot,
            tiers=ModelTiers(self.config, self.llm_client),
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
