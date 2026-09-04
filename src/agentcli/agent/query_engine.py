from __future__ import annotations

import asyncio

from agentcli.agent.agent import Agent
from agentcli.agent.orchestrator import AgentOrchestrator
from agentcli.agent.plan_execute import PlanExecuteAgent
from agentcli.config import AgentCliConfig
from agentcli.llm.base import LlmClient
from agentcli.prompt import PromptAssembler
from agentcli.tools.registry import ToolRegistry
from agentcli.types import Message, QueryResult, Usage


class QueryEngine:
    def __init__(
        self,
        *,
        llm_client: LlmClient,
        tool_registry: ToolRegistry,
        config: AgentCliConfig,
        cwd: str,
        approval_callback=None,
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.config = config
        self.cwd = cwd
        self.approval_callback = approval_callback
        self.system_prompt = PromptAssembler(
            config=config,
            cwd=cwd,
            tool_names=tool_registry.list_names(),
            model=llm_client.model_name,
            provider=llm_client.provider_name,
        ).build_static()

    async def ask(self, message: str, history: list[Message] | None = None):
        agent = Agent(
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            system_prompt=self.system_prompt,
            cwd=self.cwd,
            config=self.config,
            approval_callback=self.approval_callback,
        )
        agent.history = list(history or [])
        async for event in agent.run(message):
            yield event

    async def plan(self, message: str):
        agent = PlanExecuteAgent(
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            config=self.config,
            cwd=self.cwd,
            approval_callback=self.approval_callback,
        )
        async for event in agent.run(message):
            yield event

    async def team(self, message: str, *, worker_mode: str = "react"):
        orchestrator = AgentOrchestrator(
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            config=self.config,
            cwd=self.cwd,
            approval_callback=self.approval_callback,
            default_worker_mode=worker_mode,
        )
        async for event in orchestrator.run(message):
            yield event

    async def ask_complete_async(
        self,
        message: str,
        history: list[Message] | None = None,
    ) -> QueryResult:
        text = ""
        tokens = 0
        turns = 0
        usage = Usage()
        cost = {}
        async for event in self.ask(message, history):
            if event.get("type") == "text_delta":
                text += str(event.get("text") or "")
            elif event.get("type") == "error":
                raise event["error"]
            elif event.get("type") == "done":
                tokens = int(event.get("total_tokens") or 0)
                turns = int(event.get("total_turns") or 0)
                usage = Usage.from_mapping(event.get("usage") or {})
                cost = dict(event.get("cost") or {})
        return QueryResult(text=text, total_tokens=tokens, turns=turns, usage=usage, cost=cost)

    async def plan_complete_async(self, message: str) -> QueryResult:
        return await self._complete_from_events(self.plan(message))

    async def team_complete_async(self, message: str, *, worker_mode: str = "react") -> QueryResult:
        return await self._complete_from_events(self.team(message, worker_mode=worker_mode))

    def ask_complete(self, message: str, history: list[Message] | None = None) -> QueryResult:
        return asyncio.run(self.ask_complete_async(message, history))

    def plan_complete(self, message: str) -> QueryResult:
        return asyncio.run(self.plan_complete_async(message))

    def team_complete(self, message: str, *, worker_mode: str = "react") -> QueryResult:
        return asyncio.run(self.team_complete_async(message, worker_mode=worker_mode))

    async def _complete_from_events(self, events) -> QueryResult:
        text = ""
        tokens = 0
        turns = 0
        usage = Usage()
        cost = {}
        async for event in events:
            if event.get("type") == "text_delta":
                text += str(event.get("text") or "")
            elif event.get("type") == "error":
                raise event["error"]
            elif event.get("type") == "done":
                tokens += int(event.get("total_tokens") or 0)
                turns += int(event.get("total_turns") or 0)
                usage = usage + Usage.from_mapping(event.get("usage") or {})
                cost = dict(event.get("cost") or cost)
        return QueryResult(text=text, total_tokens=tokens, turns=turns, usage=usage, cost=cost)
