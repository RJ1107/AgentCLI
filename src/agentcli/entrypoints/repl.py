from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from agentcli import __version__
from agentcli.agent import Agent, AgentOrchestrator, PlanExecuteAgent
from agentcli.bootstrap import build_tool_registry
from agentcli.config import AgentCliConfig, config_to_public_dict
from agentcli.context.spill import use_session_folder
from agentcli.entrypoints.model_selector import ModelSelectorState, run_model_selector
from agentcli.entrypoints.session_selector import pick_session
from agentcli.llm import create_llm_client
from agentcli.llm.model_profiles import (
    DEFAULT_MODEL_PROFILES,
    PROVIDER_DEFAULTS,
    CustomModelStore,
    ModelProfile,
)
from agentcli.memory import FileMemory
from agentcli.paths import agentcli_home
from agentcli.policy import AuditLog
from agentcli.prompt import PromptAssembler
from agentcli.rag import CodeIndex
from agentcli.render import RichRenderer
from agentcli.routing import IntentRouter, ModelTiers, RouteDecision
from agentcli.runtime import DurableTaskManager
from agentcli.sandbox import close_all as close_sandboxes
from agentcli.sandbox import prepare_image, sandbox_status
from agentcli.sandbox.docker import docker_available
from agentcli.session import Session, SessionStore
from agentcli.skill import SkillRegistry
from agentcli.snapshot import SnapshotService
from agentcli.tools import ToolRegistry
from agentcli.types import Message, Usage

COMMAND_HELP: list[tuple[str, str, str]] = [
    # (command, usage, what it does)
    ("/help", "/help", "列出所有命令"),
    ("/exit", "/exit 或 /quit", "退出（也可以按 Ctrl+D）"),
    ("/new", "/new", "开一个新会话（当前会话自动保存，随时可以 /resume 回来）"),
    ("/clear", "/clear", "同 /new，并清屏"),
    ("/resume", "/resume [会话 id]", "恢复本项目之前的会话；不带参数打开选择器"),
    ("/sessions", "/sessions [delete <id>]", "列出（或删除）本项目保存的会话"),
    ("/rename", "/rename <名字>", "给当前会话改名"),
    ("/compact", "/compact [重点]", "立刻把对话压缩成摘要；可以写明要重点保留的内容"),
    ("/context", "/context", "当前模型、上下文大小、空闲时长等"),
    ("/usage", "/usage", "上一次请求的 token 用量、缓存命中和费用"),
    ("/model", "/model [模型] 或 /model <提供商> <模型>", "查看或切换模型；不带参数打开选择器"),
    ("/plan", "/plan <任务>", "先规划成 DAG，再按依赖并行执行"),
    ("/team", "/team [--plan] <任务>", "多 Agent：Planner 拆解、Worker 并行、Reviewer 审核"),
    (
        "/memory",
        "/memory [search <词>|show <名字>|delete <名字>|stats|path|clear]",
        "查看和管理长期记忆（每条一个文件）",
    ),
    ("/save", "/save <事实>", "手动存一条长期记忆"),
    ("/skill", "/skill [list|show|on|off|reload] [名字]", "查看、启用、停用 Skill"),
    ("/tools", "/tools", "列出当前可用的工具"),
    ("/mcp", "/mcp", "MCP 服务相关的提示"),
    (
        "/hitl",
        "/hitl default|auto",
        "审批模式：default 需要审批，auto 全部放行（也可按 Shift+Tab）",
    ),
    ("/policy", "/policy", "查看当前安全策略"),
    ("/sandbox", "/sandbox", "查看命令在哪里执行：Docker 沙箱还是本机"),
    ("/audit", "/audit [N]", "查看最近 N 条审计日志"),
    ("/snapshot", "/snapshot [clean]", "列出（或清空）工作区快照"),
    ("/restore", "/restore <编号或 id>", "把工作区恢复到某个快照"),
    ("/task", "/task [add|cancel|log] ...", "后台任务：添加、取消、查看日志"),
    ("/index", "/index [路径]", "为代码建立本地搜索索引"),
    ("/search", "/search <词>", "在代码索引里搜索"),
    ("/config", "/config", "查看当前配置（密钥已隐藏）"),
]
SLASH_COMMANDS = [command for command, _usage, _about in COMMAND_HELP]


@dataclass(slots=True)
class CacheReminder:
    level: Literal["cold", "stale"]
    idle_seconds: float
    tokens: int
    extra_cost: str


def cache_reminder(
    *, idle_seconds: float | None, tokens: int, config: AgentCliConfig, llm_client: Any = None
) -> CacheReminder | None:
    """Whether the next message will likely re-read a large context at full price.

    Providers do not report cache expiry, so this assumes memory.cache_ttl_minutes. Small
    contexts are skipped: re-reading them costs little and compacting them saves little.
    """

    memory = config.memory
    if idle_seconds is None or tokens < memory.idle_reminder_min_tokens:
        return None
    if idle_seconds < memory.cache_ttl_minutes * 60:
        return None
    level = "stale" if idle_seconds >= memory.stale_session_hours * 3600 else "cold"
    return CacheReminder(level, idle_seconds, tokens, _extra_cost(llm_client, tokens))


def _extra_cost(llm_client: Any, tokens: int) -> str:
    """How much more a full-price re-read costs than a cached one, when prices are known."""

    calculate = getattr(llm_client, "calculate_cost", None)
    if not callable(calculate) or getattr(llm_client, "price_profile", None) is None:
        return ""
    for currency, symbol in (("cny", "¥"), ("usd", "$")):
        try:
            miss = calculate(
                Usage(input_tokens=tokens, cache_miss_tokens=tokens), currency=currency
            )
            hit = calculate(Usage(input_tokens=tokens, cache_hit_tokens=tokens), currency=currency)
        except (KeyError, TypeError, ValueError):
            continue
        return f"{symbol}{miss.total_cost - hit.total_cost:.2f}"
    return ""


def _duration(seconds: float) -> str:
    hours, minutes = int(seconds // 3600), int(seconds % 3600 // 60)
    return f"{hours} 小时 {minutes} 分钟" if hours else f"{minutes} 分钟"


def _confirm_after_idle(console: Console, reminder: CacheReminder) -> str:
    """Ask what to do with a likely cold cache: send, compact first, or cancel."""

    if not sys.stdin.isatty():
        return "y"
    cost = f"，比命中缓存多花约 {reminder.extra_cost}" if reminder.extra_cost else ""
    console.print(
        f"[yellow]距离上次请求已经 {_duration(reminder.idle_seconds)}[/yellow]，"
        "服务商的提示缓存很可能已经过期。"
        f"当前上下文约 {reminder.tokens / 10_000:.1f} 万 token，这次会按全价重新读取{cost}。"
    )
    if reminder.level == "stale":
        console.print("离开时间较长，早期的对话细节可能已经不再需要，建议先压缩再继续。")
    default = "c" if reminder.level == "stale" else "y"
    console.print("[dim]y 直接发送 · c 先压缩再发送 · n 取消[/dim]")
    return Prompt.ask("继续？", choices=["y", "c", "n"], default=default)


_MODE_PITCH = {
    "plan": "/plan：先拆成多个步骤，再按依赖顺序（能并行的并行）执行",
    "team": "/team：拆给多个 Worker 并行做，每一步都有 Reviewer 把关",
}


def _confirm_mode(console: Console, decision: RouteDecision) -> str:
    """Offer /plan or /team for a request that looks large. Returns r, p, t, or x."""

    if not sys.stdin.isatty():
        return "r"
    why = "；".join(decision.reasons) or "需求较复杂"
    source = {"rules": "规则判断", "model": "模型判断", "jev": "Jev 判断"}.get(
        decision.source, decision.source
    )
    console.print(f"[yellow]这个任务看起来比较大[/yellow]（{source}）：{why}。")
    console.print(f"建议用 {_MODE_PITCH[decision.mode]}。")
    console.print("[dim]r 直接执行 · p 用 /plan · t 用 /team · x 直接执行，本次会话不再提示[/dim]")
    return Prompt.ask("怎么做？", choices=["r", "p", "t", "x"], default=decision.mode[0])


async def _compact(agent: Agent, console: Console, focus: str = "") -> None:
    before = agent.context_tokens()
    if not agent.history:
        console.print("当前没有对话，不需要压缩。")
        return
    with console.status("正在压缩对话……"):
        result = await agent.compact(focus)
    if result is None:
        console.print("当前没有对话，不需要压缩。")
        return
    how = {"llm": "模型摘要", "extractive": "规则抽取"}.get(result.method, result.method)
    console.print(
        f"已压缩：{before:,} → {agent.context_tokens():,} token，"
        f"{result.summarized_messages} 条较早的消息合成了摘要（{how}）"
        + (
            f"，清理了 {result.cleared_tool_results} 条旧工具结果"
            if result.cleared_tool_results
            else ""
        )
        + "。"
    )


PermissionMode = Literal["default", "auto"]


@dataclass
class PermissionModeController:
    """Apply one of the two interactive permission modes to the live config."""

    config: AgentCliConfig
    mode: PermissionMode = "default"

    def __post_init__(self) -> None:
        self._default_hitl_mode = self.config.policy.hitl_mode
        self._default_path_guard_enabled = self.config.policy.path_guard_enabled
        self._default_command_guard_enabled = self.config.policy.command_guard_enabled
        self.set(self.mode)

    def set(self, mode: PermissionMode) -> PermissionMode:
        self.mode = mode
        if mode == "auto":
            self.config.policy.hitl_mode = "never"
            self.config.policy.path_guard_enabled = False
            self.config.policy.command_guard_enabled = False
        else:
            self.config.policy.hitl_mode = self._default_hitl_mode
            self.config.policy.path_guard_enabled = self._default_path_guard_enabled
            self.config.policy.command_guard_enabled = self._default_command_guard_enabled
        return self.mode

    def toggle(self) -> PermissionMode:
        return self.set("auto" if self.mode == "default" else "default")


class _ChatSession:
    """The saved session behind the REPL: resumable, switchable, written after each turn.

    Nothing is written until the first message is sent, so opening AgentCLI and leaving
    leaves no empty session behind.
    """

    def __init__(
        self,
        store: SessionStore,
        cwd: str,
        agent: Agent,
        system_prompt: Callable[[], str] | None = None,
    ):
        self.store = store
        self.cwd = cwd
        self.agent = agent
        # Rebuilds the system prompt when a conversation starts over, so the memory index
        # in it includes what the last one saved. Within a conversation it never changes.
        self.system_prompt = system_prompt
        self.current = self._fresh()

    def _fresh(self) -> Session:
        session = self.store.create(self.cwd, self.agent.llm_client.model_name)
        use_session_folder(session.folder)
        return session

    def record(self, user_text: str) -> None:
        self.current.record_turn(
            user_text,
            _last_answer(self.agent.history),
            self.agent.history,
            self.agent.llm_client.model_name,
        )

    def save(self) -> None:
        self.current.save_history(self.agent.history)

    def start_new(self) -> None:
        self.save()
        self.agent.clear_history()
        self._refresh_prompt()
        self.current = self._fresh()

    def load(self, session: Session) -> None:
        self.save()
        self.agent.clear_history()
        self.agent.history = session.load_history()
        self._refresh_prompt()
        # The idle reminder then knows how long ago this conversation last reached the model.
        self.agent.last_active_at = session.meta.updated_at or None
        self.current = session
        use_session_folder(session.folder)

    def _refresh_prompt(self) -> None:
        if self.system_prompt:
            self.agent.system_prompt = self.system_prompt()

    async def resume(self, target: str, console: Console) -> bool:
        """target: "last", "pick", or a session id (or a unique prefix of one)."""

        if target == "last":
            session = self.store.latest(self.cwd)
        elif target == "pick":
            meta = await pick_session(self.store.list(self.cwd), self.current.id)
            if meta is None:
                return False
            session = self.store.open(meta.id)
        else:
            session = self.store.find(target, self.cwd) or self.store.find(target)
        if session is None:
            console.print("[yellow]没有找到可以恢复的会话。[/yellow]")
            return False
        self.load(session)
        meta = session.meta
        console.print(
            f"[green]已恢复会话[/green]：{meta.title or '(未命名)'}"
            f"[dim]（{meta.turns} 轮，最后更新 {meta.updated_text}，id {meta.id}）[/dim]"
        )
        return True


def _last_answer(history: list[Message]) -> str:
    for message in reversed(history):
        if message.role == "assistant" and isinstance(message.content, str) and message.content:
            return message.content
    return ""


async def start_repl(cwd: str, config: AgentCliConfig, resume: str | None = None) -> None:
    console = Console()
    _prepare_sandbox(config, console)
    permission_mode = PermissionModeController(config)
    registry, mcp_manager = await build_tool_registry(config=config, cwd=cwd)
    client = create_llm_client(config.llm)
    assembler = PromptAssembler(
        config=config,
        cwd=cwd,
        tool_names=registry.list_names(),
        model=client.model_name,
        provider=client.provider_name,
    )
    system_prompt = assembler.build_static()
    tool_count = len(registry.list_names())
    mcp_server_count = _count_mcp_servers(mcp_manager)
    skill_count = len(SkillRegistry(cwd).list())
    # The files that are really in the system prompt, not every AGENTS.md in the tree.
    agents_file_count = len(assembler.instruction_files())
    renderer = RichRenderer(context_window=client.max_context_window)
    renderer.banner(
        model=client.model_name,
        provider=client.provider_name,
        cwd=cwd,
        tools=tool_count,
        version=__version__,
        api_key_configured=bool(config.llm.api_key),
        mcp_servers=mcp_server_count,
        skills=skill_count,
        agents_files=agents_file_count,
        hitl_mode=config.policy.hitl_mode,
    )
    agent = Agent(
        llm_client=client,
        tool_registry=registry,
        system_prompt=system_prompt,
        cwd=cwd,
        config=config,
        approval_callback=lambda request: _approval_prompt(request, console, permission_mode),
    )
    store = SessionStore()
    store.cleanup(config.memory.session_retention_days)
    chat = _ChatSession(store, cwd, agent, assembler.build_static)
    if resume:
        await chat.resume(resume, console)

    history_path = agentcli_home() / "history" / "prompt_history.txt"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    session = PromptSession(
        message=lambda: _prompt_message(
            cwd=cwd,
            model=agent.llm_client.model_name,
            tools=len(registry.definitions()),
            deferred_tools=_on_demand_tool_count(registry),
            agents_files=agents_file_count,
            mcp_servers=mcp_server_count,
            skills=skill_count,
            stats=renderer.toolbar_status(),
            permission_mode=permission_mode.mode,
        ),
        history=FileHistory(str(history_path)),
        completer=WordCompleter(SLASH_COMMANDS, ignore_case=True),
        placeholder=[("class:placeholder", "Type your message or @path/to/file")],
        style=Style.from_dict(
            {
                "prompt": "bold #ffffff bg:#262626",
                "placeholder": "#9a9a9a bg:#262626",
                "prompt.dim": "#a3a3a3 bg:#000000",
                "prompt.count.agents": "bold #22d3ee bg:#000000",
                "prompt.count.mcp": "bold #c084fc bg:#000000",
                "prompt.count.skills": "bold #facc15 bg:#000000",
                "prompt.tools": "bold #22d3ee bg:#000000",
                "toolbar.model": "noreverse bold #ffffff bg:#000000",
                "toolbar.ctx.bar": "noreverse #22c55e bg:#000000",
                "toolbar.ctx.value": "noreverse #ffffff bg:#000000",
                "toolbar.cwd.value": "noreverse #c084fc bg:#000000",
                "toolbar.mode.default": "noreverse bold #22c55e bg:#000000",
                "toolbar.mode.auto": "noreverse bold #f59e0b bg:#000000",
                "toolbar.gap": "noreverse #ffffff bg:#000000",
            }
        ),
        key_bindings=_permission_key_bindings(permission_mode),
    )

    # Set when the user picks "don't suggest again" for /plan or /team this session.
    mode_hints_off = False
    try:
        while True:
            try:
                user_input = await session.prompt_async()
            except (EOFError, KeyboardInterrupt):
                console.print()
                return
            message = user_input.strip()
            if not message:
                continue
            if not message.startswith("/"):
                reminder = cache_reminder(
                    idle_seconds=(
                        None if agent.last_active_at is None else time.time() - agent.last_active_at
                    ),
                    tokens=agent.context_tokens(),
                    config=config,
                    llm_client=agent.llm_client,
                )
                if reminder:
                    choice = _confirm_after_idle(console, reminder)
                    if choice == "n":
                        continue
                    if choice == "c":
                        await _compact(agent, console)
                if config.routing.suggest_modes and not mode_hints_off:
                    router = IntentRouter(config, ModelTiers(config, agent.llm_client))
                    with console.status("判断任务规模……"):
                        decision = await router.route(message)
                    if decision.mode != "react":
                        pick = _confirm_mode(console, decision)
                        if pick == "x":
                            mode_hints_off = True
                        elif pick in {"p", "t"}:
                            message = f"{'/plan' if pick == 'p' else '/team'} {message}"
            if message.startswith("/"):
                should_exit = await _handle_slash(
                    message,
                    console,
                    cwd,
                    config,
                    agent,
                    registry,
                    permission_mode,
                    renderer,
                    chat,
                )
                if should_exit:
                    return
                continue
            await _run_agent(agent, renderer, message)
            chat.record(message)
    finally:
        # MCP servers stay connected for the whole session (a browser keeps its page between
        # tool calls); stop them, and the browsers they launched, on the way out.
        if mcp_manager:
            await mcp_manager.aclose()
        await close_sandboxes()


def _prepare_sandbox(config: AgentCliConfig, console: Console) -> None:
    """Build the sandbox image the first time (a one-off download), then say where commands
    will run, before the banner, so the tool descriptions built next match it."""

    if config.sandbox.mode == "docker" and docker_available():
        ready, _ = sandbox_status(config)
        if not ready:
            with console.status("首次使用沙箱：正在构建 Docker 镜像（约 1~2 分钟，只需一次）……"):
                ok, log = prepare_image(config)
            if not ok:
                console.print(f"[red]沙箱镜像构建失败[/red]，命令将在本机执行并逐条审批。\n{log}")
    active, message = sandbox_status(config)
    console.print(f"[{'green' if active else 'yellow'}]{message}[/]")


async def _run_agent(agent: Agent, renderer: RichRenderer, message: str) -> None:
    await _run_events(
        agent.run(message),
        renderer,
        agent.llm_client.max_context_window,
        model=agent.llm_client.model_name,
    )


async def _run_events(
    events,
    renderer: RichRenderer,
    context_window: int | None = None,
    model: str = "",
) -> None:
    renderer.set_context_window(context_window)
    renderer.start_run(model=model)
    renderer.newline()
    async for event in events:
        renderer.handle(event)
        if event.get("type") == "error":
            break
    renderer.newline()


async def _handle_slash(
    raw: str,
    console: Console,
    cwd: str,
    config: AgentCliConfig,
    agent: Agent,
    registry: ToolRegistry,
    permission_mode: PermissionModeController,
    renderer: RichRenderer,
    chat: _ChatSession | None = None,
) -> bool:
    command, _, rest = raw.partition(" ")
    arg = rest.strip()
    if command in {"/exit", "/quit"}:
        return True
    if command == "/help":
        table = Table(title="AgentCLI 命令", show_lines=False)
        table.add_column("用法", style="cyan", no_wrap=True)
        table.add_column("作用")
        for _command, usage, about in COMMAND_HELP:
            table.add_row(usage, about)
        console.print(table)
        console.print(
            "[dim]Shift+Tab 切换审批模式 · 审批时按 y 允许、n 拒绝、s 跳过、a 本会话全部放行[/dim]"
        )
    elif command == "/compact":
        await _compact(agent, console, arg)
        if chat:
            chat.save()
    elif command in {"/clear", "/new"}:
        if chat:
            chat.start_new()
        else:
            agent.clear_history()
        if command == "/clear":
            console.clear()
        console.print(
            "[green]已开始新会话。[/green][dim]之前的会话已保存，/resume 可以回去。[/dim]"
        )
    elif command == "/resume":
        if chat:
            await chat.resume(arg or "pick", console)
    elif command == "/sessions":
        if chat:
            _sessions_command(arg, console, chat)
    elif command == "/sandbox":
        active, message = sandbox_status(config)
        console.print(f"[{'green' if active else 'yellow'}]{message}[/]")
        if active:
            console.print(
                f"[dim]镜像 {config.sandbox.image} · 内存 {config.sandbox.memory} · "
                f"CPU {config.sandbox.cpus:g} · 进程上限 {config.sandbox.pids}；"
                "要关闭，在配置里设 sandbox.mode = off，或设环境变量 AGENTCLI_SANDBOX=off[/dim]"
            )
    elif command == "/rename":
        if not arg:
            console.print("[red]Usage:[/red] /rename <名字>")
        elif chat:
            chat.current.rename(arg)
            console.print(f"当前会话已改名为：{chat.current.meta.title}")
    elif command == "/context":
        memories = _file_memory(cwd, config).list()
        table = Table(title="AgentCLI Context")
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("cwd", cwd)
        table.add_row("model", f"{config.llm.model} ({config.llm.provider})")
        table.add_row("context window", f"{agent.llm_client.max_context_window:,}")
        table.add_row("context now (est.)", f"{agent.context_tokens():,} tokens")
        idle = (
            "no request yet"
            if agent.last_active_at is None
            else _duration(time.time() - agent.last_active_at)
        )
        table.add_row("idle since last request", idle)
        table.add_row("assumed cache lifetime", f"{config.memory.cache_ttl_minutes} min")
        table.add_row("render", config.render_mode)
        table.add_row("memory", f"{len(memories)} saved")
        table.add_row("tools", str(len(registry.list_names())))
        console.print(table)
    elif command == "/memory":
        await _memory_command(arg, console, cwd, config)
    elif command == "/save":
        if not arg:
            console.print("[red]Usage:[/red] /save <fact>")
        else:
            record = _file_memory(cwd, config).save(arg, source="manual", importance=0.8)
            console.print(f"已保存记忆 {record.name}.md：{record.title}")
    elif command == "/config":
        console.print_json(json.dumps(config_to_public_dict(config), ensure_ascii=False))
    elif command == "/tools":
        console.print("\n".join(registry.list_names()))
    elif command == "/hitl":
        _hitl_command(arg, console, permission_mode)
    elif command == "/policy":
        console.print_json(json.dumps(config_to_public_dict(config)["policy"], ensure_ascii=False))
    elif command == "/audit":
        limit = int(arg or "20") if (arg or "20").isdigit() else 20
        console.print_json(
            json.dumps(AuditLog(config.policy.audit_log_path).tail(limit), ensure_ascii=False)
        )
    elif command == "/index":
        count = CodeIndex(cwd).rebuild(arg or ".")
        console.print(f"Indexed {count} code lines.")
    elif command == "/search":
        results = CodeIndex(cwd).search(arg, limit=20)
        output = "\n".join(f"{r.path}:{r.line}: {r.snippet}" for r in results)
        console.print(output or "(no matches)")
    elif command == "/plan":
        if not arg:
            console.print("[red]Usage:[/red] /plan <task>")
        else:
            plan_agent = PlanExecuteAgent(
                llm_client=agent.llm_client,
                tool_registry=registry,
                config=config,
                cwd=cwd,
                approval_callback=agent.approval_callback,
                tiers=ModelTiers(config, agent.llm_client),
            )
            await _run_events(
                plan_agent.run(arg),
                RichRenderer(),
                agent.llm_client.max_context_window,
                model=_run_model_label(agent.llm_client, config),
            )
    elif command == "/team":
        if not arg:
            console.print("[red]Usage:[/red] /team <task>")
        else:
            try:
                worker_mode, team_task = _parse_mode_argument(arg)
            except ValueError as exc:
                console.print(f"[red]{exc}[/red]")
                return False
            orchestrator = AgentOrchestrator(
                llm_client=agent.llm_client,
                tool_registry=registry,
                config=config,
                cwd=cwd,
                approval_callback=agent.approval_callback,
                default_worker_mode=worker_mode,
                tiers=ModelTiers(config, agent.llm_client),
                worker_count=config.routing.team_workers,
            )
            await _run_events(
                orchestrator.run(team_task),
                RichRenderer(),
                agent.llm_client.max_context_window,
                model=_run_model_label(agent.llm_client, config),
            )
    elif command == "/model":
        await _model_command(arg, console, cwd, config, agent, registry, renderer)
    elif command == "/usage":
        payload = {
            "usage": agent.last_usage.to_dict(),
            "cost": agent.last_cost,
            "pricing_note": "Built-in provider prices are dated defaults and may change.",
        }
        console.print_json(json.dumps(payload, ensure_ascii=False))
    elif command == "/skill":
        _skill_command(arg, console, cwd)
    elif command == "/mcp":
        console.print(
            "Use `agentcli mcp serve --transport stdio|http --port 3000` to expose tools."
        )
    elif command == "/task":
        _task_command(arg, console, cwd)
    elif command == "/snapshot":
        _snapshot_command(arg, console, cwd)
    elif command == "/restore":
        if not arg:
            console.print("[red]Usage:[/red] /restore <snapshot-id-or-index>")
        else:
            record = SnapshotService(cwd).restore(arg)
            console.print(f"Restored {record.id}")
    else:
        console.print(f"[red]Unknown command:[/red] {command}")
    return False


def _sessions_command(arg: str, console: Console, chat: _ChatSession) -> None:
    sub_command, _, rest = arg.partition(" ")
    if sub_command == "delete" and rest.strip():
        target = chat.store.find(rest.strip(), chat.cwd)
        if target is None or target.id == chat.current.id:
            console.print("[yellow]没有找到这个会话（当前会话不能删除）。[/yellow]")
        else:
            chat.store.delete(target.id)
            console.print(f"已删除会话 {target.id}")
        return
    metas = chat.store.list(chat.cwd)
    if not metas:
        console.print("(本项目还没有保存的会话)")
        return
    table = Table(title="本项目的会话")
    for column in ("id", "标题", "最后更新", "轮数", "模型"):
        table.add_column(column)
    for meta in metas:
        mark = " ←" if meta.id == chat.current.id else ""
        title = (meta.title or "(未命名)") + mark
        table.add_row(meta.id, title, meta.updated_text, str(meta.turns), meta.model)
    console.print(table)


def _file_memory(cwd: str, config: AgentCliConfig) -> FileMemory:
    return FileMemory(
        cwd,
        max_entries=config.memory.max_long_term_entries,
        max_chars=config.memory.max_memory_chars,
        legacy_db=config.memory.long_term_db_path,
    )


async def _memory_command(arg: str, console: Console, cwd: str, config: AgentCliConfig) -> None:
    memory = _file_memory(cwd, config)
    sub, _, rest = arg.partition(" ")
    rest = rest.strip()
    if sub == "clear":
        console.print(f"已删除 {memory.clear()} 条记忆。")
    elif sub == "search":
        hits = memory.search(rest, limit=config.memory.search_limit)
        console.print(
            "\n".join(
                f"{hit.record.name}  [{hit.record.kind}] {hit.record.title}  "
                f"(覆盖率 {hit.coverage:.2f})"
                for hit in hits
            )
            or "(没有匹配的记忆)"
        )
    elif sub == "show" and rest:
        record = memory.get(rest)
        console.print(record.content if record else "(没有这条记忆)")
    elif sub == "stats":
        console.print_json(json.dumps(memory.stats(), ensure_ascii=False))
    elif sub == "delete" and rest:
        console.print("已删除。" if memory.delete(rest) else "(没有这条记忆)")
    elif sub == "path":
        console.print(str(memory.folder))
    else:
        console.print(memory.index_text() or "(还没有记忆)")
        console.print(f"[dim]文件在 {memory.folder}[/dim]")


def _hitl_command(
    arg: str,
    console: Console,
    permission_mode: PermissionModeController,
) -> None:
    aliases: dict[str, PermissionMode] = {
        "default": "default",
        "on": "default",
        "auto": "auto",
        "off": "auto",
    }
    if arg in aliases:
        permission_mode.set(aliases[arg])
    elif arg:
        console.print("[red]Usage:[/red] /hitl default|auto")
        return
    console.print(f"Permission mode: {_permission_mode_label(permission_mode.mode)}")


async def _model_command(
    arg: str,
    console: Console,
    cwd: str,
    config: AgentCliConfig,
    agent: Agent,
    registry: ToolRegistry,
    renderer: RichRenderer,
) -> None:
    if arg:
        parts = arg.split(maxsplit=1)
        provider = config.llm.provider if len(parts) == 1 else parts[0]
        model = parts[0] if len(parts) == 1 else parts[1]
        profile = next(
            (
                item
                for item in DEFAULT_MODEL_PROFILES
                if item.provider == provider.lower() and item.model == model
            ),
            None,
        )
        if profile is None:
            base_url = (
                config.llm.base_url
                if len(parts) == 1 and config.llm.base_url
                else _provider_defaults(provider)[1]
            )
            profile = ModelProfile(
                id="command-line-selection",
                name=model,
                provider=provider,
                model=model,
                base_url=base_url,
                context_window=config.llm.context_window or 128_000,
                description="Selected from /model arguments",
                api_key_env=_provider_api_key_env(provider),
            )
        _activate_model(profile, config, agent, registry, renderer, cwd)
        console.print(f"[green]Switched model:[/green] {model} ({provider})")
        return

    store = CustomModelStore()
    while True:
        state = ModelSelectorState(
            defaults=list(DEFAULT_MODEL_PROFILES),
            custom=store.list(),
            current_provider=agent.llm_client.provider_name,
            current_model=agent.llm_client.model_name,
        )
        action = await run_model_selector(state)
        if action is None:
            return
        if action.kind == "add":
            profile = _prompt_custom_model(console)
            if profile is not None:
                store.add(profile)
                _activate_model(profile, config, agent, registry, renderer, cwd)
                console.print(
                    f"[green]Saved and switched to custom model:[/green] {profile.name} "
                    f"[dim]({store.path})[/dim]"
                )
                return
            continue
        if action.kind == "delete" and action.profile is not None:
            if store.delete(action.profile.id):
                console.print(f"Deleted custom model: {action.profile.name}")
            continue
        if action.profile is not None:
            _activate_model(action.profile, config, agent, registry, renderer, cwd)
            console.print(
                f"[green]Switched model:[/green] {action.profile.name} "
                f"[dim]({action.profile.provider}/{action.profile.model})[/dim]"
            )
            return


def _prompt_custom_model(console: Console) -> ModelProfile | None:
    console.print("\n[bold]Add custom model[/bold]")
    provider = Prompt.ask(
        "Provider",
        choices=list(PROVIDER_DEFAULTS),
        default="openai-compatible",
    )
    provider_label, default_base_url, default_context = _provider_defaults(provider)
    model = Prompt.ask("Model ID").strip()
    if not model:
        console.print("[red]Model ID is required.[/red]")
        return None
    name = Prompt.ask("Display name", default=f"{provider_label} · {model}").strip()
    base_url = Prompt.ask("Base URL", default=default_base_url).strip()
    api_key_env = Prompt.ask(
        "API key environment variable",
        default=_provider_api_key_env(provider),
    ).strip()
    api_key = Prompt.ask(
        f"API key (optional; leave blank to use ${api_key_env})",
        default="",
        password=True,
        show_default=False,
    )
    context_text = Prompt.ask("Context window", default=str(default_context)).replace(",", "")
    try:
        context_window = int(context_text)
        return ModelProfile.custom_profile(
            name=name,
            provider=provider,
            model=model,
            base_url=base_url,
            context_window=context_window,
            api_key=api_key,
            api_key_env=api_key_env,
        )
    except ValueError as exc:
        console.print(f"[red]Invalid custom model:[/red] {exc}")
        return None


def _activate_model(
    profile: ModelProfile,
    config: AgentCliConfig,
    agent: Agent,
    registry: ToolRegistry,
    renderer: RichRenderer,
    cwd: str,
) -> None:
    old_provider = config.llm.provider.lower()
    old_api_key = config.llm.api_key
    config.llm.provider = profile.provider
    config.llm.model = profile.model
    config.llm.base_url = profile.base_url
    config.llm.context_window = profile.context_window
    config.llm.api_key = profile.resolve_api_key(
        current_provider=old_provider,
        current_api_key=old_api_key,
    )
    client = create_llm_client(config.llm)
    agent.llm_client = client
    agent.system_prompt = PromptAssembler(
        config=config,
        cwd=cwd,
        tool_names=registry.list_names(),
        model=client.model_name,
        provider=client.provider_name,
    ).build_static()
    renderer.set_context_window(client.max_context_window)


def _provider_defaults(provider: str) -> tuple[str, str, int]:
    normalized = provider.lower()
    if normalized in PROVIDER_DEFAULTS:
        return PROVIDER_DEFAULTS[normalized]
    return (provider, config_base_url(provider), 128_000)


def config_base_url(provider: str) -> str:
    from agentcli.llm.factory import DEEPSEEK_BASE_URL, OPENAI_BASE_URL, PROVIDER_BASE_URLS

    normalized = provider.lower()
    if normalized == "deepseek":
        return DEEPSEEK_BASE_URL
    return PROVIDER_BASE_URLS.get(normalized, OPENAI_BASE_URL)


def _provider_api_key_env(provider: str) -> str:
    return {
        "deepseek": "DEEPSEEK_API_KEY",
        "glm": "ZAI_API_KEY",
        "zhipu": "ZAI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openai-compatible": "AGENTCLI_API_KEY",
    }.get(provider.lower(), "AGENTCLI_API_KEY")


def _skill_command(arg: str, console: Console, cwd: str) -> None:
    registry = SkillRegistry(cwd)
    sub, _, rest = arg.partition(" ")
    if sub == "show" and rest:
        skill = registry.load(rest.strip())
        if not skill:
            console.print(f'Skill "{rest.strip()}" not found.')
            return
        console.print(skill.content[:12_000])
        return
    if sub == "on" and rest:
        console.print("enabled" if registry.enable(rest.strip()) else "skill not found")
        return
    if sub == "off" and rest:
        console.print("disabled" if registry.disable(rest.strip()) else "skill not found")
        return
    if sub == "reload":
        registry.reload()
        console.print("skills reloaded")
        return
    rows = registry.all_skills()
    lines = [
        f"{item.name}\t{item.source}\t{'on' if item.enabled else 'off'}\t{item.description}"
        for item in rows
    ]
    console.print("\n".join(lines) or "(no skills)")


def _task_command(arg: str, console: Console, cwd: str) -> None:
    manager = DurableTaskManager(agentcli_home() / "tasks" / "tasks.db", scope=cwd)
    sub, _, rest = arg.partition(" ")
    if sub == "add" and rest:
        try:
            mode, prompt = _parse_mode_argument(rest, allowed={"react", "plan", "team"})
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        task_id = manager.add(prompt, mode=mode)
        console.print(
            f"Queued {task_id} ({mode}). Run `agentcli worker` or `agentcli serve` to consume it."
        )
    elif sub == "cancel" and rest:
        console.print(f"Canceled: {manager.cancel(rest.strip())}")
    elif sub == "log" and rest:
        task = manager.get(rest.strip())
        if not task:
            console.print("(task not found)")
        else:
            console.print(task.result or task.error or f"Task {task.id} is {task.status}")
    else:
        rows = manager.list(limit=20)
        console.print(
            "\n".join(
                f"{task.id} {task.status} {task.mode} attempts={task.attempts} {task.prompt[:80]}"
                for task in rows
            )
            or "(no tasks)"
        )


def _snapshot_command(arg: str, console: Console, cwd: str) -> None:
    service = SnapshotService(cwd)
    if arg == "clean":
        console.print(f"Cleaned {service.clean()} snapshots.")
        return
    rows = service.list(limit=20)
    output = "\n".join(
        f"{index}. {row.id} {row.phase} {row.created_at}" for index, row in enumerate(rows, 1)
    )
    console.print(output or "(no snapshots)")


def _approval_prompt(
    request: dict[str, Any],
    console: Console,
    permission_mode: PermissionModeController,
) -> str:
    if not sys.stdin.isatty():
        return "deny"
    console.print(
        f"[yellow]Approval required[/yellow] {request['tool_name']} "
        f"({request['danger_level']})\n{request['input']}"
    )
    choices = ["y", "n", "a", "s"]
    host = ""
    if request["tool_name"] == "web_fetch":
        from agentcli.web.domains import host_of

        host = host_of(str((request.get("input") or {}).get("url") or ""))
    hint = "y 允许一次 · n 拒绝 · s 跳过 · a 本会话全部放行"
    if host:
        choices.append("d")
        hint += f" · d 本项目以后都允许 {host}"
    console.print(f"[dim]{hint}[/dim]")
    answer = Prompt.ask("Approve?", choices=choices, default="n")
    if answer == "d" and host:
        from agentcli.web.domains import DomainPolicy

        DomainPolicy(request.get("cwd") or os.getcwd()).remember(host)
        return "approve"
    if answer == "a":
        permission_mode.set("auto")
        return "approve"
    if answer == "y":
        return "approve"
    if answer == "s":
        return "skip"
    return "deny"


def _count_mcp_servers(manager: Any) -> int:
    if manager is None:
        return 0
    return sum(1 for spec in manager.specs.values() if spec.enabled)


def _run_model_label(session_client: Any, config: AgentCliConfig) -> str:
    """The model named in a /plan or /team summary: one model, or several tiers."""

    if ModelTiers(config, session_client).configured:
        return "tiered models"
    return session_client.model_name


def _on_demand_tool_count(registry: Any) -> int:
    """Kinds of deferred tool not loaded yet: only their names are in the request.

    Two servers offering the same tool (the two browsers both have navigate_page) count
    once, since they differ only in which browser they drive.
    """

    return len(
        {
            tool.name.split("__")[-1]
            for tool in registry.deferred_tools()
            if not registry.is_active(tool.name)
        }
    )


def _parse_mode_argument(
    value: str,
    *,
    allowed: set[str] | None = None,
) -> tuple[str, str]:
    modes = allowed or {"react", "plan"}
    parts = value.strip().split(maxsplit=2)
    if len(parts) >= 2 and parts[0] in {"--mode", "-m"}:
        mode = parts[1].lower()
        if mode not in modes:
            raise ValueError(f"mode must be one of: {', '.join(sorted(modes))}")
        prompt = parts[2].strip() if len(parts) == 3 else ""
        if not prompt:
            raise ValueError("task text is required after --mode")
        return mode, prompt
    if parts and parts[0] == "--plan":
        prompt = value.strip()[len("--plan") :].strip()
        if not prompt:
            raise ValueError("task text is required after --plan")
        return "plan", prompt
    return "react", value.strip()


def _permission_key_bindings(permission_mode: PermissionModeController) -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add(Keys.BackTab)
    def _toggle_permission_mode(event) -> None:
        permission_mode.toggle()
        event.app.invalidate()

    return bindings


def _permission_mode_label(mode: PermissionMode) -> str:
    return "Auto (full access)" if mode == "auto" else "Default"


def _prompt_message(
    *,
    cwd: str,
    model: str,
    tools: int,
    agents_files: int,
    mcp_servers: int,
    skills: int,
    stats: dict[str, Any] | None = None,
    permission_mode: PermissionMode = "default",
    deferred_tools: int = 0,
) -> list[tuple[str, str]]:
    # tools: definitions sent with every request; deferred_tools: loaded only when needed.
    on_demand = [("class:prompt.dim", f" +{deferred_tools} on demand")] if deferred_tools else []
    return [
        ("class:prompt.count.agents", str(agents_files)),
        ("class:prompt.dim", f" {_plural_label(agents_files, 'instruction file')} · "),
        ("class:prompt.count.mcp", str(mcp_servers)),
        ("class:prompt.dim", f" {_plural_label(mcp_servers, 'MCP server')} · "),
        ("class:prompt.count.skills", str(skills)),
        ("class:prompt.dim", f" {_plural_label(skills, 'skill')} · Tools "),
        ("class:prompt.tools", str(tools)),
        *on_demand,
        ("class:prompt.dim", "\n"),
        *_bottom_toolbar(cwd, model, stats, permission_mode=permission_mode),
        ("class:prompt.dim", "\n\n"),
        ("class:prompt", "* "),
    ]


def _bottom_toolbar(
    cwd: str,
    model: str,
    stats: dict[str, Any] | None = None,
    *,
    permission_mode: PermissionMode = "default",
) -> list[tuple[str, str]]:
    stats = stats or {}
    has_usage = bool(stats.get("has_usage"))
    context_ratio = float(stats.get("context_ratio") or 0)
    context_text = _format_toolbar_percent(context_ratio) if has_usage else "0%"
    return [
        ("class:toolbar.model", model),
        ("class:toolbar.gap", "    "),
        ("class:toolbar.ctx.bar", _format_toolbar_bar(context_ratio if has_usage else 0)),
        ("class:toolbar.gap", " "),
        ("class:toolbar.ctx.value", context_text),
        ("class:toolbar.gap", "  "),
        ("class:toolbar.cwd.value", _shorten_home(cwd)),
        ("class:toolbar.gap", "  "),
        (
            f"class:toolbar.mode.{permission_mode}",
            _permission_mode_label(permission_mode),
        ),
        ("class:toolbar.gap", "  Shift+Tab"),
    ]


def _plural_label(count: int, singular: str) -> str:
    return singular if count == 1 else singular + "s"


def _shorten_home(path: str) -> str:
    home = str(Path.home())
    if path == home:
        return "~"
    prefix = home + os.sep
    if path.startswith(prefix):
        return "~/" + path[len(prefix) :]
    return path


def _format_toolbar_bar(value: float, *, width: int = 12) -> str:
    bounded = max(0.0, min(value, 1.0))
    filled = round(bounded * width)
    if bounded > 0 and filled == 0:
        filled = 1
    return "█" * filled + "░" * (width - filled)


def _format_toolbar_percent(value: float) -> str:
    if value <= 0:
        return "0%"
    if value < 0.01:
        return "<1%"
    return f"{value:.0%}"
