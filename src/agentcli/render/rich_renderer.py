from __future__ import annotations

import json
from typing import Any

from rich import box
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agentcli import __version__


class RichRenderer:
    def __init__(
        self,
        console: Console | None = None,
        *,
        live_markdown: bool = False,
        context_window: int | None = None,
    ):
        self.console = console or Console()
        self._buffer: list[str] = []
        self._thinking_buffer: list[str] = []
        self._thinking_scope: str | None = None
        self._live_markdown = live_markdown
        self._live: Live | None = None
        self._thinking_live: Live | None = None
        self._context_window = context_window or 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._last_input_tokens = 0
        self._last_turns = 0
        self._last_total_tokens = 0
        self._last_context_ratio = 0.0
        self._last_has_usage = False

    def set_context_window(self, context_window: int | None) -> None:
        self._context_window = context_window or self._context_window

    def start_run(self) -> None:
        self._buffer.clear()
        self._thinking_buffer.clear()
        self._thinking_scope = None
        self._stop_live_markdown()
        self._stop_live_thinking()
        self._input_tokens = 0
        self._output_tokens = 0
        self._last_input_tokens = 0

    def toolbar_status(self) -> dict[str, Any]:
        return {
            "turns": self._last_turns,
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "total_tokens": self._last_total_tokens,
            "context_ratio": self._last_context_ratio,
            "has_usage": self._last_has_usage,
        }

    def banner(
        self,
        *,
        model: str,
        provider: str,
        cwd: str,
        tools: int,
        version: str = __version__,
        api_key_configured: bool = False,
        mcp_servers: int = 0,
        skills: int = 0,
        agents_files: int = 0,
        hitl_mode: str = "auto",
    ) -> None:
        self.console.print()
        self.console.print(
            self._identity_panel(version=version, api_key_configured=api_key_configured)
        )
        self.console.print()
        self.console.print(self._guide_panel())
        self.console.print()
        # Model, tools and the rest live in the status line under the prompt, which stays
        # current; the banner only introduces the program.
        _ = model, provider, cwd, tools, mcp_servers, skills, agents_files, hitl_mode
        self.console.rule(style="grey23")
        self.console.print()

    def handle(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "text_delta":
            self._flush_thinking()
            text = str(event.get("text") or "")
            self._buffer.append(text)
            self._update_live_markdown()
        elif event_type == "thinking_delta":
            scope = _thinking_scope(event)
            if self._thinking_buffer and scope != self._thinking_scope:
                self._flush_thinking()
            self._thinking_scope = scope
            thinking = str(event.get("thinking") or "")
            self._thinking_buffer.append(thinking)
            self._update_live_thinking()
        elif event_type == "plan_status":
            self._flush_thinking()
            self._flush_markdown(title="Plan")
        elif event_type == "plan_task_started":
            self._flush_thinking()
            self._flush_markdown(title="Plan")
            task_id = str(event.get("task_id") or "task")
            description = str(event.get("task_description") or "")
            self.console.print(
                _output_panel(
                    Text(description, style="#e5e7eb"),
                    title=Text(f"Running {task_id}", style="bold #22d3ee"),
                    border_style="#0891b2",
                )
            )
        elif event_type == "usage":
            self._record_usage(event.get("usage") or {})
        elif event_type == "turn_complete":
            stop_reason = str(event.get("stop_reason") or "end_turn")
            title = "Assistant Output" if stop_reason == "tool_use" else "Final Output"
            self._flush_thinking()
            self._flush_markdown(title=title)
        elif event_type == "tool_call":
            self._flush_thinking()
            self._flush_markdown(title="Assistant Output")
            self._print_tool_call(event)
        elif event_type == "tool_result":
            self._flush_thinking()
            self._flush_markdown(title="Assistant Output")
            self._print_tool_result(event)
        elif event_type == "error":
            self._flush_thinking()
            self._flush_markdown(title="Assistant Output")
            self.console.print(f"[red]Error:[/red] {event.get('error')}")
        elif event_type == "done":
            self._flush_thinking()
            self._flush_markdown(title="Final Output")
            self._record_run_summary(event)

    def markdown(self, text: str) -> None:
        self.console.print(Markdown(text))

    def newline(self) -> None:
        self._flush_thinking()
        self._flush_markdown(title="Final Output")
        self.console.print()

    def _flush_markdown(self, *, title: str) -> None:
        if not self._buffer:
            return
        text = "".join(self._buffer)
        self._buffer.clear()
        self._stop_live_markdown()
        if text.strip():
            self.console.print(
                _output_panel(
                    Markdown(text),
                    title=Text(title, style="bold #a8ff60"),
                    border_style="#3f3f46",
                )
            )

    def _update_live_markdown(self) -> None:
        if not self._live_markdown or not self.console.is_terminal:
            return
        text = "".join(self._buffer)
        if not text.strip():
            return
        renderable = _output_panel(
            Markdown(text),
            title=Text("Assistant Output", style="bold #a8ff60"),
            border_style="#3f3f46",
        )
        if self._live is None:
            self._live = Live(
                renderable,
                console=self.console,
                refresh_per_second=12,
                transient=True,
                vertical_overflow="visible",
            )
            self._live.start(refresh=True)
            return
        self._live.update(renderable, refresh=True)

    def _stop_live_markdown(self) -> None:
        if self._live is None:
            return
        self._live.stop()
        self._live = None

    def _flush_thinking(self) -> None:
        if not self._thinking_buffer:
            return
        text = "".join(self._thinking_buffer)
        self._thinking_buffer.clear()
        scope = self._thinking_scope
        self._thinking_scope = None
        self._stop_live_thinking()
        if text.strip():
            self.console.print(
                _output_panel(
                    Text(text, style="dim"),
                    title=Text(_thinking_title(scope), style="bold #c084fc"),
                    border_style="#6d28d9",
                )
            )

    def _update_live_thinking(self) -> None:
        if not self.console.is_terminal:
            return
        text = "".join(self._thinking_buffer)
        if not text.strip():
            return
        renderable = _output_panel(
            Text(text, style="dim"),
            title=Text(_thinking_title(self._thinking_scope), style="bold #c084fc"),
            border_style="#6d28d9",
        )
        if self._thinking_live is None:
            self._thinking_live = Live(
                renderable,
                console=self.console,
                refresh_per_second=12,
                transient=True,
                vertical_overflow="visible",
            )
            self._thinking_live.start(refresh=True)
            return
        self._thinking_live.update(renderable, refresh=True)

    def _stop_live_thinking(self) -> None:
        if self._thinking_live is None:
            return
        self._thinking_live.stop()
        self._thinking_live = None

    def _record_usage(self, usage: dict[str, Any]) -> None:
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens
        if input_tokens:
            self._last_input_tokens = input_tokens

    def _print_tool_call(self, event: dict[str, Any]) -> None:
        name = str(event.get("name") or "unknown")
        payload = event.get("input") or {}
        body = Table.grid(padding=(0, 1))
        body.add_column(style="dim", no_wrap=True)
        body.add_column()
        body.add_row("name", Text(name, style="bold #facc15"))
        body.add_row("input", Text(_format_payload(payload), style="#e5e7eb"))
        self.console.print(
            _output_panel(
                body,
                title=Text(_scoped_title("Tool Use", event), style="bold #facc15"),
                border_style="#facc15",
            )
        )

    def _print_tool_result(self, event: dict[str, Any]) -> None:
        is_error = bool(event.get("is_error"))
        name = str(event.get("name") or "unknown")
        result = str(event.get("result") or "")
        if len(result) > 1200:
            result = result[:1200] + "\n... [truncated]"
        title_style = "bold #ff4d5a" if is_error else "bold #22c55e"
        border_style = "#ff4d5a" if is_error else "#22c55e"
        status = "error" if is_error else "ok"
        self.console.print(
            _output_panel(
                result or "(empty result)",
                title=Text(
                    _scoped_title(f"Tool Result · {name} · {status}", event),
                    style=title_style,
                ),
                border_style=border_style,
            )
        )

    def _record_run_summary(self, event: dict[str, Any]) -> None:
        total_tokens = int(event.get("total_tokens") or self._input_tokens + self._output_tokens)
        turns = int(event.get("total_turns") or 0)
        has_usage = total_tokens > 0 or self._input_tokens > 0 or self._output_tokens > 0
        context_ratio = (
            self._last_input_tokens / self._context_window if self._context_window > 0 else 0
        )
        self._last_turns = turns
        self._last_total_tokens = total_tokens
        self._last_context_ratio = context_ratio
        self._last_has_usage = has_usage

    def _identity_panel(self, *, version: str, api_key_configured: bool) -> Table:
        logo = Text()
        for row, color in zip(_RJ_LOGO, _LOGO_GRADIENT, strict=True):
            logo.append(row + "\n", style=f"bold {color}")

        identity = Text()
        identity.append("'s ", style="bold #a3e635")
        identity.append("AgentCLI", style="bold white")
        identity.append(f"  v{version}", style="#6b7280")
        identity.append("\n终端里的编程 Agent", style="#9ca3af")
        if not api_key_configured:
            identity.append("\n未配置 API Key", style="bold red")
            identity.append("：设置 OPENROUTER_API_KEY 或 DEEPSEEK_API_KEY", style="#9ca3af")

        grid = Table.grid(padding=(0, 1))
        grid.add_column(no_wrap=True)
        grid.add_column(vertical="bottom")
        grid.add_row(logo, identity)
        return grid

    def _guide_panel(self) -> Table:
        guide = Table.grid(padding=(0, 2))
        guide.add_column(style="bold #22d3ee", no_wrap=True)
        guide.add_column(style="#9ca3af")
        for key, rest in _QUICK_START:
            guide.add_row(key, rest)
        return guide


# Two-cell-wide blocks, so the letters keep their proportions in a terminal.
_RJ_LOGO = (
    "██████       ██",
    "██   ██      ██",
    "██████       ██",
    "██  ██   ██  ██",
    "██   ██   ████ ",
)
_LOGO_GRADIENT = ("#bef264", "#86efac", "#5eead4", "#38bdf8", "#818cf8")

_QUICK_START = (
    ("直接提需求", "读代码、改文件、跑命令；写文件和执行命令前会先问你"),
    ("/plan  /team", "大任务先规划再执行，/team 多个 Agent 分工并交叉审核"),
    ("网页 · 文档", "打不开的网页会换浏览器读；PDF、Word、Excel、PPT 交给技能"),
    ("/model", "换模型    /compact 压缩上下文    /skill 看技能    /help 全部命令"),
    ("Shift+Tab", "切换审批模式    Ctrl+D 退出"),
)


def _format_payload(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except TypeError:
        return str(payload)


def _thinking_scope(event: dict[str, Any]) -> str | None:
    task_id = str(event.get("task_id") or "").strip()
    if task_id:
        return task_id
    if event.get("phase") == "planning":
        return "planning"
    return None


def _thinking_title(scope: str | None) -> str:
    if scope == "planning":
        return "Thinking · planning"
    if scope:
        return f"Thinking · {scope}"
    return "Thinking"


def _scoped_title(title: str, event: dict[str, Any]) -> str:
    task_id = str(event.get("task_id") or "").strip()
    return f"{title} · {task_id}" if task_id else title


def _output_panel(renderable: Any, *, title: Text, border_style: str) -> Panel:
    return Panel(
        renderable,
        title=title,
        border_style=border_style,
        box=box.ROUNDED,
        expand=True,
    )
