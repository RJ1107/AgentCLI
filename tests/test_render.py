from __future__ import annotations

import asyncio
from io import StringIO

from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from agentcli.config import load_config
from agentcli.entrypoints.repl import (
    PermissionModeController,
    _bottom_toolbar,
    _permission_key_bindings,
    _prompt_message,
)
from agentcli.render import RichRenderer


def test_banner_renders_pi_home_layout():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=200)
    renderer = RichRenderer(console=console)

    renderer.banner(
        model="deepseek-v4-flash",
        provider="deepseek",
        cwd="/tmp/project",
        tools=12,
        version="0.1.0",
        api_key_configured=True,
        mcp_servers=1,
        skills=3,
        agents_files=2,
        hitl_mode="never",
    )

    output = stream.getvalue()
    assert "████████████" in output
    assert "  ██    ██" in output
    assert "AgentCLI v0.1.0" in output
    assert "Signed in API Key" in output
    assert "What's new (v0.1.0)" in output


def test_prompt_message_keeps_status_and_input_together():
    prompt = _prompt_message(
        cwd="/tmp/project",
        model="deepseek-v4-flash",
        tools=12,
        agents_files=2,
        mcp_servers=1,
        skills=3,
        stats={"total_tokens": 13187, "context_ratio": 0.013, "has_usage": True},
    )
    plain = "".join(text for _style, text in prompt)

    assert "2 AGENTS.md files" in plain
    assert "1 MCP server" in plain
    assert "3 skills · Tools 12" in plain
    assert "Default  Shift+Tab" in plain
    assert "deepseek-v4-flash" in plain
    assert "█░░░░░░░░░░░ 1%" in plain
    assert "/tmp/project" in plain
    assert "\n\n* " in plain
    assert plain.endswith("\n* ")


def test_permission_mode_toggle_applies_and_restores_full_access_policy(tmp_path):
    config = load_config(project_root=tmp_path)
    config.policy.hitl_mode = "always"
    controller = PermissionModeController(config)

    assert controller.mode == "default"
    assert config.policy.hitl_mode == "always"
    assert config.policy.path_guard_enabled
    assert config.policy.command_guard_enabled

    assert controller.toggle() == "auto"
    assert config.policy.hitl_mode == "never"
    assert not config.policy.path_guard_enabled
    assert not config.policy.command_guard_enabled

    assert controller.toggle() == "default"
    assert config.policy.hitl_mode == "always"
    assert config.policy.path_guard_enabled
    assert config.policy.command_guard_enabled


def test_shift_tab_is_bound_to_permission_mode_toggle(tmp_path):
    controller = PermissionModeController(load_config(project_root=tmp_path))
    bindings = _permission_key_bindings(controller)

    assert any(binding.keys == (Keys.BackTab,) for binding in bindings.bindings)


def test_shift_tab_input_toggles_live_permission_mode(tmp_path):
    controller = PermissionModeController(load_config(project_root=tmp_path))

    async def run_prompt() -> None:
        with create_pipe_input() as pipe_input:
            session = PromptSession(
                input=pipe_input,
                output=DummyOutput(),
                key_bindings=_permission_key_bindings(controller),
            )
            pipe_input.send_text("\x1b[Z\r")
            await session.prompt_async()

    asyncio.run(run_prompt())

    assert controller.mode == "auto"
    assert controller.config.policy.hitl_mode == "never"


def test_bottom_toolbar_uses_runtime_summary_segments():
    toolbar = _bottom_toolbar(
        "/Users/me/project",
        "deepseek-v4-flash",
        {"turns": 1, "total_tokens": 13187, "context_ratio": 0.013, "has_usage": True},
    )

    assert ("class:toolbar.model", "deepseek-v4-flash") in toolbar
    assert ("class:toolbar.ctx.bar", "█░░░░░░░░░░░") in toolbar
    assert ("class:toolbar.ctx.value", "1%") in toolbar
    assert ("class:toolbar.cwd.value", "/Users/me/project") in toolbar
    assert not any(text == " TURN " for _style, text in toolbar)
    assert not any("Token" in text for _style, text in toolbar)


def test_text_deltas_render_as_markdown_on_turn_complete():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=120)
    renderer = RichRenderer(console=console, context_window=1000)

    renderer.handle({"type": "thinking_delta", "thinking": "需要先确认项目结构"})
    renderer.handle({"type": "text_delta", "text": "你好，我是 **Agent"})
    renderer.handle({"type": "text_delta", "text": "CLI**\n\n- `read_file`\n- **网页搜索**"})
    renderer.handle({"type": "usage", "usage": {"input_tokens": 250, "output_tokens": 50}})
    renderer.handle({"type": "turn_complete"})
    renderer.handle({"type": "done", "total_turns": 1, "total_tokens": 300})

    output = stream.getvalue()
    assert "Thinking" in output
    assert "需要先确认项目结构" in output
    assert "Final Output" in output
    assert "AgentCLI" in output
    assert "read_file" in output
    assert "网页搜索" in output
    assert "Run Summary" not in output
    assert "**AgentCLI**" not in output
    assert "`read_file`" not in output

    stats = renderer.toolbar_status()
    assert stats["turns"] == 1
    assert stats["input_tokens"] == 250
    assert stats["output_tokens"] == 50
    assert stats["total_tokens"] == 300
    assert stats["context_ratio"] == 0.25
    assert stats["has_usage"] is True


def test_interleaved_thinking_does_not_repeat_assistant_output_panels():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=120)
    renderer = RichRenderer(console=console)

    renderer.handle({"type": "text_delta", "text": "第一段"})
    renderer.handle({"type": "thinking_delta", "thinking": "中途补充思考"})
    renderer.handle({"type": "text_delta", "text": "第二段"})
    renderer.handle({"type": "turn_complete"})

    output = stream.getvalue()
    assert output.count("Assistant Output") == 0
    assert output.count("Final Output") == 1
    assert output.count("Thinking") == 1
    assert "第一段第二段" in output


def test_plan_status_and_scoped_thinking_render_with_task_identity():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=120)
    renderer = RichRenderer(console=console)

    renderer.handle({"type": "text_delta", "text": "正在规划任务"})
    renderer.handle({"type": "plan_status", "phase": "planning"})
    renderer.handle(
        {
            "type": "thinking_delta",
            "thinking": "先拆分任务",
            "phase": "planning",
        }
    )
    renderer.handle(
        {
            "type": "plan_task_started",
            "task_id": "task_1",
            "task_description": "检查模型配置",
        }
    )
    renderer.handle(
        {
            "type": "thinking_delta",
            "thinking": "读取配置文件",
            "phase": "execution",
            "task_id": "task_1",
        }
    )
    renderer.handle(
        {
            "type": "tool_call",
            "name": "read_file",
            "input": {"path": "config.py"},
            "task_id": "task_1",
        }
    )

    output = stream.getvalue()
    assert "Plan" in output
    assert "正在规划任务" in output
    assert "Thinking · planning" in output
    assert "先拆分任务" in output
    assert "Running task_1" in output
    assert "检查模型配置" in output
    assert "Thinking · task_1" in output
    assert "读取配置文件" in output
    assert "Tool Use · task_1" in output


def test_streaming_text_waits_for_turn_boundary_by_default():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=120, force_terminal=True)
    renderer = RichRenderer(console=console)

    renderer.handle({"type": "text_delta", "text": "chunk 1"})
    renderer.handle({"type": "text_delta", "text": "chunk 2"})

    assert "Assistant Output" not in stream.getvalue()
    renderer.handle({"type": "turn_complete"})
    assert stream.getvalue().count("Final Output") == 1


def test_tool_use_and_result_render_as_structured_panels():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=120)
    renderer = RichRenderer(console=console)

    renderer.handle({"type": "tool_call", "name": "list_dir", "input": {"path": "."}})
    renderer.handle(
        {
            "type": "tool_result",
            "name": "list_dir",
            "result": "README.md\nsrc/",
            "is_error": False,
        }
    )

    output = stream.getvalue()
    assert "Tool Use" in output
    assert "list_dir" in output
    assert '"path": "."' in output
    assert "Tool Result · list_dir · ok" in output
    assert "README.md" in output


def test_start_run_resets_token_usage():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=120)
    renderer = RichRenderer(console=console, context_window=1000)

    renderer.handle({"type": "usage", "usage": {"input_tokens": 900, "output_tokens": 10}})
    renderer.start_run()
    renderer.handle({"type": "usage", "usage": {"input_tokens": 100, "output_tokens": 20}})
    renderer.handle({"type": "done", "total_turns": 1, "total_tokens": 120})

    assert "900" not in stream.getvalue()
    stats = renderer.toolbar_status()
    assert stats["input_tokens"] == 100
    assert stats["output_tokens"] == 20
    assert stats["total_tokens"] == 120
    assert stats["context_ratio"] == 0.1


def test_missing_usage_keeps_toolbar_tokens_unavailable():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=120)
    renderer = RichRenderer(console=console, context_window=1000)

    renderer.handle({"type": "done", "total_turns": 1, "total_tokens": 0})

    assert "Run Summary" not in stream.getvalue()
    toolbar = _bottom_toolbar("/tmp/project", "deepseek-v4-flash", renderer.toolbar_status())
    assert ("class:toolbar.model", "deepseek-v4-flash") in toolbar
    assert ("class:toolbar.ctx.bar", "░░░░░░░░░░░░") in toolbar
    assert ("class:toolbar.ctx.value", "0%") in toolbar
