from __future__ import annotations

import asyncio

from agentcli.agent import QueryEngine
from agentcli.config import load_config
from agentcli.memory import MemoryManager
from agentcli.prompt import PromptAssembler
from agentcli.tools import ToolRegistry


class CapturingClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 100_000

    def __init__(self):
        self.system_prompts: list[str] = []

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        self.system_prompts.append(system_prompt)
        yield {"type": "text_delta", "text": "ok"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


def test_static_prompt_is_stable_and_custom_project_instructions_are_loaded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    custom = tmp_path / "rules.md"
    custom.write_text("Always run the focused test first.", encoding="utf-8")
    config = load_config(project_root=tmp_path)
    config.prompt.custom_prompt_paths = [str(custom)]
    assembler = PromptAssembler(config, str(tmp_path), ["read_file"], "model", "provider")

    first = assembler.build_static()
    second = assembler.build_static()

    assert first == second
    assert "Always run the focused test first" in first
    assert "Current time" not in first


def test_dynamic_prompt_recall_is_rebuilt_for_each_request(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    client = CapturingClient()
    engine = QueryEngine(
        llm_client=client,
        tool_registry=ToolRegistry(),
        config=config,
        cwd=str(tmp_path),
    )

    asyncio.run(engine.ask_complete_async("第一次请求，不涉及测试偏好"))
    MemoryManager(config.memory.long_term_db_path, scope=str(tmp_path)).save(
        "用户偏好使用 uv run python -m pytest 执行测试",
        kind="preference",
        importance=0.9,
    )
    asyncio.run(engine.ask_complete_async("我偏好怎么执行测试？"))

    assert len(client.system_prompts) == 2
    assert "Current time" in client.system_prompts[1]
    assert "recalled-memory" in client.system_prompts[1]
    assert "uv run python -m pytest" in client.system_prompts[1]


def test_dynamic_memory_is_bounded_and_marked_untrusted(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    manager = MemoryManager(config.memory.long_term_db_path, scope=str(tmp_path))
    manager.save("项目测试统一使用 pytest", kind="constraint")
    assembler = PromptAssembler(config, str(tmp_path), [], "model", "provider")

    dynamic = assembler.build_dynamic("项目测试怎么运行")

    assert '<recalled-memory trust="untrusted-data">' in dynamic
    assert "项目测试统一使用 pytest" in dynamic
