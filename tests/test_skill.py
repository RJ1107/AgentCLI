from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agentcli.config import load_config
from agentcli.skill import SkillContextBuffer, SkillMatcher, SkillRegistry, SkillStateStore
from agentcli.tools import ToolRegistry, get_builtin_tools
from agentcli.tools.base import ToolContext
from agentcli.tools.builtins import load_skill
from agentcli.tools.executor import ToolExecutor


def test_skill_registry_layers_and_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    builtin = tmp_path / "builtin"
    user = tmp_path / "user"
    project = tmp_path / "project"
    _write_skill(builtin, "web-access", "builtin desc", "v0")
    _write_skill(user, "web-access", "user desc", "v1")
    _write_skill(project / ".agentcli" / "skills", "project-only", "project desc", "v2")
    state = SkillStateStore(tmp_path / "skills.json")
    state.disable("web-access")

    registry = SkillRegistry(
        project,
        builtin_root=builtin,
        user_root=user,
        state_store=state,
    )

    assert [skill.name for skill in registry.all_skills()] == ["project-only", "web-access"]
    assert registry.load("web-access") is None
    assert registry.load("web-access", include_disabled=True).source == "user"
    assert [skill.name for skill in registry.enabled_skills()] == ["project-only"]

    assert registry.enable("web-access")
    assert registry.load("web-access").description == "user desc"


def test_skill_context_buffer_is_one_shot_and_capped():
    buffer = SkillContextBuffer(limit=3)
    buffer.push("a", "A")
    buffer.push("b", "B")
    buffer.push("c", "C")
    buffer.push("d", "D")

    drained = buffer.drain()

    assert "Loaded Skill: a" not in drained
    assert "Loaded Skill: b" in drained
    assert "Loaded Skill: c" in drained
    assert "Loaded Skill: d" in drained
    assert buffer.drain() == ""


def test_load_skill_pushes_body_into_context_buffer(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _write_skill(tmp_path / ".agentcli" / "skills", "demo", "demo desc", "v1", body="demo body")
    config = load_config(project_root=tmp_path)
    buffer = SkillContextBuffer()
    context = ToolContext(cwd=str(tmp_path), config=config, skill_context_buffer=buffer)

    result = asyncio.run(load_skill({"name": "demo"}, context))

    assert not result.is_error
    drained = buffer.drain()
    assert "Loaded Skill: demo" in drained
    assert "demo body" in drained


def test_skill_matcher_ranks_explicit_names_and_chinese_english_terms(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = tmp_path / "project"
    user = tmp_path / "user"
    builtin = tmp_path / "builtin"
    _write_skill(
        project / ".agentcli" / "skills",
        "pdf-ocr",
        "从扫描图片和 PDF 文档中提取文字",
        "v1",
        tags=["ocr", "文档识别"],
    )
    _write_skill(
        user,
        "web-access",
        "Live web research and webpage fetching",
        "v1",
        tags=["web", "research"],
    )
    _write_skill(builtin, "disabled-web", "current web research", "v1", tags=["web"])
    state = SkillStateStore(tmp_path / "skills.json")
    state.disable("disabled-web")
    registry = SkillRegistry(
        project,
        builtin_root=builtin,
        user_root=user,
        state_store=state,
    )

    assert registry.match("请从扫描图片里提取文字", top_k=1)[0].name == "pdf-ocr"
    assert registry.match("research the latest webpage", top_k=1)[0].name == "web-access"
    explicit = registry.match("请用 pdf-ocr，再 research the latest webpage", top_k=2)
    assert explicit[0].name == "pdf-ocr"
    assert "disabled-web" not in [skill.name for skill in registry.match("web research")]

    matcher = SkillMatcher(registry.all_skills())
    assert len(matcher.match("web research", top_k=1)) == 1


def test_skill_registry_create_update_and_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = tmp_path / "project"
    user = tmp_path / "user"
    registry = SkillRegistry(
        project,
        builtin_root=tmp_path / "builtin",
        user_root=user,
        state_store=SkillStateStore(tmp_path / "skills.json"),
    )

    created = registry.create(
        "release-check",
        description="Validate a release before publishing",
        body="# Release Check\n\nRun tests and inspect the build.",
        tags=["release", "verification"],
    )
    assert created.source == "project"
    assert created.path == project / ".agentcli" / "skills" / "release-check" / "SKILL.md"
    assert registry.load("release-check").tags == ["release", "verification"]

    with pytest.raises(FileExistsError):
        registry.create(
            "release-check",
            description="Do not overwrite",
            body="replacement",
        )

    updated = registry.update("release-check", body="updated instructions", version="2.0.0")
    assert updated.body == "updated instructions"
    assert updated.description == "Validate a release before publishing"
    assert updated.version == "2.0.0"

    user_skill = registry.create(
        "personal-style",
        description="Apply my reusable writing style",
        body="Use concise paragraphs.",
        scope="user",
    )
    assert user_skill.source == "user"
    assert user_skill.path == user / "personal-style" / "SKILL.md"


@pytest.mark.parametrize("name", ["../escape", "bad/name", "/tmp/escape", "Bad Name", "."])
def test_skill_registry_rejects_unsafe_names(tmp_path, name):
    registry = SkillRegistry(
        tmp_path / "project",
        builtin_root=tmp_path / "builtin",
        user_root=tmp_path / "user",
        state_store=SkillStateStore(tmp_path / "skills.json"),
    )

    with pytest.raises(ValueError):
        registry.create(name, description="unsafe", body="unsafe")


def test_save_skill_tool_requires_approval_and_persists_after_approval(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    tool = registry.get("save_skill")
    assert tool is not None
    assert tool.requires_approval
    assert not tool.is_read_only
    assert not tool.is_concurrency_safe
    assert set(tool.required_keys) == {"name", "description", "content"}

    call = {
        "id": "save_1",
        "function": {
            "name": "save_skill",
            "arguments": json.dumps(
                {
                    "name": "test-workflow",
                    "description": "Reusable test workflow",
                    "content": "Run the focused tests.",
                }
            ),
        },
    }
    executor = ToolExecutor(registry)
    denied = asyncio.run(
        executor.execute_all([call], ToolContext(cwd=str(tmp_path), config=config))
    )[0]
    assert denied.is_error
    assert not (tmp_path / ".agentcli" / "skills" / "test-workflow" / "SKILL.md").exists()

    approved = asyncio.run(
        executor.execute_all(
            [call],
            ToolContext(
                cwd=str(tmp_path),
                config=config,
                approval_callback=lambda _request: "approve",
            ),
        )
    )[0]
    assert not approved.is_error
    assert (tmp_path / ".agentcli" / "skills" / "test-workflow" / "SKILL.md").is_file()


def _write_skill(
    root: Path,
    name: str,
    desc: str,
    version: str,
    *,
    body: str | None = None,
    tags: list[str] | None = None,
) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_dir.joinpath("SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {desc}\nversion: {version}\n"
        f"tags: {json.dumps(tags or [], ensure_ascii=False)}\n---\n"
        f"{body or f'body for {name}'}\n",
        encoding="utf-8",
    )
