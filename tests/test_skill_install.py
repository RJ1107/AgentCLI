from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agentcli.bootstrap import build_tool_registry
from agentcli.config import load_config
from agentcli.skill import SkillRegistry
from agentcli.tools import ToolRegistry, get_builtin_tools
from agentcli.tools.base import ToolContext

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "skills"


def _context(project: Path) -> ToolContext:
    return ToolContext(cwd=str(project), config=load_config(project_root=project))


def _run(registry: ToolRegistry, name: str, payload: dict, project: Path):
    return asyncio.run(registry.get(name).execute(payload, _context(project)))


def test_install_copies_the_whole_skill_folder(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    registry = SkillRegistry(project)

    skill = registry.install(EXAMPLES / "finance-qa", scope="user")

    folder = skill.path.parent
    assert skill.name == "finance-qa" and skill.source == "user"
    assert (folder / "scripts" / "fin_calc.py").is_file()
    assert (folder / "references" / "metrics.md").is_file()
    with pytest.raises(FileExistsError):
        registry.install(EXAMPLES / "finance-qa", scope="user")
    registry.install(EXAMPLES / "finance-qa", scope="user", overwrite=True)


def test_install_rejects_folders_without_a_described_skill(tmp_path):
    empty = tmp_path / "not-a-skill"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        SkillRegistry(tmp_path).install(empty)

    nameless = tmp_path / "nameless"
    nameless.mkdir()
    (nameless / "SKILL.md").write_text("---\nname: nameless\n---\nbody\n", encoding="utf-8")
    with pytest.raises(ValueError, match="description"):
        SkillRegistry(tmp_path).install(nameless)


def test_loading_a_skill_points_at_its_files_and_they_are_readable(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    skill = SkillRegistry(project).install(EXAMPLES / "finance-qa", scope="user")
    tools = ToolRegistry()
    tools.register_all(get_builtin_tools())

    loaded = _run(tools, "load_skill", {"name": "finance-qa"}, project)
    assert f"Skill folder: {skill.path.parent}" in loaded.content
    assert "- scripts/fin_calc.py" in loaded.content

    # The user skill folder is outside the workspace, yet its files can be read.
    reference = skill.path.parent / "references" / "metrics.md"
    read = _run(tools, "read_file", {"path": str(reference)}, project)
    assert not read.is_error and "CAGR" in read.content

    # Anything else outside the workspace is still refused.
    secret = tmp_path / "secret.txt"
    secret.write_text("private", encoding="utf-8")
    with pytest.raises(Exception, match="escapes workspace"):
        _run(tools, "read_file", {"path": str(secret)}, project)


def test_every_skill_is_listed_in_load_skill(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    SkillRegistry(project).install(EXAMPLES / "trump-style", scope="user")
    config = load_config(project_root=project)
    config.features.mcp = False

    registry, _ = asyncio.run(build_tool_registry(config=config, cwd=str(project)))

    description = registry.get("load_skill").description
    assert "- trump-style:" in description
    assert "- web-access:" in description
