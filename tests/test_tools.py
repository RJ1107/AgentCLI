from __future__ import annotations

import asyncio

from agentcli.config import load_config
from agentcli.tools import ToolRegistry, get_builtin_tools
from agentcli.tools.base import ToolContext
from agentcli.tools.builtins import save_memory, search_memory
from agentcli.tools.file_ops import directory_tree, edit_file, glob_files, grep


def test_read_write_file_tool(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.policy.hitl_mode = "never"
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    context = ToolContext(cwd=str(tmp_path), config=config)

    async def run():
        write = registry.get("write_file")
        read = registry.get("read_file")
        assert write and read
        write_result = await write.execute(
            {"path": "hello.txt", "content": "hello\nworld\n"},
            context,
        )
        read_result = await read.execute({"path": "hello.txt"}, context)
        return write_result, read_result

    write_result, read_result = asyncio.run(run())
    assert not write_result.is_error
    assert "1: hello" in read_result.content
    assert "2: world" in read_result.content


def test_builtin_tools_include_memory_recall_and_skill_sedimentation():
    names = {tool.name for tool in get_builtin_tools()}

    assert "search_memory" in names
    assert "save_skill" in names


def test_memory_tools_save_metadata_and_recall_relevant_items(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    context = ToolContext(cwd=str(tmp_path), config=config)

    saved = asyncio.run(
        save_memory(
            {
                "content": "用户偏好用 uv 执行 Python 测试",
                "kind": "preference",
                "importance": 0.9,
            },
            context,
        )
    )
    recalled = asyncio.run(search_memory({"query": "怎么执行测试"}, context))

    assert not saved.is_error
    assert not recalled.is_error
    assert "uv" in recalled.content
    assert "preference" in recalled.content


def test_edit_file_dry_run_returns_diff_without_modifying_file(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("print('old')\n", encoding="utf-8")

    result = edit_file(str(tmp_path), "app.py", "old", "new", dry_run=True)

    assert not result.is_error
    assert "[DRY RUN]" in result.content
    assert "+new" in result.content
    assert target.read_text(encoding="utf-8") == "print('old')\n"


def test_edit_file_reports_exact_match_miss(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("print('old')\n", encoding="utf-8")

    result = edit_file(str(tmp_path), "app.py", "missing", "new")

    assert result.is_error
    assert "old_text" in result.content
    assert target.read_text(encoding="utf-8") == "print('old')\n"


def test_directory_tree_excludes_requested_directories(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('ok')", encoding="utf-8")
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "bundle.js").write_text("ignored", encoding="utf-8")

    result = directory_tree(str(tmp_path), ".", exclude_patterns=("dist",))

    assert not result.is_error
    assert "src/" in result.content
    assert "app.py" in result.content
    assert "dist" not in result.content


def test_glob_rejects_parent_directory_escape(tmp_path):
    result = glob_files(str(tmp_path), "../*.py")

    assert result.is_error
    assert "inside workspace" in result.content


def test_grep_invalid_regex_returns_error(tmp_path):
    (tmp_path / "app.py").write_text("print('ok')", encoding="utf-8")

    result = grep(str(tmp_path), "(", path="app.py")

    assert result.is_error
    assert "invalid regex" in result.content
