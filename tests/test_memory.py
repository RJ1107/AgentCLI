from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from agentcli.memory import FileMemory, rank


def _memory(tmp_path: Path, project: str = "project", **kwargs) -> FileMemory:
    return FileMemory(str(tmp_path / project), root=tmp_path / "memory", **kwargs)


def test_a_memory_is_a_readable_file_listed_in_the_index(tmp_path):
    memory = _memory(tmp_path)

    record = memory.save(
        "测试数据库端口是 6544，跑集成测试前先确认。",
        title="测试数据库端口 6544",
        kind="constraint",
        importance=0.8,
        keywords=["数据库", "DB", "database", "端口", "port"],
    )

    text = record.path.read_text(encoding="utf-8")
    assert text.startswith("---\ntitle: 测试数据库端口 6544\nkind: constraint")
    assert "keywords: [数据库, DB, database, 端口, port]" in text
    index = (memory.folder / "MEMORY.md").read_text(encoding="utf-8")
    assert f"- [constraint] 测试数据库端口 6544 ({record.name}.md)" in index
    assert memory.get(record.name).content.startswith("测试数据库端口是 6544")


def test_saving_the_same_title_or_content_updates_instead_of_duplicating(tmp_path):
    memory = _memory(tmp_path)
    first = memory.save("端口 5433", title="测试端口", importance=0.4, keywords=["port"])
    second = memory.save("端口 6544", title="测试端口", importance=0.9, keywords=["DB"])
    memory.save("端口 6544", title="另一种说法")

    assert len(memory.list()) == 1
    record = memory.list()[0]
    assert record.name == first.name == second.name
    assert record.content == "端口 6544"
    assert record.importance == 0.9
    assert record.keywords == ["port", "DB"]


def test_keywords_let_other_words_find_a_memory(tmp_path):
    with_keywords = _memory(tmp_path, "a")
    with_keywords.save(
        "数据库连接池上限是 20",
        keywords=["DB", "database", "connection pool", "连接池"],
    )
    without = _memory(tmp_path, "b")
    without.save("数据库连接池上限是 20")

    assert with_keywords.search("DB pool size")
    assert not without.search("DB pool size")


def test_relevance_decides_the_order_not_importance(tmp_path):
    memory = _memory(tmp_path)
    memory.save("发布分支统一叫 release/kestrel", title="发布分支命名", importance=0.2)
    memory.save("金额字段一律用 Decimal", title="金额类型", importance=1.0)

    hits = memory.search("发布分支叫什么")

    assert hits[0].record.title == "发布分支命名"
    assert all(hit.record.title != "金额类型" for hit in hits)


def test_coverage_is_comparable_across_queries_and_gates_recall(tmp_path):
    memory = _memory(tmp_path)
    memory.save("CI 超时是 38 分钟", keywords=["CI", "timeout", "超时"])

    exact = memory.search("CI 超时多少分钟")
    loose = memory.search("CI 的缓存目录和并发数和超时和重试次数都是多少")

    assert exact[0].coverage > loose[0].coverage
    assert not memory.search("CI 的缓存目录和并发数和超时和重试次数都是多少", min_coverage=0.6)


def test_expired_memories_disappear(tmp_path):
    memory = _memory(tmp_path)
    memory.save("临时：今天下午数据库维护", expires="2000-01-01")
    memory.save("长期：金额用 Decimal")

    assert [r.content for r in memory.list()] == ["长期：金额用 Decimal"]
    assert len(list(memory.folder.glob("*.md"))) == 2  # the other file and MEMORY.md


def test_quota_drops_the_least_important_first(tmp_path):
    memory = _memory(tmp_path, max_entries=2)
    memory.save("a fact", title="low", importance=0.1)
    memory.save("b fact", title="high", importance=0.9)
    memory.save("c fact", title="new", importance=0.5)

    assert sorted(r.title for r in memory.list()) == ["high", "new"]


def test_index_is_capped_and_says_what_it_left_out(tmp_path):
    memory = _memory(tmp_path)
    for index in range(5):
        memory.save(f"fact number {index}", title=f"fact {index}", importance=index / 10)

    text = memory.index_text(max_lines=3)

    assert text.count("\n- ") + text.startswith("- ") == 3
    assert "fact 4" in text and "fact 0" not in text
    assert "(+2 more; use search_memory)" in text


def test_projects_are_separate(tmp_path):
    _memory(tmp_path, "one").save("only in one")
    assert _memory(tmp_path, "two").list() == []


def test_old_sqlite_memories_are_imported_once(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    db = tmp_path / "memory.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "create table memories (scope text, content text, kind text, importance real, "
            "expires_at text)"
        )
        conn.execute(
            "insert into memories values (?, ?, ?, ?, ?)",
            (str(project.resolve()), "测试统一用 pytest", "constraint", 0.7, None),
        )
        conn.execute(
            "insert into memories values (?, ?, ?, ?, ?)",
            (str(tmp_path / "elsewhere"), "别的项目", "fact", 0.5, None),
        )

    memory = FileMemory(str(project), root=tmp_path / "memory", legacy_db=str(db))

    assert [(r.content, r.kind, r.source) for r in memory.list()] == [
        ("测试统一用 pytest", "constraint", "migrated")
    ]


def test_empty_or_oversized_content_is_refused(tmp_path):
    memory = _memory(tmp_path, max_chars=100)
    with pytest.raises(ValueError):
        memory.save("   ")
    with pytest.raises(ValueError):
        memory.save("x" * 101)


def test_rank_returns_nothing_for_unrelated_text(tmp_path):
    memory = _memory(tmp_path)
    memory.save("金额字段一律用 Decimal")
    assert rank("zzz qqq", memory.list()) == []
