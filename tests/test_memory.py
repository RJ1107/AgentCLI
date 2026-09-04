from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from agentcli.memory import MemoryManager


def test_legacy_schema_is_migrated_and_normalized_duplicates_are_merged(tmp_path):
    db_path = tmp_path / "memory.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            create table memories (
                id integer primary key autoincrement,
                scope text not null,
                content text not null,
                created_at text not null
            )
            """
        )
        conn.execute(
            "insert into memories(scope, content, created_at) values (?, ?, ?)",
            ("project", "AgentCLI   Uses Memory", "2026-01-01T00:00:00+00:00"),
        )
        conn.execute(
            "insert into memories(scope, content, created_at) values (?, ?, ?)",
            ("project", "agentcli uses memory", "2026-01-02T00:00:00+00:00"),
        )

    manager = MemoryManager(db_path, scope="project")
    rows = manager.list()

    assert len(rows) == 1
    assert rows[0].source == "legacy"
    assert rows[0].updated_at == "2026-01-02T00:00:00+00:00"
    assert len(rows[0].content_hash) == 64
    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("pragma table_info(memories)")}
    assert {
        "kind",
        "source",
        "importance",
        "confidence",
        "updated_at",
        "expires_at",
        "access_count",
        "content_hash",
    } <= columns


def test_save_rejects_empty_oversized_and_invalid_scores(tmp_path):
    manager = MemoryManager(
        tmp_path / "memory.db",
        scope="project",
        max_content_length=5,
    )

    with pytest.raises(ValueError, match="must not be empty"):
        manager.save("  \n ")
    with pytest.raises(ValueError, match="exceeds 5"):
        manager.save("123456")
    with pytest.raises(ValueError, match="importance must be between 0 and 1"):
        manager.save("valid", importance=1.1)


def test_save_deduplicates_normalized_content_and_updates_metadata(tmp_path):
    manager = MemoryManager(tmp_path / "memory.db", scope="project")

    first_id = manager.save(
        "AgentCLI   Uses Memory",
        source="user",
        importance=0.4,
        confidence=0.6,
    )
    second_id = manager.save(
        "agentcli uses memory",
        kind="preference",
        source="agent",
        importance=0.9,
        confidence=0.8,
    )

    assert first_id == second_id
    rows = manager.list()
    assert len(rows) == 1
    assert rows[0].content == "agentcli uses memory"
    assert rows[0].kind == "preference"
    assert rows[0].source == "agent"
    assert rows[0].importance == 0.9
    assert rows[0].confidence == 0.8


def test_scope_quota_evicts_low_value_memory_and_keeps_new_entry(tmp_path):
    manager = MemoryManager(tmp_path / "memory.db", scope="project", max_entries=2)

    low_id = manager.save("low value", importance=0.1, confidence=0.2)
    high_id = manager.save("high value", importance=0.9, confidence=1.0)
    new_id = manager.save("new value", importance=0.5, confidence=0.8)

    ids = {entry.id for entry in manager.list(limit=10)}
    assert ids == {high_id, new_id}
    assert low_id not in ids
    assert manager.stats()["total"] == 2


def test_recall_handles_chinese_and_english_and_ignores_unrelated_memory(tmp_path):
    manager = MemoryManager(tmp_path / "memory.db", scope="project")
    chinese_id = manager.save(
        "长期记忆通过相关度和重要度进行召回",
        kind="architecture",
        importance=0.8,
    )
    english_id = manager.save(
        "Python memory uses normalized content hashes for deduplication",
        kind="architecture",
        importance=0.7,
    )
    manager.save("The release checklist requires a clean build", importance=1.0)

    chinese = manager.recall("长期记忆怎么召回", limit=2)
    english = manager.search("python memory deduplication", limit=2)

    assert chinese[0].id == chinese_id
    assert chinese[0].access_count == 1
    assert english[0].id == english_id
    assert manager.search("completely unrelated phrase") == []


def test_recall_orders_equally_relevant_memories_by_importance(tmp_path):
    manager = MemoryManager(tmp_path / "memory.db", scope="project")
    manager.save("memory retrieval uses keyword matching", importance=0.2)
    high_id = manager.save("memory retrieval uses semantic matching", importance=0.9)

    results = manager.recall("memory retrieval", limit=2, mark_access=False)

    assert results[0].id == high_id


def test_expired_memories_are_purged_from_reads_and_stats(tmp_path):
    manager = MemoryManager(tmp_path / "memory.db", scope="project")
    expired = datetime.now(UTC) - timedelta(seconds=1)

    manager.save("already expired", expires_at=expired)

    assert manager.list() == []
    assert manager.recall("expired") == []
    assert manager.stats()["total"] == 0


def test_mark_access_stats_delete_clear_and_scope_isolation(tmp_path):
    db_path = tmp_path / "memory.db"
    project = MemoryManager(db_path, scope="project")
    other = MemoryManager(db_path, scope="other")
    first_id = project.save("first fact", kind="fact")
    second_id = project.save("second preference", kind="preference")
    other_id = other.save("other scope fact")

    assert project.mark_access([first_id, first_id, second_id]) == 2
    stats = project.stats()
    assert stats["total"] == 2
    assert stats["total_access_count"] == 2
    assert stats["by_kind"] == {"fact": 1, "preference": 1}
    assert not project.delete(other_id)
    assert project.delete(first_id)
    assert project.clear() == 1
    assert other.stats()["total"] == 1
