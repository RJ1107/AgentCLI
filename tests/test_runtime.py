from __future__ import annotations

import threading

import pytest

from agentcli.config import load_config
from agentcli.runtime import DurableTaskManager
from agentcli.runtime.api import RuntimeApiServer, _engine_events, runtime_api_key


def test_durable_task_lifecycle(tmp_path):
    manager = DurableTaskManager(tmp_path / "tasks.db")
    task_id = manager.add("do work")

    task = manager.claim_next()
    assert task is not None
    assert task.id == task_id
    assert task.status == "running"

    assert manager.complete(task_id, "done")
    completed = manager.get(task_id)
    assert completed is not None
    assert completed.status == "completed"
    assert completed.result == "done"


def test_durable_task_cancel(tmp_path):
    manager = DurableTaskManager(tmp_path / "tasks.db")
    task_id = manager.add("do work")

    assert manager.cancel(task_id)
    assert manager.get(task_id).status == "canceled"  # type: ignore[union-attr]


def test_durable_task_claim_is_atomic_across_workers(tmp_path):
    manager = DurableTaskManager(tmp_path / "tasks.db")
    task_id = manager.add("do work", mode="plan")
    barrier = threading.Barrier(2)
    claimed = []

    def claim(worker_id: str) -> None:
        barrier.wait()
        claimed.append(manager.claim_next(worker_id))

    workers = [threading.Thread(target=claim, args=(f"worker-{index}",)) for index in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    records = [record for record in claimed if record is not None]
    assert len(records) == 1
    assert records[0].id == task_id
    assert records[0].mode == "plan"
    assert records[0].attempts == 1


def test_cancel_running_task_cannot_be_overwritten_by_worker_completion(tmp_path):
    manager = DurableTaskManager(tmp_path / "tasks.db")
    task_id = manager.add("do work")
    task = manager.claim_next("worker-1")
    assert task is not None

    assert manager.cancel(task_id)
    assert not manager.complete(task_id, "late result", worker_id="worker-1")

    canceled = manager.get(task_id)
    assert canceled is not None
    assert canceled.status == "canceled"
    assert canceled.result is None


def test_expired_worker_lease_is_reclaimed(tmp_path):
    manager = DurableTaskManager(tmp_path / "tasks.db")
    task_id = manager.add("do work", mode="team")
    first = manager.claim_next("worker-1", lease_seconds=-1)
    assert first is not None

    second = manager.claim_next("worker-2")

    assert second is not None
    assert second.id == task_id
    assert second.worker_id == "worker-2"
    assert second.attempts == 2


def test_task_mode_validation(tmp_path):
    manager = DurableTaskManager(tmp_path / "tasks.db")

    try:
        manager.add("bad", mode="unknown")
    except ValueError as exc:
        assert "task mode" in str(exc)
    else:  # pragma: no cover - explicit failure message
        raise AssertionError("invalid task mode was accepted")


def test_runtime_thread_history_persists_between_turns(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    server = RuntimeApiServer(
        cwd=str(tmp_path),
        config=load_config(project_root=tmp_path),
        api_key="test-key",
        workers=1,
    )
    thread_id = server._create_thread()
    server._append_thread_message(thread_id, "user", "first question")
    server._append_thread_message(thread_id, "assistant", "first answer")

    history = server._thread_history(thread_id)

    assert [(message.role, message.content) for message in history] == [
        ("user", "first question"),
        ("assistant", "first answer"),
    ]


def test_runtime_routes_react_plan_and_team_modes():
    class Engine:
        def ask(self, message, history=None):
            return ("react", message, history)

        def plan(self, message):
            return ("plan", message)

        def team(self, message):
            return ("team", message)

    engine = Engine()

    assert _engine_events(engine, "react", "x", history=["h"])[0] == "react"
    assert _engine_events(engine, "plan", "x")[0] == "plan"
    assert _engine_events(engine, "team", "x")[0] == "team"


def test_task_workers_do_not_consume_another_project_scope(tmp_path):
    db = tmp_path / "tasks.db"
    project_a = DurableTaskManager(db, scope=tmp_path / "a")
    project_b = DurableTaskManager(db, scope=tmp_path / "b")
    task_a = project_a.add("A")
    task_b = project_b.add("B")

    claimed_a = project_a.claim_next("worker-a")
    claimed_b = project_b.claim_next("worker-b")

    assert claimed_a is not None and claimed_a.id == task_a
    assert claimed_b is not None and claimed_b.id == task_b
    assert claimed_a.scope != claimed_b.scope


def test_runtime_api_key_prefers_explicit_value(monkeypatch):
    monkeypatch.setenv("AGENTCLI_RUNTIME_API_KEY", "env-key")

    assert runtime_api_key("explicit-key") == "explicit-key"


def test_runtime_api_key_requires_value(monkeypatch):
    monkeypatch.delenv("AGENTCLI_RUNTIME_API_KEY", raising=False)

    with pytest.raises(ValueError, match="AGENTCLI_RUNTIME_API_KEY"):
        runtime_api_key()


def test_task_completion_requires_matching_worker_id(tmp_path):
    manager = DurableTaskManager(tmp_path / "tasks.db")
    task_id = manager.add("do work")
    assert manager.claim_next("worker-1") is not None

    assert not manager.complete(task_id, "wrong worker", worker_id="worker-2")
    assert manager.complete(task_id, "right worker", worker_id="worker-1")


def test_requeue_expired_only_affects_matching_scope(tmp_path):
    db = tmp_path / "tasks.db"
    project_a = DurableTaskManager(db, scope=tmp_path / "a")
    project_b = DurableTaskManager(db, scope=tmp_path / "b")
    task_a = project_a.add("A")
    task_b = project_b.add("B")
    assert project_a.claim_next("worker-a", lease_seconds=-1) is not None
    assert project_b.claim_next("worker-b", lease_seconds=-1) is not None

    assert project_a.requeue_expired() == 1

    assert project_a.get(task_a).status == "queued"  # type: ignore[union-attr]
    assert project_b.get(task_b).status == "running"  # type: ignore[union-attr]
