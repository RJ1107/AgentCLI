from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

TASK_MODES = {"react", "plan", "team"}
TERMINAL_STATUSES = {"completed", "failed", "canceled"}


@dataclass(slots=True)
class TaskRecord:
    id: str
    prompt: str
    status: str
    created_at: str
    updated_at: str
    result: str | None = None
    error: str | None = None
    mode: str = "react"
    attempts: int = 0
    worker_id: str | None = None
    lease_expires_at: str | None = None
    scope: str = ""


class DurableTaskManager:
    """SQLite-backed queue with atomic claims, leases, and cancellation-safe completion."""

    def __init__(self, db_path: str | Path, *, scope: str | Path | None = None):
        self.db_path = Path(db_path).expanduser()
        self.scope = str(Path(scope).resolve()) if scope else ""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def add(self, prompt: str, *, mode: str = "react") -> str:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("task prompt cannot be empty")
        mode = _validate_mode(mode)
        task_id = _new_id("task")
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                insert into tasks(id, prompt, status, created_at, updated_at, mode, scope)
                values (?, ?, 'queued', ?, ?, ?, ?)
                """,
                (task_id, prompt, now, now, mode, self.scope),
            )
        return task_id

    def claim_next(
        self,
        worker_id: str | None = None,
        *,
        lease_seconds: float = 300,
    ) -> TaskRecord | None:
        """Atomically lease the oldest queued task or an abandoned expired task."""

        owner = worker_id or "anonymous-worker"
        now = _now()
        lease_expires = _future(lease_seconds)
        conn = self._connect()
        try:
            conn.execute("begin immediate")
            scope_clause = " and scope = ?" if self.scope else ""
            select_params: list[str] = [now]
            if self.scope:
                select_params.append(self.scope)
            row = conn.execute(
                f"""
                select {_TASK_COLUMNS}
                from tasks
                where (status = 'queued'
                   or (status = 'running'
                       and (lease_expires_at is null or lease_expires_at <= ?)))
                  {scope_clause}
                order by case status when 'queued' then 0 else 1 end, created_at
                limit 1
                """,
                select_params,
            ).fetchone()
            if not row:
                conn.commit()
                return None
            task_id = str(row[0])
            update_scope_clause = " and scope = ?" if self.scope else ""
            update_params = [now, owner, lease_expires, task_id, now]
            if self.scope:
                update_params.append(self.scope)
            cursor = conn.execute(
                f"""
                update tasks
                set status = 'running', updated_at = ?, worker_id = ?, lease_expires_at = ?,
                    attempts = attempts + 1, error = null
                where id = ?
                  and (status = 'queued'
                       or (status = 'running'
                           and (lease_expires_at is null or lease_expires_at <= ?)))
                  {update_scope_clause}
                """,
                update_params,
            )
            if cursor.rowcount != 1:
                conn.rollback()
                return None
            claimed = conn.execute(
                f"select {_TASK_COLUMNS} from tasks where id = ?", (task_id,)
            ).fetchone()
            conn.commit()
            return _task_from_row(claimed) if claimed else None
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def heartbeat(self, task_id: str, worker_id: str, *, lease_seconds: float = 300) -> bool:
        scope_clause = " and scope = ?" if self.scope else ""
        params: list[str] = [_now(), _future(lease_seconds), task_id, worker_id]
        if self.scope:
            params.append(self.scope)
        with self._connect() as conn:
            cursor = conn.execute(
                f"""
                update tasks
                set updated_at = ?, lease_expires_at = ?
                where id = ? and status = 'running' and worker_id = ?
                {scope_clause}
                """,
                params,
            )
            return cursor.rowcount == 1

    def complete(self, task_id: str, result: str, *, worker_id: str | None = None) -> bool:
        return self._finish(
            task_id,
            "completed",
            result=result,
            error=None,
            worker_id=worker_id,
        )

    def fail(self, task_id: str, error: str, *, worker_id: str | None = None) -> bool:
        return self._finish(
            task_id,
            "failed",
            result=None,
            error=error,
            worker_id=worker_id,
        )

    def cancel(self, task_id: str) -> bool:
        scope_clause = " and scope = ?" if self.scope else ""
        params = [_now(), task_id]
        if self.scope:
            params.append(self.scope)
        with self._connect() as conn:
            cursor = conn.execute(
                f"""
                update tasks
                set status = 'canceled', updated_at = ?, worker_id = null,
                    lease_expires_at = null
                where id = ? and status in ('queued', 'running')
                {scope_clause}
                """,
                params,
            )
            return cursor.rowcount > 0

    def is_canceled(self, task_id: str) -> bool:
        scope_clause = " and scope = ?" if self.scope else ""
        params = [task_id, *([self.scope] if self.scope else [])]
        with self._connect() as conn:
            row = conn.execute(
                f"select status from tasks where id = ?{scope_clause}", params
            ).fetchone()
        return bool(row and row[0] == "canceled")

    def requeue_expired(self) -> int:
        now = _now()
        scope_clause = " and scope = ?" if self.scope else ""
        params = [now, now, *([self.scope] if self.scope else [])]
        with self._connect() as conn:
            cursor = conn.execute(
                f"""
                update tasks
                set status = 'queued', updated_at = ?, worker_id = null,
                    lease_expires_at = null,
                    error = coalesce(error, 'worker lease expired; task requeued')
                where status = 'running'
                  and (lease_expires_at is null or lease_expires_at <= ?)
                  {scope_clause}
                """,
                params,
            )
            return int(cursor.rowcount)

    def list(self, limit: int = 50) -> list[TaskRecord]:
        scope_clause = " where scope = ?" if self.scope else ""
        params: list[str | int] = [self.scope, limit] if self.scope else [limit]
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                select {_TASK_COLUMNS}
                from tasks
                {scope_clause}
                order by created_at desc
                limit ?
                """,
                params,
            ).fetchall()
        return [_task_from_row(row) for row in rows]

    def get(self, task_id: str) -> TaskRecord | None:
        scope_clause = " and scope = ?" if self.scope else ""
        params = [task_id, *([self.scope] if self.scope else [])]
        with self._connect() as conn:
            row = conn.execute(
                f"select {_TASK_COLUMNS} from tasks where id = ?{scope_clause}", params
            ).fetchone()
        return _task_from_row(row) if row else None

    def _finish(
        self,
        task_id: str,
        status: str,
        *,
        result: str | None,
        error: str | None,
        worker_id: str | None,
    ) -> bool:
        owner_clause = " and worker_id = ?" if worker_id else ""
        scope_clause = " and scope = ?" if self.scope else ""
        params: list[str | None] = [status, result, error, _now(), task_id]
        if worker_id:
            params.append(worker_id)
        if self.scope:
            params.append(self.scope)
        with self._connect() as conn:
            cursor = conn.execute(
                f"""
                update tasks
                set status = ?, result = ?, error = ?, updated_at = ?,
                    worker_id = null, lease_expires_at = null
                where id = ? and status = 'running'{owner_clause}
                {scope_clause}
                """,
                params,
            )
            return cursor.rowcount == 1

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                create table if not exists tasks (
                    id text primary key,
                    prompt text not null,
                    status text not null,
                    created_at text not null,
                    updated_at text not null,
                    result text,
                    error text,
                    mode text not null default 'react',
                    attempts integer not null default 0,
                    worker_id text,
                    lease_expires_at text,
                    scope text not null default ''
                )
                """
            )
            existing = {row[1] for row in conn.execute("pragma table_info(tasks)").fetchall()}
            migrations = {
                "mode": "alter table tasks add column mode text not null default 'react'",
                "attempts": "alter table tasks add column attempts integer not null default 0",
                "worker_id": "alter table tasks add column worker_id text",
                "lease_expires_at": "alter table tasks add column lease_expires_at text",
                "scope": "alter table tasks add column scope text not null default ''",
            }
            for column, statement in migrations.items():
                if column not in existing:
                    conn.execute(statement)
            conn.execute("create index if not exists idx_tasks_status on tasks(status, created_at)")
            conn.execute(
                "create index if not exists idx_tasks_lease on tasks(status, lease_expires_at)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=30, isolation_level=None)


_TASK_COLUMNS = (
    "id, prompt, status, created_at, updated_at, result, error, mode, attempts, "
    "worker_id, lease_expires_at, scope"
)


def _task_from_row(row) -> TaskRecord:
    return TaskRecord(*row)


def _validate_mode(mode: str) -> str:
    value = str(mode or "react").strip().lower()
    if value not in TASK_MODES:
        raise ValueError(f"task mode must be one of: {', '.join(sorted(TASK_MODES))}")
    return value


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _future(seconds: float) -> str:
    return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}"
