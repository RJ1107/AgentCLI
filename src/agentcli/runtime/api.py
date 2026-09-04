from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
from dataclasses import asdict
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from agentcli.agent import QueryEngine
from agentcli.bootstrap import build_tool_registry
from agentcli.config import AgentCliConfig
from agentcli.llm import create_llm_client
from agentcli.runtime.tasks import TASK_MODES, DurableTaskManager, TaskRecord
from agentcli.types import Message


class TaskCanceledError(RuntimeError):
    pass


class RuntimeApiServer:
    def __init__(
        self,
        *,
        cwd: str,
        config: AgentCliConfig,
        api_key: str,
        port: int = 8080,
        workers: int = 2,
    ):
        self.cwd = str(Path(cwd).resolve())
        self.config = config
        self.api_key = api_key
        self.port = port
        self.db_path = Path.home() / ".agentcli" / "runtime" / "runtime.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.task_manager = DurableTaskManager(
            Path.home() / ".agentcli" / "tasks" / "tasks.db", scope=self.cwd
        )
        self.workers = workers
        self._stop = threading.Event()
        self._ensure_schema()

    def serve_forever(self) -> None:
        self._start_workers()

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                outer._handle(self)

            def do_GET(self) -> None:  # noqa: N802
                outer._handle(self)

            def log_message(self, _format: str, *args: Any) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        print(f"AgentCLI Runtime API listening on http://127.0.0.1:{self.port}", flush=True)
        try:
            server.serve_forever()
        finally:
            self._stop.set()

    def work_forever(self) -> None:
        """Run only durable task workers, without exposing the HTTP API."""

        self._start_workers()
        print(f"AgentCLI task worker running with {self.workers} workers", flush=True)
        try:
            while not self._stop.wait(0.5):
                pass
        except KeyboardInterrupt:
            self._stop.set()

    def _start_workers(self) -> None:
        self.task_manager.requeue_expired()
        for index in range(self.workers):
            worker_id = f"{os.getpid()}-{index}"
            thread = threading.Thread(
                target=self._worker_loop,
                args=(worker_id,),
                name=f"agentcli-task-{index}",
                daemon=True,
            )
            thread.start()

    def _handle(self, request: BaseHTTPRequestHandler) -> None:
        if not self._authorized(request):
            _send_json(request, 401, {"error": "unauthorized"})
            return
        method = request.command
        path = request.path.split("?", 1)[0]
        body = _read_json(request)
        try:
            if method == "POST" and path == "/v1/threads":
                thread_id = self._create_thread()
                _send_json(request, 200, {"id": thread_id})
            elif method == "POST" and path.startswith("/v1/threads/") and path.endswith("/turns"):
                thread_id = path.split("/")[3]
                message = str(body.get("message") or body.get("prompt") or "")
                if not message:
                    _send_json(request, 400, {"error": "message is required"})
                    return
                mode = _mode_from_body(body)
                result = asyncio.run(self._run_turn(thread_id, message, mode=mode))
                _send_json(request, 200, result)
            elif method == "GET" and path.startswith("/v1/threads/") and path.endswith("/events"):
                thread_id = path.split("/")[3]
                self._send_events(request, thread_id)
            elif method == "POST" and path == "/v1/tasks":
                prompt = str(body.get("message") or body.get("prompt") or "")
                if not prompt:
                    _send_json(request, 400, {"error": "message is required"})
                    return
                mode = _mode_from_body(body)
                task_id = self.task_manager.add(prompt, mode=mode)
                _send_json(request, 200, {"id": task_id, "status": "queued", "mode": mode})
            elif method == "GET" and path == "/v1/tasks":
                _send_json(request, 200, {"tasks": [asdict(t) for t in self.task_manager.list()]})
            elif method == "GET" and path.startswith("/v1/tasks/"):
                task = self.task_manager.get(path.split("/")[3])
                payload = asdict(task) if task else {"error": "not found"}
                _send_json(request, 200 if task else 404, payload)
            elif method == "POST" and path.startswith("/v1/tasks/") and path.endswith("/cancel"):
                task_id = path.split("/")[3]
                _send_json(request, 200, {"canceled": self.task_manager.cancel(task_id)})
            else:
                _send_json(request, 404, {"error": "not found"})
        except ValueError as exc:
            _send_json(request, 400, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - API boundary
            _send_json(request, 500, {"error": str(exc)})

    async def _run_turn(
        self, thread_id: str, message: str, *, mode: str = "react"
    ) -> dict[str, Any]:
        self._ensure_llm_key()
        mode = _validate_mode(mode)
        self._append_event(thread_id, "turn.started", {"message": message, "mode": mode})
        registry, _manager = await build_tool_registry(config=self.config, cwd=self.cwd)
        engine = QueryEngine(
            llm_client=create_llm_client(self.config.llm),
            tool_registry=registry,
            config=self.config,
            cwd=self.cwd,
        )
        history = self._thread_history(thread_id) if mode == "react" else []
        text = ""
        events = _engine_events(engine, mode, message, history=history)
        total_tokens = 0
        async for event in events:
            event_type = str(event.get("type"))
            if event_type == "text_delta":
                text += str(event.get("text") or "")
                self._append_event(thread_id, "message.delta", {"text": event.get("text") or ""})
            elif event_type in {"tool_call", "tool_result", "error", "done"}:
                self._append_event(thread_id, event_type, _jsonable(event))
                if event_type == "error":
                    raise event["error"]
                if event_type == "done":
                    total_tokens = int(event.get("total_tokens") or 0)
        self._append_thread_message(thread_id, "user", message)
        self._append_thread_message(thread_id, "assistant", text)
        self._append_event(
            thread_id,
            "turn.completed",
            {"text": text, "mode": mode, "total_tokens": total_tokens},
        )
        return {
            "thread_id": thread_id,
            "text": text,
            "mode": mode,
            "total_tokens": total_tokens,
        }

    def _worker_loop(self, worker_id: str) -> None:
        while not self._stop.is_set():
            task = self.task_manager.claim_next(worker_id, lease_seconds=300)
            if not task:
                self._stop.wait(0.5)
                continue
            heartbeat_stop = threading.Event()
            heartbeat = threading.Thread(
                target=self._heartbeat_loop,
                args=(task.id, worker_id, heartbeat_stop),
                name=f"agentcli-heartbeat-{task.id}",
                daemon=True,
            )
            heartbeat.start()
            try:
                result = asyncio.run(self._run_task(task))
                self.task_manager.complete(task.id, result, worker_id=worker_id)
            except TaskCanceledError:
                # cancel() owns the terminal state; never overwrite it from the worker.
                pass
            except Exception as exc:  # noqa: BLE001
                self.task_manager.fail(task.id, str(exc), worker_id=worker_id)
            finally:
                heartbeat_stop.set()
                heartbeat.join(timeout=1)

    def _heartbeat_loop(
        self,
        task_id: str,
        worker_id: str,
        stop: threading.Event,
    ) -> None:
        while not stop.wait(30):
            if not self.task_manager.heartbeat(task_id, worker_id, lease_seconds=300):
                return

    async def _run_task(self, task: TaskRecord) -> str:
        self._ensure_llm_key()
        if self.task_manager.is_canceled(task.id):
            raise TaskCanceledError(task.id)
        registry, _manager = await build_tool_registry(config=self.config, cwd=self.cwd)
        engine = QueryEngine(
            llm_client=create_llm_client(self.config.llm),
            tool_registry=registry,
            config=self.config,
            cwd=self.cwd,
        )
        text = ""
        async for event in _engine_events(engine, task.mode, task.prompt):
            if self.task_manager.is_canceled(task.id):
                raise TaskCanceledError(task.id)
            if event.get("type") == "text_delta":
                text += str(event.get("text") or "")
            elif event.get("type") == "error":
                raise event["error"]
        return text

    def _ensure_llm_key(self) -> None:
        if not self.config.llm.api_key:
            raise ValueError(
                "AGENTCLI_API_KEY is not configured. Runtime turns/tasks need a working LLM key."
            )

    def _authorized(self, request: BaseHTTPRequestHandler) -> bool:
        auth = request.headers.get("authorization", "")
        token = request.headers.get("x-api-key", "")
        return auth == f"Bearer {self.api_key}" or token == self.api_key

    def _create_thread(self) -> str:
        thread_id = f"thread_{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}"
        with self._connect() as conn:
            conn.execute(
                "insert into threads(id, created_at) values (?, ?)",
                (thread_id, datetime.now(UTC).isoformat()),
            )
        self._append_event(thread_id, "thread.created", {"id": thread_id})
        return thread_id

    def _append_event(self, thread_id: str, event_type: str, payload: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                insert into events(thread_id, type, payload, created_at)
                values (?, ?, ?, ?)
                """,
                (
                    thread_id,
                    event_type,
                    json.dumps(payload, ensure_ascii=False),
                    datetime.now(UTC).isoformat(),
                ),
            )

    def _append_thread_message(self, thread_id: str, role: str, content: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                insert into messages(thread_id, role, content, created_at)
                values (?, ?, ?, ?)
                """,
                (thread_id, role, content, datetime.now(UTC).isoformat()),
            )

    def _thread_history(self, thread_id: str, limit: int = 100) -> list[Message]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                select role, content
                from messages
                where thread_id = ?
                order by id desc
                limit ?
                """,
                (thread_id, limit),
            ).fetchall()
        return [Message(role=role, content=content) for role, content in reversed(rows)]

    def _send_events(self, request: BaseHTTPRequestHandler, thread_id: str) -> None:
        with self._connect() as conn:
            rows = conn.execute(
                """
                select type, payload, created_at
                from events
                where thread_id = ?
                order by id
                """,
                (thread_id,),
            ).fetchall()
        body = "".join(
            f"event: {event_type}\ndata: {payload}\n\n" for event_type, payload, _created_at in rows
        ).encode("utf-8")
        request.send_response(200)
        request.send_header("content-type", "text/event-stream")
        request.send_header("content-length", str(len(body)))
        request.end_headers()
        request.wfile.write(body)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                create table if not exists threads (
                    id text primary key,
                    created_at text not null
                )
                """
            )
            conn.execute(
                """
                create table if not exists messages (
                    id integer primary key autoincrement,
                    thread_id text not null,
                    role text not null,
                    content text not null,
                    created_at text not null
                )
                """
            )
            conn.execute(
                "create index if not exists idx_messages_thread on messages(thread_id, id)"
            )
            conn.execute(
                """
                create table if not exists events (
                    id integer primary key autoincrement,
                    thread_id text not null,
                    type text not null,
                    payload text not null,
                    created_at text not null
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)


def _read_json(request: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(request.headers.get("content-length") or 0)
    if length == 0:
        return {}
    try:
        value = json.loads(request.rfile.read(length).decode("utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _send_json(request: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request.send_response(status)
    request.send_header("content-type", "application/json")
    request.send_header("content-length", str(len(body)))
    request.end_headers()
    request.wfile.write(body)


def _jsonable(event: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in event.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        else:
            result[key] = str(value)
    return result


def runtime_api_key(explicit: str | None = None) -> str:
    key = explicit or os.environ.get("AGENTCLI_RUNTIME_API_KEY")
    if not key:
        raise ValueError("AGENTCLI_RUNTIME_API_KEY is required for Runtime API")
    return key


def _mode_from_body(body: dict[str, Any]) -> str:
    return _validate_mode(str(body.get("mode") or "react"))


def _validate_mode(mode: str) -> str:
    value = mode.strip().lower()
    if value not in TASK_MODES:
        raise ValueError(f"mode must be one of: {', '.join(sorted(TASK_MODES))}")
    return value


def _engine_events(engine: QueryEngine, mode: str, message: str, *, history=None):
    if mode == "plan":
        return engine.plan(message)
    if mode == "team":
        return engine.team(message)
    return engine.ask(message, history=history)
