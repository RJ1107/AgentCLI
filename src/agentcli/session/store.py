"""Saved conversations, so a session can be resumed after AgentCLI exits.

Each session is a folder under <AGENTCLI_HOME>/sessions/<id>/:

- meta.json         title, project folder, model, times, turn count
- history.json      the messages the model sees (after any compaction): what a resume loads
- transcript.jsonl  every user message and final answer, append-only, for reading back
- tool-results/     full copies of tool results that compaction cleared (see context.spill)

A session folder is written only once the first message is sent, so starting AgentCLI and
leaving creates nothing.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from agentcli.paths import agentcli_home
from agentcli.types import Message

TITLE_CHARS = 60


def sessions_root() -> Path:
    return agentcli_home() / "sessions"


def new_session_id() -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


@dataclass(slots=True)
class SessionMeta:
    id: str
    cwd: str
    title: str = ""
    model: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    turns: int = 0
    # Set by /rename; a renamed session keeps its name instead of the first message.
    named: bool = False

    @property
    def updated_text(self) -> str:
        return datetime.fromtimestamp(self.updated_at).strftime("%m-%d %H:%M")


class Session:
    def __init__(self, folder: Path, meta: SessionMeta):
        self.folder = folder
        self.meta = meta

    @property
    def id(self) -> str:
        return self.meta.id

    def record_turn(self, user_text: str, answer: str, history: list[Message], model: str) -> None:
        """Save one exchange: the transcript line, the model-facing history, and the meta."""

        self.folder.mkdir(parents=True, exist_ok=True)
        now = time.time()
        if not self.meta.created_at:
            self.meta.created_at = now
        if not self.meta.named and not self.meta.title:
            self.meta.title = _title(user_text)
        self.meta.updated_at = now
        self.meta.turns += 1
        self.meta.model = model
        with (self.folder / "transcript.jsonl").open("a", encoding="utf-8") as handle:
            for role, text in (("user", user_text), ("assistant", answer)):
                entry = {"time": now, "role": role, "text": text}
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.save_history(history)

    def save_history(self, history: list[Message]) -> None:
        """Rewrite history.json (after a turn or a /compact) and the meta."""

        if not self.meta.created_at:
            return  # nothing sent yet: nothing worth keeping
        self.folder.mkdir(parents=True, exist_ok=True)
        _write_json(self.folder / "history.json", [_message_to_dict(m) for m in history])
        _write_json(self.folder / "meta.json", asdict(self.meta))

    def load_history(self) -> list[Message]:
        path = self.folder / "history.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return [_message_from_dict(item) for item in raw if isinstance(item, dict)]

    def rename(self, title: str) -> None:
        self.meta.title = " ".join(title.split())[:TITLE_CHARS]
        self.meta.named = True
        if self.meta.created_at:
            _write_json(self.folder / "meta.json", asdict(self.meta))


class SessionStore:
    def __init__(self, root: Path | None = None):
        self.root = root or sessions_root()

    def create(self, cwd: str, model: str = "") -> Session:
        meta = SessionMeta(id=new_session_id(), cwd=_norm(cwd), model=model)
        return Session(self.root / meta.id, meta)

    def open(self, session_id: str) -> Session | None:
        meta = self._read_meta(self.root / session_id)
        return Session(self.root / session_id, meta) if meta else None

    def list(self, cwd: str | None = None, limit: int = 50) -> list[SessionMeta]:
        """Saved sessions, newest first; only those of one project folder when cwd is given."""

        metas = []
        try:
            folders = list(self.root.iterdir())
        except OSError:
            return []
        for folder in folders:
            meta = self._read_meta(folder)
            if meta and (cwd is None or meta.cwd == _norm(cwd)):
                metas.append(meta)
        metas.sort(key=lambda m: m.updated_at, reverse=True)
        return metas[:limit]

    def latest(self, cwd: str) -> Session | None:
        metas = self.list(cwd, limit=1)
        return self.open(metas[0].id) if metas else None

    def find(self, prefix: str, cwd: str | None = None) -> Session | None:
        """Open by full id or a unique prefix of one."""

        matches = [m for m in self.list(cwd, limit=10_000) if m.id.startswith(prefix.strip())]
        return self.open(matches[0].id) if len(matches) == 1 else None

    def delete(self, session_id: str) -> bool:
        folder = self.root / session_id
        if not (folder / "meta.json").exists():
            return False
        shutil.rmtree(folder, ignore_errors=True)
        return True

    def cleanup(self, retention_days: float = 30.0) -> int:
        """Delete sessions not used for retention_days. Returns how many were removed."""

        if retention_days <= 0:
            return 0
        cutoff = time.time() - retention_days * 86_400
        removed = 0
        try:
            folders = list(self.root.iterdir())
        except OSError:
            return 0
        for folder in folders:
            try:
                if folder.is_dir() and _last_used(folder) < cutoff:
                    shutil.rmtree(folder, ignore_errors=True)
                    removed += 1
            except OSError:
                continue
        return removed

    @staticmethod
    def _read_meta(folder: Path) -> SessionMeta | None:
        try:
            raw = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
            known = SessionMeta.__dataclass_fields__
            return SessionMeta(**{k: v for k, v in raw.items() if k in known})
        except (OSError, json.JSONDecodeError, TypeError):
            return None


def _title(text: str) -> str:
    line = " ".join(text.split())
    return line if len(line) <= TITLE_CHARS else line[: TITLE_CHARS - 1] + "…"


def _norm(cwd: str) -> str:
    return os.path.normcase(str(Path(cwd).resolve()))


def _last_used(folder: Path) -> float:
    meta = folder / "meta.json"
    return (meta if meta.exists() else folder).stat().st_mtime


def _write_json(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(temporary, path)


def _message_to_dict(message: Message) -> dict[str, Any]:
    data: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.name:
        data["name"] = message.name
    if message.tool_call_id:
        data["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        data["tool_calls"] = message.tool_calls
    return data


def _message_from_dict(data: dict[str, Any]) -> Message:
    return Message(
        role=data.get("role", "user"),
        content=data.get("content", ""),
        name=data.get("name"),
        tool_call_id=data.get("tool_call_id"),
        tool_calls=list(data.get("tool_calls") or []),
    )
