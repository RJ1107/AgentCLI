"""Long-term memory as Markdown files, one per memory, with a generated MEMORY.md index.

<AGENTCLI_HOME>/memory/<project key>/
    MEMORY.md     one line per memory, rebuilt on every change (do not edit by hand)
    <name>.md     frontmatter (title, kind, importance, keywords, dates) and the details

Files instead of a database: a person can read, fix, or delete a memory in any editor, the
agent can open one with read_file, and the index is small enough to sit in the system
prompt, so the model sees what exists and fetches what it needs, the way Claude Code and
Codex keep memory.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from agentcli.memory.retrieval import Hit, rank
from agentcli.paths import agentcli_home

INDEX = "MEMORY.md"
# Index order: rules the agent must not break first, loose facts last.
KINDS = ("constraint", "correction", "preference", "decision", "fact")


@dataclass(slots=True)
class MemoryRecord:
    name: str
    title: str
    content: str
    kind: str = "fact"
    importance: float = 0.5
    keywords: list[str] = field(default_factory=list)
    created: str = ""
    updated: str = ""
    expires: str = ""
    source: str = "agent"
    path: Path | None = None

    @property
    def expired(self) -> bool:
        if not self.expires:
            return False
        try:
            when = datetime.fromisoformat(self.expires)
        except ValueError:
            return False
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        return when <= datetime.now(UTC)


def memory_root() -> Path:
    return agentcli_home() / "memory"


def project_key(cwd: str) -> str:
    """D:\\AgentCLI -> D-AgentCLI: readable, and unique per project folder."""

    resolved = str(Path(cwd).resolve())
    return re.sub(r"[^\w]+", "-", resolved).strip("-")[:120] or "root"


class FileMemory:
    def __init__(
        self,
        cwd: str,
        *,
        root: Path | None = None,
        max_entries: int = 1_000,
        max_chars: int = 8_000,
        legacy_db: str = "",
    ):
        self.cwd = str(Path(cwd).resolve())
        self.folder = (root or memory_root()) / project_key(cwd)
        self.max_entries = max(1, max_entries)
        self.max_chars = max(100, max_chars)
        if legacy_db and not self.folder.exists():
            self._migrate(Path(legacy_db))

    # ------------------------------------------------------------------ read

    def list(self) -> list[MemoryRecord]:
        """Live memories in index order; expired ones are deleted on the way."""

        records = []
        for path in self._files():
            record = _read(path)
            if record is None:
                continue
            if record.expired:
                path.unlink(missing_ok=True)
                continue
            records.append(record)
        records.sort(key=_index_order)
        return records

    def get(self, name: str) -> MemoryRecord | None:
        path = self.folder / f"{_clean_name(name)}.md"
        return _read(path) if path.exists() else None

    def search(
        self,
        query: str,
        limit: int = 8,
        *,
        kinds: list[str] | None = None,
        min_coverage: float = 0.0,
    ) -> list[Hit]:
        records = [r for r in self.list() if not kinds or r.kind in kinds]
        hits = [hit for hit in rank(query, records) if hit.coverage >= min_coverage]
        return hits[: max(0, limit)]

    def index_text(self, max_lines: int = 200, max_chars: int = 8_000) -> str:
        """The index as the model sees it: most important first when it must be cut."""

        records = self.list()
        chosen = sorted(records, key=lambda r: -r.importance)[:max_lines]
        chosen.sort(key=_index_order)
        lines = [_index_line(record) for record in chosen]
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars].rsplit("\n", 1)[0]
        shown = text.count("\n") + 1 if text else 0
        if shown < len(records):
            text += f"\n(+{len(records) - shown} more; use search_memory)"
        return text

    def stats(self) -> dict[str, Any]:
        records = self.list()
        return {
            "folder": str(self.folder),
            "memories": len(records),
            "by_kind": {kind: sum(r.kind == kind for r in records) for kind in KINDS},
        }

    # ------------------------------------------------------------------ write

    def save(
        self,
        content: str,
        *,
        title: str = "",
        name: str = "",
        kind: str = "fact",
        importance: float = 0.5,
        keywords: list[str] | tuple[str, ...] = (),
        expires: str = "",
        source: str = "agent",
    ) -> MemoryRecord:
        """Add a memory, or update the one with the same name, title, or content."""

        content = content.strip()
        if not content:
            raise ValueError("memory content is empty")
        if len(content) > self.max_chars:
            raise ValueError(f"memory content is longer than {self.max_chars} characters")
        kind = kind if kind in KINDS else "fact"
        importance = min(1.0, max(0.0, float(importance)))
        title = _one_line(title) or _one_line(content)[:80]
        today = date.today().isoformat()

        existing = self._find_same(name, title, content)
        if existing:
            record = existing
            record.title = title
            record.content = content
            record.kind = kind
            record.importance = max(record.importance, importance)
            record.keywords = _merge(record.keywords, keywords)
            record.expires = expires or record.expires
            record.updated = today
        else:
            record = MemoryRecord(
                name=self._free_name(name or title),
                title=title,
                content=content,
                kind=kind,
                importance=importance,
                keywords=_merge([], keywords),
                created=today,
                updated=today,
                expires=expires,
                source=source,
            )
        self.folder.mkdir(parents=True, exist_ok=True)
        record.path = self.folder / f"{record.name}.md"
        record.path.write_text(_render(record), encoding="utf-8")
        self._enforce_quota(keep=record.name)
        self._write_index()
        return record

    def delete(self, name: str) -> bool:
        path = self.folder / f"{_clean_name(name)}.md"
        if not path.exists():
            return False
        path.unlink()
        self._write_index()
        return True

    def clear(self) -> int:
        files = self._files()
        for path in files:
            path.unlink(missing_ok=True)
        self._write_index()
        return len(files)

    # ------------------------------------------------------------------ helpers

    def _files(self) -> list[Path]:
        if not self.folder.exists():
            return []
        return [p for p in self.folder.glob("*.md") if p.name != INDEX]

    def _find_same(self, name: str, title: str, content: str) -> MemoryRecord | None:
        wanted_name = _clean_name(name) if name else ""
        for record in self.list():
            if (
                (wanted_name and record.name == wanted_name)
                or record.title.casefold() == title.casefold()
                or _normal(record.content) == _normal(content)
            ):
                return record
        return None

    def _free_name(self, seed: str) -> str:
        base = _clean_name(seed)
        if not re.search(r"[a-z]", base):
            base = "memory-" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]
        candidate, counter = base, 2
        while (self.folder / f"{candidate}.md").exists():
            candidate, counter = f"{base}-{counter}", counter + 1
        return candidate

    def _enforce_quota(self, keep: str) -> None:
        records = self.list()
        excess = len(records) - self.max_entries
        if excess <= 0:
            return
        # Least important first, then the longest untouched.
        candidates = sorted(
            (r for r in records if r.name != keep), key=lambda r: (r.importance, r.updated)
        )
        for record in candidates[:excess]:
            if record.path:
                record.path.unlink(missing_ok=True)

    def _write_index(self) -> None:
        if not self.folder.exists():
            return
        lines = [
            "# Memory index",
            "",
            "Generated by AgentCLI from the files in this folder; edit those, not this.",
            "",
            *[_index_line(record) for record in self.list()],
        ]
        (self.folder / INDEX).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _migrate(self, legacy_db: Path) -> None:
        """Import this project's rows from the old SQLite store, once."""

        if not legacy_db.exists():
            return
        try:
            with sqlite3.connect(f"file:{legacy_db}?mode=ro", uri=True) as conn:
                rows = conn.execute(
                    "select content, kind, importance, expires_at from memories where scope = ?",
                    (self.cwd,),
                ).fetchall()
        except sqlite3.Error:
            return
        for content, kind, importance, expires in rows:
            try:
                self.save(
                    str(content),
                    kind=str(kind or "fact"),
                    importance=float(importance or 0.5),
                    expires=str(expires or ""),
                    source="migrated",
                )
            except ValueError:
                continue


def _index_order(record: MemoryRecord) -> tuple[int, float, str]:
    kind = KINDS.index(record.kind) if record.kind in KINDS else len(KINDS)
    return (kind, -record.importance, record.title)


def _index_line(record: MemoryRecord) -> str:
    return f"- [{record.kind}] {record.title} ({record.name}.md)"


def _render(record: MemoryRecord) -> str:
    keywords = ", ".join(k.replace(",", " ") for k in record.keywords)
    lines = [
        "---",
        f"title: {record.title}",
        f"kind: {record.kind}",
        f"importance: {record.importance:.2f}",
        f"keywords: [{keywords}]",
        f"created: {record.created}",
        f"updated: {record.updated}",
        f"expires: {record.expires}",
        f"source: {record.source}",
        "---",
        "",
        record.content,
        "",
    ]
    return "\n".join(lines)


def _read(path: Path) -> MemoryRecord | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.S)
    if not match:
        return MemoryRecord(name=path.stem, title=path.stem, content=text.strip(), path=path)
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip()
    try:
        importance = float(meta.get("importance") or 0.5)
    except ValueError:
        importance = 0.5
    raw_keywords = meta.get("keywords", "").strip().strip("[]")
    return MemoryRecord(
        name=path.stem,
        title=meta.get("title") or path.stem,
        content=match.group(2).strip(),
        kind=meta.get("kind") or "fact",
        importance=importance,
        keywords=[k.strip() for k in raw_keywords.split(",") if k.strip()],
        created=meta.get("created", ""),
        updated=meta.get("updated", ""),
        expires=meta.get("expires", ""),
        source=meta.get("source", "agent"),
        path=path,
    )


def _clean_name(value: str) -> str:
    ascii_words = re.findall(r"[a-z0-9]+", value.lower())
    return "-".join(ascii_words)[:60]


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def _normal(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _merge(old: list[str], new: list[str] | tuple[str, ...]) -> list[str]:
    merged: list[str] = []
    for keyword in [*old, *new]:
        keyword = _one_line(keyword)
        if keyword and keyword.casefold() not in {k.casefold() for k in merged}:
            merged.append(keyword)
    return merged[:20]
