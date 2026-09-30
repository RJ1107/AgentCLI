"""Workspace snapshots, kept in a shadow git repository outside the project.

A snapshot used to be a full copy of the project. That is simple and restores anything,
but it costs the project's size every time: 53 s and 165 MB for this repository, whose
gitignored eval scratch was copied too, and on Windows a path over 260 characters (an npm
cache) made the copy fail, silently, so that request had no snapshot at all.

Now each project gets a git repository of its own under AGENTCLI_HOME/snapshots/<id>/,
with the project as its work tree. The project's own .git is never touched and the project
need not be a git repository. A snapshot is `git add -A`, `git write-tree`, and a
parentless `git commit-tree`, named by a ref. Git stores each file once by the hash of its
content, so a snapshot adds only the files that changed since any earlier one, and `add`
skips unchanged files by their size and modification time without reading them. The
project's .gitignore applies, plus the always-skipped folders below. Line endings and
filters are switched off in the shadow repository, so a restore gives back the exact
bytes. When git is not installed, the old full copy is used.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from agentcli.paths import agentcli_home

SKIP_DIRS = {".git", ".venv", "node_modules", "dist", "build", "target", "__pycache__"}
# Past KEEP_SNAPSHOTS + TRIM_BATCH, the oldest are dropped down to KEEP_SNAPSHOTS and git gc
# prunes the content no remaining snapshot shares; in batches, so gc does not run every time.
KEEP_SNAPSHOTS = 100
TRIM_BATCH = 20
_GIT_ENV = {
    "GIT_AUTHOR_NAME": "AgentCLI",
    "GIT_AUTHOR_EMAIL": "snapshots@agentcli.local",
    "GIT_COMMITTER_NAME": "AgentCLI",
    "GIT_COMMITTER_EMAIL": "snapshots@agentcli.local",
    # The user's global git config must not change what a snapshot stores.
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_TERMINAL_PROMPT": "0",
}


class SnapshotError(RuntimeError):
    pass


@dataclass(slots=True)
class SnapshotRecord:
    id: str
    phase: str
    created_at: str
    # The commit in the shadow repository; a folder for snapshots from the old full copy.
    path: Path
    commit: str = ""


class SnapshotService:
    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()
        digest = hashlib.sha256(str(self.project_root).encode("utf-8")).hexdigest()[:16]
        self.root = agentcli_home() / "snapshots" / digest
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"
        self.git_dir = self.root / "repo.git"
        self.use_git = shutil.which("git") is not None

    # ------------------------------------------------------------------ public

    def create(self, phase: str) -> SnapshotRecord:
        snapshot_id = f"{phase}_{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}"
        created_at = datetime.now(UTC).isoformat()
        if self.use_git:
            commit = self._git_snapshot(snapshot_id)
            record = SnapshotRecord(snapshot_id, phase, created_at, self.git_dir, commit)
        else:
            target = self.root / snapshot_id
            target.mkdir(parents=True, exist_ok=True)
            self._copy_tree(self.project_root, target)
            record = SnapshotRecord(snapshot_id, phase, created_at, target)
        with self.index_path.open("a", encoding="utf-8") as handle:
            entry = {"id": record.id, "phase": phase, "created_at": created_at,
                     "path": str(record.path), "commit": record.commit}
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._trim()
        return record

    def list(self, limit: int = 20) -> list[SnapshotRecord]:
        return self._records()[-limit:][::-1]

    def read(self, record: SnapshotRecord, relative: str) -> bytes | None:
        """One file's content as the snapshot holds it, or None when it did not exist."""

        if record.commit:
            try:
                return self._git("show", f"{record.commit}:{Path(relative).as_posix()}").stdout
            except SnapshotError:
                return None
        path = record.path / relative
        return path.read_bytes() if path.is_file() else None

    def restore(self, snapshot_ref: str) -> SnapshotRecord:
        records = self.list(limit=10_000)
        record = None
        if snapshot_ref.isdigit():
            index = int(snapshot_ref) - 1
            if 0 <= index < len(records):
                record = records[index]
        else:
            record = next((item for item in records if item.id == snapshot_ref), None)
        if not record:
            raise ValueError(f"snapshot not found: {snapshot_ref}")
        current = self.create("pre-restore")
        if record.commit:
            self._git_restore(record.commit, current.commit)
        else:
            self._restore_tree(record.path, self.project_root)
        return record

    def clean(self) -> int:
        count = len(self._records())
        if self.root.exists():
            _remove_tree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        return count

    # ------------------------------------------------------------------ git

    def _git(self, *args: str, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
        command = [
            "git",
            f"--git-dir={self.git_dir}",
            f"--work-tree={self.project_root}",
            *args,
        ]
        done = subprocess.run(
            command,
            input=input_bytes,
            capture_output=True,
            env={**os.environ, **_GIT_ENV},
            cwd=self.project_root,
            timeout=300,
        )
        if done.returncode != 0:
            message = done.stderr.decode("utf-8", "replace").strip()[:400]
            raise SnapshotError(f"git {args[0]} failed: {message}")
        return done

    def _ensure_repo(self) -> None:
        if (self.git_dir / "HEAD").exists():
            return
        subprocess.run(
            ["git", "init", "--bare", "--quiet", str(self.git_dir)],
            check=True,
            capture_output=True,
            env={**os.environ, **_GIT_ENV},
        )
        settings = {
            "core.bare": "false",
            "core.autocrlf": "false",  # store and restore the bytes as they are
            "core.safecrlf": "false",
            "core.longpaths": "true",  # Windows paths over 260 characters
            "core.fsmonitor": "false",
            "core.quotepath": "false",
            "commit.gpgsign": "false",  # internal objects, never pushed anywhere
            "gc.auto": "0",  # pruning is done after trimming old snapshots
        }
        for key, value in settings.items():
            self._git("config", key, value)
        info = self.git_dir / "info"
        info.mkdir(parents=True, exist_ok=True)
        # Beats the project's .gitattributes: no end-of-line or LFS conversion.
        (info / "attributes").write_text("* -text -filter -ident -working-tree-encoding\n",
                                         encoding="utf-8")
        excludes = [f"{name}/" for name in sorted(SKIP_DIRS)]
        try:
            # The snapshot store itself, when AGENTCLI_HOME lies inside the project.
            inside = self.root.resolve().relative_to(self.project_root)
            excludes.append("/" + inside.parts[0] + "/")
        except ValueError:
            pass
        (info / "exclude").write_text("\n".join(excludes) + "\n", encoding="utf-8")

    def _git_snapshot(self, snapshot_id: str) -> str:
        self._ensure_repo()
        # A nested repository is recorded as a link to its commit, not as its files.
        self._git("add", "--all", "--ignore-errors", "--", ".")
        tree = self._git("write-tree").stdout.decode().strip()
        commit = self._git("commit-tree", tree, "-m", snapshot_id).stdout.decode().strip()
        self._git("update-ref", f"refs/snapshots/{snapshot_id}", commit)
        return commit

    def _git_restore(self, commit: str, current: str) -> None:
        # Files the project gained since the snapshot: remove them.
        added = self._git("diff", "--name-only", "--no-renames", "--diff-filter=A", "-z",
                          commit, current).stdout.decode("utf-8").split("\0")
        for name in filter(None, added):
            path = self.project_root / name
            if path.is_file() or path.is_symlink():
                path.unlink()
        # Everything the snapshot held, byte for byte.
        self._git("read-tree", commit)
        self._git("checkout-index", "--all", "--force")

    def _trim(self) -> None:
        records = self._records()
        if len(records) <= KEEP_SNAPSHOTS + TRIM_BATCH:
            return
        dropped, kept = records[:-KEEP_SNAPSHOTS], records[-KEEP_SNAPSHOTS:]
        for record in dropped:
            if record.commit:
                try:
                    self._git("update-ref", "-d", f"refs/snapshots/{record.id}")
                except SnapshotError:
                    pass
            elif record.path.is_dir():
                shutil.rmtree(record.path, ignore_errors=True)
        with self.index_path.open("w", encoding="utf-8") as handle:
            for record in kept:
                entry = {"id": record.id, "phase": record.phase, "created_at": record.created_at,
                         "path": str(record.path), "commit": record.commit}
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if self.use_git and (self.git_dir / "HEAD").exists():
            try:
                self._git("gc", "--quiet", "--prune=now")
            except SnapshotError:
                pass

    def _records(self) -> list[SnapshotRecord]:
        if not self.index_path.exists():
            return []
        records = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(
                SnapshotRecord(
                    id=item["id"],
                    phase=item["phase"],
                    created_at=item["created_at"],
                    path=Path(item["path"]),
                    commit=item.get("commit", ""),
                )
            )
        return records

    # ------------------------------------------------------------------ full copy (no git)

    def _copy_tree(self, source: Path, target: Path) -> None:
        for item in source.iterdir():
            if _skip(item) or self._holds_store(item):
                continue
            destination = target / item.name
            if item.is_dir():
                shutil.copytree(item, destination, ignore=_ignore)
            elif item.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, destination)

    def _restore_tree(self, source: Path, target: Path) -> None:
        for item in target.iterdir():
            if _skip(item) or self._holds_store(item):
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        self._copy_tree(source, target)

    def _holds_store(self, item: Path) -> bool:
        # When the project contains the snapshot store (e.g. the home directory is the
        # project), copying it would copy snapshots into themselves and restoring would
        # delete them.
        return self.root == item.resolve() or item.resolve() in self.root.parents


class TurnSnapshot:
    """At most one snapshot per user request, taken right before its first write.

    A request that only reads (read_file, grep, ...) never pays for a snapshot. When a write
    does come, the snapshot still shows the workspace exactly as it was when the request
    started, because nothing before that point could have changed it. No snapshot is taken
    after the request either: /restore records a "pre-restore" snapshot of the current state
    first, so undoing a restore stays possible without it.
    """

    def __init__(self, project_root: str | Path):
        self.project_root = project_root
        self.record: SnapshotRecord | None = None
        # Why the snapshot could not be taken; the edit still goes ahead.
        self.error: str | None = None
        self._attempted = False

    def before_write(self) -> None:
        # Synchronous on purpose: in one event loop no other coroutine can run between the
        # check and the copy, so parallel workers cannot take two snapshots for one request.
        if self._attempted:
            return
        self._attempted = True
        try:
            self.record = SnapshotService(self.project_root).create("pre-write")
        except Exception as exc:  # noqa: BLE001 - a failed snapshot must not block the edit
            self.record = None
            self.error = f"{type(exc).__name__}: {exc}"[:300]


def _skip(path: Path) -> bool:
    return path.name in SKIP_DIRS


def _ignore(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in SKIP_DIRS}


def _remove_tree(path: Path) -> None:
    # Git marks its objects read-only; Windows refuses to delete them until that is undone.
    def retry(func, target, *_exc) -> None:
        os.chmod(target, 0o700)
        func(target)

    try:
        shutil.rmtree(path, onexc=retry)  # Python 3.12+
    except TypeError:
        shutil.rmtree(path, onerror=retry)
