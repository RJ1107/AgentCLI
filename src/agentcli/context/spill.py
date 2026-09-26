"""Full copies of tool results that compaction clears from the conversation.

Clearing an old tool result replaces it with a one-line stub. For a file read the agent can
simply read the file again, but a command's output, a fetched page, or a browser snapshot
cannot be reproduced later, so the full text is written here first and the stub points to
it. Files live under AGENTCLI_HOME (so on whatever drive that is) and are deleted after
`retention_days`.
"""

from __future__ import annotations

import os
import re
import shutil
import time
import uuid
from pathlib import Path

from agentcli.paths import agentcli_home

# One folder per process: a REPL session, a single prompt, or a runtime worker.
_SESSION = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
_cleaned = False


def saved_results_root() -> Path:
    return agentcli_home() / "sessions"


class ToolResultStore:
    def __init__(self, root: Path | None = None, *, retention_days: float = 7.0):
        self.root = root or saved_results_root()
        self.retention_days = retention_days
        self.folder = self.root / _SESSION / "tool-results"

    def save(self, call_id: str, content: str) -> Path | None:
        """Write the full result; None if it could not be written (the stub then says so)."""

        self._clean_once()
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", call_id or uuid.uuid4().hex)[:80]
        path = self.folder / f"{safe}.txt"
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text(content, encoding="utf-8")
        except OSError:
            return None
        return path

    def _clean_once(self) -> None:
        """Delete session folders older than the retention period, once per process."""

        global _cleaned
        if _cleaned or self.retention_days <= 0:
            return
        _cleaned = True
        cutoff = time.time() - self.retention_days * 86_400
        try:
            sessions = list(self.root.iterdir())
        except OSError:
            return
        for session in sessions:
            try:
                if session.is_dir() and session.stat().st_mtime < cutoff:
                    shutil.rmtree(session, ignore_errors=True)
            except OSError:
                continue
