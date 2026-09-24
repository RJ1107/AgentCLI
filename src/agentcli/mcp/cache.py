from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from agentcli.mcp.config import McpServerSpec
from agentcli.paths import agentcli_home

CACHE_TTL_SECONDS = 7 * 24 * 3600


class ToolListCache:
    """Remember each MCP server's tool list on disk so startup does not launch servers.

    The key covers everything that decides which program runs (transport, command, args, URL,
    working directory, environment), so editing a server's config, including bumping a pinned
    package version, misses the cache and the list is fetched again. Values of env vars only
    feed the hash; they are never written to the file.
    """

    def __init__(self, root: Path | None = None):
        self._root = root

    @property
    def root(self) -> Path:
        # Resolved late so a changed home directory (tests, service accounts) is respected.
        return self._root or agentcli_home() / "mcp-cache"

    def get(self, spec: McpServerSpec) -> list[dict[str, Any]] | None:
        path = self._path(spec)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if time.time() - float(data.get("cached_at") or 0) > CACHE_TTL_SECONDS:
            return None
        tools = data.get("tools")
        return tools if isinstance(tools, list) else None

    def put(self, spec: McpServerSpec, tools: list[dict[str, Any]]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for stale in self.root.glob(f"{_safe(spec.name)}-*.json"):
            stale.unlink(missing_ok=True)
        path = self._path(spec)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"cached_at": time.time(), "tools": tools}, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(path)

    def clear(self) -> int:
        if not self.root.exists():
            return 0
        removed = 0
        for path in self.root.glob("*.json"):
            path.unlink(missing_ok=True)
            removed += 1
        return removed

    def _path(self, spec: McpServerSpec) -> Path:
        identity = {
            "type": spec.type,
            "command": spec.command,
            "args": spec.args,
            "url": spec.url,
            "cwd": spec.cwd,
            "env": spec.env,
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16]
        return self.root / f"{_safe(spec.name)}-{digest}.json"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name) or "server"
