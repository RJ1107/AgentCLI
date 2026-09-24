from __future__ import annotations

import os
from pathlib import Path


def agentcli_home() -> Path:
    """Where AgentCLI keeps its own data: config, memory, snapshots, skills, caches, logs.

    AGENTCLI_HOME moves all of it at once (for example to another drive); unset, it is
    ~/.agentcli. Read on every call, so a changed environment (tests, services) is respected.
    """

    override = os.environ.get("AGENTCLI_HOME", "").strip()
    return Path(override).expanduser() if override else Path.home() / ".agentcli"
