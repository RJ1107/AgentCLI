from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentcli.paths import agentcli_home


@dataclass(slots=True)
class McpServerSpec:
    name: str
    type: str = "stdio"
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    timeout: float = 30.0
    # Whether this server's own readOnlyHint may skip approval. A server describes itself;
    # only a server the user vouches for should be believed.
    trusted: bool = False
    # Deferred tools cost only their names in each request until the model loads them.
    defer: bool = True
    # One line on what the server is for, shown in the load_tools index.
    description: str = ""


def load_mcp_server_specs(project_root: str | Path) -> dict[str, McpServerSpec]:
    root = Path(project_root).resolve()
    merged: dict[str, Any] = {}
    for path in [agentcli_home() / "mcp.json", root / ".agentcli" / "mcp.json"]:
        data = _read_json(path)
        if not data:
            continue
        servers = data.get("mcpServers", data)
        if isinstance(servers, dict):
            merged.update(servers)
    return {
        name: _spec_from_raw(name, raw, root)
        for name, raw in merged.items()
        if isinstance(raw, dict)
    }


# Pinned so the browser server only changes when AgentCLI is updated on purpose; "@latest"
# would run whatever npm serves that day.
CHROME_DEVTOOLS_MCP_VERSION = "1.10.1"


# The visible browser's profile: kept between sessions so logins survive, and separate from
# the user's own Chrome so the agent only ever holds the logins made here on purpose.
AGENT_BROWSER_PROFILE = "${AGENTCLI_HOME}/browser-profile"


def write_chrome_devtools_config(
    *,
    scope_root: str | Path | None = None,
    browser_url: str | None = None,
    headless: bool = True,
    slim: bool = False,
    no_usage_statistics: bool = True,
    isolated: bool = True,
    visible: bool = True,
) -> Path:
    """Write two browser servers, both deferred and started only when first used.

    - chrome-devtools: headless, blank throwaway profile; for public pages that only need
      JavaScript to render.
    - chrome-visible: a window the user can see and use, with a persistent AgentCLI-only
      profile; for login walls and bot checks, where the user signs in or completes the check
      themselves.
    """

    config_dir = Path(scope_root).resolve() / ".agentcli" if scope_root else agentcli_home()
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "mcp.json"
    data = _read_json(path) or {"mcpServers": {}}
    servers = data.setdefault("mcpServers", {})
    base = ["-y", f"chrome-devtools-mcp@{CHROME_DEVTOOLS_MCP_VERSION}"]
    if no_usage_statistics:
        base.extend(["--no-usage-statistics", "--no-performance-crux"])
    if slim:
        base.append("--slim")

    args = list(base)
    if headless:
        args.append("--headless")
    if browser_url:
        args.append(f"--browser-url={browser_url}")
    elif isolated:
        # A throwaway profile: this browser never sees any logins, cookies, or history.
        args.append("--isolated")
    servers["chrome-devtools"] = _browser_entry(
        args,
        "background browser, blank profile each session: public pages that need JavaScript",
    )
    if visible:
        servers["chrome-visible"] = _browser_entry(
            [*base, f"--userDataDir={AGENT_BROWSER_PROFILE}"],
            "visible browser with a saved AgentCLI-only profile: pages behind a login or bot "
            "check; the user signs in and completes any CAPTCHA themselves",
        )
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _browser_entry(args: list[str], description: str) -> dict[str, Any]:
    return {
        "type": "stdio",
        "command": "npx",
        "args": args,
        "description": description,
        # First start downloads the package and launches Chrome.
        "timeout": 120,
        # Official Google server: its own read-only hints (list_pages, wait_for) may skip
        # approval. Navigation, snapshots, clicks, and scripts still ask.
        "trusted": True,
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _spec_from_raw(name: str, raw: dict[str, Any], project_root: Path) -> McpServerSpec:
    default_type = "streamable_http" if raw.get("url") else "stdio"
    server_type = str(raw.get("type") or raw.get("transport") or default_type)
    env = {
        key: _expand(str(value), project_root) for key, value in dict(raw.get("env") or {}).items()
    }
    args = [_expand(str(arg), project_root) for arg in raw.get("args") or []]
    cwd = raw.get("cwd")
    return McpServerSpec(
        name=name,
        type=server_type,
        command=_expand(str(raw["command"]), project_root) if raw.get("command") else None,
        args=args,
        env=env,
        cwd=_expand(str(cwd), project_root) if cwd else None,
        url=_expand(str(raw["url"]), project_root) if raw.get("url") else None,
        headers={
            key: _expand(str(value), project_root)
            for key, value in dict(raw.get("headers") or {}).items()
        },
        enabled=bool(raw.get("enabled", True)),
        timeout=float(raw.get("timeout", raw.get("startup_timeout", 30.0)) or 30.0),
        trusted=bool(raw.get("trusted", False)),
        defer=bool(raw.get("defer", True)),
        description=str(raw.get("description") or ""),
    )


def _expand(value: str, project_root: Path) -> str:
    replacements = {
        "PROJECT_DIR": str(project_root),
        "HOME": str(Path.home()),
        "AGENTCLI_HOME": str(agentcli_home()),
    }

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        return replacements.get(name, os.environ.get(name, ""))

    return re.sub(r"\$\{([^}]+)\}", replace, value)
