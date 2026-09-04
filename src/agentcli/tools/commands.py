from __future__ import annotations

import asyncio
import os
import re
import shlex
from dataclasses import dataclass

from agentcli.policy import CommandGuard

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

MAX_OUTPUT_CHARS = 20_000
TRUNCATION_SUFFIX = "\n... [output truncated]"


@dataclass(slots=True)
class CommandResult:
    """Structured result of a command execution."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False
    truncated: bool = False

    @property
    def combined(self) -> str:
        return self.stdout + self.stderr

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    @property
    def summary(self) -> str:
        if self.timed_out:
            return "(timed out)"
        return f"(exit {self.exit_code}, no output)" if not self.combined.strip() else ""


# ---------------------------------------------------------------------------
# Sensitive-command heuristics for HITL prompts
# ---------------------------------------------------------------------------

# Patterns that indicate a command modifies the system or has side effects
_WRITE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(rm|mv|cp|chmod|chown|ln)\b"),
    re.compile(r"\b(mkdir|touch|truncate)\b"),
    re.compile(r"\b(wget|curl)\s+.*[-]O\b"),
    re.compile(r"\b(git\s+push|git\s+commit|git\s+reset|git\s+rebase|git\s+merge)\b"),
    re.compile(
        r"\b(pip|npm|brew|apt|yum|dnf|pacman)\s+(install|uninstall|remove|update|upgrade)\b"
    ),
    re.compile(r"\b(kubectl|helm|docker|podman)\s+(apply|create|delete|run|exec|port-forward)\b"),
    re.compile(r"\b(>|>>)\s*[^\s]"),
    re.compile(r"\|.*\b(sh|bash)\b"),
]

_DESTRUCTIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\brm\s+-[rf]+\s"),
    re.compile(r"\bdd\s+if="),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bshutdown\b"),
    re.compile(r"\breboot\b"),
    re.compile(r"\bchmod\s+-R\s+777\s+/"),
    re.compile(r":\(\)\s*\{"),
    re.compile(r"\bfind\s+/\s"),
]


def classify_command(command: str) -> str:
    """Classify a command into a danger level string for HITL display."""
    norm = command.strip()
    if not norm:
        return "safe"

    for pattern in _DESTRUCTIVE_PATTERNS:
        if pattern.search(norm):
            return "high"

    for pattern in _WRITE_PATTERNS:
        if pattern.search(norm):
            return "medium"

    return "safe"


def sensitive_command_summary(command: str) -> str | None:
    """Return a human-readable warning if the command looks sensitive."""
    level = classify_command(command)
    if level == "high":
        return "⚠️  Destructive command — may permanently modify or damage the system."
    if level == "medium":
        return "⚡ Command has write or side-effect potential."
    return None


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

_ENV_KEY_BLOCKLIST: tuple[str, ...] = (
    "token",
    "secret",
    "password",
    "passwd",
    "authorization",
    "bearer",
    "access_key",
    "secret_key",
    "private_key",
)


def _sanitize_env(env: dict[str, str]) -> dict[str, str]:
    """Return a copy of *env* with sensitive keys masked for logging."""
    return {
        k: ("***" if any(marker in k.lower() for marker in _ENV_KEY_BLOCKLIST) else v)
        for k, v in env.items()
    }


# ---------------------------------------------------------------------------
# CommandExecutor
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CommandExecutionOptions:
    """Fine-grained options for a single command execution."""

    cwd: str | None = None
    timeout: float = 60.0
    env: dict[str, str] | None = None
    max_output_chars: int = MAX_OUTPUT_CHARS
    suppress_shell: bool = False  # When true, use Popen with list-form instead of shell


class CommandExecutor:
    """Encapsulates subprocess execution with security, timeout, and output handling.

    Usage::

        executor = CommandExecutor(command_guard=CommandGuard())
        result = await executor.execute("ls -la", cwd="/workspace")
    """

    def __init__(
        self,
        command_guard: CommandGuard | None = None,
        default_timeout: float = 60.0,
        default_max_output: int = MAX_OUTPUT_CHARS,
    ) -> None:
        self.command_guard = command_guard or CommandGuard()
        self.default_timeout = default_timeout
        self.default_max_output = default_max_output

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        max_output_chars: int | None = None,
    ) -> CommandResult:
        """Execute *command* via a shell subprocess.

        Parameters
        ----------
        command:
            Shell command string.
        cwd:
            Working directory (default: CWD of current process).
        timeout:
            Maximum seconds the command is allowed to run.
        env:
            Extra or override environment variables.
        max_output_chars:
            Maximum characters kept from combined stdout+stderr.

        Raises
        ------
        CommandPolicyError
            If the command is blocked by the command guard policy.
        """
        self._validate(command)

        resolved_cwd = cwd or os.getcwd()
        resolved_timeout = timeout if timeout is not None else self.default_timeout
        resolved_max_output = (
            max_output_chars if max_output_chars is not None else self.default_max_output
        )

        merged_env = self._build_env(env)

        return await self._run_subprocess(
            command=command,
            cwd=resolved_cwd,
            timeout=resolved_timeout,
            env=merged_env,
            max_output_chars=resolved_max_output,
        )

    async def execute_list(
        self,
        cmd_parts: list[str],
        *,
        cwd: str | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        max_output_chars: int | None = None,
    ) -> CommandResult:
        """Execute a command as a list (bypasses shell). Safer for subprocess calls.

        Parameters
        ----------
        cmd_parts:
            Command as a list of arguments (e.g. ``["ls", "-la"]``).
        """
        command_str = _list_to_shell(cmd_parts)
        self._validate(command_str)

        resolved_cwd = cwd or os.getcwd()
        resolved_timeout = timeout if timeout is not None else self.default_timeout
        resolved_max_output = (
            max_output_chars if max_output_chars is not None else self.default_max_output
        )
        merged_env = self._build_env(env)

        return await self._run_subprocess_list(
            cmd_parts=cmd_parts,
            cwd=resolved_cwd,
            timeout=resolved_timeout,
            env=merged_env,
            max_output_chars=resolved_max_output,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, command: str) -> None:
        if self.command_guard:
            self.command_guard.validate(command)

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    @staticmethod
    def _build_env(overrides: dict[str, str] | None) -> dict[str, str]:
        env = os.environ.copy()
        if overrides:
            env.update(overrides)
        return env

    # ------------------------------------------------------------------
    # Subprocess runners
    # ------------------------------------------------------------------

    async def _run_subprocess(
        self,
        command: str,
        cwd: str,
        timeout: float,
        env: dict[str, str],
        max_output_chars: int,
    ) -> CommandResult:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        return await self._handle_process(proc, command, timeout, max_output_chars)

    async def _run_subprocess_list(
        self,
        cmd_parts: list[str],
        cwd: str,
        timeout: float,
        env: dict[str, str],
        max_output_chars: int,
    ) -> CommandResult:
        proc = await asyncio.create_subprocess_exec(
            *cmd_parts,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        return await self._handle_process(
            proc,
            _list_to_shell(cmd_parts),
            timeout,
            max_output_chars,
        )

    async def _handle_process(
        self,
        proc: asyncio.subprocess.Process,
        command: str,
        timeout: float,
        max_output_chars: int,
    ) -> CommandResult:
        timed_out = False
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            timed_out = True
            stdout, stderr = b"", b""

        stdout_str = _decode(stdout)
        stderr_str = _decode(stderr)

        truncated = False
        combined_len = len(stdout_str) + len(stderr_str)
        if combined_len > max_output_chars:
            truncated = True
            # Prefer to keep stdout over stderr
            avail = max_output_chars - len(TRUNCATION_SUFFIX)
            if len(stdout_str) >= avail:
                stdout_str = stdout_str[:avail] + TRUNCATION_SUFFIX
                stderr_str = ""
            else:
                stderr_str = stderr_str[: avail - len(stdout_str)] + TRUNCATION_SUFFIX

        return CommandResult(
            stdout=stdout_str,
            stderr=stderr_str,
            exit_code=-1 if timed_out else (proc.returncode or 0),
            timed_out=timed_out,
            truncated=truncated,
        )


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _decode(data: bytes) -> str:
    """Decode bytes with UTF-8 fallback."""
    return data.decode("utf-8", errors="replace")


def _list_to_shell(cmd_parts: list[str]) -> str:
    """Convert a command list to a human-readable shell string (for logging/validation)."""
    return " ".join(shlex.quote(part) for part in cmd_parts)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_executor(
    blacklist: list[str] | None = None,
    default_timeout: float = 60.0,
) -> CommandExecutor:
    """Build a CommandExecutor wired to a CommandGuard with the given blacklist."""
    return CommandExecutor(
        command_guard=CommandGuard(blacklist=blacklist or []),
        default_timeout=default_timeout,
    )
