"""Run the agent's shell commands in a Docker container instead of on the host.

One container per project per AgentCLI process, started on the first command and removed
when AgentCLI exits (or, if it is killed, when the container's own timer runs out). The
container sees only the project folder, mounted at /workspace; it has no network, none of
the host's environment variables (so no API keys), no Linux capabilities, a read-only
system, and caps on memory, CPU, and processes. A command that needs the network runs in
a fresh container of the same shape with the network on, after the user approves it.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

IMAGE_VERSION = "1"
DEFAULT_IMAGE = f"agentcli-sandbox:{IMAGE_VERSION}"
DOCKERFILE_DIR = Path(__file__).resolve().parent
WORKDIR = "/workspace"
TIMEOUT_EXIT = 137  # what `timeout -s KILL` leaves behind


@dataclass(slots=True, frozen=True)
class SandboxOptions:
    image: str = DEFAULT_IMAGE
    memory: str = "2g"
    cpus: float = 2.0
    pids: int = 256
    # The container removes itself after this long even if AgentCLI never gets to.
    max_hours: float = 12.0


def volume_name(workspace: str) -> str:
    digest = hashlib.sha1(str(Path(workspace).resolve()).encode("utf-8")).hexdigest()[:12]
    return f"agentcli-venv-{digest}"


def container_args(
    workspace: str,
    options: SandboxOptions,
    *,
    network: bool,
    name: str = "",
    command: list[str] | None = None,
) -> list[str]:
    """`docker run` arguments for a sandbox container (without the leading "docker")."""

    args = ["run", "--rm", "--label", "org.agentcli.sandbox=1"]
    if name:
        args += ["--name", name, "-d"]
    args += [
        "--network",
        "bridge" if network else "none",
        "--mount",
        f"type=bind,source={Path(workspace).resolve()},target={WORKDIR}",
        "--mount",
        f"type=volume,source={volume_name(workspace)},target=/opt/venv",
        "-w",
        WORKDIR,
        "--read-only",
        "--tmpfs",
        "/tmp:rw,exec,size=1g",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--memory",
        options.memory,
        "--cpus",
        f"{options.cpus:g}",
        "--pids-limit",
        str(options.pids),
    ]
    if hasattr(os, "getuid"):
        # On Linux and macOS hosts, files the command creates belong to the user, not root.
        args += ["--user", f"{os.getuid()}:{os.getgid()}"]
    args.append(options.image)
    args += command or ["sleep", str(int(options.max_hours * 3600))]
    return args


def _timed(command: str, timeout: float) -> list[str]:
    # Enforced inside the container: killing the docker client would leave it running.
    return ["timeout", "-s", "KILL", str(max(1, int(timeout))), "sh", "-c", command]


class DockerSandbox:
    def __init__(self, workspace: str, options: SandboxOptions, docker: str = "docker"):
        self.workspace = str(Path(workspace).resolve())
        self.options = options
        self.docker = docker
        self.name = f"agentcli-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self._started = False
        self._lock = asyncio.Lock()

    async def run(self, command: str, timeout: float, *, network: bool = False) -> tuple[int, str]:
        """Run a shell command; returns (exit code, combined output)."""

        if network:
            args = container_args(
                self.workspace, self.options, network=True, command=_timed(command, timeout)
            )
        else:
            await self._ensure_started()
            args = ["exec", "-w", WORKDIR, self.name, *_timed(command, timeout)]
        code, output = await _docker(self.docker, args, timeout + 30)
        if code == TIMEOUT_EXIT:
            return code, output + f"\n[command killed after {timeout:.0f}s]"
        return code, output

    async def _ensure_started(self) -> None:
        async with self._lock:
            if self._started:
                return
            args = container_args(self.workspace, self.options, network=False, name=self.name)
            code, output = await _docker(self.docker, args, 120)
            if code != 0:
                raise RuntimeError(f"could not start the sandbox container: {output.strip()}")
            self._started = True

    async def close(self) -> None:
        if self._started:
            self._started = False
            await _docker(self.docker, ["rm", "-f", self.name], 30)


async def _docker(docker: str, args: list[str], timeout: float) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        docker, *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return TIMEOUT_EXIT, "[docker did not respond in time]"
    return proc.returncode or 0, stdout.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------- availability

_checks: dict[str, bool] = {}


def docker_available(docker: str = "docker") -> bool:
    """A Docker engine that runs Linux containers is reachable (checked once per process)."""

    if docker not in _checks:
        try:
            result = subprocess.run(
                [docker, "info", "--format", "{{.OSType}}"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            _checks[docker] = result.returncode == 0 and result.stdout.strip() == "linux"
        except (OSError, subprocess.SubprocessError):
            _checks[docker] = False
    return _checks[docker]


def image_exists(image: str, docker: str = "docker") -> bool:
    try:
        result = subprocess.run(
            [docker, "image", "inspect", image], capture_output=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def build_image(image: str, docker: str = "docker") -> tuple[bool, str]:
    """Build the sandbox image from the bundled Dockerfile (downloads the base image)."""

    try:
        result = subprocess.run(
            [docker, "build", "-t", image, str(DOCKERFILE_DIR)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    return result.returncode == 0, (result.stdout + result.stderr)[-2000:]
