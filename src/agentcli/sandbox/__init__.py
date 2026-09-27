"""Where the agent's shell commands run: a Docker sandbox, or the host.

sandbox_for() gives the sandbox for a project, or None when commands run on the host
(sandbox off, or Docker or the image not available). Callers treat None as "ask the user
before every command", which is how AgentCLI worked before the sandbox.
"""

from __future__ import annotations

from pathlib import Path

from agentcli.config import AgentCliConfig
from agentcli.sandbox.docker import (
    DEFAULT_IMAGE,
    DockerSandbox,
    SandboxOptions,
    build_image,
    docker_available,
    image_exists,
)

_sandboxes: dict[str, DockerSandbox] = {}
_images_ready: dict[str, bool] = {}


def options_from(config: AgentCliConfig) -> SandboxOptions:
    sandbox = config.sandbox
    return SandboxOptions(
        image=sandbox.image or DEFAULT_IMAGE,
        memory=sandbox.memory,
        cpus=sandbox.cpus,
        pids=sandbox.pids,
        max_hours=sandbox.max_hours,
    )


def sandbox_status(config: AgentCliConfig) -> tuple[bool, str]:
    """(commands will run in the sandbox, a line saying so or why not)."""

    if config.sandbox.mode != "docker":
        return False, "沙箱已关闭（sandbox.mode = off）：命令在本机执行，逐条审批。"
    if not docker_available():
        return (
            False,
            "没有检测到 Docker：命令在本机执行，逐条审批。安装并启动 Docker Desktop 即可启用沙箱。",
        )
    image = options_from(config).image
    if image not in _images_ready:
        _images_ready[image] = image_exists(image)
    if not _images_ready[image]:
        return False, f"沙箱镜像 {image} 还没有构建：命令在本机执行，逐条审批。"
    return True, "沙箱：Docker 容器，只能看到本项目，默认断网，命令无需逐条审批。"


def prepare_image(config: AgentCliConfig) -> tuple[bool, str]:
    """Build the image if it is missing. Returns (ready, build log on failure)."""

    image = options_from(config).image
    if image_exists(image):
        _images_ready[image] = True
        return True, ""
    ok, log = build_image(image)
    _images_ready[image] = ok
    return ok, log


def sandbox_for(cwd: str, config: AgentCliConfig) -> DockerSandbox | None:
    active, _ = sandbox_status(config)
    if not active:
        return None
    key = str(Path(cwd).resolve())
    if key not in _sandboxes:
        _sandboxes[key] = DockerSandbox(key, options_from(config))
    return _sandboxes[key]


async def close_all() -> None:
    sandboxes = list(_sandboxes.values())
    _sandboxes.clear()
    for sandbox in sandboxes:
        await sandbox.close()


__all__ = [
    "DockerSandbox",
    "SandboxOptions",
    "close_all",
    "prepare_image",
    "sandbox_for",
    "sandbox_status",
]
