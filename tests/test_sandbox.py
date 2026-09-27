from __future__ import annotations

import asyncio

import pytest

from agentcli.config import load_config
from agentcli.sandbox import docker as docker_module
from agentcli.sandbox.docker import (
    DEFAULT_IMAGE,
    DockerSandbox,
    SandboxOptions,
    container_args,
    docker_available,
    image_exists,
)
from agentcli.tools.base import ToolContext
from agentcli.tools.builtins import _bash_needs_approval


def test_container_sees_only_the_project_without_network_or_privileges(tmp_path):
    args = container_args(str(tmp_path), SandboxOptions(), network=False, name="box")

    joined = " ".join(args)
    assert "--network none" in joined
    assert f"type=bind,source={tmp_path.resolve()},target=/workspace" in joined
    assert "--read-only" in args and "--cap-drop" in args and "ALL" in args
    assert "no-new-privileges" in args
    assert "--memory 2g" in joined and "--pids-limit 256" in joined
    assert "-e" not in args  # nothing from the host environment, so no API keys
    assert args[-2:] == ["sleep", str(12 * 3600)]  # removes itself if AgentCLI never does


def test_network_is_only_on_for_a_command_that_asks(tmp_path):
    args = container_args(str(tmp_path), SandboxOptions(), network=True, command=["true"])
    assert "--network bridge" in " ".join(args)
    assert "-d" not in args  # a one-off container, gone when the command ends


def test_approval_inside_the_sandbox_only_for_network_or_host(tmp_path, monkeypatch):
    context = ToolContext(cwd=str(tmp_path), config=load_config(project_root=tmp_path))
    monkeypatch.setattr("agentcli.sandbox.sandbox_for", lambda cwd, config: None)
    assert _bash_needs_approval({"command": "ls"}, context)

    monkeypatch.setattr("agentcli.sandbox.sandbox_for", lambda cwd, config: object())
    assert not _bash_needs_approval({"command": "ls"}, context)
    assert _bash_needs_approval({"command": "pip install x", "network": True}, context)
    assert _bash_needs_approval({"command": "ls", "sandbox": False}, context)


def test_without_docker_commands_fall_back_to_the_host(tmp_path, monkeypatch):
    config = load_config(project_root=tmp_path)
    config.sandbox.mode = "docker"
    monkeypatch.setattr(docker_module, "_checks", {"docker": False})
    from agentcli.sandbox import sandbox_for, sandbox_status

    assert sandbox_for(str(tmp_path), config) is None
    assert "没有检测到 Docker" in sandbox_status(config)[1]


needs_docker = pytest.mark.skipif(
    not (docker_available() and image_exists(DEFAULT_IMAGE)),
    reason="needs Docker and the agentcli-sandbox image",
)


@needs_docker
def test_real_sandbox_isolates_the_command(tmp_path, monkeypatch):
    # tmp_path is under the user folder, which may contain non-ASCII characters.
    (tmp_path / "in.txt").write_text("from host", encoding="utf-8")
    monkeypatch.setenv("FAKE_SECRET", "sk-should-not-leak")
    sandbox = DockerSandbox(str(tmp_path), SandboxOptions())

    async def scenario():
        try:
            read = await sandbox.run("cat in.txt && echo made > out.txt", 30)
            env = await sandbox.run('echo "secret=[$FAKE_SECRET]"', 30)
            net = await sandbox.run(
                "python -c \"import socket; socket.create_connection(('1.1.1.1', 53), 3)\"", 30
            )
            system = await sandbox.run("touch /usr/evil", 30)
            outside = await sandbox.run("ls /home /root 2>&1; ls / ", 30)
            slow = await sandbox.run("sleep 20", 2)
            return read, env, net, system, outside, slow
        finally:
            await sandbox.close()

    read, env, net, system, outside, slow = asyncio.run(scenario())

    assert read == (0, "from host")
    assert (tmp_path / "out.txt").read_text().strip() == "made"  # writes reach the project
    assert env[1].strip() == "secret=[]"  # host environment (API keys) not visible
    assert net[0] != 0  # no network
    assert system[0] != 0  # system folders are read-only
    assert "workspace" in outside[1] and "Users" not in outside[1]
    assert slow[0] == docker_module.TIMEOUT_EXIT
