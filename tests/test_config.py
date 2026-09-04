from __future__ import annotations

import json

from agentcli.config import config_to_public_dict, load_config


def test_config_precedence(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = tmp_path / "project"
    (home / ".agentcli").mkdir(parents=True)
    (project / ".agentcli").mkdir(parents=True)
    (home / ".agentcli" / "config.json").write_text(
        json.dumps({"llm": {"provider": "home", "model": "home-model"}}),
        encoding="utf-8",
    )
    (project / ".agentcli" / "config.json").write_text(
        json.dumps({"llm": {"provider": "project", "model": "project-model"}}),
        encoding="utf-8",
    )
    (project / ".env").write_text("AGENTCLI_MODEL=env-file-model\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("AGENTCLI_PROVIDER", "process")

    config = load_config(
        project_root=project,
        overrides={"llm": {"model": "cli-model"}},
    )

    assert config.llm.provider == "process"
    assert config.llm.model == "cli-model"


def test_provider_specific_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("AGENTCLI_PROVIDER", "deepseek")
    monkeypatch.delenv("AGENTCLI_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")

    config = load_config(project_root=tmp_path)

    assert config.llm.api_key == "deepseek-key"


def test_generic_api_key_takes_precedence_over_provider_specific_key(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("AGENTCLI_PROVIDER", "deepseek")
    monkeypatch.setenv("AGENTCLI_API_KEY", "generic-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "provider-key")

    config = load_config(project_root=tmp_path)

    assert config.llm.api_key == "generic-key"


def test_env_flags_toggle_features_render_mode_and_hitl(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(
        project_root=tmp_path,
        env={
            "AGENTCLI_MCP": "false",
            "AGENTCLI_SKILL": "true",
            "AGENTCLI_MEMORY": "false",
            "AGENTCLI_RENDER_MODE": "plain",
            "AGENTCLI_HITL": "never",
        },
    )

    assert not config.features.mcp
    assert config.features.skill
    assert not config.features.memory
    assert config.render_mode == "plain"
    assert config.policy.hitl_mode == "never"


def test_public_config_redacts_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path, env={"AGENTCLI_API_KEY": "secret-value"})

    public = config_to_public_dict(config)

    assert public["llm"]["api_key"] == "***"
