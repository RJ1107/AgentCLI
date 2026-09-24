from __future__ import annotations

import json
from pathlib import Path

from agentcli.config import load_config
from agentcli.mcp.config import _spec_from_raw, write_chrome_devtools_config
from agentcli.paths import agentcli_home
from agentcli.skill import SkillRegistry
from agentcli.snapshot import SnapshotService


def test_agentcli_home_moves_every_data_file(tmp_path, monkeypatch):
    data = tmp_path / "D-drive" / "agentcli-data"
    monkeypatch.setenv("AGENTCLI_HOME", str(data))
    (data).mkdir(parents=True)
    (data / "config.json").write_text(
        json.dumps({"routing": {"fast_model": "openrouter:openai/gpt-6-luna"}}), encoding="utf-8"
    )
    project = tmp_path / "project"
    project.mkdir()

    config = load_config(project_root=project)

    assert agentcli_home() == data
    assert config.routing.fast_model == "openrouter:openai/gpt-6-luna"  # user config read there
    assert config.memory.long_term_db_path == str(data / "memory.db")
    assert config.policy.audit_log_path == str(data / "audit.jsonl")
    assert SnapshotService(project).root.parent == data / "snapshots"
    assert SkillRegistry(project).user_root == data / "skills"

    # The visible browser's saved profile follows it too.
    path = write_chrome_devtools_config(scope_root=None)
    raw = json.loads(path.read_text(encoding="utf-8"))["mcpServers"]["chrome-visible"]
    spec = _spec_from_raw("chrome-visible", raw, project)
    assert path == data / "mcp.json"
    profile = next(arg for arg in spec.args if arg.startswith("--userDataDir="))
    assert Path(profile.split("=", 1)[1]) == data / "browser-profile"


def test_without_agentcli_home_data_stays_in_the_home_folder(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENTCLI_HOME", raising=False)
    config = load_config(project_root=tmp_path)

    assert agentcli_home().name == ".agentcli"
    assert config.memory.long_term_db_path.endswith(".agentcli" + "\\memory.db") or (
        config.memory.long_term_db_path.endswith(".agentcli/memory.db")
    )
