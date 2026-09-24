from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path_factory, monkeypatch):
    """Keep every test out of the real ~/.agentcli (memory.db, snapshots, audit log).

    Path.home() and expanduser() read USERPROFILE on Windows and HOME elsewhere, so both are
    redirected. Tests that set HOME themselves still override it for their own needs.
    """

    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
