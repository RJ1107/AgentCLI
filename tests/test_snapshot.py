from __future__ import annotations

import pytest

from agentcli.snapshot import SnapshotService
from agentcli.snapshot import service as snapshot_service


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTCLI_HOME", str(tmp_path / "home"))
    root = tmp_path / "project"
    root.mkdir()
    return root


def test_snapshot_restore(project):
    file_path = project / "note.txt"
    file_path.write_text("before", encoding="utf-8")

    service = SnapshotService(project)
    first = service.create("pre-turn")
    file_path.write_text("after", encoding="utf-8")

    restored = service.restore(first.id)

    assert restored.id == first.id
    assert file_path.read_text(encoding="utf-8") == "before"


def test_restore_removes_files_added_since_and_brings_back_deleted_ones(project):
    (project / "src").mkdir()
    (project / "src" / "kept.py").write_text("x = 1\n", encoding="utf-8")
    (project / "gone.txt").write_text("bye", encoding="utf-8")
    service = SnapshotService(project)
    first = service.create("pre-write")

    (project / "gone.txt").unlink()
    (project / "src" / "new.py").write_bytes(b"y = 2\n")
    service.restore(first.id)

    assert (project / "gone.txt").read_text(encoding="utf-8") == "bye"
    assert not (project / "src" / "new.py").exists()
    # Restoring kept a "pre-restore" snapshot, so the restore itself can be undone.
    assert service.list()[0].phase == "pre-restore"
    assert service.read(service.list()[0], "src/new.py") == b"y = 2\n"


def test_bytes_come_back_exactly_whatever_the_line_endings_or_attributes(project):
    (project / ".gitattributes").write_text("* text=auto eol=lf\n", encoding="utf-8")
    crlf = b"line one\r\nline two\r\n"
    (project / "win.txt").write_bytes(crlf)
    service = SnapshotService(project)
    first = service.create("pre-write")
    (project / "win.txt").write_bytes(b"changed")

    service.restore(first.id)

    assert (project / "win.txt").read_bytes() == crlf


def test_gitignored_and_heavy_folders_are_not_stored(project):
    (project / ".gitignore").write_text("scratch/\n*.log\n", encoding="utf-8")
    (project / "scratch").mkdir()
    (project / "scratch" / "big.bin").write_bytes(b"0" * 1000)
    (project / "run.log").write_text("noise", encoding="utf-8")
    (project / "node_modules").mkdir()
    (project / "node_modules" / "lib.js").write_text("x", encoding="utf-8")
    (project / "app.py").write_bytes(b"print(1)\n")

    record = SnapshotService(project).create("pre-write")
    service = SnapshotService(project)

    assert service.read(record, "app.py") == b"print(1)\n"
    for skipped in ("scratch/big.bin", "run.log", "node_modules/lib.js"):
        assert service.read(record, skipped) is None


def test_unchanged_files_are_stored_once(project):
    for index in range(20):
        (project / f"f{index}.txt").write_text(f"content {index}\n" * 50, encoding="utf-8")
    service = SnapshotService(project)
    service.create("one")
    objects_after_first = _object_count(service)

    (project / "f0.txt").write_text("edited\n", encoding="utf-8")
    service.create("two")

    # One new blob for the edited file, one new tree, one new commit: nothing else.
    assert _object_count(service) - objects_after_first == 3


def test_the_projects_own_git_repository_is_left_alone(project):
    import subprocess

    subprocess.run(["git", "init", "--quiet", str(project)], check=True)
    (project / "a.txt").write_text("a", encoding="utf-8")
    SnapshotService(project).create("pre-write")

    status = subprocess.run(["git", "-C", str(project), "status", "--porcelain"],
                            capture_output=True, text=True, check=True).stdout
    assert status.strip() == "?? a.txt"
    refs = subprocess.run(["git", "-C", str(project), "for-each-ref"],
                          capture_output=True, text=True, check=True).stdout
    assert refs.strip() == ""


def test_old_snapshots_are_trimmed_in_batches(project, monkeypatch):
    monkeypatch.setattr(snapshot_service, "KEEP_SNAPSHOTS", 3)
    monkeypatch.setattr(snapshot_service, "TRIM_BATCH", 2)
    (project / "a.txt").write_text("a", encoding="utf-8")
    service = SnapshotService(project)
    for index in range(6):
        service.create(f"s{index}")

    kept = [record.phase for record in service.list(limit=100)]
    assert kept == ["s5", "s4", "s3"]
    refs = service._git("for-each-ref", "--format=%(refname)", "refs/snapshots").stdout.decode()
    assert len(refs.split()) == 3


def test_without_git_the_full_copy_still_works(project, monkeypatch):
    monkeypatch.setattr(snapshot_service.shutil, "which", lambda _name: None)
    (project / "note.txt").write_text("before", encoding="utf-8")
    service = SnapshotService(project)
    record = service.create("pre-write")
    (project / "note.txt").write_text("after", encoding="utf-8")

    service.restore(record.id)

    assert not record.commit and service.read(record, "note.txt") == b"before"
    assert (project / "note.txt").read_text(encoding="utf-8") == "before"


def _object_count(service: SnapshotService) -> int:
    out = service._git("count-objects", "-v").stdout.decode()
    fields = dict(line.split(": ") for line in out.strip().splitlines())
    return int(fields["count"]) + int(fields["in-pack"])
