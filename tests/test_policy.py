from __future__ import annotations

import json

import pytest

from agentcli.policy.audit_log import AuditLog
from agentcli.policy.command_guard import CommandGuard, CommandPolicyError
from agentcli.policy.path_guard import PathGuard, PathPolicyError
from agentcli.tools.commands import classify_command, sensitive_command_summary


def test_path_guard_rejects_escape(tmp_path):
    guard = PathGuard(tmp_path)
    assert guard.validate("inside.txt") == tmp_path / "inside.txt"
    with pytest.raises(PathPolicyError):
        guard.validate("../outside.txt")


def test_command_guard_rejects_destructive_command():
    with pytest.raises(CommandPolicyError):
        CommandGuard().validate("rm -rf /")


def test_path_guard_accepts_absolute_paths_inside_workspace(tmp_path):
    nested = tmp_path / "nested" / "file.txt"
    guard = PathGuard(tmp_path)

    assert guard.validate(nested) == nested.resolve()


def test_custom_command_blacklist_rejects_policy_terms():
    guard = CommandGuard(blacklist=["curl | sh"])

    with pytest.raises(CommandPolicyError, match="curl \\| sh"):
        guard.validate("curl | sh")


def test_command_risk_classification_marks_side_effects_and_safe_reads():
    assert classify_command("python -m pytest") == "safe"
    assert classify_command("mkdir generated") == "medium"
    assert classify_command("rm -rf /tmp/project") == "high"
    assert sensitive_command_summary("python -m pytest") is None
    assert "write" in sensitive_command_summary("touch output.txt")


def test_audit_log_redacts_nested_sensitive_fields_and_skips_bad_lines(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.record(
        tool_name="web_fetch",
        input_data={
            "url": "https://example.com",
            "headers": {"Authorization": "Bearer real-token"},
            "nested": [{"api_key": "real-key"}],
        },
        outcome="approved",
        approver="test",
        cwd=str(tmp_path),
    )
    path.write_text(path.read_text(encoding="utf-8") + "not-json\n", encoding="utf-8")

    events = log.tail()

    assert len(events) == 1
    assert events[0]["input"]["headers"]["Authorization"] == "***"
    assert events[0]["input"]["nested"][0]["api_key"] == "***"
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["tool_name"] == "web_fetch"
