from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from agentcli.config import load_config
from agentcli.memory import FileMemory
from agentcli.memory.rerank import jev_recall
from agentcli.prompt import PromptAssembler


def _jev(scores: dict[str, float] | None = None, *, status: int = 200, delay: float = 0.0):
    seen: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if delay:
            await asyncio.sleep(delay)
        if status != 200:
            return httpx.Response(status, text="down")
        answers = {name: {"type": "noul", "noul": (scores or {}).get(name, 0.0)} for name in body["questions"]}
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": answers, "usage": {}})

    return httpx.MockTransport(handler), seen


def _assembler(config, cwd):
    return PromptAssembler(config=config, cwd=str(cwd), tool_names=[], model="m", provider="p")


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTCLI_HOME", str(tmp_path / "home"))
    cwd = tmp_path / "project"
    cwd.mkdir()
    memory = FileMemory(str(cwd))
    port = memory.save("测试环境 PostgreSQL 跑在 5433 端口", title="测试数据库端口", keywords=["DB", "port"])
    checks = memory.save("上线前必须通过 lint 和单测", title="发布检查", keywords=["release"])
    return cwd, memory, port, checks


def test_jev_keeps_what_it_finds_relevant_in_its_order(project, monkeypatch):
    _cwd, memory, port, checks = project
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    shortlist = memory.search("数据库 端口 上线", limit=15)
    transport, seen = _jev({port.name: 0.6, checks.name: 0.97})

    hits = asyncio.run(
        jev_recall("部署前要跑哪些检查", shortlist, threshold=0.85, limit=3, timeout=2, transport=transport)
    )

    assert [hit.record.name for hit in hits] == [checks.name]
    # One request: the whole shortlist in the state, one noul per memory.
    assert len(seen) == 1 and set(seen[0]["questions"]) == {h.record.name for h in shortlist}
    assert seen[0]["state"]["question"] == "部署前要跑哪些检查"


def test_recall_uses_jev_when_a_key_is_set(project, monkeypatch):
    cwd, _memory, _port, checks = project
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    transport, seen = _jev({checks.name: 0.95})
    assembler = _assembler(load_config(project_root=str(cwd)), cwd)

    # "上线" shares one word with the release memory: far below the coverage gate alone.
    text = asyncio.run(assembler.recall("上线前我该注意什么", transport=transport))

    assert seen and "发布检查" in text and "5433" not in text


@pytest.mark.parametrize("failure", [{"status": 503}, {"delay": 2.0}])
def test_recall_falls_back_to_the_coverage_gate_when_jev_fails(project, monkeypatch, failure):
    cwd, _memory, _port, _checks = project
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    config = load_config(project_root=str(cwd))
    config.memory.recall_jev_timeout = 0.3
    transport, _seen = _jev(**failure)
    assembler = _assembler(config, cwd)

    text = asyncio.run(assembler.recall("测试数据库端口是多少", transport=transport))

    assert "5433" in text


def test_recall_without_a_key_never_calls_jev(project, monkeypatch):
    cwd, _memory, _port, _checks = project
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    transport, seen = _jev({})
    assembler = _assembler(load_config(project_root=str(cwd)), cwd)

    text = asyncio.run(assembler.recall("测试数据库端口是多少", transport=transport))

    assert not seen and "5433" in text
