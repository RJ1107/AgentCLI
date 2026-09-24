from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from agentcli.mcp.client import McpClientManager
from agentcli.mcp.config import McpServerSpec


class _FakeSession:
    """Stands in for a stateful MCP server, like a browser that remembers its page."""

    def __init__(self):
        self.page = "about:blank"


def _manager_with_fake_server(tmp_path, opened: list[_FakeSession], closed: list[_FakeSession]):
    manager = McpClientManager(tmp_path)

    @asynccontextmanager
    async def connect(_spec):
        session = _FakeSession()
        opened.append(session)
        try:
            yield session
        finally:
            closed.append(session)

    manager._connect = connect  # type: ignore[method-assign]
    return manager


def test_calls_share_one_connection_so_server_state_survives(tmp_path):
    opened: list[_FakeSession] = []
    closed: list[_FakeSession] = []
    manager = _manager_with_fake_server(tmp_path, opened, closed)
    spec = McpServerSpec(name="browser", command="fake")

    async def run():
        async with manager._session(spec) as session:
            session.page = "https://example.com"  # navigate_page
        async with manager._session(spec) as session:
            snapshot = session.page  # take_snapshot
        await manager.aclose()
        return snapshot

    assert asyncio.run(run()) == "https://example.com"
    assert len(opened) == 1
    assert closed == opened


def test_each_event_loop_gets_its_own_connection(tmp_path):
    opened: list[_FakeSession] = []
    closed: list[_FakeSession] = []
    manager = _manager_with_fake_server(tmp_path, opened, closed)
    spec = McpServerSpec(name="browser", command="fake")

    async def use_once():
        async with manager._session(spec):
            pass
        await manager.aclose()

    asyncio.run(use_once())
    asyncio.run(use_once())  # e.g. a second Runtime API task in a new asyncio.run

    assert len(opened) == 2
    assert len(closed) == 2


def test_failed_connection_is_retried_on_next_use(tmp_path):
    manager = McpClientManager(tmp_path)
    attempts = {"count": 0}

    @asynccontextmanager
    async def flaky(_spec):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("server failed to start")
        yield _FakeSession()

    manager._connect = flaky  # type: ignore[method-assign]
    spec = McpServerSpec(name="flaky", command="fake")

    async def run():
        try:
            async with manager._session(spec):
                pass
        except RuntimeError as exc:
            first_error = str(exc)
        async with manager._session(spec) as session:
            assert session.page == "about:blank"
        await manager.aclose()
        return first_error

    assert asyncio.run(run()) == "server failed to start"
    assert attempts["count"] == 2
