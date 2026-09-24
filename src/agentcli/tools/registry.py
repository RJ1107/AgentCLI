from __future__ import annotations

from agentcli.tools.base import Tool


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}
        # Deferred tools the model has loaded; they stay loaded for the rest of the session.
        self._activated: set[str] = set()

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def register_all(self, tools: list[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        return sorted(self._tools)

    def definitions(self) -> list[dict]:
        """Schemas sent to the model: every regular tool plus deferred tools loaded so far."""

        return [
            self._tools[name].definition()
            for name in self.list_names()
            if not self._tools[name].deferred or name in self._activated
        ]

    def deferred_tools(self) -> list[Tool]:
        return [self._tools[name] for name in self.list_names() if self._tools[name].deferred]

    def is_active(self, name: str) -> bool:
        tool = self._tools.get(name)
        return tool is not None and (not tool.deferred or name in self._activated)

    def activate(self, names: list[str]) -> list[str]:
        """Load deferred tools; returns the names that were newly activated."""

        added = [
            name
            for name in names
            if name in self._tools and self._tools[name].deferred and name not in self._activated
        ]
        self._activated.update(added)
        return added
