from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style

from agentcli.llm.model_profiles import ModelProfile

ModelTab = Literal["default", "custom"]
ActionKind = Literal["select", "add", "delete"]


@dataclass(frozen=True, slots=True)
class ModelSelectorAction:
    kind: ActionKind
    profile: ModelProfile | None = None


@dataclass(slots=True)
class ModelSelectorState:
    defaults: list[ModelProfile]
    custom: list[ModelProfile]
    current_provider: str
    current_model: str
    tab: ModelTab = "default"
    indexes: dict[ModelTab, int] = field(default_factory=lambda: {"default": 0, "custom": 0})

    def __post_init__(self) -> None:
        for tab in ("default", "custom"):
            profiles = self.defaults if tab == "default" else self.custom
            for index, profile in enumerate(profiles):
                if (
                    profile.provider == self.current_provider.lower()
                    and profile.model == self.current_model
                ):
                    self.tab = tab
                    self.indexes[tab] = index
                    break

    @property
    def item_count(self) -> int:
        return len(self.defaults) if self.tab == "default" else len(self.custom) + 1

    @property
    def index(self) -> int:
        return self.indexes[self.tab]

    def switch_tab(self, delta: int = 1) -> None:
        tabs: tuple[ModelTab, ...] = ("default", "custom")
        self.tab = tabs[(tabs.index(self.tab) + delta) % len(tabs)]
        self.indexes[self.tab] = min(self.indexes[self.tab], self.item_count - 1)

    def move(self, delta: int) -> None:
        self.indexes[self.tab] = (self.index + delta) % max(self.item_count, 1)

    def selected_action(self) -> ModelSelectorAction:
        if self.tab == "custom" and self.index == len(self.custom):
            return ModelSelectorAction("add")
        profiles = self.defaults if self.tab == "default" else self.custom
        return ModelSelectorAction("select", profiles[self.index])

    def delete_action(self) -> ModelSelectorAction | None:
        if self.tab != "custom" or self.index >= len(self.custom):
            return None
        return ModelSelectorAction("delete", self.custom[self.index])

    def render(self) -> StyleAndTextTuples:
        fragments: StyleAndTextTuples = [("class:command", "> /model\n\n")]
        fragments.extend([("class:title", "Model  ·  ")])
        fragments.extend(self._tab("default", f"Default ({len(self.defaults)})"))
        fragments.append(("", "   "))
        fragments.extend(self._tab("custom", f"Custom ({len(self.custom)})"))
        fragments.extend(
            [
                ("", "\n"),
                ("class:line", "─" * 78 + "\n"),
                ("class:heading", "Current\n"),
                ("class:muted", "  Provider : "),
                ("", self.current_provider + "\n"),
                ("class:muted", "  Model    : "),
                ("", self.current_model + "\n\n"),
            ]
        )
        profiles = self.defaults if self.tab == "default" else self.custom
        if not profiles and self.tab == "custom":
            fragments.append(("class:muted", "  No custom models yet.\n\n"))
        for index, profile in enumerate(profiles):
            selected = index == self.index
            current = (
                profile.provider == self.current_provider.lower()
                and profile.model == self.current_model
            )
            marker = "> " if selected else "  "
            style = "class:selected" if selected else "class:model"
            check = " ✓" if current else ""
            fragments.append((style, f"{marker}{profile.name}{check}\n"))
            fragments.append(
                (
                    "class:muted",
                    f"    {profile.provider} · modelID: {profile.model} · "
                    f"context: {profile.context_window:,}\n",
                )
            )
            if profile.description:
                fragments.append(("class:muted", f"    {profile.description}\n"))
        if self.tab == "custom":
            selected = self.index == len(self.custom)
            marker = "> " if selected else "  "
            style = "class:selected" if selected else "class:model"
            fragments.extend(
                [
                    (style, f"{marker}[+] Add custom model...\n"),
                    ("class:muted", "    Add a new Bring Your Own Key (BYOK) model\n"),
                ]
            )
        fragments.extend(
            [
                ("class:footer", "\nTab/←→ switch · ↑↓ navigate · Enter select"),
                (
                    "class:footer",
                    " · d delete · Esc back" if self.tab == "custom" else " · Esc back",
                ),
            ]
        )
        return fragments

    def _tab(self, tab: ModelTab, label: str) -> StyleAndTextTuples:
        return [("class:tab.active" if self.tab == tab else "class:tab", f" {label} ")]


async def run_model_selector(state: ModelSelectorState) -> ModelSelectorAction | None:
    bindings = KeyBindings()
    control = FormattedTextControl(text=state.render, focusable=True, show_cursor=False)

    def refresh(event) -> None:
        event.app.invalidate()

    @bindings.add("tab")
    @bindings.add("right")
    def _next_tab(event) -> None:
        state.switch_tab(1)
        refresh(event)

    @bindings.add("s-tab")
    @bindings.add("left")
    def _previous_tab(event) -> None:
        state.switch_tab(-1)
        refresh(event)

    @bindings.add("up")
    def _up(event) -> None:
        state.move(-1)
        refresh(event)

    @bindings.add("down")
    def _down(event) -> None:
        state.move(1)
        refresh(event)

    @bindings.add("enter")
    def _select(event) -> None:
        event.app.exit(result=state.selected_action())

    @bindings.add("d")
    def _delete(event) -> None:
        action = state.delete_action()
        if action:
            event.app.exit(result=action)

    @bindings.add("escape")
    @bindings.add("c-c")
    def _cancel(event) -> None:
        event.app.exit(result=None)

    application: Application[ModelSelectorAction | None] = Application(
        layout=Layout(Window(control, wrap_lines=False, always_hide_cursor=True)),
        key_bindings=bindings,
        style=Style.from_dict(
            {
                "command": "#c084fc",
                "title": "bold #ffffff",
                "heading": "bold #ffffff",
                "line": "#555555",
                "tab": "#9a9a9a",
                "tab.active": "bold #22c55e bg:#12351f",
                "model": "#f3f4f6",
                "selected": "bold #22c55e",
                "muted": "#9a9a9a",
                "footer": "italic #9a9a9a",
            }
        ),
        full_screen=False,
        mouse_support=False,
    )
    return await application.run_async()
