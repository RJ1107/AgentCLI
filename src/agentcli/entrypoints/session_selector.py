from __future__ import annotations

from dataclasses import dataclass

from prompt_toolkit.application import Application, get_app
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style

from agentcli.entrypoints.model_selector import _window
from agentcli.session import SessionMeta


@dataclass(slots=True)
class SessionPickerState:
    sessions: list[SessionMeta]
    current_id: str = ""
    index: int = 0

    def move(self, delta: int) -> None:
        if self.sessions:
            self.index = (self.index + delta) % len(self.sessions)

    def render(self, visible: int | None = None) -> StyleAndTextTuples:
        fragments: StyleAndTextTuples = [
            ("class:command", "> /resume\n\n"),
            ("class:title", f"Sessions in this project ({len(self.sessions)})\n"),
            ("class:line", "─" * 78 + "\n"),
        ]
        if not self.sessions:
            fragments.append(("class:muted", "  No saved sessions yet.\n"))
        first, last = _window(len(self.sessions), self.index, visible)
        if first > 0:
            fragments.append(("class:muted", f"  ↑ {first} more above\n"))
        for position, meta in list(enumerate(self.sessions))[first:last]:
            selected = position == self.index
            marker = "> " if selected else "  "
            current = "  (current)" if meta.id == self.current_id else ""
            fragments.append(
                (
                    "class:selected" if selected else "class:item",
                    f"{marker}{meta.title or '(untitled)'}{current}\n",
                )
            )
            fragments.append(
                (
                    "class:muted",
                    f"    {meta.updated_text} · {meta.turns} turns · {meta.model} · {meta.id}\n",
                )
            )
        if last < len(self.sessions):
            fragments.append(("class:muted", f"  ↓ {len(self.sessions) - last} more below\n"))
        fragments.append(("class:footer", "\n↑↓ navigate · Enter resume · Esc cancel"))
        return fragments


def _sessions_that_fit() -> int:
    # Each session takes two lines; the header and footer about eight.
    try:
        rows = get_app().output.get_size().rows
    except Exception:  # noqa: BLE001 - no terminal size (tests, pipes): show them all
        return 1_000
    return max(3, (rows - 8) // 2)


async def pick_session(sessions: list[SessionMeta], current_id: str = "") -> SessionMeta | None:
    state = SessionPickerState(sessions, current_id)
    bindings = KeyBindings()
    control = FormattedTextControl(
        text=lambda: state.render(visible=_sessions_that_fit()), focusable=True, show_cursor=False
    )

    @bindings.add("up")
    def _up(event) -> None:
        state.move(-1)
        event.app.invalidate()

    @bindings.add("down")
    def _down(event) -> None:
        state.move(1)
        event.app.invalidate()

    @bindings.add("enter")
    def _select(event) -> None:
        event.app.exit(result=state.sessions[state.index] if state.sessions else None)

    @bindings.add("escape")
    @bindings.add("c-c")
    def _cancel(event) -> None:
        event.app.exit(result=None)

    application: Application[SessionMeta | None] = Application(
        layout=Layout(Window(control, wrap_lines=False, always_hide_cursor=True)),
        key_bindings=bindings,
        style=Style.from_dict(
            {
                "command": "#c084fc",
                "title": "bold #ffffff",
                "line": "#555555",
                "item": "#f3f4f6",
                "selected": "bold #22c55e",
                "muted": "#9a9a9a",
                "footer": "italic #9a9a9a",
            }
        ),
        full_screen=False,
        mouse_support=False,
    )
    return await application.run_async()
