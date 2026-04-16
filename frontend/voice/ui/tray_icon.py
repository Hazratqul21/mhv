"""System tray icon using pystray.

Shows MIYA's current state (Idle / Listening / Processing / Speaking)
in the system tray with a right-click context menu for mute, settings,
and quit.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Callable

from PIL import Image, ImageDraw

log = logging.getLogger("miya.voice.tray")

STATUS_COLORS = {
    "idle": "#4CAF50",       # green
    "listening": "#2196F3",  # blue
    "processing": "#FF9800", # orange
    "speaking": "#9C27B0",   # purple
    "error": "#F44336",      # red
    "muted": "#9E9E9E",      # gray
}


def _make_icon(color: str, size: int = 64) -> Image.Image:
    """Generate a simple colored circle icon."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = 4
    draw.ellipse(
        [margin, margin, size - margin, size - margin],
        fill=color,
        outline="#FFFFFF",
        width=2,
    )
    draw.text(
        (size // 2 - 6, size // 2 - 8),
        "M",
        fill="#FFFFFF",
    )
    return img


class TrayIcon:
    """Manages the system tray icon and menu."""

    def __init__(
        self,
        on_quit: Callable | None = None,
        on_mute_toggle: Callable | None = None,
    ) -> None:
        self._on_quit = on_quit
        self._on_mute_toggle = on_mute_toggle
        self._status = "idle"
        self._muted = False
        self._icon = None
        self._thread: threading.Thread | None = None

    def _build_menu(self):
        import pystray

        mute_label = "Unmute" if self._muted else "Mute"
        return pystray.Menu(
            pystray.MenuItem(f"Status: {self._status}", lambda: None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(mute_label, self._handle_mute_toggle),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit MIYA Voice", self._handle_quit),
        )

    def _handle_quit(self, icon, item) -> None:
        log.info("quit requested from tray")
        if self._icon:
            self._icon.stop()
        if self._on_quit:
            self._on_quit()

    def _handle_mute_toggle(self, icon, item) -> None:
        self._muted = not self._muted
        log.info("mute toggled: %s", self._muted)
        color = STATUS_COLORS["muted"] if self._muted else STATUS_COLORS[self._status]
        if self._icon:
            self._icon.icon = _make_icon(color)
            self._icon.menu = self._build_menu()
        if self._on_mute_toggle:
            self._on_mute_toggle(self._muted)

    def update_status(self, status: str) -> None:
        self._status = status
        if self._icon and not self._muted:
            color = STATUS_COLORS.get(status, STATUS_COLORS["idle"])
            self._icon.icon = _make_icon(color)
            self._icon.title = f"MIYA Voice - {status}"
            self._icon.menu = self._build_menu()

    def start(self) -> None:
        """Start the tray icon in a background thread."""
        import pystray

        color = STATUS_COLORS["idle"]
        self._icon = pystray.Icon(
            name="miya-voice",
            icon=_make_icon(color),
            title="MIYA Voice - idle",
            menu=self._build_menu(),
        )
        self._thread = threading.Thread(target=self._icon.run, daemon=True)
        self._thread.start()
        log.info("system tray icon started")

    def stop(self) -> None:
        if self._icon:
            self._icon.stop()
            self._icon = None
