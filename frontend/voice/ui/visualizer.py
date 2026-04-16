"""Terminal-based voice activity visualizer.

Prints a live audio-level bar to the terminal so the user can see when
the microphone is picking up speech.  Works in any terminal without
requiring a GUI toolkit.
"""

from __future__ import annotations

import asyncio
import logging
import sys

import numpy as np

log = logging.getLogger("miya.voice.viz")

BAR_WIDTH = 40
BLOCK_CHARS = " ▏▎▍▌▋▊▉█"


class VoiceVisualizer:
    """Simple terminal-based audio level visualizer."""

    def __init__(self, bar_width: int = BAR_WIDTH, enabled: bool = True) -> None:
        self.bar_width = bar_width
        self.enabled = enabled
        self._status = "idle"

    def set_status(self, status: str) -> None:
        self._status = status

    def _level_bar(self, rms: float) -> str:
        level = min(rms * 10, 1.0)  # scale up typical speech levels
        filled = level * self.bar_width
        full_blocks = int(filled)
        frac = filled - full_blocks
        frac_char = BLOCK_CHARS[int(frac * (len(BLOCK_CHARS) - 1))]
        empty = self.bar_width - full_blocks - 1

        bar = "█" * full_blocks + frac_char + " " * max(empty, 0)
        return bar[: self.bar_width]

    def display(self, audio_chunk: np.ndarray | None = None) -> None:
        if not self.enabled:
            return

        if audio_chunk is not None and audio_chunk.size > 0:
            rms = float(np.sqrt(np.mean(audio_chunk**2)))
        else:
            rms = 0.0

        status_colors = {
            "idle": "\033[32m",       # green
            "listening": "\033[34m",  # blue
            "processing": "\033[33m", # yellow
            "speaking": "\033[35m",   # magenta
        }
        color = status_colors.get(self._status, "\033[0m")
        reset = "\033[0m"

        bar = self._level_bar(rms)
        line = f"\r{color}[{self._status:^11}]{reset} |{bar}| {rms:.3f}"
        sys.stdout.write(line)
        sys.stdout.flush()

    def clear(self) -> None:
        sys.stdout.write("\r" + " " * (self.bar_width + 30) + "\r")
        sys.stdout.flush()
