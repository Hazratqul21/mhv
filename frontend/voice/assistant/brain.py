"""Voice assistant conversation brain.

Manages the conversation flow: decides whether to use a quick command or
forward the request to the MIYA API, tracks mode (idle / listening /
processing / speaking), and keeps a short local context for voice-specific
needs.
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field

from assistant.commands import QuickCommands
from core.api_client import MiyaAPIClient

log = logging.getLogger("miya.voice.brain")


class VoiceMode(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


@dataclass
class Turn:
    role: str  # "user" or "assistant"
    text: str
    ts: float = field(default_factory=time.time)


class VoiceBrain:
    """High-level conversation controller for the voice daemon."""

    def __init__(
        self,
        api_client: MiyaAPIClient,
        quick_commands: QuickCommands | None = None,
        max_history: int = 10,
        *,
        use_quick_commands: bool = True,
    ) -> None:
        self.api = api_client
        self.commands = quick_commands or QuickCommands()
        self._use_quick_commands = use_quick_commands
        self.mode = VoiceMode.IDLE
        self._history: list[Turn] = []
        self._max_history = max_history

    @property
    def history(self) -> list[Turn]:
        return list(self._history)

    def set_mode(self, mode: VoiceMode) -> None:
        if mode != self.mode:
            log.debug("mode %s -> %s", self.mode.value, mode.value)
            self.mode = mode

    async def process(self, user_text: str) -> str:
        """Process user speech text and return a response string."""
        if not user_text.strip():
            return ""

        self._history.append(Turn(role="user", text=user_text))
        self._trim_history()

        self.set_mode(VoiceMode.PROCESSING)

        quick = None
        if self._use_quick_commands:
            quick = self.commands.match(user_text)
        if quick is not None:
            response = quick
            log.info("quick command -> %s", response[:80])
        else:
            response = await self.api.chat(user_text)

        self._history.append(Turn(role="assistant", text=response))
        self._trim_history()

        return response

    def _trim_history(self) -> None:
        if len(self._history) > self._max_history * 2:
            self._history = self._history[-self._max_history * 2 :]

    def reset(self) -> None:
        self._history.clear()
        self.set_mode(VoiceMode.IDLE)
