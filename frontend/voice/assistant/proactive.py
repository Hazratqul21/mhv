"""Proactive background monitor.

Periodically checks for system events and conditions that warrant an
unsolicited voice notification — e.g. high CPU, low battery, scheduled
reminders, or completed background tasks.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger("miya.voice.proactive")


@dataclass
class Reminder:
    time: datetime.datetime
    message: str
    spoken: bool = False


@dataclass
class Alert:
    kind: str
    message: str
    cooldown_s: float = 300.0
    last_fired: float = 0.0


class ProactiveMonitor:
    """Runs background checks and returns messages to speak aloud."""

    def __init__(
        self,
        speak_callback: Callable[[str], Any] | None = None,
        check_interval: float = 30.0,
    ) -> None:
        self._speak = speak_callback
        self._interval = check_interval
        self._reminders: list[Reminder] = []
        self._running = False
        self._task: asyncio.Task | None = None

    def add_reminder(self, time: datetime.datetime, message: str) -> None:
        self._reminders.append(Reminder(time=time, message=message))
        log.info("reminder added: %s at %s", message, time.isoformat())

    async def _check_system(self) -> list[str]:
        """Return a list of proactive messages based on system state."""
        messages: list[str] = []

        try:
            import psutil

            cpu = psutil.cpu_percent(interval=1)
            if cpu > 90:
                messages.append(f"Diqqat: CPU yuklamasi juda yuqori — {cpu:.0f}%")

            mem = psutil.virtual_memory()
            if mem.percent > 90:
                messages.append(
                    f"Diqqat: Xotira deyarli to'ldi — {mem.percent:.0f}% ishlatilmoqda"
                )

            battery = psutil.sensors_battery()
            if battery and not battery.power_plugged and battery.percent < 15:
                messages.append(
                    f"Batareya kam qoldi — {battery.percent}%. Quvvat ulang."
                )
        except ImportError:
            pass
        except Exception as e:
            log.debug("system check error: %s", e)

        return messages

    def _check_reminders(self) -> list[str]:
        now = datetime.datetime.now()
        fired: list[str] = []
        for rem in self._reminders:
            if not rem.spoken and now >= rem.time:
                fired.append(f"Eslatma: {rem.message}")
                rem.spoken = True
        return fired

    async def _loop(self) -> None:
        while self._running:
            try:
                msgs = self._check_reminders()
                msgs.extend(await self._check_system())

                for msg in msgs:
                    log.info("proactive alert: %s", msg)
                    if self._speak:
                        await self._speak(msg) if asyncio.iscoroutinefunction(self._speak) else self._speak(msg)

            except Exception as e:
                log.error("proactive loop error: %s", e)

            await asyncio.sleep(self._interval)

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())
        log.info("proactive monitor started (interval=%.0fs)", self._interval)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
