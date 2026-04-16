from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Callable, Awaitable

from app.utils.logger import get_logger

log = get_logger(__name__)

Listener = Callable[..., Awaitable[None]]


class EventBus:
    """Simple async pub/sub event bus for inter-component communication."""

    def __init__(self) -> None:
        self._listeners: dict[str, list[Listener]] = defaultdict(list)
        self._history: list[dict[str, Any]] = []
        self._max_history = 1000

    def on(self, event: str, callback: Listener) -> None:
        self._listeners[event].append(callback)
        log.debug("event_listener_added", event=event, callback=callback.__name__)

    def off(self, event: str, callback: Listener) -> None:
        listeners = self._listeners.get(event, [])
        if callback in listeners:
            listeners.remove(callback)

    async def emit(self, event: str, **data: Any) -> None:
        record = {"event": event, **data}
        self._history.append(record)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        listeners = list(self._listeners.get(event, ()))
        if not listeners:
            return

        log.debug("event_emitted", event=event, listener_count=len(listeners))
        tasks = [asyncio.create_task(cb(**data)) for cb in listeners]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                log.error(
                    "event_listener_error",
                    event=event,
                    listener=listeners[i].__name__,
                    error=str(result),
                )

    def get_history(self, event: str | None = None, limit: int = 50) -> list[dict]:
        if event:
            filtered = [r for r in self._history if r["event"] == event]
        else:
            filtered = self._history
        return filtered[-limit:]

    def clear(self) -> None:
        self._listeners.clear()
        self._history.clear()
