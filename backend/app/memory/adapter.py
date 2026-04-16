"""Memory adapter that bridges the dict-based memory layer to the
interface expected by MiyaOrchestrator (store / get_recent / search_relevant).
"""

from __future__ import annotations

from typing import Any

from app.utils.logger import get_logger

log = get_logger(__name__)


class MemoryAdapter:
    """Wraps the dict of memory components into a unified interface.

    The orchestrator calls:
      - store(session_id, query, result)
      - get_recent(session_id, limit)
      - search_relevant(query, limit)
    """

    def __init__(self, components: dict[str, Any]) -> None:
        self._components = components

    @property
    def conversation_db(self):
        return self._components.get("conversation_db")

    @property
    def vector(self):
        return self._components.get("vector")

    @property
    def cache(self):
        return self._components.get("cache")

    def get(self, key: str, default=None):
        """Allow dict-style access for backward compatibility with routes."""
        return self._components.get(key, default)

    async def store(self, session_id: str, query: str, result: dict) -> None:
        conv_db = self.conversation_db
        if conv_db:
            try:
                await conv_db.add_message(session_id, "user", query)
                await conv_db.add_message(
                    session_id, "assistant", result.get("response", result.get("output", ""))
                )
            except Exception as exc:
                log.debug("memory_store_conv_failed: %s", exc)

        vector = self.vector
        if vector:
            try:
                vector.add(
                    text=f"Q: {query}\nA: {result.get('response', result.get('output', ''))}",
                    metadata={"session_id": session_id, "type": "conversation"},
                )
            except Exception as exc:
                log.debug("memory_store_vector_failed: %s", exc)

    async def get_recent(self, session_id: str, limit: int = 10) -> list[dict]:
        conv_db = self.conversation_db
        if conv_db:
            try:
                return await conv_db.get_history(session_id, limit=limit)
            except Exception as exc:
                log.debug("memory_get_recent_failed: %s", exc)
        return []

    async def search_relevant(self, query: str, limit: int = 5) -> list[dict]:
        vector = self.vector
        if vector:
            try:
                return vector.search(query, n_results=limit)
            except Exception as exc:
                log.debug("memory_search_relevant_failed: %s", exc)
        return []
