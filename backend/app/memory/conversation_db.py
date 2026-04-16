from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from app.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    role        TEXT    NOT NULL,
    content     TEXT    NOT NULL,
    timestamp   REAL    NOT NULL,
    metadata_json TEXT  DEFAULT '{}'
);
"""
_CREATE_INDEX = """\
CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);
"""


class ConversationDB:
    """Async SQLite store for conversation history."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        settings = get_settings()
        if db_path is None:
            db_path = settings.data_dir / "sqlite" / "conversations.db"
        self._db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create the database file and schema if they don't exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute(_CREATE_TABLE)
        await self._db.execute(_CREATE_INDEX)
        await self._db.commit()
        log.info("conversation_db_ready", path=str(self._db_path))

    async def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            await self.initialize()
        assert self._db is not None
        return self._db

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Insert a message and return its row id."""
        db = await self._conn()
        cursor = await db.execute(
            "INSERT INTO conversations (session_id, role, content, timestamp, metadata_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, time.time(), json.dumps(metadata or {})),
        )
        await db.commit()
        row_id = cursor.lastrowid or 0
        log.debug("message_added", session_id=session_id, role=role, row_id=row_id)
        return row_id

    async def get_history(
        self,
        session_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return the most recent messages for a session, oldest first."""
        db = await self._conn()
        sql = (
            "SELECT id, session_id, role, content, timestamp, metadata_json "
            "FROM conversations "
            "WHERE session_id = ? "
            "ORDER BY id DESC LIMIT ? OFFSET ?"
        )
        rows = await db.execute_fetchall(sql, (session_id, limit, offset))
        results: list[dict[str, Any]] = []
        for row in reversed(rows):
            results.append({
                "id": row[0],
                "session_id": row[1],
                "role": row[2],
                "content": row[3],
                "timestamp": row[4],
                "metadata": json.loads(row[5]) if row[5] else {},
            })
        return results

    async def search_messages(
        self,
        query: str,
        session_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Simple LIKE-based full-text search across messages."""
        db = await self._conn()
        if session_id:
            sql = (
                "SELECT id, session_id, role, content, timestamp, metadata_json "
                "FROM conversations "
                "WHERE session_id = ? AND content LIKE ? "
                "ORDER BY id DESC LIMIT ?"
            )
            rows = await db.execute_fetchall(sql, (session_id, f"%{query}%", limit))
        else:
            sql = (
                "SELECT id, session_id, role, content, timestamp, metadata_json "
                "FROM conversations "
                "WHERE content LIKE ? "
                "ORDER BY id DESC LIMIT ?"
            )
            rows = await db.execute_fetchall(sql, (f"%{query}%", limit))

        return [
            {
                "id": row[0],
                "session_id": row[1],
                "role": row[2],
                "content": row[3],
                "timestamp": row[4],
                "metadata": json.loads(row[5]) if row[5] else {},
            }
            for row in rows
        ]

    async def get_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """List distinct sessions with message count and last activity."""
        db = await self._conn()
        sql = (
            "SELECT session_id, COUNT(*) as msg_count, "
            "MIN(timestamp) as created, MAX(timestamp) as last_active "
            "FROM conversations GROUP BY session_id "
            "ORDER BY last_active DESC LIMIT ?"
        )
        rows = await db.execute_fetchall(sql, (limit,))
        return [
            {
                "session_id": row[0],
                "message_count": row[1],
                "created_at": row[2],
                "last_active": row[3],
            }
            for row in rows
        ]

    async def delete_session(self, session_id: str) -> int:
        """Delete all messages for a session. Returns rows deleted."""
        db = await self._conn()
        cursor = await db.execute(
            "DELETE FROM conversations WHERE session_id = ?", (session_id,)
        )
        await db.commit()
        count = cursor.rowcount
        log.info("session_deleted", session_id=session_id, rows=count)
        return count

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
            log.info("conversation_db_closed")
