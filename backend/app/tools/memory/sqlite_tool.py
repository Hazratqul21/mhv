from __future__ import annotations

from typing import Any

import aiosqlite

from app.config import get_settings
from app.tools.registry import BaseTool
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SQLiteTool(BaseTool):
    name = "sqlite"
    description = "Run SQL queries, insert rows, and create tables in a local SQLite database"
    category = "memory"
    parameters = {
        "action": {
            "type": "string",
            "enum": ["query", "insert", "create_table"],
            "description": "Operation to perform",
        },
        "database": {
            "type": "string",
            "description": "Database file name (stored under data_dir)",
            "default": "miya.db",
        },
        "sql": {
            "type": "string",
            "description": "SQL statement to execute",
        },
        "params": {
            "type": "array",
            "description": "Positional bind parameters for the SQL statement",
        },
    }

    def __init__(self) -> None:
        settings = get_settings()
        self._data_dir = settings.data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def _db_path(self, name: str) -> str:
        return str(self._data_dir / name)

    async def execute(self, input_data: dict[str, Any]) -> Any:
        action = input_data.get("action", "")
        db_name = input_data.get("database", "miya.db")
        sql = input_data.get("sql", "")
        params = input_data.get("params", [])

        if not sql:
            return {"error": "'sql' is required"}

        try:
            async with aiosqlite.connect(self._db_path(db_name)) as db:
                if action == "query":
                    db.row_factory = aiosqlite.Row
                    cursor = await db.execute(sql, params)
                    rows = await cursor.fetchall()
                    columns = [d[0] for d in cursor.description] if cursor.description else []
                    return {
                        "columns": columns,
                        "rows": [dict(zip(columns, row)) for row in rows],
                        "count": len(rows),
                    }

                if action in ("insert", "create_table"):
                    await db.execute(sql, params)
                    await db.commit()
                    return {
                        "status": "ok",
                        "action": action,
                        "changes": db.total_changes,
                    }

                return {"error": f"Unknown action '{action}'"}

        except Exception as exc:
            logger.error("sqlite_error", action=action, error=str(exc))
            return {"error": str(exc)}
