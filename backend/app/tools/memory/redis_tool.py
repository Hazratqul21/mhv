from __future__ import annotations

from typing import Any

import redis.asyncio as aioredis

from app.config import get_settings
from app.tools.registry import BaseTool
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RedisTool(BaseTool):
    name = "redis"
    description = "Key-value cache operations: get, set, delete, and expire keys in Redis"
    category = "memory"
    parameters = {
        "action": {
            "type": "string",
            "enum": ["get", "set", "delete", "expire"],
            "description": "Operation to perform",
        },
        "key": {
            "type": "string",
            "description": "Redis key",
        },
        "value": {
            "type": "string",
            "description": "Value to set (for set action)",
        },
        "ttl": {
            "type": "integer",
            "description": "TTL in seconds (for set/expire)",
        },
    }

    def __init__(self) -> None:
        settings = get_settings()
        self._redis = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
        )

    async def execute(self, input_data: dict[str, Any]) -> Any:
        action = input_data.get("action", "")
        key = input_data.get("key", "")

        if not key:
            return {"error": "'key' is required"}

        try:
            if action == "get":
                value = await self._redis.get(key)
                return {"key": key, "value": value, "exists": value is not None}

            if action == "set":
                value = input_data.get("value", "")
                ttl = input_data.get("ttl")
                if ttl:
                    await self._redis.setex(key, int(ttl), value)
                else:
                    await self._redis.set(key, value)
                return {"status": "ok", "key": key}

            if action == "delete":
                removed = await self._redis.delete(key)
                return {"status": "ok", "key": key, "deleted": bool(removed)}

            if action == "expire":
                ttl = input_data.get("ttl", 3600)
                result = await self._redis.expire(key, int(ttl))
                return {"status": "ok", "key": key, "expire_set": bool(result)}

            return {"error": f"Unknown action '{action}'"}

        except Exception as exc:
            logger.error("redis_error", action=action, error=str(exc))
            return {"error": str(exc)}
