from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Optional

import redis.asyncio as aioredis

from app.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_TTL = 3600  # 1 hour


def _redis_url_for_log(url: str) -> str:
    """Hide user:password in redis://…@host/… for logs."""
    if "@" not in url or "://" not in url:
        return url
    try:
        scheme, rest = url.split("://", 1)
        _creds, hostpart = rest.rsplit("@", 1)
        return f"{scheme}://***@{hostpart}"
    except ValueError:
        return url


class CacheManager:
    """Async Redis cache with JSON serialization for complex values."""

    def __init__(self, redis_url: str | None = None, prefix: str = "miya") -> None:
        self._settings = get_settings()
        self._url = redis_url or self._settings.redis_url
        self._prefix = prefix
        self._redis: aioredis.Redis | None = None

    def _key(self, key: str) -> str:
        return f"{self._prefix}:{key}"

    async def connect(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(
                self._url,
                decode_responses=True,
                socket_connect_timeout=5,
            )
            await self._redis.ping()
            log.info("redis_connected", url=_redis_url_for_log(self._url))
        return self._redis

    async def _conn(self) -> aioredis.Redis:
        if self._redis is None:
            return await self.connect()
        return self._redis

    async def get(self, key: str) -> Any | None:
        """Retrieve a value. Returns ``None`` on miss."""
        r = await self._conn()
        raw = await r.get(self._key(key))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = DEFAULT_TTL,
    ) -> None:
        """Store a value with optional TTL (seconds)."""
        r = await self._conn()
        serialized = json.dumps(value, default=str)
        await r.set(self._key(key), serialized, ex=ttl if ttl > 0 else None)

    async def delete(self, key: str) -> bool:
        """Delete a key. Returns True if the key existed."""
        r = await self._conn()
        return bool(await r.delete(self._key(key)))

    async def exists(self, key: str) -> bool:
        """Check whether a key exists."""
        r = await self._conn()
        return bool(await r.exists(self._key(key)))

    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Awaitable[Any]],
        ttl: int = DEFAULT_TTL,
    ) -> Any:
        """Return cached value or call *factory* to compute, cache, and return it."""
        cached = await self.get(key)
        if cached is not None:
            return cached
        value = await factory()
        await self.set(key, value, ttl=ttl)
        return value

    async def increment(self, key: str, amount: int = 1) -> int:
        """Atomically increment an integer key."""
        r = await self._conn()
        return await r.incrby(self._key(key), amount)

    async def expire(self, key: str, ttl: int) -> bool:
        """Set a TTL on an existing key."""
        r = await self._conn()
        return bool(await r.expire(self._key(key), ttl))

    async def flush_prefix(self) -> int:
        """Delete all keys matching the configured prefix."""
        r = await self._conn()
        pattern = f"{self._prefix}:*"
        count = 0
        async for key in r.scan_iter(match=pattern, count=200):
            await r.delete(key)
            count += 1
        log.info("cache_flushed", prefix=self._prefix, count=count)
        return count

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
            self._redis = None
            log.info("redis_disconnected")
