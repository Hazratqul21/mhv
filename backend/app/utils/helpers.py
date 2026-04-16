from __future__ import annotations

import time
import uuid
import hashlib
from contextlib import asynccontextmanager
from typing import AsyncIterator, Any


def generate_id(prefix: str = "") -> str:
    raw = uuid.uuid4().hex[:16]
    return f"{prefix}{raw}" if prefix else raw


def generate_session_id() -> str:
    return generate_id(prefix="sess_")


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class Timer:
    """Measure elapsed wall-clock time in milliseconds."""

    def __init__(self) -> None:
        self._start: float = 0
        self._end: float = 0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed_ms(self) -> int:
        end = self._end or time.perf_counter()
        return int((end - self._start) * 1000)


@asynccontextmanager
async def async_timer() -> AsyncIterator[Timer]:
    t = Timer()
    t._start = time.perf_counter()
    try:
        yield t
    finally:
        t._end = time.perf_counter()


def truncate(text: str, max_tokens: int = 2048, chars_per_token: float = 3.5) -> str:
    max_chars = int(max_tokens * chars_per_token)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


def format_tool_result(name: str, result: Any, success: bool = True) -> dict:
    return {
        "tool": name,
        "success": success,
        "result": str(result) if not isinstance(result, (dict, list)) else result,
    }
