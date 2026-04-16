from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable, Optional
from uuid import uuid4

from app.utils.logger import get_logger

log = get_logger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskItem:
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    name: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0
    result: Any = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    @property
    def duration_ms(self) -> Optional[int]:
        if self.started_at and self.completed_at:
            return int((self.completed_at - self.started_at) * 1000)
        return None


class TaskQueue:
    """Priority-based async task queue with concurrency control."""

    def __init__(self, max_concurrent: int = 3) -> None:
        self._queue: asyncio.PriorityQueue[tuple[int, float, TaskItem, Callable]] = (
            asyncio.PriorityQueue()
        )
        self._tasks: dict[str, TaskItem] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None

    async def submit(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        name: str = "",
        priority: int = 0,
        **kwargs: Any,
    ) -> str:
        item = TaskItem(name=name or func.__name__, priority=priority)
        self._tasks[item.id] = item

        async def _wrapped() -> Any:
            return await func(*args, **kwargs)

        await self._queue.put((-priority, item.created_at, item, _wrapped))
        log.info("task_submitted", task_id=item.id, name=item.name, priority=priority)
        return item.id

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        log.info("task_queue_started")

    async def stop(self) -> None:
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        log.info("task_queue_stopped")

    async def _worker_loop(self) -> None:
        while self._running:
            try:
                _, _, item, func = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            asyncio.create_task(self._execute(item, func))
            self._queue.task_done()

    async def _execute(self, item: TaskItem, func: Callable) -> None:
        async with self._semaphore:
            item.status = TaskStatus.RUNNING
            item.started_at = time.time()
            log.info("task_started", task_id=item.id, name=item.name)

            try:
                item.result = await func()
                item.status = TaskStatus.COMPLETED
            except Exception as exc:
                item.status = TaskStatus.FAILED
                item.error = str(exc)
                log.error("task_failed", task_id=item.id, error=str(exc))
            finally:
                item.completed_at = time.time()

    def get_task(self, task_id: str) -> Optional[TaskItem]:
        return self._tasks.get(task_id)

    def get_all(self, status: Optional[TaskStatus] = None) -> list[TaskItem]:
        items = list(self._tasks.values())
        if status:
            items = [t for t in items if t.status == status]
        return sorted(items, key=lambda t: t.created_at, reverse=True)

    def cancel(self, task_id: str) -> bool:
        item = self._tasks.get(task_id)
        if item and item.status == TaskStatus.PENDING:
            item.status = TaskStatus.CANCELLED
            return True
        return False
