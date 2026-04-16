"""Async job runner for scheduled tasks.

Runs registered coroutines at fixed intervals (e.g. daily, weekly).
"""

from __future__ import annotations

import asyncio
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ScheduledJob:
    name: str
    coroutine_factory: Callable[[], Coroutine[Any, Any, Any]]
    interval_seconds: int
    last_run: float = 0.0
    run_count: int = 0
    last_error: str = ""
    enabled: bool = True


class JobRunner:
    """Lightweight async scheduler for periodic background tasks."""

    def __init__(self, check_interval: int = 60) -> None:
        self._jobs: dict[str, ScheduledJob] = {}
        self._check_interval = check_interval
        self._task: asyncio.Task | None = None
        self._running = False
        self._inflight: dict[str, asyncio.Task[Any]] = {}

    def register(
        self,
        name: str,
        coro_factory: Callable[[], Coroutine[Any, Any, Any]],
        interval_seconds: int,
        run_immediately: bool = False,
    ) -> None:
        """Register a coroutine to run at a fixed interval."""
        job = ScheduledJob(
            name=name,
            coroutine_factory=coro_factory,
            interval_seconds=interval_seconds,
        )
        if run_immediately:
            job.last_run = 0.0
        else:
            job.last_run = time.time()

        self._jobs[name] = job
        log.info("job_registered", name=name, interval=interval_seconds)

    def start(self) -> None:
        """Start the background scheduler loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        log.info("job_runner_started", jobs=list(self._jobs.keys()))

    async def stop(self) -> None:
        """Gracefully stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        log.info("job_runner_stopped")

    async def _loop(self) -> None:
        while self._running:
            now = time.time()

            for job in self._jobs.values():
                if not job.enabled:
                    continue

                elapsed = now - job.last_run
                if elapsed >= job.interval_seconds:
                    inflight = self._inflight.get(job.name)
                    if inflight is not None and not inflight.done():
                        continue
                    t = asyncio.create_task(self._run_job(job))
                    self._inflight[job.name] = t
                    t.add_done_callback(
                        lambda _t, name=job.name: self._inflight.pop(name, None)
                    )

            await asyncio.sleep(self._check_interval)

    async def _run_job(self, job: ScheduledJob) -> None:
        job.last_run = time.time()
        log.info("job_starting", name=job.name, run=job.run_count + 1)

        try:
            await job.coroutine_factory()
            job.run_count += 1
            job.last_error = ""
            log.info("job_completed", name=job.name, run=job.run_count)
        except Exception as exc:
            job.last_error = str(exc)
            log.error("job_failed", name=job.name, error=str(exc))
            log.debug(traceback.format_exc())

    def get_status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "jobs": {
                name: {
                    "enabled": job.enabled,
                    "interval_seconds": job.interval_seconds,
                    "run_count": job.run_count,
                    "last_run": job.last_run,
                    "next_run": job.last_run + job.interval_seconds,
                    "last_error": job.last_error,
                }
                for name, job in self._jobs.items()
            },
        }

    def enable_job(self, name: str) -> bool:
        if name in self._jobs:
            self._jobs[name].enabled = True
            return True
        return False

    def disable_job(self, name: str) -> bool:
        if name in self._jobs:
            self._jobs[name].enabled = False
            return True
        return False

    async def run_now(self, name: str) -> bool:
        """Manually trigger a job immediately."""
        if name not in self._jobs:
            return False
        await self._run_job(self._jobs[name])
        return True
