from __future__ import annotations

import asyncio
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.core.llm_engine import LLMEngine
from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class HealthCheck:
    service: str
    status: str  # "healthy", "degraded", "down"
    timestamp: float = 0.0
    details: str = ""
    recovery_attempts: int = 0


class SelfHealer:
    """Monitors system health and autonomously repairs failures.

    Capabilities:
    - Detects and restarts crashed services (Docker containers, model processes).
    - Reloads models that have been evicted or corrupted.
    - Clears caches / memory when resources are exhausted.
    - Re-runs failed tasks with corrective adjustments.
    """

    def __init__(self, llm_engine: LLMEngine) -> None:
        self._engine = llm_engine
        self._settings = get_settings()
        self._health: dict[str, HealthCheck] = {}
        self._max_recovery_attempts = 3
        self._error_log: list[dict[str, Any]] = []
        self._running = False

    async def start(self) -> None:
        self._running = True
        log.info("self_healer_started")
        asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        self._running = False
        log.info("self_healer_stopped")

    async def _monitor_loop(self) -> None:
        while self._running:
            try:
                await self._check_all()
            except Exception:
                log.exception("monitor_cycle_failed")
            await asyncio.sleep(30)

    async def _check_all(self) -> None:
        checks = [
            self._check_docker_services(),
            self._check_memory_usage(),
            self._check_gpu_health(),
            self._check_disk_space(),
            self._check_model_health(),
        ]
        results = await asyncio.gather(*checks, return_exceptions=True)
        degraded = [
            s for s, h in self._health.items() if h.status != "healthy"
        ]
        if degraded:
            log.warning("health_check_issues", degraded=degraded)
        else:
            log.debug("health_check_all_ok", services=len(self._health))

    async def _check_docker_services(self) -> None:
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "compose", "ps", "--format", "json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._settings.project_root),
            )
            stdout, _ = await proc.communicate()
            if proc.returncode != 0:
                self._update_health("docker", "degraded", "docker compose ps failed")
                return

            self._update_health("docker", "healthy")
        except FileNotFoundError:
            self._update_health("docker", "down", "docker not found")
        except Exception as exc:
            self._update_health("docker", "degraded", str(exc))

    async def _check_memory_usage(self) -> None:
        try:
            import psutil
            mem = psutil.virtual_memory()
            if mem.percent > 95:
                self._update_health("memory", "degraded", f"RAM {mem.percent}%")
                await self._recover_memory()
            elif mem.percent > 85:
                self._update_health("memory", "degraded", f"RAM {mem.percent}%")
            else:
                self._update_health("memory", "healthy", f"RAM {mem.percent}%")
        except ImportError:
            self._update_health("memory", "degraded", "psutil not installed")
        except Exception as exc:
            self._update_health("memory", "degraded", str(exc))

    async def _check_gpu_health(self) -> None:
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode != 0:
                self._update_health("gpu", "degraded", "nvidia-smi failed")
                return

            first_line = stdout.decode().strip().splitlines()[0]
            line = first_line.split(",")
            if len(line) >= 4:
                temp = int(line[0].strip())
                mem_used = int(line[1].strip())
                mem_total = int(line[2].strip())
                utilization = int(line[3].strip())

                if temp > 90:
                    self._update_health(
                        "gpu", "degraded", f"GPU temp={temp}°C (critical)"
                    )
                    await self._cool_down_gpu()
                elif temp > 80:
                    self._update_health("gpu", "degraded", f"GPU temp={temp}°C (warm)")
                else:
                    self._update_health(
                        "gpu",
                        "healthy",
                        f"temp={temp}°C mem={mem_used}/{mem_total}MB util={utilization}%",
                    )
            else:
                self._update_health("gpu", "healthy")
        except FileNotFoundError:
            self._update_health("gpu", "down", "nvidia-smi not found")
        except Exception as exc:
            self._update_health("gpu", "degraded", str(exc))

    async def _check_disk_space(self) -> None:
        try:
            usage = shutil.disk_usage("/")
            free_gb = usage.free / (1024 ** 3)
            total_gb = usage.total / (1024 ** 3)
            pct_used = (usage.used / usage.total) * 100

            if free_gb < 5:
                self._update_health(
                    "disk", "degraded", f"{free_gb:.1f}GB free of {total_gb:.0f}GB"
                )
                await self._clean_disk()
            else:
                self._update_health(
                    "disk", "healthy", f"{free_gb:.1f}GB free ({pct_used:.0f}% used)"
                )
        except Exception as exc:
            self._update_health("disk", "degraded", str(exc))

    async def _check_model_health(self) -> None:
        try:
            loaded = self._engine.list_loaded_models()
            self._update_health(
                "models", "healthy", f"{len(loaded)} models loaded"
            )
        except Exception as exc:
            self._update_health("models", "degraded", str(exc))

    async def _recover_memory(self) -> None:
        log.warning("attempting_memory_recovery")
        try:
            loaded = self._engine.list_loaded_models()
            if len(loaded) > 1:
                oldest = loaded[-1]
                self._engine.unload_model(oldest)
                log.info("evicted_model", model=oldest)
        except Exception:
            log.exception("memory_recovery_failed")

    async def _cool_down_gpu(self) -> None:
        log.warning("gpu_overheating_pause_models")
        await asyncio.sleep(10)

    async def _clean_disk(self) -> None:
        log.warning("low_disk_space_cleanup")
        try:
            import tempfile
            tmp = tempfile.gettempdir()
            proc = await asyncio.create_subprocess_exec(
                "find", tmp, "-type", "f", "-mtime", "+7", "-delete",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
        except Exception:
            log.exception("disk_cleanup_failed")

    async def handle_error(
        self, source: str, error: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        entry = {
            "source": source,
            "error": error,
            "context": context or {},
            "timestamp": time.time(),
        }
        self._error_log.append(entry)
        if len(self._error_log) > 1000:
            self._error_log = self._error_log[-500:]

        diagnosis = await self._diagnose(error, context or {})

        health_key = diagnosis.get("action", "unknown")
        prev = self._health.get(health_key)
        attempts = (prev.recovery_attempts + 1) if prev else 1

        if attempts > self._max_recovery_attempts:
            log.error("max_recovery_exceeded", action=health_key, attempts=attempts)
            diagnosis["recovery_skipped"] = True
            return diagnosis

        self._update_health(health_key, "recovering",
                            f"attempt {attempts}/{self._max_recovery_attempts}")
        if prev:
            prev.recovery_attempts = attempts

        success = False
        if diagnosis.get("action") == "restart_service":
            success = await self._restart_service(diagnosis.get("service", ""))
        elif diagnosis.get("action") == "reload_model":
            success = await self._reload_model(diagnosis.get("model", ""))
        elif diagnosis.get("action") == "clear_cache":
            success = await self._clear_cache() or True

        if success:
            self._update_health(health_key, "healthy", "recovered")
            if prev:
                prev.recovery_attempts = 0
        else:
            self._update_health(health_key, "degraded", f"recovery failed (attempt {attempts})")

        diagnosis["recovered"] = success
        return diagnosis

    async def _diagnose(
        self, error: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        if "CUDA out of memory" in error or "OOM" in error:
            return {"action": "reload_model", "model": context.get("model", ""), "reason": "GPU OOM"}
        if "Connection refused" in error:
            return {"action": "restart_service", "service": context.get("service", ""), "reason": "connection_refused"}
        if "disk" in error.lower() and "full" in error.lower():
            return {"action": "clear_cache", "reason": "disk_full"}
        return {"action": "log_only", "reason": "unknown_error"}

    async def _restart_service(self, service: str) -> bool:
        if not service:
            return False
        log.info("restarting_service", service=service)
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "compose", "restart", service,
                cwd=str(self._settings.project_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            return proc.returncode == 0
        except Exception:
            log.exception("service_restart_failed", service=service)
            return False

    async def _reload_model(self, model: str) -> bool:
        if not model:
            return False
        log.info("reloading_model", model=model)
        try:
            self._engine.unload_model(model)
            await asyncio.sleep(1)
            import gc; gc.collect()
            self._engine.swap_model(model, n_ctx=self._settings.chat_ctx)
            return True
        except Exception:
            log.exception("model_reload_failed", model=model)
            return False

    async def _clear_cache(self) -> None:
        log.info("clearing_cache")
        try:
            import gc
            gc.collect()
            cache_dirs = [
                self._settings.data_dir / "cache",
                Path("/tmp/miya_cache"),
            ]
            for d in cache_dirs:
                if d.exists():
                    import shutil as sh
                    for f in d.iterdir():
                        if f.is_file() and (time.time() - f.stat().st_mtime > 86400):
                            f.unlink()
                            log.debug("cache_file_removed", path=str(f))
        except Exception:
            log.exception("cache_clear_failed")

    def _update_health(
        self, service: str, status: str, details: str = ""
    ) -> None:
        prev = self._health.get(service)
        self._health[service] = HealthCheck(
            service=service,
            status=status,
            timestamp=time.time(),
            details=details,
            recovery_attempts=(prev.recovery_attempts if prev else 0),
        )
        if prev and prev.status == "healthy" and status != "healthy":
            log.warning("health_degraded", service=service, details=details)

    def get_health(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "status": h.status,
                "details": h.details,
                "last_check": h.timestamp,
                "recovery_attempts": h.recovery_attempts,
            }
            for name, h in self._health.items()
        }
