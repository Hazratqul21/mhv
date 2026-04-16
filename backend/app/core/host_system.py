from __future__ import annotations

import asyncio
import platform
import shutil
import time
from dataclasses import dataclass, field
from typing import Any

from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class GPUInfo:
    index: int = 0
    name: str = ""
    vram_total_mb: int = 0
    vram_used_mb: int = 0
    vram_free_mb: int = 0
    temperature: int = 0
    utilization: int = 0
    power_draw_w: float = 0.0
    power_limit_w: float = 0.0


@dataclass
class SystemSnapshot:
    timestamp: float = 0.0
    # CPU
    cpu_percent: float = 0.0
    cpu_count: int = 0
    cpu_freq_mhz: float = 0.0
    # Memory
    ram_total_gb: float = 0.0
    ram_used_gb: float = 0.0
    ram_percent: float = 0.0
    swap_total_gb: float = 0.0
    swap_used_gb: float = 0.0
    # Disk
    disk_total_gb: float = 0.0
    disk_used_gb: float = 0.0
    disk_free_gb: float = 0.0
    disk_percent: float = 0.0
    # GPU
    gpus: list[GPUInfo] = field(default_factory=list)
    # Network
    net_bytes_sent: int = 0
    net_bytes_recv: int = 0
    # OS
    os_name: str = ""
    os_version: str = ""
    hostname: str = ""
    uptime_hours: float = 0.0
    # Processes
    process_count: int = 0
    top_cpu_processes: list[dict[str, Any]] = field(default_factory=list)
    top_mem_processes: list[dict[str, Any]] = field(default_factory=list)


class HostSystem:
    """Provides real-time awareness of the host PC — MIYA's physical body.

    Periodically snapshots CPU, RAM, disk, GPU, network, and OS state so
    that agents can make resource-aware decisions (e.g. GPU Manager deciding
    which model to load/evict, System Admin cleaning disk, etc.).
    """

    def __init__(self, poll_interval: int = 30) -> None:
        self._poll_interval = poll_interval
        self._latest: SystemSnapshot = SystemSnapshot()
        self._history: list[SystemSnapshot] = []
        self._max_history = 120  # last 60 min at 30s intervals
        self._running = False

    async def start(self) -> None:
        self._running = True
        asyncio.create_task(self._monitor_loop())
        log.info("host_system_monitoring_started", interval=self._poll_interval)

    async def stop(self) -> None:
        self._running = False

    async def _monitor_loop(self) -> None:
        while self._running:
            try:
                snap = await self.snapshot()
                self._latest = snap
                self._history.append(snap)
                if len(self._history) > self._max_history:
                    self._history = self._history[-self._max_history:]
            except Exception:
                log.exception("host_snapshot_failed")
            await asyncio.sleep(self._poll_interval)

    async def snapshot(self) -> SystemSnapshot:
        snap = SystemSnapshot(timestamp=time.time())
        await asyncio.gather(
            self._collect_cpu(snap),
            self._collect_memory(snap),
            self._collect_disk(snap),
            self._collect_gpu(snap),
            self._collect_network(snap),
            self._collect_os(snap),
            self._collect_processes(snap),
            return_exceptions=True,
        )
        return snap

    async def _collect_cpu(self, snap: SystemSnapshot) -> None:
        def _run() -> None:
            try:
                import psutil

                snap.cpu_percent = psutil.cpu_percent(interval=0.5)
                snap.cpu_count = psutil.cpu_count(logical=True) or 0
                freq = psutil.cpu_freq()
                if freq:
                    snap.cpu_freq_mhz = freq.current
            except ImportError:
                pass

        await asyncio.to_thread(_run)

    async def _collect_memory(self, snap: SystemSnapshot) -> None:
        def _run() -> None:
            try:
                import psutil

                vm = psutil.virtual_memory()
                snap.ram_total_gb = round(vm.total / (1024 ** 3), 2)
                snap.ram_used_gb = round(vm.used / (1024 ** 3), 2)
                snap.ram_percent = vm.percent
                sw = psutil.swap_memory()
                snap.swap_total_gb = round(sw.total / (1024 ** 3), 2)
                snap.swap_used_gb = round(sw.used / (1024 ** 3), 2)
            except ImportError:
                pass

        await asyncio.to_thread(_run)

    async def _collect_disk(self, snap: SystemSnapshot) -> None:
        def _run() -> None:
            usage = shutil.disk_usage("/")
            snap.disk_total_gb = round(usage.total / (1024 ** 3), 2)
            snap.disk_used_gb = round(usage.used / (1024 ** 3), 2)
            snap.disk_free_gb = round(usage.free / (1024 ** 3), 2)
            snap.disk_percent = round((usage.used / usage.total) * 100, 1)

        await asyncio.to_thread(_run)

    async def _collect_gpu(self, snap: SystemSnapshot) -> None:
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,"
                "temperature.gpu,utilization.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode != 0:
                return

            for line in stdout.decode().strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 9:
                    continue
                gpu = GPUInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    vram_total_mb=int(parts[2]),
                    vram_used_mb=int(parts[3]),
                    vram_free_mb=int(parts[4]),
                    temperature=int(parts[5]),
                    utilization=int(parts[6]),
                    power_draw_w=float(parts[7]),
                    power_limit_w=float(parts[8]),
                )
                snap.gpus.append(gpu)
        except FileNotFoundError:
            pass
        except Exception:
            log.debug("gpu_collection_failed", exc_info=True)

    async def _collect_network(self, snap: SystemSnapshot) -> None:
        def _run() -> None:
            try:
                import psutil

                net = psutil.net_io_counters()
                if net:
                    snap.net_bytes_sent = net.bytes_sent
                    snap.net_bytes_recv = net.bytes_recv
            except ImportError:
                pass

        await asyncio.to_thread(_run)

    async def _collect_os(self, snap: SystemSnapshot) -> None:
        def _run() -> None:
            snap.os_name = platform.system()
            snap.os_version = platform.release()
            snap.hostname = platform.node()
            try:
                import psutil

                boot = psutil.boot_time()
                snap.uptime_hours = round((time.time() - boot) / 3600, 2)
            except ImportError:
                pass

        await asyncio.to_thread(_run)

    async def _collect_processes(self, snap: SystemSnapshot) -> None:
        def _run() -> None:
            try:
                import psutil

                procs = []
                for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
                    try:
                        info = p.info
                        if info:
                            procs.append(info)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                snap.process_count = len(procs)
                by_cpu = sorted(procs, key=lambda x: x.get("cpu_percent", 0) or 0, reverse=True)
                snap.top_cpu_processes = by_cpu[:10]
                by_mem = sorted(procs, key=lambda x: x.get("memory_percent", 0) or 0, reverse=True)
                snap.top_mem_processes = by_mem[:10]
            except ImportError:
                pass

        await asyncio.to_thread(_run)

    @property
    def latest(self) -> SystemSnapshot:
        return self._latest

    def get_summary(self) -> dict[str, Any]:
        s = self._latest
        gpu_info = []
        for g in s.gpus:
            gpu_info.append({
                "name": g.name,
                "vram": f"{g.vram_used_mb}/{g.vram_total_mb}MB",
                "temp": f"{g.temperature}°C",
                "util": f"{g.utilization}%",
                "power": f"{g.power_draw_w:.0f}/{g.power_limit_w:.0f}W",
            })

        return {
            "cpu": f"{s.cpu_percent}% ({s.cpu_count} cores @ {s.cpu_freq_mhz:.0f}MHz)",
            "ram": f"{s.ram_used_gb}/{s.ram_total_gb}GB ({s.ram_percent}%)",
            "disk": f"{s.disk_used_gb}/{s.disk_total_gb}GB ({s.disk_percent}% used, {s.disk_free_gb}GB free)",
            "gpus": gpu_info,
            "os": f"{s.os_name} {s.os_version} ({s.hostname})",
            "uptime": f"{s.uptime_hours:.1f}h",
            "processes": s.process_count,
        }

    def get_gpu_summary(self) -> list[dict[str, Any]]:
        result = []
        for g in self._latest.gpus:
            result.append({
                "index": g.index,
                "name": g.name,
                "vram_total_mb": g.vram_total_mb,
                "vram_used_mb": g.vram_used_mb,
                "vram_free_mb": g.vram_free_mb,
                "temperature": g.temperature,
                "utilization": g.utilization,
                "power_draw_w": g.power_draw_w,
                "power_limit_w": g.power_limit_w,
                "vram_percent": round(g.vram_used_mb / max(g.vram_total_mb, 1) * 100, 1),
            })
        return result

    def is_gpu_available(self, min_vram_mb: int = 2000) -> bool:
        return any(g.vram_free_mb >= min_vram_mb for g in self._latest.gpus)

    def get_history(self, last_n: int = 10) -> list[dict[str, Any]]:
        entries = self._history[-last_n:]
        return [
            {
                "timestamp": s.timestamp,
                "cpu": s.cpu_percent,
                "ram": s.ram_percent,
                "disk": s.disk_percent,
                "gpu_util": s.gpus[0].utilization if s.gpus else 0,
                "gpu_temp": s.gpus[0].temperature if s.gpus else 0,
                "gpu_vram_pct": (
                    round(s.gpus[0].vram_used_mb / max(s.gpus[0].vram_total_mb, 1) * 100, 1)
                    if s.gpus
                    else 0
                ),
            }
            for s in entries
        ]
