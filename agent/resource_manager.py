"""
ResourceManager — probes, budgets, and monitors container resources.

Usage:
    rm = ResourceManager()
    print(rm.probe())        # {vram_gb, cpu_cores, ram_gb, gpu_available}
    print(rm.can_spawn(...)) # True/False based on remaining budget
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    """Point-in-time resource availability."""
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    cpu_cores: float = 1.0
    ram_total_gb: float = 1.0
    ram_free_gb: float = 1.0
    gpu_available: bool = False
    timestamp: float = 0.0


@dataclass
class WorkerResourceDemand:
    """Resource requirements for a single worker process."""
    worker_type: str          # "deterministic" | "local" | "fireworks"
    gpu_vram_gb: float = 0.0
    cpu_cores: float = 0.5
    ram_gb: float = 0.5
    count: int = 1            # how many of this type


class ResourceManager:
    """Probes and tracks container resources for worker budgeting."""

    def __init__(self):
        self._last_probe: Optional[ResourceSnapshot] = None
        self._probe_interval = 5.0  # seconds between refreshes
        self._gpu_available = self._detect_gpu()

    def _detect_gpu(self) -> bool:
        """Check if GPU (NVIDIA) is available."""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                logger.info("GPU detected: %s", result.stdout.strip().split('\n')[0])
                return True
        except Exception:
            pass
        logger.info("No GPU detected — running CPU-only")
        return False

    def probe(self, force: bool = False) -> ResourceSnapshot:
        """Gather current resource availability."""
        now = time.monotonic()
        if not force and self._last_probe and (now - self._last_probe.timestamp) < self._probe_interval:
            return self._last_probe

        snapshot = ResourceSnapshot(gpu_available=self._gpu_available, timestamp=now)

        # GPU VRAM
        if self._gpu_available:
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.total,memory.free",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if lines and ',' in lines[0]:
                        total_str, free_str = lines[0].split(',')
                        snapshot.vram_total_gb = float(total_str.strip()) / 1024
                        snapshot.vram_free_gb = float(free_str.strip()) / 1024
            except Exception as e:
                logger.warning("VRAM probe failed: %s", e)

        # CPU
        try:
            import os
            snapshot.cpu_cores = os.cpu_count() or 1
        except Exception:
            pass

        # RAM
        try:
            import psutil
            mem = psutil.virtual_memory()
            snapshot.ram_total_gb = mem.total / (1024**3)
            snapshot.ram_free_gb = mem.available / (1024**3)
        except ImportError:
            pass

        self._last_probe = snapshot
        return snapshot

    def can_spawn(self, demand: WorkerResourceDemand, existing_usage: Optional[ResourceSnapshot] = None) -> tuple[bool, str]:
        """Check if we can spawn {demand.count} workers of {demand.worker_type}.

        Returns (can_spawn, reason).
        """
        current = existing_usage or self.probe()

        total_vram_needed = demand.gpu_vram_gb * demand.count
        total_cpu_needed = demand.cpu_cores * demand.count
        total_ram_needed = demand.ram_gb * demand.count

        if demand.gpu_vram_gb > 0 and not current.gpu_available:
            return False, f"No GPU available for {demand.worker_type}"

        if demand.gpu_vram_gb > 0 and total_vram_needed > current.vram_free_gb:
            return False, (
                f"VRAM insufficient: need {total_vram_needed:.1f}GB, "
                f"have {current.vram_free_gb:.1f}GB free"
            )

        if total_cpu_needed > current.cpu_cores:
            return False, (
                f"CPU insufficient: need {total_cpu_needed:.0f} cores, "
                f"have {current.cpu_cores:.0f} total"
            )

        if total_ram_needed > current.ram_free_gb:
            return False, (
                f"RAM insufficient: need {total_ram_needed:.1f}GB, "
                f"have {current.ram_free_gb:.1f}GB free"
            )

        return True, ""

    def budget_workers(self, worker_demands: list[WorkerResourceDemand]) -> list[WorkerResourceDemand]:
        """Given desired workers, return only those that fit within resources.

        Greedy allocation by priority order (local_llm first, then deterministic, then fireworks).
        """
        current = self.probe()

        # Simulate resource consumption
        allocated: list[WorkerResourceDemand] = []

        for demand in sorted(worker_demands, key=self._priority):
            can, reason = self.can_spawn(demand, current)
            if not can:
                logger.warning("Cannot allocate %s: %s", demand.worker_type, reason)
                continue

            # Deduct resources
            if demand.gpu_vram_gb > 0:
                current.vram_free_gb -= demand.gpu_vram_gb * demand.count
            current.cpu_cores -= demand.cpu_cores * demand.count
            current.ram_free_gb -= demand.ram_gb * demand.count

            allocated.append(demand)

        if len(allocated) < len(worker_demands):
            logger.info(
                "Budgeted %d/%d worker types (%.1fGB VRAM, %.0f cores, %.1fGB RAM remaining)",
                len(allocated), len(worker_demands),
                current.vram_free_gb, current.cpu_cores, current.ram_free_gb,
            )

        return allocated

    @staticmethod
    def _priority(demand: WorkerResourceDemand) -> int:
        """Lower number = higher priority for allocation."""
        order = {"local": 0, "deterministic": 1, "fireworks": 2}
        return order.get(demand.worker_type, 99)
