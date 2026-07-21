"""Tests for ResourceManager."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.resource_manager import ResourceManager, WorkerResourceDemand, ResourceSnapshot


def test_probe_returns_something():
    rm = ResourceManager()
    snap = rm.probe()
    assert snap.cpu_cores >= 1
    assert snap.ram_total_gb > 0
    print(f"CPU: {snap.cpu_cores} cores, RAM: {snap.ram_total_gb:.1f}GB, GPU: {snap.gpu_available}")
    assert snap.timestamp > 0


def test_can_spawn_deterministic():
    rm = ResourceManager()
    can, reason = rm.can_spawn(WorkerResourceDemand("deterministic", cpu_cores=0.1, ram_gb=0.1))
    assert can, f"Should be able to spawn deterministic: {reason}"


def test_can_spawn_local_if_gpu():
    rm = ResourceManager()
    can, reason = rm.can_spawn(WorkerResourceDemand("local", gpu_vram_gb=1.5, cpu_cores=1.0, ram_gb=1.0))
    if rm._gpu_available:
        assert can, f"GPU available but can't spawn local: {reason}"
    else:
        print(f"No GPU — local worker skipped: {reason}")


def test_budget_workers():
    rm = ResourceManager()
    demands = [
        WorkerResourceDemand("deterministic", cpu_cores=0.1, ram_gb=0.1, count=4),
        WorkerResourceDemand("local", gpu_vram_gb=1.5, cpu_cores=1.0, ram_gb=1.0, count=1),
        WorkerResourceDemand("fireworks", cpu_cores=0.2, ram_gb=0.2, count=2),
    ]
    allocated = rm.budget_workers(demands)
    assert len(allocated) > 0, "Should allocate at least some workers"
    print(f"Allocated {len(allocated)}/{len(demands)} worker types")


def test_can_spawn_insufficient_resources():
    """Verify can_spawn returns False when resources are exhausted."""
    snap = ResourceSnapshot(
        vram_total_gb=4.0,
        vram_free_gb=2.0,
        cpu_cores=2,
        ram_total_gb=4.0,
        ram_free_gb=2.0,
        gpu_available=True,
        timestamp=0.0,
    )
    rm = ResourceManager()
    # Demand more VRAM than available
    can, reason = rm.can_spawn(
        WorkerResourceDemand("local", gpu_vram_gb=10.0, cpu_cores=1.0, ram_gb=1.0),
        existing_usage=snap,
    )
    assert not can, "Should reject demand exceeding VRAM"
    assert "VRAM insufficient" in reason


def test_can_spawn_no_gpu():
    """Verify can_spawn returns False for GPU demand when no GPU."""
    snap = ResourceSnapshot(
        vram_total_gb=0.0,
        vram_free_gb=0.0,
        cpu_cores=4,
        ram_total_gb=8.0,
        ram_free_gb=8.0,
        gpu_available=False,
        timestamp=0.0,
    )
    rm = ResourceManager()
    can, reason = rm.can_spawn(
        WorkerResourceDemand("local", gpu_vram_gb=1.5, cpu_cores=1.0, ram_gb=1.0),
        existing_usage=snap,
    )
    assert not can, "Should reject GPU demand without GPU"
    assert "No GPU available" in reason


if __name__ == "__main__":
    test_probe_returns_something()
    test_can_spawn_deterministic()
    test_can_spawn_local_if_gpu()
    test_budget_workers()
    test_can_spawn_insufficient_resources()
    test_can_spawn_no_gpu()
    print("\nAll ResourceManager tests passed!")
