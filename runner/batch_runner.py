#!/usr/bin/env python3
"""Parallel task orchestration across multiple Pipeline instances.

Worker model: multiprocessing.ProcessPoolExecutor with per-task timeout.
Each task is submitted as its own future and enforced with
future.result(timeout=30) for a hard per-task deadline. A global deadline
(default 600s) limits wall-clock time across all tasks.

Graceful degradation: if every worker crashes the fallback runs tasks
sequentially in the main process.

Usage:
    from runner.batch_runner import BatchRunner, run_parallel

    runner = BatchRunner(config=cfg, n_workers=2)
    results = runner.run(tasks, deadline_s=600.0)

    # Or convenience wrapper:
    results = run_parallel(tasks, n_workers=2)
"""

from __future__ import annotations

import concurrent.futures
import logging
import sys
import time
from typing import Any, Optional

from agent import PipelineConfig

logger = logging.getLogger("batch_runner")


# ── Memory budget estimator ────────────────────────────────────────────────


def _compute_workers(max_ram_gb: float = 4.0, model_gb: float = 1.1) -> int:
    """Estimate max workers that fit in the RAM budget.

    Overhead per worker includes ~0.4 GB for the Python runtime + libraries
    plus the model itself. A 0.5 GB buffer is reserved for the OS.

    Args:
        max_ram_gb: Total available RAM in GB.
        model_gb: Estimated model memory footprint in GB.

    Returns:
        Safe number of concurrent workers (1..4).
    """
    overhead = 0.4 + model_gb
    available = max_ram_gb - 0.5
    if available <= 0:
        return 1
    return max(1, min(4, int(available / overhead)))


# ── Worker entrypoint (module-level for pickling) ──────────────────────────


def _worker_process(
    cfg_dict: dict,
    task: dict,
    worker_id: int,
) -> dict:
    """Run a single task in a child process.

    Imports Pipeline inline so the class is picklable across process
    boundaries.

    Args:
        cfg_dict: PipelineConfig fields as a plain dict (picklable).
        task: Task dict that must contain ``_idx`` plus either
              ``prompt`` or ``question``.
        worker_id: Integer worker identifier for provenance.

    Returns:
        Result dict with keys:
            _idx, task_id, answer, timing_ms, worker
    """
    from agent import Pipeline, PipelineConfig

    cfg = PipelineConfig(**cfg_dict)
    pipe = Pipeline(cfg)
    try:
        tid = task.get("task_id", f"w{worker_id}_idx_{task.get('_idx', 0)}")
        prompt = task.get("prompt", task.get("question", ""))
        t0 = time.monotonic()
        try:
            answer = pipe.process(prompt)
        except Exception:
            answer = ""
        elapsed_ms = (time.monotonic() - t0) * 1000
        return {
            "_idx": task.get("_idx", 0),
            "task_id": tid,
            "answer": answer,
            "timing_ms": elapsed_ms,
            "worker": worker_id,
        }
    finally:
        pipe.close()


# ── BatchRunner class ──────────────────────────────────────────────────────


class BatchRunner:
    """Orchestrates parallel task execution across multiple Pipeline instances.

    Features:
    - Memory-aware worker count capping (max 4 workers).
    - Per-task timeout enforced via ``future.result(timeout=30)``.
    - Global deadline enforcement (default 600 s).
    - Worker crash detection and logging.
    - Result collection with input-order preservation (sorted by ``_idx``).
    - Graceful degradation: if all workers die, falls back to sequential
      execution in the main process.

    Args:
        config: PipelineConfig instance (or ``None`` for env-var defaults).
        n_workers: Desired worker count (capped by memory budget).
        max_ram_gb: Total available RAM in GB for worker budget calculation.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        n_workers: int = 1,
        max_ram_gb: float = 4.0,
    ):
        self.config = config or PipelineConfig()
        self.max_ram_gb = max_ram_gb
        computed = _compute_workers(max_ram_gb=max_ram_gb)
        self.n_workers = min(n_workers, computed)
        logger.info(
            "BatchRunner(n_workers=%d, max_ram_gb=%.1f, computed_max=%d)",
            n_workers, max_ram_gb, computed,
        )
        # Testing hook: inject a different executor class (e.g. ThreadPoolExecutor)
        self._executor_class: Optional[type] = None

    # ── Public API ────────────────────────────────────────────────────────

    def run(
        self,
        tasks: list[dict],
        deadline_s: float = 600.0,
    ) -> list[dict]:
        """Process tasks and return results in input order.

        Each task is submitted as an independent future so a hard per-task
        timeout of 30 s can be enforced via ``future.result(timeout=30)``.
        A global deadline (``deadline_s``) limits total wall-clock time.

        Args:
            tasks: List of dicts with ``task_id`` and ``prompt`` (or
                   ``question``).
            deadline_s: Global hard deadline in seconds (default 600).

        Returns:
            List of result dicts with keys:
                task_id, answer, timing_ms, worker
        """
        if not tasks:
            return []

        deadline = time.monotonic() + deadline_s

        # Single task — run in main process to avoid spawn overhead.
        if len(tasks) == 1:
            return self._run_single(tasks[0])

        return self._run_parallel(tasks, deadline)

    # ── Internal: single task in main process ─────────────────────────────

    def _run_single(self, task: dict) -> list[dict]:
        """Process a single task in the main process (no spawn overhead)."""
        from agent import Pipeline

        tid = task.get("task_id", "idx_0")
        prompt = task.get("prompt", task.get("question", ""))
        pipe = Pipeline(self.config)
        t0 = time.monotonic()
        try:
            answer = pipe.process(prompt)
        except Exception:
            answer = ""
        elapsed_ms = (time.monotonic() - t0) * 1000
        pipe.close()
        return [{
            "task_id": tid,
            "answer": answer,
            "timing_ms": elapsed_ms,
            "worker": 0,
        }]

    # ── Internal: parallel execution ──────────────────────────────────────

    def _run_parallel(
        self,
        tasks: list[dict],
        deadline: float,
    ) -> list[dict]:
        """Submit each task as an individual future with per-task timeout."""
        n_workers = min(self.n_workers, len(tasks))

        # Serialise config to a picklable dict.
        cfg_dict = {
            k: v
            for k, v in self.config.__dict__.items()
            if not k.startswith("_")
        }

        # Check deadline before spawning workers.
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            logger.warning("Deadline already reached — returning empty results")
            return []
        if remaining_s < 10.0:
            logger.warning(
                "Deadline too close (%.1fs) — running sequentially",
                remaining_s,
            )
            return self._run_sequential(tasks, deadline)

        # Spawn the pool.
        if self._executor_class is not None:
            executor = self._executor_class(max_workers=n_workers)
        else:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=n_workers,
            )

        results: list[dict] = []
        crashed_workers: set[int] = set()
        worker_counter: int = 0
        futures: dict[concurrent.futures.Future, int] = {}

        try:
            # Submit each task as its own future so we can enforce a per-task
            # timeout via future.result(timeout=30).
            for idx, task in enumerate(tasks):
                # Check deadline before submitting the next task.
                if time.monotonic() >= deadline:
                    logger.warning(
                        "Deadline reached at task %d/%d — stopping submission",
                        idx, len(tasks),
                    )
                    break

                # Tag the task with its original index for re-ordering.
                tagged = {**task, "_idx": idx}
                wid = worker_counter % n_workers
                worker_counter += 1
                fut = executor.submit(_worker_process, cfg_dict, tagged, wid)
                futures[fut] = wid

            # Collect results with per-task timeout.
            for fut, wid in futures.items():
                if time.monotonic() >= deadline:
                    logger.warning(
                        "Deadline reached during result collection — "
                        "dropping %d pending future(s)",
                        len([f for f in futures if not f.done()]),
                    )
                    break
                try:
                    res = fut.result(timeout=30)
                    results.append(res)
                except concurrent.futures.TimeoutError:
                    logger.warning(
                        "Worker %d task timed out (>30s) — skipping",
                        wid,
                    )
                    # Cancel the future so it doesn't linger.
                    fut.cancel()
                except Exception as exc:
                    logger.warning(
                        "Worker %d crashed (%s: %s) — continuing",
                        wid, type(exc).__name__, exc,
                    )
                    crashed_workers.add(wid)

        finally:
            executor.shutdown(wait=False)

        # Graceful degradation: if all workers that produced results have
        # crashed, fall back to sequential.
        all_crashed = (
            len(crashed_workers) >= n_workers
            and not results
        )
        if all_crashed:
            logger.warning(
                "All workers crashed — falling back to sequential execution",
            )
            seq_results = self._run_sequential(tasks, deadline)
            return seq_results

        # Check for unprocessed tasks (those never submitted or submitted
        # but timed out / crashed without producing a result).
        done_indices: set[int] = {r["_idx"] for r in results if "_idx" in r}
        remaining_tasks = [
            t for i, t in enumerate(tasks) if i not in done_indices
        ]
        if remaining_tasks:
            remaining_s = deadline - time.monotonic()
            if remaining_s > 0:
                logger.info(
                    "Running %d remaining task(s) sequentially (%.1fs left)",
                    len(remaining_tasks),
                    remaining_s,
                )
                seq_results = self._run_sequential(remaining_tasks, deadline)
                results.extend(seq_results)

        # Sort by original input order.
        results.sort(key=lambda r: r.get("_idx", 0))

        # Strip the internal _idx key from output dicts.
        for r in results:
            r.pop("_idx", None)

        return results

    # ── Internal: sequential fallback ──────────────────────────────────────

    def _run_sequential(
        self,
        tasks: list[dict],
        deadline: float,
    ) -> list[dict]:
        """Fallback: run tasks sequentially in the main process.

        Creates a single Pipeline and processes tasks one-by-one until
        the deadline is reached.
        """
        from agent import Pipeline

        pipe = Pipeline(self.config)
        seq_results: list[dict] = []
        try:
            for i, task in enumerate(tasks):
                if time.monotonic() >= deadline:
                    logger.warning(
                        "Deadline reached during sequential fallback at "
                        "task %d/%d — stopping",
                        i, len(tasks),
                    )
                    break
                tid = task.get("task_id", f"seq_{len(seq_results)}")
                prompt = task.get("prompt", task.get("question", ""))
                t0 = time.monotonic()
                try:
                    answer = pipe.process(prompt)
                except Exception:
                    answer = ""
                elapsed_ms = (time.monotonic() - t0) * 1000
                seq_results.append({
                    "task_id": tid,
                    "answer": answer,
                    "timing_ms": elapsed_ms,
                    "worker": -1,
                })
        finally:
            pipe.close()
        return seq_results


# ── Convenience wrapper ────────────────────────────────────────────────────


def run_parallel(
    tasks: list[dict],
    config: Optional[Any] = None,
    n_workers: int = 1,
    deadline_s: float = 600.0,
) -> list[dict]:
    """Convenience wrapper around :class:`BatchRunner`.

    Args:
        tasks: List of task dicts with ``task_id`` and ``prompt`` (or
               ``question``).
        config: Optional :class:`~agent.pipeline.PipelineConfig` instance.
        n_workers: Desired worker count (capped by memory budget).
        deadline_s: Global hard deadline in seconds (default 600).

    Returns:
        List of result dicts with keys: task_id, answer, timing_ms, worker.
    """
    runner = BatchRunner(config=config, n_workers=n_workers)
    return runner.run(tasks, deadline_s=deadline_s)


# ── CLI entry point ────────────────────────────────────────────────────────


def main() -> int:
    """CLI entry point for batch_runner."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Run Pipeline tasks in parallel via multiprocessing.",
    )
    parser.add_argument(
        "--input", required=True, help="Path to input tasks JSON",
    )
    parser.add_argument(
        "--output", required=True, help="Path to write results JSON",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--deadline", type=float, default=600.0,
        help="Global deadline in seconds (default: 600)",
    )
    args = parser.parse_args()

    with open(args.input) as f:
        tasks = json.load(f)

    logger.info("Loaded %d tasks from %s", len(tasks), args.input)
    results = run_parallel(
        tasks,
        n_workers=args.workers,
        deadline_s=args.deadline,
    )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %d results to %s", len(results), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
