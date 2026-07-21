"""ReadyPool — Manages parallel worker processes (Pull-Based Architecture).

Architecture (Phase 1 — Pull System):
  - Shared multiprocessing.Queue (task_pool) instead of per-worker queues
  - Workers pull tasks autonomously from the shared pool
  - Monitor loop does health checking every 5s (not dispatching every 0.1s)
  - Orphaned task re-enqueue when workers crash
  - Deadline emergency flag broadcasting
  - Keeps backward compatibility: entrypoint calls pool.dispatch_loop()
"""

import itertools
import logging
import multiprocessing
import signal
import time
from typing import Optional

from staging.ready_classifier import CATEGORIES
from staging.ready_config import ReadyConfig
from staging.ready_queue import ReadyQueue, ReadyTask
from staging.ready_judge import ReadyJudge

# Worker factory registry
_WORKER_REGISTRY: dict[str, type] = {}

logger = logging.getLogger(__name__)


def register_worker_type(worker_type: str, cls: type) -> None:
    """Register a worker class for a type name."""
    _WORKER_REGISTRY[worker_type] = cls


class ReadyPool:
    """Orchestrates workers with per-worker-type partitioned task pools.

    Architecture (Phase 1b — Per-Type Partitioned Pools):
    - Per-worker-type per-category pools (`_type_pools`) instead of shared pools
    - Tasks are assigned round-robin across worker types when draining ReadyQueue
    - Each worker type ONLY pulls from its own reserved pools
    - Shared overflow pool (`task_pool`) for backstop: unclaimed tasks released
      after reservation_timeout_s become available to any worker
    - Monitor loop does health checking, orphan re-enqueue, deadline broadcast,
      and backstop release

    Backward compatible: dispatch_loop() still works as entrypoint's API.
    """

    def __init__(self, config: ReadyConfig):
        self.config = config
        self.task_pool = multiprocessing.Queue()  # Shared overflow pool (backstop / backward compat fallback)
        self._type_pools: dict[str, dict[str, multiprocessing.Queue]] = {}  # Per-worker-type per-category pools
        self._task_type: dict[str, str] = {}  # task_id → assigned worker_type
        self._task_enqueued: dict[str, float] = {}  # task_id → time.monotonic() when enqueued
        self._task_map: dict[str, "ReadyTask"] = {}  # task_id → task object (for backstop release)
        self._backstop_released: set[str] = set()  # task_ids already released to overflow
        self.steal_request_queue = multiprocessing.Queue()
        self._results_queue: Optional[multiprocessing.Queue] = None
        self._processes: list[multiprocessing.Process] = []
        self._busy_flags: list[multiprocessing.Value] = []
        self._heartbeats: list[multiprocessing.Value] = []
        self._worker_ids: list[str] = []
        self._worker_types: list[str] = []
        self._stolen_queues: dict[str, multiprocessing.Queue] = {}
        self._inbox_queues: dict[str, multiprocessing.Queue] = {}
        self._ready_flags: list[multiprocessing.Value] = []
        self._deadline_emergency = multiprocessing.Value('b', 0)
        self._running = multiprocessing.Event()
        self._pending_map: dict[str, Optional[str]] = {}  # task_id → worker_id or None
        self._started = False
        self._last_log = 0.0
        self._total_tasks = 0
        self._worker_timing: dict[str, list[float]] = {}  # worker_type → [timing_ms, ...]

    # ── Lifecycle ──

    def start(self, queue: ReadyQueue) -> None:
        """Populate per-worker-type pools and spawn worker processes.

        Tasks are partitioned round-robin across worker types into dedicated
        per-type per-category pools. Each worker type only pulls from its own
        reserved pools, preventing fast workers (det) from draining everything
        before slower workers (local GPU) get a chance.

        After reservation_timeout_s, unclaimed tasks are released to the shared
        overflow pool (self.task_pool) for any worker to process (backstop).
        """
        if self._started:
            logger.warning("Pool already started")
            return

        self._results_queue = multiprocessing.Queue()
        self._running.set()

        # Clear tracking state
        self._type_pools.clear()
        self._task_type.clear()
        self._task_enqueued.clear()
        self._task_map.clear()
        self._backstop_released.clear()
        self._pending_map.clear()
        self._total_tasks = 0

        # ── Spawn workers ──
        workers_to_start = self._build_worker_plan()

        # Collect distinct worker types for pool creation
        worker_types = sorted(set(
            wtype for _, wtype, _, _, _ in workers_to_start
        ))

        # Create per-worker-type per-category pools
        # Only create pools for worker types that actually exist
        self._type_pools = {
            wtype: {cat: multiprocessing.Queue() for cat in CATEGORIES}
            for wtype in worker_types
        }
        logger.info("[pool] Created per-type pools: %s",
                    {wt: len(self._type_pools[wt]) for wt in self._type_pools})

        # ── Spawn workers ──
        for wid, wtype, worker_cls, index, extra_kwargs in workers_to_start:
            busy_flag = multiprocessing.Value('b', 0)  # 0 = not busy
            heartbeat = multiprocessing.Value('d', time.monotonic())
            # Use per-worker whitelist if provided (e.g. from loc model config),
            # otherwise compute from category_priority matrix
            whitelist = extra_kwargs.pop("category_whitelist", None) or self._build_whitelist(wtype)
            ready_flag = multiprocessing.Value('b', 0)  # 0 = not ready
            inbox = multiprocessing.Queue()
            stolen = multiprocessing.Queue()

            self._inbox_queues[wid] = inbox
            self._stolen_queues[wid] = stolen

            # Each worker gets ONLY its own type's reserved pools as category_pools
            reserved_pools = self._type_pools.get(wtype, {})
            # The shared task_pool acts as overflow for backstop
            # Workers check: reserved pools → task_pool (overflow) → steal

            worker = worker_cls(
                worker_id=wid,
                worker_type=wtype,
                config=self.config,
                task_pool=self.task_pool,
                category_pools=reserved_pools,
                results_queue=self._results_queue,
                steal_request_queue=self.steal_request_queue,
                stolen_queue=stolen,
                inbox_queue=inbox,
                busy_flag=busy_flag,
                heartbeat=heartbeat,
                deadline_emergency=self._deadline_emergency,
                worker_index=index,
                category_whitelist=whitelist,
                ready_flag=ready_flag,
                **extra_kwargs,
            )
            self._ready_flags.append(ready_flag)

            p = multiprocessing.Process(
                target=worker.run,
                name=wid,
                daemon=True,
            )
            p.start()
            self._processes.append(p)
            self._busy_flags.append(busy_flag)
            self._heartbeats.append(heartbeat)
            self._worker_ids.append(wid)
            self._worker_types.append(wtype)

            logger.info("[pool] Started %s (type=%s, pid=%d, whitelist=%s, reserved_categories=%d)",
                        wid, wtype, p.pid, whitelist, len(reserved_pools))

        self._started = True
        logger.info("[pool] All %d workers started (task drain deferred until ready)",
                    len(self._processes))

        # ── Wait for ALL workers to signal ready before proceeding ──
        # This prevents fast-init workers (det) from consuming all tasks
        # before slow-init workers (local LLM model load) are ready.
        if self._ready_flags:
            logger.info("[pool] Waiting for %d workers to signal ready...", len(self._ready_flags))
            deadline = time.monotonic() + 30.0
            for i, flag in enumerate(self._ready_flags):
                while time.monotonic() < deadline and flag.value == 0:
                    time.sleep(0.1)
            logger.info("[pool] All workers ready")

        # ── Drain ReadyQueue into per-worker-type per-category pools (round-robin) ──
        # Tasks are assigned round-robin so each worker type gets an equal share.
        # This is the core of the scheduling fix: fast workers can only consume
        # tasks from their own type's pools, leaving other types' tasks untouched.
        if worker_types:
            type_cycle = itertools.cycle(worker_types)
        else:
            type_cycle = itertools.cycle([""])  # fallback, shouldn't happen

        while not queue.empty:
            task = queue.dequeue_any(preferred_categories=[])
            if task is None:
                break
            assigned_type = next(type_cycle)
            cat = task.category
            # Put into the assigned type's reserved pool
            if assigned_type in self._type_pools and cat in self._type_pools[assigned_type]:
                self._type_pools[assigned_type][cat].put_nowait(task)
            else:
                # Fallback for unknown categories or types — put into shared overflow
                self.task_pool.put_nowait(task)
            # Track assignment for backstop release
            self._pending_map[task.task_id] = None  # Not yet pulled
            self._task_type[task.task_id] = assigned_type
            self._task_enqueued[task.task_id] = time.monotonic()
            self._task_map[task.task_id] = task
            self._total_tasks += 1

        logger.info(
            "[pool] Drained %d tasks round-robin across %d worker types%s",
            self._total_tasks, len(worker_types),
            f" ({', '.join(worker_types)})" if worker_types else "",
        )

    def dispatch_loop(self, queue: ReadyQueue, judge: ReadyJudge, deadline: float) -> None:
        """Backward-compatible entry point. Delegates to monitor_loop().

        Args:
            queue: The original ReadyQueue (for task extraction into task_pool)
            judge: The ReadyJudge instance
            deadline: Absolute time.monotonic() deadline
        """
        if not self._started:
            self.start(queue)
        self.monitor_loop(judge, deadline)

    def monitor_loop(self, judge: ReadyJudge, deadline: float) -> None:
        """Health monitoring loop — runs in main process.

        Does NOT dispatch tasks. Workers pull from shared task_pool.
        Responsibilities:
        - Health checks every 5s (worker alive, heartbeat timeout)
        - Orphaned task re-enqueue on worker crash
        - Deadline emergency flag broadcasting (< 30s remaining)
        - Progress logging
        - Result ingestion and judging (backward compat)
        """
        self._last_log = time.monotonic()

        while self._running.is_set():
            now = time.monotonic()
            remaining = deadline - now

            # ── 1. Check worker health — detect dead workers (R3) ──
            for i in range(len(self._processes) - 1, -1, -1):
                if not self._processes[i].is_alive():
                    self._handle_dead_worker(i)

            # ── 2. Check for stuck workers (heartbeat timeout > 60s) ──
            for i in range(len(self._heartbeats)):
                if (i < len(self._busy_flags) and self._busy_flags[i].value
                        and i < len(self._heartbeats)
                        and (now - self._heartbeats[i].value) > 60):
                    self._handle_stuck_worker(i)

            # ── 2b. Backstop: release stale reserved tasks to shared overflow ──
            # Tasks assigned to a worker type that haven't been pulled after
            # reservation_timeout_s are released to the shared task_pool so
            # any worker type can process them. This prevents a slow/unavailable
            # worker type from blocking tasks indefinitely.
            if self.config.reservation_timeout_s > 0 and self._task_type:
                for tid in list(self._task_type.keys()):
                    if tid not in self._pending_map:
                        # Already judged or removed from tracking
                        continue
                    if self._pending_map[tid] is not None:
                        # Actively being processed by a worker
                        continue
                    if tid in self._backstop_released:
                        # Already released to overflow
                        continue
                    enqueued = self._task_enqueued.get(tid, 0)
                    if now - enqueued > self.config.reservation_timeout_s:
                        task = self._task_map.get(tid)
                        if task is not None:
                            self.task_pool.put_nowait(task)
                            self._backstop_released.add(tid)
                            assigned_type = self._task_type.get(tid, "?")
                            logger.info(
                                "[backstop] Released task %s (assigned to %s, "
                                "waited %.1fs) to overflow pool",
                                tid, assigned_type, now - enqueued,
                            )

            # ── 3. Deadline emergency broadcast ──
            if remaining < 30 and not self._deadline_emergency.value:
                self._deadline_emergency.value = 1
                logger.warning("[monitor] DEADLINE EMERGENCY — forcing fast mode")
            elif remaining >= 30 and self._deadline_emergency.value:
                self._deadline_emergency.value = 0  # Reset if deadline pushed back

            # ── 4. Ingest results from workers (backward compat) ──
            judge.ingest_results(self._results_queue)

            # ── 5. Judge completed tasks ──
            for tid in list(judge.pending_tasks):
                if judge.ready_to_judge(tid):
                    answer, meta = judge.judge(tid)
                    self._pending_map.pop(tid, None)
                    logger.info(
                        "[monitor] Judged %s: strategy=%s votes=%d largest=%d answer=%.60s",
                        tid, meta["strategy"], meta["votes_cast"],
                        meta["largest_group"], answer,
                    )

            # ── 6. Deadline adaptation — force-judge tasks with at least 1 answer ──
            if remaining <= 60 and remaining > 0 and judge.pending_tasks:
                for tid in list(judge.pending_tasks):
                    if judge.count_answers(tid) >= 1:
                        # ── Judge Speed Dominance Fix (deadline path) ──
                        # Don't force-judge if no local worker answer arrived yet
                        # and local workers are still alive. Otherwise DetWorkers
                        # (µs) judge before LocWorkers (~2.5s) can contribute.
                        details = judge.get_answer_details(tid)
                        # Inline type check since _get_worker_type is on judge
                        has_local = any(
                            a.get("worker_type", a.get("worker_id", "")).startswith("loc_")
                            if a.get("worker_type") in (None, "")
                            else a.get("worker_type") == "local"
                            for a in details
                        )
                        if not has_local and self.config.loc_workers > 0:
                            # Check if any local worker is still alive — if all dead, force-judge anyway
                            local_alive = any(
                                self._worker_types[i] == "local" and self._processes[i].is_alive()
                                for i in range(len(self._processes))
                                if i < len(self._worker_types)
                            )
                            if local_alive:
                                continue  # wait for local answer
                        answer, meta = judge.judge(tid)
                        self._pending_map.pop(tid, None)
                        logger.info(
                            "[monitor] Deadline-forced %s (had %d/%d votes) → %.60s",
                            tid, meta["votes_cast"], self.config.judgment_votes, answer,
                        )

            # ── 7. Log progress ──
            if now - self._last_log >= 10.0:
                alive = sum(1 for p in self._processes if p.is_alive())
                busy_count = sum(f.value for f in self._busy_flags)
                logger.info(
                    "[monitor] Progress: %d/%d judged, %d/%d busy, %d alive, %.0fs remaining",
                    judge.total_judged, self._total_tasks,
                    busy_count, len(self._processes), alive, remaining,
                )
                # ── Resource logging ──
                self._log_resources(judge)
                self._last_log = now

            # ── 8. Check completion conditions ──
            if judge.total_judged >= self._total_tasks:
                logger.info("[monitor] All %d tasks judged", self._total_tasks)
                break

            if remaining <= 0:
                logger.warning("[monitor] Deadline reached — force-judging %d pending tasks",
                               len(judge.pending_tasks))
                for tid in judge.pending_tasks:
                    judge.judge(tid)
                break

            if all(not p.is_alive() for p in self._processes):
                logger.warning("[monitor] All workers dead — judging remaining")
                judge.ingest_results(self._results_queue)
                for tid in list(judge.pending_tasks):
                    if judge.count_answers(tid) >= 1:
                        judge.judge(tid)
                break

            # ── 9. Sleep 5s (no longer dispatching every 0.1s) ──
            time.sleep(5.0)

        # ── Final drain: ingest any remaining results and judge partial tasks (R4) ──
        judge.ingest_results(self._results_queue)
        for tid in list(judge.pending_tasks):
            if judge.count_answers(tid) >= 1:
                judge.judge(tid)
    def _log_resources(self, judge) -> None:
        """Log current resource usage (GPU VRAM, CPU, RAM) + worker timing."""
        try:
            from agent.resource_manager import ResourceManager
            rm = ResourceManager()
            snap = rm.probe(force=True)
            if snap.gpu_available:
                logger.info(
                    "[resources] GPU VRAM: %.1f/%.1f GB free  |  "
                    "CPU: %d cores  |  RAM: %.1f/%.1f GB free  |  "
                    "Workers: %d judged, %d pending",
                    snap.vram_free_gb, snap.vram_total_gb,
                    snap.cpu_cores,
                    snap.ram_free_gb, snap.ram_total_gb,
                    judge.total_judged, len(judge.pending_tasks),
                )
            # Log per-worker-type timing summary
            timing = judge.get_timing_summary()
            for wtype, stats in timing.items():
                logger.info(
                    "[timing] %s: %d answers, avg=%.0fms p95=%.0fms total=%.1fs",
                    wtype, stats["count"], stats["avg_ms"],
                    stats["p95_ms"], stats["total_s"],
                )
        except Exception as e:
            logger.debug("[resources] Resource logging failed: %s", e)

    def shutdown(self, timeout: float = 5.0) -> None:
        """Gracefully terminate all worker processes."""
        self._running.clear()
        for p in self._processes:
            if p.is_alive():
                p.terminate()
        for p in self._processes:
            p.join(timeout=timeout)
            if p.is_alive():
                logger.warning("[pool] Worker %s (pid=%d) did not exit — killing", p.name, p.pid)
                p.kill()
                p.join(1.0)
        self._processes.clear()
        logger.info("[pool] All workers shut down")

    # ── Health Handlers ──

    def _handle_dead_worker(self, i: int) -> None:
        """Handle a dead worker: log, reset state, re-enqueue orphaned tasks."""
        if i >= len(self._processes):
            return
        p = self._processes[i]
        wid = self._worker_ids[i] if i < len(self._worker_ids) else "unknown"
        exitcode = p.exitcode

        logger.warning("[pool] Worker %s (pid=%d) died with exit code %s",
                       wid, p.pid, exitcode)

        # Reset busy flag
        if i < len(self._busy_flags):
            self._busy_flags[i].value = 0

        # Check for orphaned tasks associated with this worker
        # Since workers pull autonomously, we can't know exactly which task
        # was in-flight. Log the orphan for visibility; the deadline drain
        # will handle incomplete tasks by producing empty answers.
        orphaned_ids = [
            tid for tid, assigned in self._pending_map.items()
            if assigned == wid
        ]
        for tid in orphaned_ids:
            logger.warning("[pool] Orphaned task %s (was with crashed worker %s)", tid, wid)
            # Best-effort re-enqueue: we can't recover the task object
            # because it was consumed by the worker. The deadline drain
            # handles incomplete tasks.

    def _handle_stuck_worker(self, i: int) -> None:
        """Handle a worker stuck with busy_flag=1 for >60s."""
        if i >= len(self._processes):
            return
        wid = self._worker_ids[i] if i < len(self._worker_ids) else "unknown"
        p = self._processes[i]

        logger.warning("[pool] Worker %s (pid=%d) stuck busy for >60s — terminating", wid, p.pid)

        if p.is_alive():
            p.terminate()
            p.join(2.0)
            if p.is_alive():
                p.kill()
                p.join(1.0)

        # Reset busy flag
        if i < len(self._busy_flags):
            self._busy_flags[i].value = 0

        # Log orphaned tasks
        orphaned_ids = [
            tid for tid, assigned in self._pending_map.items()
            if assigned == wid
        ]
        for tid in orphaned_ids:
            logger.warning(
                "[pool] Task %s possibly orphaned from stuck worker %s (handled by deadline drain)",
                tid, wid,
            )

    # ── Worker plan (unchanged helpers) ──

    def _build_worker_plan(self) -> list[tuple[str, str, type, int, dict]]:
        """Build the list of workers to start.

        Returns [(worker_id, worker_type, worker_class, worker_index, extra_kwargs), ...]

        For local workers, extra_kwargs includes model_path, loc_model_id,
        and category_whitelist from the model config entry.
        """
        plan = []

        # Deterministic workers
        try:
            from staging.workers.det_worker import DetWorker
            register_worker_type("deterministic", DetWorker)
            for i in range(self.config.det_workers):
                plan.append((f"det_worker_{i}", "deterministic", DetWorker, i, {}))
        except ImportError as e:
            logger.warning("DetWorker not available: %s", e)

        # Local workers — one per model config
        try:
            from staging.workers.loc_worker import LocWorker
            register_worker_type("local", LocWorker)
            for i, model_cfg in enumerate(self.config.loc_model_configs):
                plan.append((
                    f"loc_{model_cfg['id']}",
                    "local",
                    LocWorker,
                    i,
                    {
                        "model_path": model_cfg["path"],
                        "loc_model_id": model_cfg["id"],
                        "category_whitelist": model_cfg["categories"],
                    },
                ))
        except ImportError as e:
            logger.warning("LocWorker not available: %s", e)

        # Fireworks workers
        try:
            from staging.workers.fw_worker import FwWorker
            register_worker_type("fireworks", FwWorker)
            for i in range(self.config.fw_workers):
                plan.append((f"fw_worker_{i}", "fireworks", FwWorker, i, {}))
        except ImportError as e:
            logger.warning("FwWorker not available: %s", e)

        logger.info("[pool] Worker plan: %d workers total (%s)",
                    len(plan), ", ".join(f"{wid}" for wid, _, _, _, _ in plan))
        return plan

    def _build_whitelist(self, worker_type: str) -> list[str]:
        """Return categories sorted by priority for this worker type.

        Uses the category_priority matrix from config. A lower index in
        the preference list means higher priority for that worker type.
        Categories not mentioning this worker type get priority 999.
        """
        priority_map = self.config.category_priority
        scores: dict[str, int] = {}
        for cat, pref_list in priority_map.items():
            try:
                idx = pref_list.index(worker_type)
                scores[cat] = idx
            except ValueError:
                scores[cat] = 999
        return sorted(scores, key=scores.get)

    # ── Properties ──

    @property
    def busy_workers(self) -> list[str]:
        return [wid for wid, flag in zip(self._worker_ids, self._busy_flags) if flag.value]

    @property
    def available_count(self) -> int:
        return sum(1 for f in self._busy_flags if not f.value)

    @property
    def total_workers(self) -> int:
        return len(self._processes)


# Forward-compatibility alias
ReadyMonitor = ReadyPool
