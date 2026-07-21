"""ReadyWorker — Abstract base class for all worker types.

Each worker runs in its own process, pulls tasks from a shared task pool,
processes each task STAGING_JUDGMENT_VOTES times (5 by default) with
variation, and pushes all answers to a shared results queue. Supports
work stealing for load balancing across heterogeneous workers.

Design contract:
- Workers are process-safe (no shared state except queues/manager)
- Each worker owns its model/API client lifecycle
- 5-try strategy is handled in process() by the subclass
- Pull priority: stolen > inbox > task_pool > steal
"""

import logging
import signal
import time
from abc import ABC, abstractmethod
from multiprocessing import Queue, Value, Event
from typing import Optional

from staging.ready_config import ReadyConfig
from staging.ready_queue import ReadyTask

logger = logging.getLogger(__name__)


class ReadyWorker(ABC):
    """Abstract base for worker implementations.

    Args:
        worker_id: Unique worker identifier (e.g. 'fw_worker_0')
        worker_type: Worker type label (e.g. 'fireworks', 'local', 'deterministic')
        config: Shared configuration
        task_pool: Shared multiprocessing.Queue — shared task pool (main pull point)
        results_queue: Shared multiprocessing.Queue for pushing results
        steal_request_queue: Shared multiprocessing.Queue — steal requests
        stolen_queue: Per-worker multiprocessing.Queue — tasks stolen FOR us
        inbox_queue: Per-worker multiprocessing.Queue — tasks assigned TO us
        busy_flag: Shared Value('b', 1/0) indicating worker availability
        heartbeat: Shared Value('d') — float timestamp for crash detection
        deadline_emergency: Shared Value('b') — flag for fast mode
    """

    def __init__(
        self,
        worker_id: str,
        worker_type: str,
        config: ReadyConfig,
        task_pool,
        results_queue,
        steal_request_queue,
        stolen_queue,
        inbox_queue,
        busy_flag,
        heartbeat,
        deadline_emergency,
        category_whitelist=None,
        category_pools=None,
        **kwargs,
    ) -> None:
        self.worker_id = worker_id
        self.worker_type = worker_type
        self.config = config
        self.task_pool = task_pool
        self.category_pools = category_pools or {}
        self.results_queue = results_queue
        self.steal_request_queue = steal_request_queue
        self.stolen_queue = stolen_queue
        self.inbox_queue = inbox_queue
        self.ready_flag = kwargs.get("ready_flag")
        self.busy_flag = busy_flag
        self.heartbeat = heartbeat
        self.deadline_emergency = deadline_emergency
        self.category_whitelist = category_whitelist or []
        self._running = True
        self._known_workers: list[str] = []

    # ── Lifecycle ──

    @abstractmethod
    def initialize(self) -> None:
        """Load model, configure API client, etc. Called once at worker start."""
        ...

    @abstractmethod
    def process(self, task: ReadyTask) -> list[dict]:
        """Process a single task 5 times (by default) with variation.

        Must return a list of answer dicts:
            [{"worker_id": str, "task_id": str, "answer": str, "timing_ms": float}, ...]

        The subclass controls the variation strategy (temperature sweep,
        prefill variations, different system prompts, etc.).
        """
        ...

    def shutdown(self) -> None:
        """Release resources. Override in subclass if needed."""
        self._running = False

    # ── Main loop ──

    def run(self) -> None:
        """Main loop with heartbeat for crash detection.

        Pulls tasks, processes them (normal or emergency fast mode),
        and pushes results. Heartbeat is updated every iteration to
        allow the pool monitor to detect crashed workers.
        """
        signal.signal(signal.SIGTERM, self._handle_sigterm)

        try:
            self.initialize()
        except Exception as exc:
            logger.error("[%s] Initialization failed: %s", self.worker_id, exc)
            return

        # Signal that this worker is ready for task processing
        if self.ready_flag is not None:
            self.ready_flag.value = 1

        logger.info("[%s] Worker started (type=%s)", self.worker_id, self.worker_type)

        while self._running:
            # Update heartbeat FIRST — shows we're alive even when busy
            self.heartbeat.value = time.monotonic()

            # PULL — worker drives the work cycle
            task = self._pull_task()
            if task is None:
                time.sleep(0.2)
                continue

            # PROCESS (mark busy)
            self.busy_flag.value = 1
            task.status = "in_progress"

            try:
                # Check deadline_emergency for "fast mode"
                if self.deadline_emergency.value:
                    # Single try instead of full judgment_votes loop
                    answers = self._process_single(task)
                else:
                    answers = self.process(task)  # Full judgment_votes loop
            except Exception as exc:
                logger.exception("[%s] Task %s failed: %s",
                                 self.worker_id, task.task_id, exc)
                answers = []

            # PUSH results — always done, even on failure
            self._push_results(task, answers)
            self.busy_flag.value = 0
            task.status = "judged"

        self.shutdown()
        logger.info("[%s] Worker stopped", self.worker_id)

    def _process_single(self, task: ReadyTask) -> list[dict]:
        """Single fast try for deadline emergency mode.

        Default implementation: runs a single iteration of process().
        Subclasses should override to skip the full temperature sweep
        and produce one answer quickly.
        """
        # Fallback: take the first result from full processing
        # (subclasses should override with a true single-try path)
        results = self.process(task)
        return results[:1]

    # ── Internal helpers ──

    def _pull_task(self) -> Optional[ReadyTask]:
        """Pull next task with work stealing.

        Priority: stolen > inbox > [reserved pools] > overflow > steal

        With per-worker-type partitioning:
        - Reserved pools (category_pools) contain tasks assigned to this worker type
        - Overflow pool (task_pool) contains backstop tasks released after timeout
        - Workers always check overflow after their own reserved pools
        """
        import queue as _queue

        # 1. Check stolen queue (fastest path — already transferred to us)
        try:
            return self.stolen_queue.get_nowait()
        except _queue.Empty:
            pass

        # 2. Check inbox (tasks explicitly assigned to us)
        try:
            return self.inbox_queue.get_nowait()
        except _queue.Empty:
            pass

        # 3. Pull from per-type reserved pools (respects category_whitelist)
        if self.category_pools and self.category_whitelist:
            for cat in self.category_whitelist:
                pool = self.category_pools.get(cat)
                if pool is not None:
                    try:
                        task = pool.get_nowait()
                        if task is not None:
                            return task
                    except _queue.Empty:
                        continue

        # 4. Check overflow pool (shared fallback / backstop release)
        if self.task_pool is not None:
            try:
                task = self.task_pool.get_nowait()
                if task is not None:
                    return task
            except _queue.Empty:
                pass

        # 5. Attempt work stealing (we're fully idle)
        return self._attempt_steal()

    def _attempt_steal(self) -> Optional[ReadyTask]:
        """Try to steal work from a busy worker.

        Sends a steal request to the shared queue, waits briefly for a response.
        """
        import queue as _queue

        try:
            self.steal_request_queue.put_nowait({
                "thief_id": self.worker_id,
                "thief_type": self.worker_type,
                "whitelist": self.category_whitelist,
                "timestamp": time.monotonic(),
            })
        except _queue.Full:
            return None

        # Wait briefly for a stolen task
        try:
            return self.stolen_queue.get(timeout=0.5)
        except _queue.Empty:
            return None

    def _push_results(self, task: ReadyTask, answers: list[dict]) -> None:
        """Push results for a completed task to the results queue."""
        if not answers:
            answers = [{
                "worker_id": self.worker_id,
                "task_id": task.task_id,
                "answer": "",
                "timing_ms": 0,
                "prompt": task.prompt,
                "category": task.category,
            }]
        for a in answers:
            # Attach task metadata for the judge (needed for Fireworks escalation)
            a["worker_type"] = self.worker_type
            a["prompt"] = task.prompt
            a["category"] = task.category
            self.results_queue.put_nowait(a)

    def _handle_sigterm(self, signum, frame) -> None:
        # NOT signal-safe to log here (logging lock may be held).
        self._running = False
