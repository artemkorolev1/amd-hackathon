"""ReadyQueue — Multi-category task queue for worker dispatch.

Each task enqueued after bulk classification is wrapped as a ReadyTask
with its category, classification scores, and answer accumulator.

Workers pull tasks from the queue (push model), process them
sequentially (5 tries per task), and push results to a shared
results queue.
"""

import queue
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ReadyTask:
    """A classified task ready for worker processing.

    The `answers` list is populated by the worker as it tries the task
    up to STAGING_JUDGMENT_VOTES times.
    """

    task_id: str
    prompt: str
    category: str
    category_4way: str
    raw_scores: dict = field(default_factory=dict)
    confidence: float = 0.0
    score_delta: float = 0.0
    answers: list = field(default_factory=list)
    status: str = "pending"  # pending → in_progress → judged

    def add_answer(self, worker_id: str, answer: str, timing_ms: float) -> None:
        """Record one answer from a worker."""
        self.answers.append({
            "worker_id": worker_id,
            "answer": answer,
            "timing_ms": timing_ms,
        })

    @property
    def num_answers(self) -> int:
        return len(self.answers)

    @property
    def elapsed_ms(self) -> float:
        """Total elapsed processing time across all tries."""
        return sum(a["timing_ms"] for a in self.answers)

    def to_dict(self) -> dict:
        """Serialize to dict for final output."""
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "category": self.category,
            "category_4way": self.category_4way,
            "confidence": self.confidence,
            "score_delta": self.score_delta,
            "answers": self.answers,
            "status": self.status,
            "num_answers": self.num_answers,
            "elapsed_ms": self.elapsed_ms,
        }


class ReadyQueue:
    """Multi-category task queue. Worker-safe via multiprocessing.Queue internally.

    Tasks are enqueued after bulk classification and dequeued by workers
    based on their preferred categories.
    """

    def __init__(self):
        self._queues: dict[str, queue.Queue] = {}
        self._task_map: dict[str, ReadyTask] = {}

    def _ensure_category(self, category: str) -> None:
        if category not in self._queues:
            self._queues[category] = queue.Queue()

    def enqueue(self, task: ReadyTask) -> None:
        """Add a single classified task to its category queue."""
        self._ensure_category(task.category)
        self._queues[task.category].put_nowait(task)
        self._task_map[task.task_id] = task

    def enqueue_batch(self, tasks: list[ReadyTask]) -> None:
        """Bulk enqueue classified tasks."""
        for t in tasks:
            self.enqueue(t)

    def dequeue(self, category: str, timeout: float = 1.0) -> Optional[ReadyTask]:
        """Get next task for a specific category (blocking with timeout).

        Returns None if no task available within timeout.
        """
        self._ensure_category(category)
        try:
            return self._queues[category].get(timeout=timeout)
        except queue.Empty:
            return None

    def dequeue_any(self, preferred_categories: list[str]) -> Optional[ReadyTask]:
        """Get next task from preferred categories (non-blocking).

        Tries each preferred category in order, then falls back
        to any non-empty queue.
        """
        for cat in preferred_categories:
            self._ensure_category(cat)
            try:
                return self._queues[cat].get_nowait()
            except queue.Empty:
                continue
        # Fallback: any non-empty queue not already tried
        for cat, q in self._queues.items():
            if cat in preferred_categories:
                continue
            try:
                return q.get_nowait()
            except queue.Empty:
                continue
        return None

    def task_counts_by_category(self) -> dict[str, int]:
        """Return {category: count} for non-empty queues."""
        return {cat: q.qsize()
                for cat, q in self._queues.items()
                if q.qsize() > 0}

    @property
    def empty(self) -> bool:
        return all(q.empty() for q in self._queues.values())

    @property
    def total_pending(self) -> int:
        return sum(q.qsize() for q in self._queues.values())

    def drain_to_pool(self, task_pool: "multiprocessing.Queue") -> int:
        """Drain all queued tasks into a shared multiprocessing.Queue.

        Returns the count of tasks drained.
        """
        import multiprocessing

        count = 0
        for cat, q in self._queues.items():
            while True:
                try:
                    task = q.get_nowait()
                    task_pool.put(task)
                    count += 1
                except queue.Empty:
                    break
        return count
