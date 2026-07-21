"""DetWorker — Deterministic solver worker.

Fast path for tasks that don't need LLM inference.
Processes tasks 5 times (all same, deterministic = no variation needed).
"""

import logging
import time

from staging.ready_config import ReadyConfig
from staging.ready_queue import ReadyTask
from staging.ready_worker import ReadyWorker

logger = logging.getLogger(__name__)


class DetWorker(ReadyWorker):
    """Deterministic solver worker — near-instant for suitable categories."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._solvers = {}

    def initialize(self) -> None:
        """Import deterministic solvers from the existing agent module.

        Handles missing dependencies (spacy) gracefully — disables
        solvers that depend on unavailable packages.
        """
        self._solvers = {}
        try:
            from agent.solvers.deterministic import (
                solve_arithmetic,
                solve_factual_qa,
                solve_sentiment,
                solve_summarization,
                solve_code_debugging,
                solve_code_generation,
                solve_ner,
                solve_logic,
            )
            self._solvers = {
                "math":          solve_arithmetic,
                "factual":       solve_factual_qa,
                "sentiment":     solve_sentiment,
                "summarization": solve_summarization,
                "code_debug":    solve_code_debugging,
                "code_gen":      solve_code_generation,
                "ner":           solve_ner,
                "logic":         solve_logic,
            }
            logger.info("[%s] Loaded %d deterministic solvers",
                        self.worker_id, len(self._solvers))
        except ImportError as exc:
            logger.warning("[%s] Deterministic solvers unavailable: %s",
                           self.worker_id, exc)
        except Exception as exc:
            logger.warning("[%s] Failed to load deterministic solvers: %s",
                           self.worker_id, exc)

    def process(self, task: ReadyTask) -> list[dict]:
        """Process task 5 times deterministically (same answer each time).

        Variation is not meaningful for deterministic solvers — they always
        return the same output for the same input.
        """
        solver = self._solvers.get(task.category)
        if not solver:
            logger.warning("[%s] No deterministic solver for category '%s'",
                           self.worker_id, task.category)
            return [{
                "worker_id": self.worker_id,
                "task_id": task.task_id,
                "answer": "",
                "timing_ms": 0,
            }] * self.config.judgment_votes

        answers = []
        for _ in range(self.config.judgment_votes):
            t0 = time.monotonic()
            try:
                answer = solver(task.prompt, task.category) or ""
            except Exception as exc:
                logger.warning("[%s] Solver failed for task %s: %s",
                               self.worker_id, task.task_id, exc)
                answer = ""
            elapsed = (time.monotonic() - t0) * 1000
            answers.append({
                "worker_id": self.worker_id,
                "task_id": task.task_id,
                "model_id": task.category,  # which solver was used
                "try_index": 0,
                "temperature": None,
                "answer": answer,
                "timing_ms": elapsed,
            })

        return answers

    def _process_single(self, task: ReadyTask) -> dict:
        """Deadline-emergency fast mode — run the solver once."""
        solver = self._solvers.get(task.category)
        if not solver:
            return {
                "worker_id": self.worker_id,
                "task_id": task.task_id,
                "answer": "",
                "timing_ms": 0,
            }
        t0 = time.monotonic()
        try:
            answer = solver(task.prompt, task.category) or ""
        except Exception as exc:
            logger.warning("[%s] _process_single solver failed for task %s: %s",
                           self.worker_id, task.task_id, exc)
            answer = ""
        elapsed = (time.monotonic() - t0) * 1000
        return {
            "worker_id": self.worker_id,
            "task_id": task.task_id,
            "model_id": task.category,
            "try_index": 0,
            "temperature": None,
            "answer": answer,
            "timing_ms": elapsed,
        }
