"""FwWorker — Fireworks API worker.

Stateless, fast, no model loading. Handles rate limiting with
exponential backoff. Varies temperature across 5 sequential tries
to produce diverse answers for majority voting.
"""

import logging
import time

from staging.ready_config import ReadyConfig
from staging.ready_queue import ReadyTask
from staging.ready_worker import ReadyWorker

logger = logging.getLogger(__name__)

# Temperature sweep across 5 tries — varies from conservative to creative
_TEMPERATURE_SWEEP = [0.1, 0.3, 0.5, 0.7, 0.9]


class FwWorker(ReadyWorker):
    """Fireworks API worker — fast, stateless, 5-try with temperature sweep."""

    def __init__(self, *args, worker_index: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.worker_index = worker_index
        self._solver = None
        self._model_id = ""

    def initialize(self) -> None:
        """Configure Fireworks solver and pick model."""
        from agent.solvers.fireworks import FireworksSolver
        self._solver = FireworksSolver(api_key=self.config.fw_api_key)
        # Pick this worker's model from the config list (round-robin)
        models = self.config.fw_models
        self._model_id = models[self.worker_index % len(models)]
        logger.info("[%s] Using Fireworks model: %s", self.worker_id, self._model_id)

    def process(self, task: ReadyTask) -> list[dict]:
        """Process task 5 times with a temperature sweep.

        Tries 0.1 → 0.9 temperature for answer diversity.
        Uses agent.solvers.fw_router.route for per-category config.
        """
        from agent.solvers.fw_router import route

        cfg = route(task.category, task.prompt, 0.5)
        answers = []

        for i in range(self.config.judgment_votes):
            temperature = _TEMPERATURE_SWEEP[i % len(_TEMPERATURE_SWEEP)]

            t0 = time.monotonic()
            try:
                answer = self._solver.solve(
                    self._model_id,
                    cfg.system_prompt,
                    task.prompt,
                    max_tokens=cfg.max_tokens,
                    temperature=temperature,
                    prefill=cfg.prefill,
                    task_type=task.category,
                    timeout=int(self.config.worker_timeout_s),
                )
            except Exception as exc:
                logger.warning("[%s] FW try %d failed for task %s: %s",
                               self.worker_id, i, task.task_id, exc)
                answer = ""

            elapsed = (time.monotonic() - t0) * 1000
            answers.append({
                "worker_id": self.worker_id,
                "task_id": task.task_id,
                "model_id": self._model_id,
                "try_index": i,
                "temperature": temperature,
                "answer": answer,
                "timing_ms": elapsed,
            })

        return answers

    def _process_single(self, task: ReadyTask) -> dict:
        """Deadline-emergency fast mode — run one API call at temp 0.1."""
        from agent.solvers.fw_router import route

        cfg = route(task.category, task.prompt, 0.5)
        t0 = time.monotonic()
        try:
            answer = self._solver.solve(
                self._model_id,
                cfg.system_prompt,
                task.prompt,
                max_tokens=cfg.max_tokens,
                temperature=0.1,
                prefill=cfg.prefill,
                task_type=task.category,
                timeout=int(self.config.worker_timeout_s),
            )
        except Exception as exc:
            logger.warning("[%s] _process_single failed for task %s: %s",
                           self.worker_id, task.task_id, exc)
            answer = ""
        elapsed = (time.monotonic() - t0) * 1000
        return {
            "worker_id": self.worker_id,
            "task_id": task.task_id,
            "model_id": self._model_id,
            "try_index": 0,
            "temperature": 0.1,
            "answer": answer,
            "timing_ms": elapsed,
        }
