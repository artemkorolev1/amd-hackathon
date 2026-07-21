"""LocWorker — Local GGUF model worker.

Memory-intensive (separate process, own model instance). Varies
generation parameters across 5 tries for answer diversity.
"""

import logging
import time

from staging.ready_config import ReadyConfig
from staging.ready_queue import ReadyTask
from staging.ready_worker import ReadyWorker

logger = logging.getLogger(__name__)

_TEMPERATURE_SWEEP = [0.1, 0.3, 0.5, 0.7, 0.9]


class LocWorker(ReadyWorker):
    """Local GGUF model worker — offline-capable, temperature sweep."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._llm = None
        self.model_path = kwargs.get("model_path", self.config.loc_model_path)
        self.loc_model_id = kwargs.get("loc_model_id", "unknown")

    def initialize(self) -> None:
        """Load llama-cpp-python model in this worker process."""
        from llama_cpp import Llama

        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=2048,
            n_gpu_layers=self.config.loc_n_gpu_layers,
            n_threads=self.config.loc_n_threads,
            flash_attn=True,
            verbose=False,
        )
        logger.info("[%s] Local model loaded: %s (id=%s)", self.worker_id,
                    self.model_path, self.loc_model_id)

    def _get_system_prompt(self, category: str) -> str:
        """Build a minimal system prompt for the category.

        Replicates the pipeline's prompt construction without importing Pipeline.
        """
        # Minimal per-category system prompt (same approach as fw_router)
        prompts = {
            "sentiment":     "Output EXACTLY one word: Positive, Negative, or Neutral. No explanation.",
            "ner":           "Entities: Person=..., Org=..., Loc=..., Date=... No prose.",
            "math":          "Output ONLY the number. No units. No explanation.",
            "code_gen":      "Write the code. Fenced block. No explanation.",
            "code_debug":    "Point out the bug in 1 sentence. Fix in fenced block.",
            "factual":       "Answer briefly. One sentence max.",
            "logic":         "Think step by step. End with Answer: ...",
            "summarization": "Summarize in 2-3 sentences.",
        }
        return prompts.get(category, "Answer concisely.")

    def process(self, task: ReadyTask) -> list[dict]:
        """Process task 5 times with temperature sweep."""
        if self._llm is None:
            logger.error("[%s] Model not initialized", self.worker_id)
            return [{
                "worker_id": self.worker_id,
                "task_id": task.task_id,
                "answer": "",
                "timing_ms": 0,
            }] * self.config.judgment_votes

        sys_prompt = self._get_system_prompt(task.category)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": task.prompt},
        ]

        answers = []
        for i in range(self.config.judgment_votes):
            temperature = _TEMPERATURE_SWEEP[i % len(_TEMPERATURE_SWEEP)]

            t0 = time.monotonic()
            try:
                result = self._llm.create_chat_completion(
                    messages=messages,
                    max_tokens=512,
                    temperature=temperature,
                    stop=None,
                )
                answer = result["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                logger.warning("[%s] Local try %d failed for task %s: %s",
                               self.worker_id, i, task.task_id, exc)
                answer = ""

            elapsed = (time.monotonic() - t0) * 1000
            answers.append({
                "worker_id": self.worker_id,
                "task_id": task.task_id,
                "model_id": self.loc_model_id,
                "try_index": i,
                "temperature": temperature,
                "answer": answer,
                "timing_ms": elapsed,
            })

        return answers

    def _process_single(self, task: ReadyTask) -> dict:
        """Deadline-emergency fast mode — run one inference at temp 0.1."""
        if self._llm is None:
            return {
                "worker_id": self.worker_id,
                "task_id": task.task_id,
                "answer": "",
                "timing_ms": 0,
            }
        sys_prompt = self._get_system_prompt(task.category)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": task.prompt},
        ]
        t0 = time.monotonic()
        try:
            result = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=512,
                temperature=0.1,
                stop=None,
            )
            answer = result["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.warning("[%s] _process_single failed for task %s: %s",
                           self.worker_id, task.task_id, exc)
            answer = ""
        elapsed = (time.monotonic() - t0) * 1000
        return {
            "worker_id": self.worker_id,
            "task_id": task.task_id,
            "model_id": self.loc_model_id,
            "try_index": 0,
            "temperature": 0.1,
            "answer": answer,
            "timing_ms": elapsed,
        }

    def shutdown(self) -> None:
        """Release llama-cpp-python model."""
        if self._llm is not None:
            self._llm = None
        super().shutdown()
