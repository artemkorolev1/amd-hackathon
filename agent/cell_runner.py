"""
cell_runner.py — Universal cell executor for the answer-centered pipeline.

The CellRunner is the core orchestrator. Given a CellConfig and AnswerRequest,
it dispatches to the role-specific implementation and returns an AnswerResponse.

Usage:
    runner = CellRunner()
    request = AnswerRequest(prompt="What is 2+2?", category="math")
    config = CellConfig.from_workcell(workcell)
    response = runner.execute(config, request)
    # response.answer, response.confidence, response.metrics
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Callable

import agent.config as cfg
from agent.contracts import AnswerRequest, AnswerResponse, CellMetrics, CellConfig, WorkflowRequest

logger = logging.getLogger(__name__)


class CellRunner:
    """Executes a work cell given its config + request.

    The runner looks up the role-specific executor in its registry,
    runs it with timing and error handling, and returns an AnswerResponse.
    """

    def __init__(self, resource_manager=None, pipeline=None):
        self._resource_manager = resource_manager
        self._pipeline = pipeline  # Optional pipeline instance for local_llm inference
        self._executors: dict[str, Callable] = {
            "deterministic": self._run_deterministic,
            "local_llm": self._run_local_llm,
            "api_llm": self._run_api_llm,
            "workflow": self._run_workflow,
            "aggregator": self._run_aggregator,
        }

    def execute(self, config: CellConfig, request: AnswerRequest) -> AnswerResponse:
        """Execute a cell and return a structured response.

        Args:
            config: Cell configuration (role, model, prompts, steps)
            request: Input prompt and context

        Returns:
            AnswerResponse with answer, metrics, and metadata
        """
        t0 = time.monotonic()

        executor = self._executors.get(config.role)
        if executor is None:
            return AnswerResponse(
                answer="",
                confidence=0.0,
                metadata={"error": f"Unknown role: {config.role}"},
            )

        try:
            result = executor(config, request)
            elapsed = (time.monotonic() - t0) * 1000
            result.metrics.elapsed_ms = round(elapsed, 1)
            return result
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            logger.exception("Cell %s (%s) failed: %s", config.name, config.role, e)
            return AnswerResponse(
                answer="",
                confidence=0.0,
                metrics=CellMetrics(elapsed_ms=round(elapsed, 1)),
                metadata={"error": str(e), "cell_name": config.name, "role": config.role},
            )

    def register_executor(self, role: str, fn: Callable) -> None:
        """Register or override an executor for a role."""
        self._executors[role] = fn

    # ═══════════════════════════════════════════════════════════════
    # Role-specific executors
    # ═══════════════════════════════════════════════════════════════

    def _run_deterministic(self, config: CellConfig, request: AnswerRequest) -> AnswerResponse:
        """Run deterministic solvers (regex/heuristic). Zero LLM cost."""
        from agent.solvers.deterministic import (
            solve_arithmetic, solve_sentiment, solve_factual_qa,
            solve_ner, solve_summarization, solve_logic, solve_code_debugging,
        )

        category = request.category
        prompt = request.prompt

        solver_map = {
            "math": ("math_arithmetic", solve_arithmetic),
            "sentiment": ("sentiment", solve_sentiment),
            "factual": ("other_complex", solve_factual_qa),
            "ner": ("ner", solve_ner),
            "summarization": ("summarization", solve_summarization),
            "logic": ("logic", solve_logic),
            "code_debug": ("code_debugging", solve_code_debugging),
            "code_gen": ("code_debugging", solve_code_debugging),
        }

        if category in solver_map:
            solver_cat, solver_fn = solver_map[category]
            t0 = time.monotonic()
            answer = solver_fn(prompt, solver_cat)
            elapsed = (time.monotonic() - t0) * 1000
            if answer:
                return AnswerResponse(
                    answer=answer,
                    confidence=0.9,
                    metrics=CellMetrics(elapsed_ms=round(elapsed, 1), tool="deterministic"),
                    metadata={"solver": solver_fn.__name__},
                )

        return AnswerResponse(answer="", confidence=0.0, metadata={"note": "no deterministic solver matched"})

    def _run_local_llm(self, config: CellConfig, request: AnswerRequest) -> AnswerResponse:
        """Run local GGUF inference via Pipeline or direct llama-cpp-python call."""
        if self._pipeline is None:
            return AnswerResponse(answer="", confidence=0.0, metadata={"error": "no pipeline available"})

        t0 = time.monotonic()
        # Use the Pipeline's process() — the most battle-tested inference path
        answer = self._pipeline.process(request.prompt)
        elapsed = (time.monotonic() - t0) * 1000

        if answer:
            return AnswerResponse(
                answer=answer,
                confidence=0.8,
                metrics=CellMetrics(
                    elapsed_ms=round(elapsed, 1),
                    model=config.model_key or getattr(self._pipeline, "cfg", None) and getattr(self._pipeline.cfg, "model_path", None) or None,
                    tokens_out=len(answer.split()),
                ),
            )
        return AnswerResponse(answer="", confidence=0.0, metrics=CellMetrics(elapsed_ms=round(elapsed, 1)))

    def _run_api_llm(self, config: CellConfig, request: AnswerRequest) -> AnswerResponse:
        """Run inference via external API (Fireworks)."""
        from agent.solvers.fireworks import FireworksSolver
        from agent.solvers.fw_router import route as fw_route

        if not request.prompt.strip():
            return AnswerResponse(answer="", confidence=0.0)

        t0 = time.monotonic()
        try:
            fw = FireworksSolver()
            if not fw.api_key:
                return AnswerResponse(answer="", confidence=0.0, metadata={"error": "no API key"})

            complexity = 0.5  # default complexity for API routing
            route = fw_route(request.category, request.prompt, complexity)

            answer = fw.solve(
                route.model_id, route.system_prompt, request.prompt,
                max_tokens=route.max_tokens,
                temperature=route.temperature,
                prefill=route.prefill,
                task_type=request.category,
                timeout=30.0,
            )
            elapsed = (time.monotonic() - t0) * 1000

            if answer:
                return AnswerResponse(
                    answer=answer,
                    confidence=0.9,
                    metrics=CellMetrics(
                        elapsed_ms=round(elapsed, 1),
                        model=route.model_id,
                        tokens_out=len(answer.split()),
                    ),
                )
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            logger.warning("API LLM call failed: %s", e)
            return AnswerResponse(answer="", confidence=0.0, metrics=CellMetrics(elapsed_ms=round(elapsed, 1)), metadata={"error": str(e)})

        return AnswerResponse(answer="", confidence=0.0, metadata={"note": "API returned empty"})

    def _run_workflow(self, config: CellConfig, request: AnswerRequest) -> AnswerResponse:
        """Run a multi-step Plan-and-Solve workflow."""
        from agent.cell import Cell, StepConfig
        from agent.workflow import WorkflowEngine

        if not config.steps:
            return AnswerResponse(answer="", confidence=0.0, metadata={"error": "workflow cell has no steps"})

        # Build a temporary Cell for WorkflowEngine
        step_configs = [StepConfig.from_dict(s) if isinstance(s, dict) else s for s in config.steps]
        cell = Cell(
            task_id=request.category,
            model_key=config.model_key,
            system_prompt=config.system_prompt,
            steps=step_configs,
        )

        # Need an inference function for LLM steps
        if self._pipeline is None:
            return AnswerResponse(answer="", confidence=0.0, metadata={"error": "no pipeline for workflow LLM steps"})

        engine = WorkflowEngine(self._pipeline._infer)
        t0 = time.monotonic()
        result = engine.run(cell, request.prompt)
        elapsed = (time.monotonic() - t0) * 1000

        final_answer = result.get("final_answer", "")
        step_results = result.get("step_results", [])

        if final_answer:
            total_tokens = sum(s.get("tokens_est", 0) for s in step_results)
            return AnswerResponse(
                answer=final_answer,
                confidence=0.85,
                metrics=CellMetrics(
                    elapsed_ms=round(elapsed, 1),
                    model=config.model_key,
                    tokens_out=total_tokens,
                ),
                artifacts={s.get("step", f"step_{i}"): s.get("output", "") for i, s in enumerate(step_results)},
                metadata={"steps": len(step_results), "step_details": step_results},
            )

        return AnswerResponse(answer="", confidence=0.0, metrics=CellMetrics(elapsed_ms=round(elapsed, 1)))

    def _run_aggregator(self, config: CellConfig, request: AnswerRequest) -> AnswerResponse:
        """Run consensus aggregation on multiple answers.

        Expects request.metadata to contain 'answers' list.
        """
        answers = request.metadata.get("answers", [])
        if not answers:
            return AnswerResponse(answer="", confidence=0.0, metadata={"error": "no answers to aggregate"})

        # Simple majority vote
        from collections import Counter
        t0 = time.monotonic()

        texts = [a.get("answer", "") if isinstance(a, dict) else str(a) for a in answers]
        if not texts:
            return AnswerResponse(answer="", confidence=0.0)

        # Tally non-empty answers
        non_empty = [t for t in texts if t.strip()]
        if not non_empty:
            return AnswerResponse(answer="", confidence=0.0)

        counter = Counter(non_empty)
        most_common = counter.most_common(1)[0]
        winner = most_common[0]
        agreement = most_common[1] / len(non_empty)

        elapsed = (time.monotonic() - t0) * 1000

        return AnswerResponse(
            answer=winner,
            confidence=round(agreement, 2),
            metrics=CellMetrics(elapsed_ms=round(elapsed, 1)),
            metadata={
                "votes": len(non_empty),
                "agreement": round(agreement, 2),
                "tally": dict(counter.most_common()),
                "total_submitted": len(texts),
            },
        )
