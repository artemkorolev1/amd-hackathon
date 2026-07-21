#!/usr/bin/env python3
"""Tests for batch_runner — parallel task orchestration via multiprocessing."""

from __future__ import annotations

import concurrent.futures
import os
import sys
import time
import unittest
from typing import Any
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from runner.batch_runner import (
    BatchRunner,
    _compute_workers,
    _worker_process,
    run_parallel,
)

# =============================================================================
# _compute_workers — memory budget math
# =============================================================================


class TestComputeWorkers(unittest.TestCase):
    def test_default_ram_returns_two(self):
        """4 GB RAM, 1.1 GB model → 2 workers."""
        assert _compute_workers(max_ram_gb=4.0, model_gb=1.1) == 2

    def test_large_ram_capped_at_four(self):
        """8 GB RAM, 1.1 GB model → 4 (hard cap)."""
        assert _compute_workers(max_ram_gb=8.0, model_gb=1.1) == 4

    def test_huge_ram_still_capped(self):
        """100 GB RAM → still max 4."""
        assert _compute_workers(max_ram_gb=100, model_gb=1.1) == 4

    def test_small_ram_returns_one(self):
        """2 GB RAM, 1.1 GB model → 1."""
        assert _compute_workers(max_ram_gb=2.0, model_gb=1.1) == 1

    def test_very_small_ram_min_one(self):
        """0.5 GB RAM → floor at 1."""
        assert _compute_workers(max_ram_gb=0.5, model_gb=1.1) == 1

    def test_small_model_allows_more_workers(self):
        """4 GB RAM, 0.3 GB model → 4 workers."""
        assert _compute_workers(max_ram_gb=4.0, model_gb=0.3) == 4

    def test_custom_model_gb(self):
        """4 GB RAM, 0.5 GB model → 3 workers (4.0 - 0.5) / (0.4 + 0.5) = 3.5/0.9 = 3."""
        assert _compute_workers(max_ram_gb=4.0, model_gb=0.5) == 3


# =============================================================================
# _worker_process — module-level picklable entrypoint
# =============================================================================


class TestWorkerProcess(unittest.TestCase):
    """Direct test of the picklable worker function with mocked Pipeline."""

    def test_worker_process_basic(self):
        """Worker processes a single task and returns a result dict."""
        with patch("agent.Pipeline") as MockPipeline, \
             patch("agent.PipelineConfig") as MockPC:
            instance = MockPipeline.return_value
            instance.process.return_value = "ans1"

            task = {"_idx": 0, "task_id": "t1", "prompt": "q1"}
            result = _worker_process({"model_path": "x.gguf"}, task, 0)

            assert result["_idx"] == 0
            assert result["task_id"] == "t1"
            assert result["answer"] == "ans1"
            assert result["timing_ms"] >= 0
            assert result["worker"] == 0
            instance.process.assert_called_once_with("q1")
            instance.close.assert_called_once()

    def test_worker_process_handles_exception(self):
        """A task that throws should return empty answer."""
        with patch("agent.Pipeline") as MockPipeline, \
             patch("agent.PipelineConfig") as MockPC:
            instance = MockPipeline.return_value
            instance.process.side_effect = RuntimeError("OOM")

            task = {"_idx": 0, "task_id": "t1", "prompt": "q1"}
            result = _worker_process({"model_path": "x.gguf"}, task, 1)

            assert result["answer"] == ""
            assert result["worker"] == 1
            instance.close.assert_called_once()

    def test_worker_process_with_question_fallback(self):
        """Worker falls back to 'question' key when 'prompt' missing."""
        with patch("agent.Pipeline") as MockPipeline, \
             patch("agent.PipelineConfig") as MockPC:
            instance = MockPipeline.return_value
            instance.process.return_value = "fallback_ans"

            task = {"_idx": 0, "task_id": "t1", "question": "What?"}
            result = _worker_process({}, task, 0)
            instance.process.assert_called_once_with("What?")
            assert result["answer"] == "fallback_ans"


# =============================================================================
# BatchRunner — empty tasks
# =============================================================================


class TestEmptyTasks(unittest.TestCase):
    def test_empty_list_returns_empty(self):
        """Empty input → empty output immediately."""
        runner = BatchRunner()
        assert runner.run([]) == []

    def test_empty_list_run_parallel(self):
        """Convenience wrapper also handles empty."""
        assert run_parallel([]) == []


# =============================================================================
# BatchRunner — single task (main-process path)
# =============================================================================


class TestSingleTask(unittest.TestCase):
    """Single-task path avoids process-spawn overhead."""

    def test_single_task_returns_result(self):
        """Single task is processed and result dict has all fields."""
        with patch("agent.Pipeline") as MockPipeline, \
             patch("agent.PipelineConfig") as MockPC:
            instance = MockPipeline.return_value
            instance.process.return_value = "test answer"

            runner = BatchRunner()
            tasks = [{"task_id": "t1", "prompt": "What is 2+2?"}]
            results = runner.run(tasks)

            assert len(results) == 1
            r = results[0]
            assert r["task_id"] == "t1"
            assert r["answer"] == "test answer"
            assert "timing_ms" in r
            assert r["timing_ms"] >= 0
            assert r["worker"] == 0
            instance.process.assert_called_once_with("What is 2+2?")
            instance.close.assert_called_once()

    def test_single_task_handles_crash(self):
        """If single task crashes, answer is empty."""
        with patch("agent.Pipeline") as MockPipeline, \
             patch("agent.PipelineConfig") as MockPC:
            instance = MockPipeline.return_value
            instance.process.side_effect = RuntimeError("OOM")

            runner = BatchRunner()
            results = runner.run([{"task_id": "t1", "prompt": "test"}])

            assert len(results) == 1
            assert results[0]["answer"] == ""
            assert results[0]["task_id"] == "t1"
            instance.close.assert_called_once()

    def test_single_task_uses_question_fallback(self):
        """Falls back to 'question' key when 'prompt' missing."""
        with patch("agent.Pipeline") as MockPipeline, \
             patch("agent.PipelineConfig") as MockPC:
            instance = MockPipeline.return_value
            instance.process.return_value = "ans"

            runner = BatchRunner()
            results = runner.run([{"task_id": "t1", "question": "What?"}])

            instance.process.assert_called_once_with("What?")
            assert results[0]["answer"] == "ans"

    def test_single_task_generates_task_id(self):
        """Missing task_id gets auto-generated."""
        with patch("agent.Pipeline") as MockPipeline, \
             patch("agent.PipelineConfig") as MockPC:
            instance = MockPipeline.return_value
            instance.process.return_value = "ans"

            runner = BatchRunner()
            results = runner.run([{"prompt": "test"}])

            assert results[0]["task_id"] == "idx_0"


# =============================================================================
# BatchRunner — parallel execution (via ThreadPoolExecutor for testing)
# =============================================================================


class TestParallelExecution(unittest.TestCase):
    """Tests the parallel path using ThreadPoolExecutor (same-process)."""

    def test_preserves_input_order(self):
        """Results must be returned in the same order as input tasks."""
        with patch("agent.Pipeline") as MockPipeline, \
             patch("agent.PipelineConfig") as MockPC:
            instance = MockPipeline.return_value
            # Use prompt-based answers so they are deterministic across threads
            instance.process.side_effect = lambda p: f"ans_{p}"

            runner = BatchRunner(n_workers=2)
            runner._executor_class = concurrent.futures.ThreadPoolExecutor
            tasks = [
                {"task_id": "t1", "prompt": "q1"},
                {"task_id": "t2", "prompt": "q2"},
                {"task_id": "t3", "prompt": "q3"},
                {"task_id": "t4", "prompt": "q4"},
            ]
            results = runner.run(tasks)

            assert len(results) == 4
            assert [r["task_id"] for r in results] == ["t1", "t2", "t3", "t4"]
            assert [r["answer"] for r in results] == [
                "ans_q1", "ans_q2", "ans_q3", "ans_q4",
            ]

    def test_worker_crash_handling(self):
        """Worker crash should not crash the runner."""
        with patch("agent.Pipeline") as MockPipeline, \
             patch("agent.PipelineConfig") as MockPC:
            instance = MockPipeline.return_value
            instance.process.side_effect = RuntimeError("Worker crashed")

            runner = BatchRunner(n_workers=2)
            runner._executor_class = concurrent.futures.ThreadPoolExecutor
            tasks = [
                {"task_id": "t1", "prompt": "q1"},
                {"task_id": "t2", "prompt": "q2"},
                {"task_id": "t3", "prompt": "q3"},
            ]
            results = runner.run(tasks)

            # All tasks should have empty answers, but all results present
            assert len(results) == 3
            for r in results:
                assert r["answer"] == ""

    def test_some_workers_crash_still_returns_all(self):
        """Partial worker crash should still return all task results."""
        with patch("agent.Pipeline") as MockPipeline, \
             patch("agent.PipelineConfig") as MockPC:
            instance = MockPipeline.return_value
            call_log: list[str] = []

            def process_side(prompt: str) -> str:
                call_log.append(prompt)
                if prompt == "q1":
                    raise RuntimeError("Worker 0 crash")
                return f"answer_{prompt}"

            instance.process.side_effect = process_side

            runner = BatchRunner(n_workers=2)
            runner._executor_class = concurrent.futures.ThreadPoolExecutor
            tasks = [
                {"task_id": "t1", "prompt": "q1"},
                {"task_id": "t2", "prompt": "q2"},
                {"task_id": "t3", "prompt": "q3"},
            ]
            results = runner.run(tasks)

            # Even if some crash, all tasks should have results
            assert len(results) == 3
            # Map by task_id
            by_id = {r["task_id"]: r for r in results}
            assert by_id["t1"]["answer"] == ""
            assert by_id["t2"]["answer"] == "answer_q2"
            assert by_id["t3"]["answer"] == "answer_q3"

    def test_uneven_chunks(self):
        """Uneven task distribution across workers should still work."""
        with patch("agent.Pipeline") as MockPipeline, \
             patch("agent.PipelineConfig") as MockPC:
            instance = MockPipeline.return_value
            instance.process.side_effect = lambda p: f"ans_{p}"

            runner = BatchRunner(n_workers=3)
            runner._executor_class = concurrent.futures.ThreadPoolExecutor
            tasks = [{"task_id": f"t{i}", "prompt": f"q{i}"} for i in range(7)]
            results = runner.run(tasks)

            assert len(results) == 7
            assert [r["task_id"] for r in results] == [f"t{i}" for i in range(7)]


# =============================================================================
# BatchRunner — global deadline enforcement
# =============================================================================


class TestDeadline(unittest.TestCase):
    """Tests for global deadline enforcement."""

    def test_immediate_deadline_returns_empty(self):
        """deadline_s=0 means deadline already passed → no tasks processed."""
        with patch("agent.Pipeline") as MockPipeline, \
             patch("agent.PipelineConfig") as MockPC:
            instance = MockPipeline.return_value
            instance.process.return_value = "ans"

            runner = BatchRunner(n_workers=2)
            runner._executor_class = concurrent.futures.ThreadPoolExecutor
            tasks = [{"task_id": "t1", "prompt": "q1"}, {"task_id": "t2", "prompt": "q2"}]
            results = runner.run(tasks, deadline_s=0.0)

            # deadline = time.monotonic() + 0 is already ≤ time.monotonic()
            # → sequential fallback, first check breaks
            assert len(results) == 0

    def test_deadline_near_switches_to_sequential(self):
        """When deadline is near (≤10s), runner switches to sequential execution."""
        with patch("agent.Pipeline") as MockPipeline, \
             patch("agent.PipelineConfig") as MockPC:
            instance = MockPipeline.return_value
            instance.process.return_value = "ans"

            runner = BatchRunner(n_workers=2)
            runner._executor_class = concurrent.futures.ThreadPoolExecutor
            tasks = [{"task_id": f"t{i}", "prompt": f"q{i}"} for i in range(3)]
            # deadline_s=5 means sequential path (>0 remaining but <10)
            results = runner.run(tasks, deadline_s=5.0)

            assert len(results) == 3
            for r in results:
                assert r["answer"] == "ans"

    def test_deadline_hard_stop_during_sequential(self):
        """Sequential fallback stops mid-way when deadline expires."""
        real_monotonic = time.monotonic

        with patch("agent.Pipeline") as MockPipeline, \
             patch("agent.PipelineConfig") as MockPC:
            instance = MockPipeline.return_value

            # Simulate time advancing past deadline after 1 task
            call_times: list[float] = []

            def clock() -> float:
                idx = len(call_times)
                if idx == 0:
                    call_times.append(1000.0)  # deadline base
                    return 1000.0
                elif idx < 4:
                    call_times.append(1000.0 + idx * 0.001)  # first few calls ~ same time
                    return 1000.0 + idx * 0.001
                else:
                    call_times.append(1015.0)  # deadline exceeded
                    return 1015.0

            def process_side(prompt: str) -> str:
                return f"ans_{prompt}"

            instance.process.side_effect = process_side

            with patch("time.monotonic", side_effect=clock):
                runner = BatchRunner(n_workers=2)
                runner._executor_class = concurrent.futures.ThreadPoolExecutor
                tasks = [
                    {"task_id": "t1", "prompt": "q1"},
                    {"task_id": "t2", "prompt": "q2"},
                    {"task_id": "t3", "prompt": "q3"},
                ]
                # deadline = 1000.0 + 10.0 = 1010.0
                results = runner.run(tasks, deadline_s=10.0)

            # After q1, time advances to 1015 which is past deadline 1010 → stop
            assert len(results) == 1
            assert results[0]["task_id"] == "t1"


# =============================================================================
# BatchRunner — per-task timeout
# =============================================================================


class TestPerTaskTimeout(unittest.TestCase):
    """Per-task timeout is enforced via pipeline's inference_timeout_s."""

    def test_task_timeout_returns_empty(self):
        """When pipeline.process times out, answer is empty."""
        with patch("agent.Pipeline") as MockPipeline, \
             patch("agent.PipelineConfig") as MockPC:
            instance = MockPipeline.return_value
            instance.process.side_effect = TimeoutError("Task timed out")

            runner = BatchRunner()
            tasks = [{"task_id": "t1", "prompt": "long task"}]
            results = runner.run(tasks)

            assert len(results) == 1
            assert results[0]["answer"] == ""


# =============================================================================
# BatchRunner — convenience wrapper
# =============================================================================


class TestRunParallel(unittest.TestCase):
    """run_parallel convenience function."""

    def test_run_parallel_empty(self):
        """Convenience function handles empty."""
        assert run_parallel([]) == []

    def test_run_parallel_single(self):
        """Convenience function with single task."""
        with patch("agent.Pipeline") as MockPipeline, \
             patch("agent.PipelineConfig") as MockPC:
            instance = MockPipeline.return_value
            instance.process.return_value = "ans"

            results = run_parallel([{"task_id": "t1", "prompt": "q"}])
            assert len(results) == 1
            assert results[0]["answer"] == "ans"

    def test_run_parallel_with_config(self):
        """Convenience function passes config."""
        with patch("agent.Pipeline") as MockPipeline, \
             patch("agent.PipelineConfig") as MockPC:
            instance = MockPipeline.return_value
            instance.process.return_value = "ans"

            # Should not crash with custom n_workers
            results = run_parallel(
                [{"task_id": "t1", "prompt": "q"}, {"task_id": "t2", "prompt": "q2"}],
                n_workers=2,
            )
            assert len(results) == 2


# =============================================================================
# BatchRunner — n_workers capping
# =============================================================================


class TestWorkerCapping(unittest.TestCase):
    """n_workers is capped by memory budget."""

    def test_n_workers_capped(self):
        """Requesting more workers than memory allows should cap."""
        runner = BatchRunner(n_workers=10, max_ram_gb=4.0)
        assert runner.n_workers == 2  # capped by _compute_workers

    def test_memory_budget_limits(self):
        """Small memory budget limits workers regardless of request."""
        runner = BatchRunner(n_workers=5, max_ram_gb=1.5)
        assert runner.n_workers == 1  # only 1 fits in 1.5 GB


# =============================================================================
# BatchRunner — edge cases
# =============================================================================


class TestEdgeCases(unittest.TestCase):
    def test_tasks_without_task_id(self):
        """Tasks without task_id get auto-generated IDs."""
        with patch("agent.Pipeline") as MockPipeline, \
             patch("agent.PipelineConfig") as MockPC:
            instance = MockPipeline.return_value
            instance.process.return_value = "ans"

            runner = BatchRunner(n_workers=2)
            runner._executor_class = concurrent.futures.ThreadPoolExecutor
            tasks = [
                {"prompt": "q1"},
                {"prompt": "q2"},
            ]
            results = runner.run(tasks)
            assert len(results) == 2
            # IDs should be unique and not empty
            assert results[0]["task_id"]
            assert results[1]["task_id"]
            assert results[0]["task_id"] != results[1]["task_id"]

    def test_mixed_prompt_and_question_keys(self):
        """Tasks can have either 'prompt' or 'question' keys."""
        with patch("agent.Pipeline") as MockPipeline, \
             patch("agent.PipelineConfig") as MockPC:
            instance = MockPipeline.return_value
            instance.process.side_effect = lambda p: f"ans_{p}"

            runner = BatchRunner(n_workers=2)
            runner._executor_class = concurrent.futures.ThreadPoolExecutor
            tasks = [
                {"task_id": "t1", "prompt": "hello"},
                {"task_id": "t2", "question": "world"},
            ]
            results = runner.run(tasks)
            assert len(results) == 2
            assert results[0]["answer"] == "ans_hello"
            assert results[1]["answer"] == "ans_world"


if __name__ == "__main__":
    unittest.main()
