#!/usr/bin/env python3
"""staging/entrypoint.py — Parallel submission container entrypoint.

Reads tasks from /input/tasks.json, runs bulk classification, dispatches
to workers, collects 5 answers per task, judges via majority vote.

When DETAILED_OUTPUT=1 is set, additionally writes results_detailed.json
(with full instrumentation per task) and timing.json alongside results.json.

Does NOT import agent.Pipeline — imports only specific agent modules
(category_filter, solvers) used by workers and judge.
"""

import json
import logging
import os
import signal
import sys
import time

import threading

from staging import ReadyConfig, ReadyQueue, ReadyMonitor, ReadyJudge
from staging.ready_queue import ReadyTask

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("staging-entrypoint")


def _read_tasks() -> list[dict]:
    """Read task list from /input/tasks.json, sys.argv[1], or stdin."""
    input_path = "/input/tasks.json"
    if os.path.exists(input_path):
        try:
            with open(input_path) as f:
                data = json.load(f)
            questions = data.get("questions", data) if isinstance(data, dict) else data
            logger.info("Read %d tasks from %s", len(questions), input_path)
            return questions
        except Exception as e:
            logger.warning("Failed to read %s: %s", input_path, e)

    if len(sys.argv) > 1:
        eval_path = sys.argv[1]
        with open(eval_path) as f:
            data = json.load(f)
        questions = data.get("questions", data) if isinstance(data, dict) else data
        logger.info("Read %d tasks from %s", len(questions), eval_path)
        return questions

    tasks: list[dict] = []
    for i, line in enumerate(sys.stdin):
        line = line.strip()
        if line:
            tasks.append({"task_id": f"idx_{i}", "prompt": line})
    logger.info("Read %d tasks from stdin", len(tasks))
    return tasks


def _write_output(results: list[dict] | dict, path: str = "/output/results.json") -> None:
    """Atomically write results to path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, ensure_ascii=False)
    os.replace(tmp, path)
    logger.info("Wrote %d results to %s", len(results), path)


def _limit_tasks(tasks: list[dict]) -> list[dict]:
    """Apply TASK_COUNT limit if set."""
    try:
        task_count = int(os.environ.get("TASK_COUNT", "0"))
        if task_count > 0 and len(tasks) > task_count:
            tasks = tasks[:task_count]
            logger.info("Limited to %d tasks via TASK_COUNT", task_count)
    except (ValueError, TypeError):
        pass
    return tasks


_output_written = False
_shutdown_results: list[dict] = []


def _sigterm_handler(signum, frame) -> None:
    """On SIGTERM/SIGINT: write whatever results we have."""
    global _output_written, _shutdown_results
    if not _output_written and _shutdown_results:
        _write_output(_shutdown_results)
    sys.exit(0)


# ════════════════════════════════════════════════════════════════════════
# Detailed output builder (used when DETAILED_OUTPUT=1)
# ════════════════════════════════════════════════════════════════════════


def _build_detailed_output(
    final_results: list[dict],
    judge: ReadyJudge,
    ready_tasks: list[ReadyTask],
    timing_data: dict,
    classify_timing_per_task: dict[str, float],
) -> list[dict]:
    """Build full instrumentation output for each task.

    Args:
        final_results: Output of judge.judge_all()
        judge: ReadyJudge instance (for get_answer_details)
        ready_tasks: Original ReadyTask objects (for classification meta)
        timing_data: Aggregate timing dict (per-stage)
        classify_timing_per_task: task_id -> classification timing_ms

    Returns:
        List of detailed per-task dicts matching the instrumentation schema.
    """
    # Build task_id -> classification meta lookup
    cls_meta: dict[str, dict] = {}
    for rt in ready_tasks:
        cls_meta[rt.task_id] = {
            "category": rt.category,
            "category_4way": rt.category_4way,
            "confidence": rt.confidence,
            "score_delta": rt.score_delta,
            "raw_scores": rt.raw_scores,
        }

    detailed = []
    for r in final_results:
        tid = r["task_id"]
        cm = cls_meta.get(tid, {})
        detailed.append({
            "task_id": tid,
            "answer": r["answer"],
            "category": cm.get("category", ""),
            "category_4way": cm.get("category_4way", ""),
            "classification": {
                "category": cm.get("category", ""),
                "category_4way": cm.get("category_4way", ""),
                "confidence": cm.get("confidence", 0.5),
                "score_delta": cm.get("score_delta", 0.0),
                "raw_scores": cm.get("raw_scores", {}),
                "timing_ms": classify_timing_per_task.get(tid, 0),
            },
            "worker_answers": judge.get_answer_details(tid),
            "judgment": r.get("_judgment", {}),
            "total_timing_ms": (
                sum(
                    a.get("timing_ms", 0) or 0
                    for a in judge.get_answer_details(tid)
                )
                + classify_timing_per_task.get(tid, 0)
            ),
            "judged_by": "ready_judge",
        })
    return detailed


def _build_timing_data(
    t_start: float,
    classify_s: float,
    judge: ReadyJudge,
    total_tasks: int,
) -> dict:
    """Build aggregate timing summary dict."""
    # Worker setup time: elapsed between start and classification end
    worker_setup_s = time.monotonic() - t_start - classify_s
    if worker_setup_s < 0:
        worker_setup_s = 0

    timing = {
        "classification_s": round(classify_s, 2),
        "worker_setup_s": round(worker_setup_s, 2),
        "total_s": round(time.monotonic() - t_start, 1),
        "total_tasks": total_tasks,
        "per_worker_type": judge.get_timing_summary(),
    }
    return timing


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════


def main() -> None:
    global _shutdown_results
    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGINT, _sigterm_handler)

    t_start = time.monotonic()
    detailed_output = int(os.environ.get("DETAILED_OUTPUT", "0"))

    # 1. Load config
    config = ReadyConfig.from_env()
    logger.info("Config: %d workers (%d FW + %d Loc + %d Det), %d votes, %.0fs deadline",
                config.total_workers, config.fw_workers, config.loc_workers,
                config.det_workers, config.judgment_votes, config.deadline_s)

    # Resource-aware worker budgeting
    if int(os.environ.get("RESOURCE_AWARE", "1")):
        from agent.resource_manager import ResourceManager
        rm = ResourceManager()
        initial = rm.probe()
        logger.info("Resources: %.1fGB VRAM, %.0f cores, %.1fGB RAM (GPU=%s)",
                    initial.vram_free_gb, initial.cpu_cores, initial.ram_free_gb,
                    initial.gpu_available)

        demands = config.build_resource_demands()
        allocated = rm.budget_workers(demands)

        # Update config counts based on what fits
        config.loc_workers = 0
        config.det_workers = 0
        config.fw_workers = 0
        for a in allocated:
            if a.worker_type == "local":
                config.loc_workers += a.count
            elif a.worker_type == "deterministic":
                config.det_workers += a.count
            elif a.worker_type == "fireworks":
                config.fw_workers += a.count

        # If local workers were reduced, slice model configs to match
        if len(config.loc_model_configs) > config.loc_workers:
            config.loc_model_configs = config.loc_model_configs[:config.loc_workers]

        logger.info("Resource budget: %d local, %d det, %d fw workers",
                    config.loc_workers, config.det_workers, config.fw_workers)
    else:
        logger.info("Resource-aware budgeting disabled via RESOURCE_AWARE=0")

    deadline = time.monotonic() + config.deadline_s

    # 2. Read tasks
    tasks = _read_tasks()
    if not tasks:
        logger.warning("No tasks — writing empty output")
        _write_output([])
        return

    tasks = _limit_tasks(tasks)
    total_tasks = len(tasks)

    # 3. Bulk classify all tasks (with timing)
    logger.info("Bulk classifying %d tasks...", total_tasks)
    prompts = [t.get("prompt", t.get("question", "")) for t in tasks]

    t_classify = time.monotonic()
    classify_timing_per_task: dict[str, float] = {}
    try:
        from staging.ready_classifier import classify_batch
        classified = classify_batch(prompts)
    except Exception as exc:
        logger.error("Bulk classification failed: %s — falling back to 'factual' for all tasks", exc)
        classified = [{"category": "factual", "category_4way": "knowledge",
                       "raw_scores": {}, "confidence": 0.5, "score_delta": 0.0}
                      for _ in prompts]
    t_classify_elapsed = time.monotonic() - t_classify
    classify_s = t_classify_elapsed
    classify_ms_per_task = (t_classify_elapsed * 1000 / max(len(prompts), 1))
    logger.info("Bulk classification complete in %.2fs (%.1fms/task)",
                t_classify_elapsed, classify_ms_per_task)

    # 4. Build ReadyQueue
    queue = ReadyQueue()
    ready_tasks = []
    for i, (task, cls) in enumerate(zip(tasks, classified)):
        tid = task.get("task_id", f"task_{i}")
        prompt = task.get("prompt", task.get("question", ""))
        ready_task = ReadyTask(
            task_id=tid,
            prompt=prompt,
            category=cls.get("category", "factual"),
            category_4way=cls.get("category_4way", "knowledge"),
            raw_scores=cls.get("raw_scores", {}),
            confidence=cls.get("confidence", 0.5),
            score_delta=cls.get("score_delta", 0.0),
        )
        ready_tasks.append(ready_task)
        classify_timing_per_task[tid] = classify_ms_per_task

    queue.enqueue_batch(ready_tasks)
    logger.info("Enqueued %d tasks by category: %s",
                total_tasks, queue.task_counts_by_category())

    # 5. Start monitor + judge (Pull-System orchestration)
    judge = ReadyJudge(config)
    monitor = ReadyMonitor(config)
    stop_event = threading.Event()

    # Drain queue to shared task_pool and spawn workers
    monitor.start(queue)

    # Run judge in autonomous daemon thread
    judge_thread = threading.Thread(
        target=judge.consume_loop,
        args=(monitor._results_queue, monitor._deadline_emergency, stop_event),
        daemon=True,
    )
    judge_thread.start()

    try:
        # Monitor loop (blocking in main thread) — health checks, re-enqueue, deadline
        monitor.monitor_loop(judge, deadline)
    except Exception as exc:
        logger.exception("Fatal error in monitor loop: %s", exc)
    finally:
        # Stop judge thread
        stop_event.set()
        judge_thread.join(timeout=2.0)

        # Final drain of any remaining results
        monitor._running.clear()
        judge.ingest_results(monitor._results_queue)

        # 6. Judge all remaining tasks
        logger.info("Judging %d pending tasks...", len(judge.pending_tasks))
        final_results = judge.judge_all()
        _shutdown_results = [{"task_id": r["task_id"], "answer": r["answer"]} for r in final_results]
        logger.info("Final: %d/%d tasks judged", len(final_results), total_tasks)

        # 7. Log judgment strategy distribution
        strategy_counts = {}
        for r in final_results:
            strat = r.get("_judgment", {}).get("strategy", "unknown")
            strategy_counts[strat] = strategy_counts.get(strat, 0) + 1
        for strat, count in sorted(strategy_counts.items()):
            logger.info("  Strategy '%s': %d tasks", strat, count)

        # 8. Write standard output (strip _judgment metadata for grader contract)
        output = [{"task_id": r["task_id"], "answer": r["answer"]} for r in final_results]
        _write_output(output)
        _output_written = True

        # 9. Write detailed output (when DETAILED_OUTPUT=1)
        if detailed_output:
            timing_data = _build_timing_data(t_start, classify_s, judge, total_tasks)
            detailed = _build_detailed_output(
                final_results, judge, ready_tasks, timing_data, classify_timing_per_task,
            )
            _write_output(detailed, "/output/results_detailed.json")
            # Write timing.json as a single-object JSON (wrapped for type consistency)
            _write_output([timing_data], "/output/timing.json")

        elapsed = time.monotonic() - t_start
        logger.info("Total time: %.1fs", elapsed)

        # 10. Shutdown
        monitor.shutdown()


if __name__ == "__main__":
    main()
