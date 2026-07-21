#!/usr/bin/env python3
"""AMD ACT II Track 1 — CLI entrypoint (thin wrapper around agent.Pipeline).

Reads tasks from /input/tasks.json (grader mount), sys.argv[1], or stdin.
Writes answers to /output/results.json and prints one line per answer to stdout.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

from agent import Pipeline

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger("harness")


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


def _write_output(results: list[dict]) -> None:
    """Atomically write results to /output/results.json."""
    os.makedirs("/output", exist_ok=True)
    tmp = "/output/results.json.tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, ensure_ascii=False)
    os.replace(tmp, "/output/results.json")


def main() -> None:
    tasks = _read_tasks()
    try:
        task_count = int(os.environ.get("TASK_COUNT", "0"))
        if task_count > 0 and len(tasks) > task_count:
            tasks = tasks[:task_count]
            logger.info("Limited to %d tasks via TASK_COUNT", task_count)
    except (ValueError, TypeError):
        pass
    if not tasks:
        logger.warning("No tasks to process — writing empty output")
        _write_output([])
        return

    pipe = Pipeline()
    deadline = time.monotonic() + pipe.cfg.deadline_s
    results: list[dict] = []

    try:
        for i, q in enumerate(tasks):
            if time.monotonic() >= deadline:
                logger.warning("Deadline reached after %d/%d tasks — stopping",
                               i, len(tasks))
                break

            if isinstance(q, str):
                tid = f"task_{i}"
                prompt = q
            else:
                tid = q.get("task_id", f"task_{i}")
                prompt = q.get("prompt", q.get("question", ""))
            try:
                answer = pipe.process(prompt)
            except Exception as exc:
                logger.exception("Task %s failed — returning empty", tid)
                answer = ""

            results.append({"task_id": tid, "answer": answer})
            if (i + 1) % 5 == 0:
                _write_output(results)
            print(answer.replace("\n", "\\n"))
    finally:
        pipe.close()
        _write_output(results)
        logger.warning("Wrote %d results to /output/results.json", len(results))


if __name__ == "__main__":
    main()
