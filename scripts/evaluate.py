#!/usr/bin/env python3
"""
Evaluation harness for AMD hackathon agent.

Simulates the judging harness: reads tasks, runs the agent (or reads pre-existing
output from stdin), compares each answer against ground truth using fuzzy matching,
and reports accuracy metrics.

Usage:
    # Run agent and evaluate
    python evaluate.py --tasks tasks.txt --ground-truth ground_truth.txt

    # Evaluate from pre-existing answers (no agent run)
    python evaluate.py --no-run < agent_output.txt

    # Custom agent command
    python evaluate.py --agent-cmd python -m agent.main
"""

import argparse
import os
import subprocess
import sys
from typing import List, Tuple

# Re-export grading functions from the shared module so existing importers
# (runner/evaluate.py, tests/test_evaluate.py) continue working unchanged.
from scripts.grade_answer import extract_numbers, fuzzy_match, grade_answer, tokenize  # noqa: F401


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_tasks(path: str) -> List[str]:
    """Load tasks, grouping by blank lines and merging 'Options:' with preceding 'Problem:'."""
    with open(path) as f:
        raw = f.read()
    
    # Split on blank lines first
    groups = [g.strip() for g in raw.strip().split("\n\n") if g.strip()]
    
    # Merge: if a group starts with "Options:", merge with previous group
    merged = []
    for g in groups:
        if g.startswith("Options:") and merged:
            merged[-1] = merged[-1] + "\n" + g
        else:
            merged.append(g)
    return merged


def load_ground_truth(path: str) -> List[str]:
    """Load ground truth answers, grouping by blank lines (some answers span multiple lines)."""
    with open(path) as f:
        raw = f.read()
    groups = [g.strip() for g in raw.strip().split("\n\n") if g.strip()]
    return groups


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

def run_agent(tasks: List[str], agent_cmd: List[str]) -> List[str]:
    """Run the agent with *tasks* passed as JSON via TASKS env var (supports multi-line)."""
    import json as _json
    
    cwd = os.path.dirname(os.path.abspath(__file__))

    # Build the default command if none provided
    venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
    default_python = venv_python if os.path.exists(venv_python) else sys.executable
    cmd = agent_cmd or [default_python, "-m", "agent.main"]

    # Pass tasks as JSON via env var so multi-line tasks work
    env = os.environ.copy()
    env["TASKS"] = _json.dumps(tasks)
    env["FIREWORKS_API_KEY"] = env.get("FIREWORKS_API_KEY", os.environ.get("FIREWORKS_API_KEY", ""))

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=600,
            env=env,
        )
    except FileNotFoundError:
        print(f"[ERROR] Agent command not found: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("[ERROR] Agent timed out after 600 seconds", file=sys.stderr)
        sys.exit(1)

    if proc.returncode != 0:
        print(f"[ERROR] Agent exited with code {proc.returncode}", file=sys.stderr)
        if proc.stderr:
            print("stderr:", proc.stderr[:2000], file=sys.stderr)

    return [l.strip() for l in proc.stdout.split("\n") if l.strip()]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate AMD hackathon agent against ground truth."
    )
    parser.add_argument(
        "--tasks",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "tasks.txt"),
        help="Path to tasks file (one per line, default: tasks.txt)",
    )
    parser.add_argument(
        "--ground-truth",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "ground_truth.txt"),
        help="Path to ground truth file (one expected answer per line, "
             "default: ground_truth.txt)",
    )
    parser.add_argument(
        "--agent-cmd",
        nargs="+",
        default=None,
        help="Agent command (default: python -m agent.main)",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Skip running the agent; read answers from stdin instead",
    )
    args = parser.parse_args()

    # ---- Load data ----
    base_dir = os.path.dirname(os.path.abspath(__file__))

    tasks_path = args.tasks if os.path.isabs(args.tasks) else os.path.join(base_dir, args.tasks)
    gt_path = args.ground_truth if os.path.isabs(args.ground_truth) else os.path.join(base_dir, args.ground_truth)

    ground_truth = load_ground_truth(gt_path)
    tasks = load_tasks(tasks_path)

    print(f"[evaluate] Loaded {len(tasks)} tasks from {tasks_path}", file=sys.stderr)
    print(f"[evaluate] Loaded {len(ground_truth)} ground-truth answers from {gt_path}", file=sys.stderr)

    # Pad or trim to the shorter list
    n = min(len(tasks), len(ground_truth))
    if len(tasks) != len(ground_truth):
        print(f"[evaluate] WARNING: task count ({len(tasks)}) != ground-truth count "
              f"({len(ground_truth)}); using first {n}",
              file=sys.stderr)
        tasks = tasks[:n]
        ground_truth = ground_truth[:n]

    # ---- Run agent ----
    if args.no_run:
        answers = [l.strip() for l in sys.stdin if l.strip()]
        print(f"[evaluate] Read {len(answers)} answers from stdin", file=sys.stderr)
    else:
        agent_cmd = args.agent_cmd
        print(f"[evaluate] Running agent: {agent_cmd or 'python -m agent.main'}",
              file=sys.stderr)
        answers = run_agent(tasks, agent_cmd)
        print(f"[evaluate] Agent produced {len(answers)} answers", file=sys.stderr)

    if len(answers) > n:
        print(f"[evaluate] WARNING: agent returned {len(answers)} answers but "
              f"only {n} tasks; using first {n}", file=sys.stderr)
        answers = answers[:n]
    elif len(answers) < n:
        print(f"[evaluate] WARNING: agent returned only {len(answers)} answers "
              f"for {n} tasks; grading what we have", file=sys.stderr)

    # ---- Grade each answer ----
    results: List[Tuple[int, str, str, bool, str]] = []
    for i in range(n):
        task = tasks[i]
        expected = ground_truth[i]
        answer = answers[i] if i < len(answers) else ""
        passed, reason = grade_answer(answer, expected)
        results.append((i + 1, task, answer, passed, reason))

    # ---- Report ----
    passed_count = sum(1 for r in results if r[3])
    total = len(results)
    accuracy = (passed_count / total * 100) if total > 0 else 0.0
    passes_gate = accuracy >= 84.2

    print()
    print("=" * 80)
    print("  AMD HACKATHON — EVALUATION RESULTS")
    print("=" * 80)

    for idx, task, answer, passed, reason in results:
        icon = "✅ PASS" if passed else "❌ FAIL"
        print(f"\n  {icon}  Task {idx}: {task}")
        if answer:
            print(f"       Answer: {answer[:150]}")
        if not passed:
            print(f"       Reason: {reason}")

    print()
    print("=" * 80)
    print(f"  SUMMARY")
    print(f"    Accuracy:        {passed_count}/{total} ({accuracy:.1f}%)")
    if passes_gate:
        print(f"    84.2% gate:      ✅ PASSED ({accuracy:.1f}% ≥ 84.2%)")
    else:
        print(f"    84.2% gate:      ❌ FAILED ({accuracy:.1f}% < 84.2%)")
    print("=" * 80)

    return 0 if passes_gate else 1


if __name__ == "__main__":
    sys.exit(main())
