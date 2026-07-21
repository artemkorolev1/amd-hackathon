"""
math_cascade.py — Cascading deterministic pipeline for GSM8K math problems.

Combines the step counter and step type classifier into a single pipeline:
  1. math_step_counter.predict_step_count(question) → step_count
  2. math_step_classifier.predict_step_type(question, position, step_count) → operation
  3. Output structured plan.

Also provides evaluation against ground truth.
"""

import json
from typing import Dict, List, Optional, Union

from agent.solvers.math_step_counter import (
    predict_step_count,
    parse_ground_truth_steps as counter_parse_steps,
)
from agent.solvers.math_step_classifier import (
    predict_step_type,
    parse_ground_truth_plan,
)


def predict_pipeline(question: str) -> Dict:
    """
    Run the full deterministic cascade on a math word problem.

    Args:
        question: The GSM8K question text.

    Returns:
        Dict with:
            - step_count: predicted number of steps
            - steps: list of {pos, op} dicts
            - question: original question
    """
    # Step 1: Predict step count
    step_count = predict_step_count(question)

    # Step 2: For each position, predict operation type
    steps = []
    for pos in range(1, step_count + 1):
        op = predict_step_type(question, pos, step_count)
        steps.append({"pos": pos, "op": op})

    return {
        "question": question,
        "step_count": step_count,
        "steps": steps,
    }


def predict_pipeline_with_ground_truth(question: str, answer: str) -> Dict:
    """
    Run pipeline and compare against ground truth parsed from answer.

    Returns predicted plan, true plan, and correctness flags.
    """
    predicted = predict_pipeline(question)
    true_steps = parse_ground_truth_plan(answer)

    # Build true plan
    true_step_count = len(true_steps)
    if true_step_count == 0:
        true_step_count = 1
        true_steps = [{"pos": 1, "op": "add"}]

    true_plan = {
        "step_count": true_step_count,
        "steps": true_steps,
    }

    # Compare step count
    step_count_correct = predicted["step_count"] == true_step_count

    # Compare step types per position
    step_type_correct = {}
    for step in predicted["steps"]:
        pos = step["pos"]
        true_op = None
        for ts in true_plan["steps"]:
            if ts["pos"] == pos:
                true_op = ts["op"]
                break
        step_type_correct[pos] = step["op"] == true_op

    # Full plan match (step count + all step types)
    plan_correct = step_count_correct and all(step_type_correct.values())

    return {
        "predicted": predicted,
        "true": true_plan,
        "step_count_correct": step_count_correct,
        "step_type_correct": step_type_correct,
        "plan_correct": plan_correct,
    }


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_on_dataframe(df) -> Dict:
    """
    Run full cascade evaluation on a DataFrame with 'question' and 'answer' columns.

    Returns detailed accuracy metrics.
    """
    total = len(df)
    step_count_correct = 0
    plan_correct = 0
    step_type_correct_total = 0
    step_type_total = 0

    # Per-position step type accuracy
    per_position = {}  # pos -> {correct: N, total: N}
    # Per-operation accuracy
    per_operation = {}  # op -> {correct: N, total: N}

    results = []

    for _, row in df.iterrows():
        question = row["question"]
        answer = row["answer"]

        eval_result = predict_pipeline_with_ground_truth(question, answer)

        if eval_result["step_count_correct"]:
            step_count_correct += 1

        if eval_result["plan_correct"]:
            plan_correct += 1

        # Step type accuracy per position
        for pos, correct in eval_result["step_type_correct"].items():
            if pos not in per_position:
                per_position[pos] = {"correct": 0, "total": 0}
            per_position[pos]["total"] += 1
            step_type_total += 1
            if correct:
                per_position[pos]["correct"] += 1
                step_type_correct_total += 1

            # True operation for this position
            true_op = None
            for ts in eval_result["true"]["steps"]:
                if ts["pos"] == pos:
                    true_op = ts["op"]
                    break
            if true_op:
                if true_op not in per_operation:
                    per_operation[true_op] = {"correct": 0, "total": 0}
                per_operation[true_op]["total"] += 1
                if correct:
                    per_operation[true_op]["correct"] += 1

        results.append(eval_result)

    # Aggregated metrics
    pos_acc = {}
    for p in sorted(per_position):
        d = per_position[p]
        pos_acc[p] = round(d["correct"] / d["total"], 4) if d["total"] > 0 else 0.0

    op_acc = {}
    for op in sorted(per_operation):
        d = per_operation[op]
        op_acc[op] = round(d["correct"] / d["total"], 4) if d["total"] > 0 else 0.0

    return {
        "total_examples": total,
        "step_count_accuracy": round(step_count_correct / total, 4) if total > 0 else 0.0,
        "step_count_correct": step_count_correct,
        "step_type_accuracy": round(step_type_correct_total / step_type_total, 4) if step_type_total > 0 else 0.0,
        "step_type_correct": step_type_correct_total,
        "step_type_total": step_type_total,
        "plan_accuracy": round(plan_correct / total, 4) if total > 0 else 0.0,
        "plans_correct": plan_correct,
        "per_position_accuracy": pos_acc,
        "per_position_counts": {p: per_position[p]["total"] for p in sorted(per_position)},
        "per_operation_accuracy": op_acc,
    }


def pretty_print_results(metrics: Dict) -> str:
    """Format evaluation results as a readable string."""
    lines = []
    lines.append("=" * 60)
    lines.append("CASCADE PIPELINE EVALUATION RESULTS")
    lines.append("=" * 60)
    lines.append(f"Total examples: {metrics['total_examples']}")
    lines.append("")
    lines.append(f"Step Count Accuracy:    {metrics['step_count_accuracy']:.2%}  "
                 f"({metrics['step_count_correct']}/{metrics['total_examples']})")
    lines.append(f"Step Type Accuracy:     {metrics['step_type_accuracy']:.2%}  "
                 f"({metrics['step_type_correct']}/{metrics['step_type_total']})")
    lines.append(f"Combined Plan Accuracy: {metrics['plan_accuracy']:.2%}  "
                 f"({metrics['plans_correct']}/{metrics['total_examples']})")
    lines.append("")
    lines.append("--- Per-Position Step Type Accuracy ---")
    for p in sorted(metrics["per_position_accuracy"]):
        cnt = metrics["per_position_counts"][p]
        acc = metrics["per_position_accuracy"][p]
        lines.append(f"  Position {p}: {acc:.2%} ({cnt} examples)")
    lines.append("")
    lines.append("--- Per-Operation Step Type Accuracy ---")
    for op in sorted(metrics["per_operation_accuracy"]):
        acc = metrics["per_operation_accuracy"][op]
        lines.append(f"  {op}: {acc:.2%}")
    lines.append("=" * 60)
    return "\n".join(lines)
