#!/usr/bin/env python3
"""
Evaluation script for the logic classifier cascade.

Loads training-v1.json and other eval files, routes each logic prompt through
the cascade, and reports:
  - Per-level hit rate (% routed to each tool)
  - Overall solve rate (% where solver returned non-None)
  - Accuracy vs expected_answer using fuzzy_match
  - Per-subtype accuracy (zebra vs logiqa)
  - Rejection precision/recall on non-logic items
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agent.solvers.logic_classifier_cascade import route_logic as route_logic_cascade
from agent.solvers.logic_classifier_cascade import (
    is_truth_teller, is_sequence, is_syllogism,
    is_constraint_puzzle, is_argument_analysis,
    _ROUTE_NAMES,
)
from scripts.grade_answer import fuzzy_match

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)


def classify_cascade_level(prompt: str) -> Tuple[Optional[str], float]:
    """Determine which level of the cascade would match."""
    for level, name in sorted(_ROUTE_NAMES.items()):
        classifier = {
            1: is_truth_teller,
            2: is_sequence,
            3: is_syllogism,
            4: is_constraint_puzzle,
            5: is_argument_analysis,
        }[level]
        matched, conf = classifier(prompt)
        if matched:
            return name, conf
    return None, 0.0


def run_accuracy_eval(entries: List[Dict], label: str) -> Dict[str, Any]:
    """Run the cascade on logic items and compare against expected answers."""
    results = {
        "label": label,
        "total": len(entries),
        "route_counts": Counter(),  # level -> count
        "route_confidences": defaultdict(list),  # level -> list of confs
        "solve_counts": Counter(),  # "solved" / "unsolved"
        "accuracy_counts": Counter(),  # "correct" / "incorrect" / "no_answer"
        "by_subtype": defaultdict(lambda: {"total": 0, "solved": 0, "correct": 0}),
        "errors": [],
    }

    for entry in entries:
        prompt = entry["prompt"]
        expected = entry["expected_answer"]
        subtype = "zebra" if "zebra" in entry.get("source", "").lower() or "zebra" in entry.get("task_id", "").lower() else "logiqa"
        task_id = entry.get("task_id", "unknown")

        # Classify level
        level_name, conf = classify_cascade_level(prompt)
        results["route_counts"][level_name or "fallback"] += 1
        if level_name:
            results["route_confidences"][level_name].append(conf)

        # Run routing
        try:
            answer = route_logic_cascade(prompt)
        except Exception as e:
            answer = None
            results["errors"].append(f"{task_id}: Exception: {e}")

        # Check if solved
        if answer is not None:
            results["solve_counts"]["solved"] += 1
            results["by_subtype"][subtype]["solved"] += 1
        else:
            results["solve_counts"]["unsolved"] += 1

        # Check accuracy
        if answer is not None and expected:
            is_correct = fuzzy_match(answer, expected)
            if is_correct:
                results["accuracy_counts"]["correct"] += 1
                results["by_subtype"][subtype]["correct"] += 1
            else:
                results["accuracy_counts"]["incorrect"] += 1
                if len(results["errors"]) < 20:
                    results["errors"].append(
                        f"{task_id} ({subtype}): "
                        f"routed={level_name}, "
                        f"expected={expected[:80]!r}, "
                        f"got={answer[:80]!r}"
                    )
        elif answer is None:
            results["accuracy_counts"]["no_answer"] += 1

        results["by_subtype"][subtype]["total"] += 1

    return results


def run_rejection_eval(logic_entries: List[Dict], all_entries: List[Dict]) -> Dict[str, Any]:
    """Evaluate how well the cascade rejects non-logic items.

    A 'rejection' means the cascade returns None for a non-logic item.
    """
    logic_ids = {e.get("task_id") for e in logic_entries}
    non_logic = [e for e in all_entries if e.get("task_id") not in logic_ids]

    results = {
        "total_non_logic": len(non_logic),
        "rejected": 0,  # correctly returned None
        "false_positives": [],  # returned non-None for non-logic
        "rejection_rate": 0.0,
        "by_category": defaultdict(lambda: {"total": 0, "rejected": 0}),
    }

    for entry in non_logic:
        task_id = entry.get("task_id", "unknown")
        cat = entry.get("category", "unknown")
        results["by_category"][cat]["total"] += 1

        try:
            answer = route_logic_cascade(entry["prompt"])
        except Exception:
            answer = None

        if answer is None:
            results["rejected"] += 1
            results["by_category"][cat]["rejected"] += 1
        else:
            if len(results["false_positives"]) < 20:
                results["false_positives"].append(
                    f"{task_id} ({cat}): returned {answer[:80]!r}"
                )

    if non_logic:
        results["rejection_rate"] = results["rejected"] / len(non_logic) * 100

    return results


def print_results(accuracy: Dict[str, Any], rejection: Dict[str, Any]):
    """Pretty-print the evaluation results."""
    print("=" * 70)
    print(f"  LOGIC CASCADE EVALUATION REPORT")
    print(f"  Dataset: {accuracy['label']}")
    print("=" * 70)
    print()

    # Summary
    print(f"Total logic items:    {accuracy['total']}")
    solved = accuracy["solve_counts"].get("solved", 0)
    unsolved = accuracy["solve_counts"].get("unsolved", 0)
    correct = accuracy["accuracy_counts"].get("correct", 0)
    incorrect = accuracy["accuracy_counts"].get("incorrect", 0)
    no_answer = accuracy["accuracy_counts"].get("no_answer", 0)

    print(f"Solved by cascade:    {solved}/{accuracy['total']} ({solved/accuracy['total']*100:.1f}%)")
    print(f"  Correct:           {correct} ({correct/accuracy['total']*100:.1f}%)")
    print(f"  Incorrect:         {incorrect} ({incorrect/accuracy['total']*100:.1f}%)")
    print(f"  No answer:         {no_answer} ({no_answer/accuracy['total']*100:.1f}%)")
    print(f"Accuracy (solved):   {correct/max(solved,1)*100:.1f}%")
    print(f"Accuracy (overall):  {correct/max(accuracy['total'],1)*100:.1f}%")
    print()

    # Per-level hit rate
    print("── Cascade Level Hit Rate ──")
    total = accuracy["total"]
    for level_name in ["truth_teller", "sequence", "syllogism", "constraint_puzzle",
                        "argument_analysis", "fallback"]:
        count = accuracy["route_counts"].get(level_name, 0)
        confs = accuracy["route_confidences"].get(level_name, [])
        avg_conf = sum(confs) / max(len(confs), 1) if confs else 0
        pct = count / total * 100 if total else 0
        print(f"  {level_name:25s}: {count:4d}/{total} ({pct:5.1f}%)  avg conf={avg_conf:.2f}")
    print()

    # Per-subtype accuracy
    print("── Per-Subtype Accuracy ──")
    for subtype in ["zebra", "logiqa"]:
        sd = accuracy["by_subtype"].get(subtype, {"total": 0, "solved": 0, "correct": 0})
        if sd["total"] == 0:
            continue
        acc = sd["correct"] / sd["total"] * 100
        solve_rate = sd["solved"] / sd["total"] * 100
        print(f"  {subtype:25s}: {sd['total']:4d} items, solved={sd['solved']} ({solve_rate:.1f}%), "
              f"correct={sd['correct']} ({acc:.1f}%)")
    print()

    # Rejection stats
    print("── Rejection Stats (non-logic items) ──")
    rej_total = rejection["total_non_logic"]
    rej_count = rejection["rejected"]
    rej_rate = rejection["rejection_rate"]
    print(f"  Total non-logic: {rej_total}")
    print(f"  Rejected (correct): {rej_count} ({rej_rate:.1f}%)")
    print(f"  False positives: {len(rejection['false_positives'])}")
    if rejection["false_positives"]:
        print(f"  First 10 false positives:")
        for fp in rejection["false_positives"][:10]:
            print(f"    {fp}")

    print()
    print("── Rejection by Category ──")
    for cat in sorted(rejection["by_category"].keys()):
        cd = rejection["by_category"][cat]
        reject_pct = cd["rejected"] / cd["total"] * 100 if cd["total"] else 0
        print(f"  {cat:25s}: {cd['rejected']:4d}/{cd['total']} ({reject_pct:5.1f}%)")

    # Errors listing
    if accuracy["errors"]:
        print()
        print("── Errors/Incorrect (first 20) ──")
        for err in accuracy["errors"][:20]:
            print(f"  {err}")

    print()
    print("=" * 70)


def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/eval"))
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../eval_results"))
    os.makedirs(results_dir, exist_ok=True)

    # Load training-v1 (200 logic items, 1514 total)
    train1_path = os.path.join(data_dir, "training-v1.json")
    all_entries = load_json(train1_path)
    logic_entries = [e for e in all_entries if e.get("category") == "logic"]
    print(f"Loaded {len(all_entries)} total items, {len(logic_entries)} logic items from training-v1.json")

    # Run accuracy evaluation on logic items
    acc_results = run_accuracy_eval(logic_entries, "training-v1 (logic subset)")

    # Run rejection evaluation on non-logic items
    rej_results = run_rejection_eval(logic_entries, all_entries)

    # Print results
    print_results(acc_results, rej_results)

    # Also run on training-v3 if available
    train3_path = os.path.join(data_dir, "training-v3.json")
    if os.path.exists(train3_path):
        train3 = load_json(train3_path)
        logic3 = [e for e in train3 if e.get("category") == "logic"]
        print(f"\nLoaded {len(train3)} total items, {len(logic3)} logic items from training-v3.json")
        if logic3:
            acc3 = run_accuracy_eval(logic3, "training-v3 (logic subset)")
            rej3 = run_rejection_eval(logic3, train3)
            print_results(acc3, rej3)

    # Run on validation-v3 if available
    val3_path = os.path.join(data_dir, "validation-v3.json")
    if os.path.exists(val3_path):
        val3 = load_json(val3_path)
        logic_val = [e for e in val3 if e.get("category") == "logic"]
        print(f"\nLoaded {len(val3)} total items, {len(logic_val)} logic items from validation-v3.json")
        if logic_val:
            acc_val = run_accuracy_eval(logic_val, "validation-v3 (logic subset)")
            rej_val = run_rejection_eval(logic_val, val3)
            print_results(acc_val, rej_val)

    # Save results to file
    report = {
        "train_v1": {
            "logic_total": len(logic_entries),
            "solved": acc_results["solve_counts"].get("solved", 0),
            "correct": acc_results["accuracy_counts"].get("correct", 0),
            "incorrect": acc_results["accuracy_counts"].get("incorrect", 0),
            "no_answer": acc_results["accuracy_counts"].get("no_answer", 0),
            "route_counts": dict(acc_results["route_counts"]),
            "by_subtype": {k: dict(v) for k, v in acc_results["by_subtype"].items()},
            "rejection_rate": rej_results["rejection_rate"],
            "rejected": rej_results["rejected"],
            "total_non_logic": rej_results["total_non_logic"],
        }
    }
    report_path = os.path.join(results_dir, "logic_cascade_eval.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")

    # Return a summary dict for the caller
    return {
        "accuracy_overall": acc_results["accuracy_counts"].get("correct", 0) / max(len(logic_entries), 1) * 100,
        "solve_rate": acc_results["solve_counts"].get("solved", 0) / max(len(logic_entries), 1) * 100,
        "rejection_rate": rej_results["rejection_rate"],
        "total_logic": len(logic_entries),
        "total_correct": acc_results["accuracy_counts"].get("correct", 0),
    }


if __name__ == "__main__":
    main()
