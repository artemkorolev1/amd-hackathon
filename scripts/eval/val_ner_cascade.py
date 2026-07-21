#!/usr/bin/env python3
"""
Evaluation script for the NER classifier cascade.

Loads training-v1.json and other eval files, routes each NER prompt through
the cascade, and reports:
  - Per-level hit rate (% routed to each tool)
  - Overall solve rate (% where solver returned non-None)
  - F1 score vs expected_answer
  - Exact match rate
  - Rejection precision/recall on non-NER items
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agent.solvers.ner_classifier_cascade import route_ner
from agent.solvers.ner_classifier_cascade import (
    has_tweet_markers, is_ner_extraction,
    _ROUTE_NAMES,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)


def classify_cascade_level(prompt: str) -> Tuple[Optional[str], float]:
    """Determine which level of the cascade would match."""
    for level, name in sorted(_ROUTE_NAMES.items()):
        classifier = {
            1: has_tweet_markers,
            2: is_ner_extraction,
        }[level]
        matched, conf = classifier(prompt)
        if matched:
            return name, conf
    return None, 0.0


def compute_ner_f1(result_text: str, expected_text: str) -> Tuple[float, float, float]:
    """Compute precision, recall, F1 for NER output.
    
    Both result and expected are in TYPE: entity format (one per line).
    We normalize by lowercasing and matching type+entity pairs.
    """
    def parse_lines(text):
        lines = set()
        for line in text.strip().split('\n'):
            line = line.strip()
            if ':' in line:
                parts = line.split(':', 1)
                etype = parts[0].strip().lower()
                entity = parts[1].strip().lower()
                # Normalize: remove {@...@} markers for comparison
                entity = entity.replace('{@', '').replace('@}', '').strip()
                lines.add((etype, entity))
        return lines
    
    result_pairs = parse_lines(result_text)
    expected_pairs = parse_lines(expected_text)
    
    if not expected_pairs:
        if not result_pairs:
            return 1.0, 1.0, 1.0
        return 0.0, 0.0, 0.0
    
    intersection = result_pairs & expected_pairs
    
    precision = len(intersection) / len(result_pairs) if result_pairs else 0.0
    recall = len(intersection) / len(expected_pairs) if expected_pairs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def run_accuracy_eval(entries: List[Dict], label: str) -> Dict[str, Any]:
    """Run the cascade on NER items and evaluate against expected answers."""
    results = {
        "label": label,
        "total": len(entries),
        "route_counts": Counter(),
        "route_confidences": defaultdict(list),
        "solve_counts": Counter(),
        "f1_scores": [],
        "exact_matches": 0,
        "partial_matches": 0,
        "precision_total": 0.0,
        "recall_total": 0.0,
        "errors": [],
    }
    
    for entry in entries:
        prompt = entry["prompt"]
        expected = entry["expected_answer"]
        task_id = entry.get("task_id", "unknown")
        
        # Classify level
        level_name, conf = classify_cascade_level(prompt)
        results["route_counts"][level_name or "fallback"] += 1
        if level_name:
            results["route_confidences"][level_name].append(conf)
        
        # Run routing
        try:
            answer = route_ner(prompt)
        except Exception as e:
            answer = None
            results["errors"].append(f"{task_id}: Exception: {e}")
        
        # Check if solved
        if answer is not None:
            results["solve_counts"]["solved"] += 1
        else:
            results["solve_counts"]["unsolved"] += 1
        
        # Compute F1
        if answer is not None and expected:
            precision, recall, f1 = compute_ner_f1(answer, expected)
            results["f1_scores"].append(f1)
            results["precision_total"] += precision
            results["recall_total"] += recall
            
            if answer.strip() == expected.strip():
                results["exact_matches"] += 1
                results["partial_matches"] += 1
            elif f1 > 0:
                results["partial_matches"] += 1
            else:
                if len(results["errors"]) < 20:
                    results["errors"].append(
                        f"{task_id}: F1={f1:.3f}, "
                        f"expected={expected[:80]!r}, "
                        f"got={answer[:80]!r}"
                    )
        elif answer is None:
            results["f1_scores"].append(0.0)
    
    return results


def run_rejection_eval(ner_entries: List[Dict], all_entries: List[Dict]) -> Dict[str, Any]:
    """Evaluate how well the cascade rejects non-NER items.
    
    A 'rejection' means the cascade returns None for a non-NER item.
    """
    ner_ids = {e.get("task_id") for e in ner_entries}
    non_ner = [e for e in all_entries if e.get("task_id") not in ner_ids]
    
    results = {
        "total_non_ner": len(non_ner),
        "rejected": 0,
        "false_positives": [],
        "rejection_rate": 0.0,
        "by_category": defaultdict(lambda: {"total": 0, "rejected": 0}),
    }
    
    for entry in non_ner:
        task_id = entry.get("task_id", "unknown")
        cat = entry.get("category", "unknown")
        results["by_category"][cat]["total"] += 1
        
        try:
            answer = route_ner(entry["prompt"])
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
    
    if non_ner:
        results["rejection_rate"] = results["rejected"] / len(non_ner) * 100
    
    return results


def print_results(accuracy: Dict[str, Any], rejection: Dict[str, Any]):
    """Pretty-print the evaluation results."""
    print("=" * 70)
    print(f"  NER CASCADE EVALUATION REPORT")
    print(f"  Dataset: {accuracy['label']}")
    print("=" * 70)
    print()
    
    # Summary
    total = accuracy["total"]
    solved = accuracy["solve_counts"].get("solved", 0)
    unsolved = accuracy["solve_counts"].get("unsolved", 0)
    
    print(f"Total NER items:    {total}")
    print(f"Solved by cascade:  {solved}/{total} ({solved/total*100:.1f}%)")
    print(f"Unsolved:           {unsolved}/{total} ({unsolved/total*100:.1f}%)")
    print()
    
    # F1 scores
    f1_scores = accuracy["f1_scores"]
    if f1_scores:
        avg_f1 = sum(f1_scores) / len(f1_scores)
        exact = accuracy["exact_matches"]
        partial = accuracy["partial_matches"]
        avg_precision = accuracy["precision_total"] / total if total else 0
        avg_recall = accuracy["recall_total"] / total if total else 0
        
        print(f"Average F1:         {avg_f1:.3f}")
        print(f"Average Precision:  {avg_precision:.3f}")
        print(f"Average Recall:     {avg_recall:.3f}")
        print(f"Exact matches:      {exact}/{total} ({exact/total*100:.1f}%)")
        print(f"Partial (F1 > 0):   {partial}/{total} ({partial/total*100:.1f}%)")
        print()
        
        # F1 distribution
        ranges = [(0.0, 0.0), (0.01, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.99), (1.0, 1.0)]
        print("  F1 Distribution:")
        for lo, hi in ranges:
            if lo == hi:
                count = sum(1 for f in f1_scores if f == lo)
                label = f"F1 = {lo:.0%}" if lo == 1.0 else f"F1 = {lo:.2f}"
            else:
                count = sum(1 for f in f1_scores if lo < f <= hi)
                label = f"F1 in ({lo:.2f}, {hi:.2f}]"
            if count > 0:
                bar = "#" * count
                print(f"    {label:25s}: {count:4d} {bar}")
    
    # Per-level hit rate
    print()
    print("── Cascade Level Hit Rate ──")
    for level_name in ["tweet_ner", "general_ner", "fallback"]:
        count = accuracy["route_counts"].get(level_name, 0)
        confs = accuracy["route_confidences"].get(level_name, [])
        avg_conf = sum(confs) / max(len(confs), 1) if confs else 0
        pct = count / total * 100 if total else 0
        print(f"  {level_name:20s}: {count:4d}/{total} ({pct:5.1f}%)  avg conf={avg_conf:.2f}")
    
    # Rejection stats
    print()
    print("── Rejection Stats (non-NER items) ──")
    rej_total = rejection["total_non_ner"]
    rej_count = rejection["rejected"]
    rej_rate = rejection["rejection_rate"]
    print(f"  Total non-NER: {rej_total}")
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
        print("── Errors/Misses (first 20) ──")
        for err in accuracy["errors"][:20]:
            print(f"  {err}")
    
    print()
    print("=" * 70)


def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/eval"))
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../eval_results"))
    os.makedirs(results_dir, exist_ok=True)
    
    # Load training-v1
    train1_path = os.path.join(data_dir, "training-v1.json")
    all_entries = load_json(train1_path)
    ner_entries = [e for e in all_entries if e.get("category") == "ner"]
    print(f"Loaded {len(all_entries)} total items, {len(ner_entries)} NER items from training-v1.json")
    
    # Run accuracy evaluation on NER items
    acc_results = run_accuracy_eval(ner_entries, "training-v1 (NER subset)")
    
    # Run rejection evaluation on non-NER items
    rej_results = run_rejection_eval(ner_entries, all_entries)
    
    # Print results
    print_results(acc_results, rej_results)
    
    # Also run on training-v3 if available
    train3_path = os.path.join(data_dir, "training-v3.json")
    if os.path.exists(train3_path):
        train3 = load_json(train3_path)
        ner3 = [e for e in train3 if e.get("category") == "ner"]
        print(f"\nLoaded {len(train3)} total items, {len(ner3)} NER items from training-v3.json")
        if ner3:
            acc3 = run_accuracy_eval(ner3, "training-v3 (NER subset)")
            rej3 = run_rejection_eval(ner3, train3)
            print_results(acc3, rej3)
    
    # Run on validation-v3 if available
    val3_path = os.path.join(data_dir, "validation-v3.json")
    if os.path.exists(val3_path):
        val3 = load_json(val3_path)
        ner_val = [e for e in val3 if e.get("category") == "ner"]
        print(f"\nLoaded {len(val3)} total items, {len(ner_val)} NER items from validation-v3.json")
        if ner_val:
            acc_val = run_accuracy_eval(ner_val, "validation-v3 (NER subset)")
            rej_val = run_rejection_eval(ner_val, val3)
            print_results(acc_val, rej_val)
    
    # Save results to file
    report = {
        "train_v1": {
            "ner_total": len(ner_entries),
            "solved": acc_results["solve_counts"].get("solved", 0),
            "avg_f1": sum(acc_results["f1_scores"]) / max(len(acc_results["f1_scores"]), 1),
            "exact_matches": acc_results["exact_matches"],
            "partial_matches": acc_results["partial_matches"],
            "route_counts": dict(acc_results["route_counts"]),
            "rejection_rate": rej_results["rejection_rate"],
            "rejected": rej_results["rejected"],
            "total_non_ner": rej_results["total_non_ner"],
        }
    }
    report_path = os.path.join(results_dir, "ner_cascade_eval.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
