#!/usr/bin/env python3
"""Compare two evaluation runs — detect regressions, improvements, and accuracy deltas.

Usage:
    # Compare two report.json files from instrumented_evaluate
    python -m runner.regression \\
        --baseline eval_results/v14_report.json \\
        --candidate eval_results/v15_report.json \\
        --output-dir eval_results/compare/ \\
        --label-baseline V14 \\
        --label-candidate V15

    # Alternative: compare raw results.json files directly
    python -m runner.regression \\
        --baseline-results results-v14.json \\
        --candidate-results results-v15.json \\
        --gold data/eval/validation-v1.json \\
        --output-dir eval_results/compare/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.grade_answer import grade_answer
from runner.evaluate import load_gold

try:
    from runner.instrumented_evaluate import build_diff_html, load_results, evaluate_tasks_detailed, compute_timing_detailed, build_detailed_report
except ImportError:
    # Fallback if instrumented_evaluate not available yet
    build_diff_html = None

logger = logging.getLogger("regression")

TEMPLATE_DIR = Path(__file__).resolve().parent / "report_templates"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_report(path: str) -> dict:
    """Load a saved report.json from instrumented_evaluate output."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------


def compute_deltas(
    baseline: dict,
    candidate: dict,
) -> dict:
    """Compare two evaluation reports, returning per-task and aggregate deltas.

    Args:
        baseline: Report dict from instrumented_evaluate or raw results dict
        candidate: Report dict from instrumented_evaluate or raw results dict

    Returns:
        dict with keys: overall, by_category, regressions, improvements, unchanged
    """
    # Extract per-task lists
    baseline_tasks = _task_map(baseline)
    candidate_tasks = _task_map(candidate)

    all_task_ids = set(baseline_tasks.keys()) | set(candidate_tasks.keys())

    regressions: list[dict] = []
    improvements: list[dict] = []
    unchanged: list[dict] = []

    for tid in sorted(all_task_ids):
        b = baseline_tasks.get(tid, {})
        c = candidate_tasks.get(tid, {})

        b_correct = b.get("correct", False)
        c_correct = c.get("correct", False)

        entry = {
            "task_id": tid,
            "category": c.get("category", b.get("category", "")),
            "difficulty": c.get("difficulty", b.get("difficulty", "")),
            "expected": c.get("expected", b.get("expected", "")),
            "baseline_correct": b_correct,
            "candidate_correct": c_correct,
            "baseline_answer": b.get("answer", ""),
            "candidate_answer": c.get("answer", ""),
        }

        if b_correct and not c_correct:
            regressions.append(entry)
        elif not b_correct and c_correct:
            improvements.append(entry)
        else:
            unchanged.append(entry)

    # Aggregate deltas
    def _get_overall(report: dict) -> dict:
        """Extract overall metrics, handling both full report and raw results."""
        if "overall" in report:
            return report["overall"]
        # Raw results dict — compute on the fly
        tasks = report.get("per_task", report.get("results", []))
        if isinstance(tasks, dict):
            tasks = list(tasks.values())
        total = len(tasks)
        correct = sum(1 for t in tasks if t.get("correct", False))
        acc = correct / total if total > 0 else 0.0
        return {"accuracy": acc, "total": total, "correct": correct, "gate_pass": acc >= 0.842}

    b_overall = _get_overall(baseline)
    c_overall = _get_overall(candidate)

    b_acc = b_overall.get("accuracy", 0.0)
    c_acc = c_overall.get("accuracy", 0.0)

    # Per-category deltas
    def _get_categories(report: dict) -> dict:
        if "by_category" in report:
            return report["by_category"]
        # Compute from raw tasks
        tasks = report.get("per_task", report.get("results", []))
        if isinstance(tasks, dict):
            tasks = list(tasks.values())
        cats: dict[str, dict] = {}
        for t in tasks:
            cat = t.get("category", "unknown")
            if cat not in cats:
                cats[cat] = {"total": 0, "correct": 0}
            cats[cat]["total"] += 1
            if t.get("correct", False):
                cats[cat]["correct"] += 1
        for cat in cats:
            t = cats[cat]["total"]
            c = cats[cat]["correct"]
            cats[cat]["accuracy"] = (c / t) if t > 0 else 0.0
        return cats

    b_cats = _get_categories(baseline)
    c_cats = _get_categories(candidate)
    all_cats = sorted(set(list(b_cats.keys()) + list(c_cats.keys())))

    by_category: dict[str, dict] = {}
    for cat in all_cats:
        b = b_cats.get(cat, {"accuracy": 0.0})
        c = c_cats.get(cat, {"accuracy": 0.0})
        by_category[cat] = {
            "baseline": b.get("accuracy", 0.0),
            "candidate": c.get("accuracy", 0.0),
            "delta": c.get("accuracy", 0.0) - b.get("accuracy", 0.0),
        }

    return {
        "overall": {
            "baseline_accuracy": b_acc,
            "candidate_accuracy": c_acc,
            "delta": c_acc - b_acc,
            "gate_baseline": b_acc >= 0.842,
            "gate_candidate": c_acc >= 0.842,
        },
        "by_category": by_category,
        "regressions": regressions,
        "improvements": improvements,
        "unchanged": unchanged,
    }


def _task_map(report: dict) -> dict[str, dict]:
    """Extract a task_id -> task dict from either a full report or raw results."""
    if "per_task" in report:
        tasks = report["per_task"]
        if isinstance(tasks, list):
            return {t.get("task_id", f"idx_{i}"): t for i, t in enumerate(tasks)}
        if isinstance(tasks, dict):
            return tasks
    # Try results key (raw instrumented_evaluate output)
    tasks = report.get("results", [])
    if isinstance(tasks, list):
        return {t.get("task_id", f"idx_{i}"): t for i, t in enumerate(tasks)}
    return {}


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def save_diff_json(deltas: dict, path: str) -> str:
    """Save delta report as JSON."""
    with open(path, "w") as f:
        json.dump(deltas, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Saved diff JSON to %s", path)
    return path


def save_diff_html(
    deltas: dict,
    path: str,
    label_baseline: str = "Baseline",
    label_candidate: str = "Candidate",
) -> str:
    """Render and save HTML diff report."""
    if build_diff_html:
        html = build_diff_html(deltas, label_baseline, label_candidate)
    else:
        html = _fallback_html(deltas, label_baseline, label_candidate)
    with open(path, "w") as f:
        f.write(html)
    logger.info("Saved diff HTML to %s", path)
    return path


def _fallback_html(deltas: dict, label_baseline: str, label_candidate: str) -> str:
    """Minimal HTML when instrumented_evaluate's builder isn't available."""
    o = deltas.get("overall", {})
    regs = deltas.get("regressions", [])
    imps = deltas.get("improvements", [])
    delta_class = "delta-up" if o.get("delta", 0) > 0 else "delta-down"

    reg_rows = "".join(
        f"<tr><td>{r['task_id']}</td><td>{r['category']}</td>"
        f"<td>{r['expected'][:80]}</td><td>{r['baseline_answer'][:80]}</td>"
        f"<td>{r['candidate_answer'][:80]}</td></tr>"
        for r in regs
    )
    imp_rows = "".join(
        f"<tr><td>{r['task_id']}</td><td>{r['category']}</td>"
        f"<td>{r['expected'][:80]}</td><td>{r['baseline_answer'][:80]}</td>"
        f"<td>{r['candidate_answer'][:80]}</td></tr>"
        for r in imps
    )

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Diff: {label_baseline} → {label_candidate}</title>
<style>
body {{ font-family: sans-serif; padding: 20px; }}
.delta-up {{ color: #1a7d1a; }} .delta-down {{ color: #c41e1e; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
th {{ background: #4472C4; color: #fff; padding: 8px; text-align: left; }}
td {{ padding: 6px 8px; border-bottom: 1px solid #e0e0e0; }}
</style></head><body>
<h1>Regression Diff: {label_baseline} → {label_candidate}</h1>
<h2>Overall Delta: <span class="{delta_class}">{o.get('delta',0)*100:+.1f}pp</span></h2>
<p>Baseline: {o.get('baseline_accuracy',0)*100:.1f}% | Candidate: {o.get('candidate_accuracy',0)*100:.1f}%</p>
<h2>Regressions ({len(regs)})</h2>
{"<table><tr><th>ID</th><th>Category</th><th>Expected</th><th>Baseline</th><th>Candidate</th></tr>" + reg_rows + "</table>" if regs else "<p>No regressions.</p>"}
<h2>Improvements ({len(imps)})</h2>
{"<table><tr><th>ID</th><th>Category</th><th>Expected</th><th>Baseline</th><th>Candidate</th></tr>" + imp_rows + "</table>" if imps else "<p>No improvements.</p>"}
</body></html>"""


# ---------------------------------------------------------------------------
# Direct results comparison (without pre-built reports)
# ---------------------------------------------------------------------------


def compare_results(
    baseline_results_path: str,
    candidate_results_path: str,
    gold_path: str,
) -> dict:
    """Grade two pipeline output files against the same ground truth and compare.

    Args:
        baseline_results_path: Path to baseline results.json
        candidate_results_path: Path to candidate results.json
        gold_path: Path to ground-truth JSON

    Returns:
        Same structure as compute_deltas()
    """
    from runner.instrumented_evaluate import grade_results_detailed

    # Grade both against the same gold
    base_report = grade_results_detailed(
        results_path=baseline_results_path,
        gold_path=gold_path,
        title="Baseline",
        save_html=False,
        save_json=False,
    )
    cand_report = grade_results_detailed(
        results_path=candidate_results_path,
        gold_path=gold_path,
        title="Candidate",
        save_html=False,
        save_json=False,
    )
    return compute_deltas(base_report, cand_report)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two evaluation runs — detect regressions and improvements."
    )
    # Compare pre-built reports
    parser.add_argument("--baseline", help="Path to baseline report.json")
    parser.add_argument("--candidate", help="Path to candidate report.json")
    # Or compare raw results directly
    parser.add_argument("--baseline-results", help="Path to baseline results.json")
    parser.add_argument("--candidate-results", help="Path to candidate results.json")
    parser.add_argument("--gold", help="Path to ground-truth JSON (required with --*-results)")

    parser.add_argument("--output-dir", default=".",
                        help="Directory for output reports (default: current dir)")
    parser.add_argument("--label-baseline", default="Baseline",
                        help="Label for baseline (default: Baseline)")
    parser.add_argument("--label-candidate", default="Candidate",
                        help="Label for candidate (default: Candidate)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stderr,
        format="%(levelname)s %(message)s",
    )

    # Determine mode
    if args.baseline and args.candidate:
        # Compare pre-built reports
        logger.info("Loading baseline from %s", args.baseline)
        baseline = load_report(args.baseline)
        logger.info("Loading candidate from %s", args.candidate)
        candidate = load_report(args.candidate)
        deltas = compute_deltas(baseline, candidate)
    elif args.baseline_results and args.candidate_results and args.gold:
        # Compare raw results
        logger.info("Comparing raw results against gold %s", args.gold)
        deltas = compare_results(
            args.baseline_results,
            args.candidate_results,
            args.gold,
        )
    else:
        parser.print_help()
        print("\nERROR: Provide either --baseline+--candidate for report comparison, "
              "or --baseline-results+--candidate-results+--gold for raw comparison.",
              file=sys.stderr)
        return 1

    # Save output
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        json_path = os.path.join(args.output_dir, "regression_diff.json")
        save_diff_json(deltas, json_path)
        html_path = os.path.join(args.output_dir, "regression_diff.html")
        save_diff_html(deltas, html_path, args.label_baseline, args.label_candidate)

    # Print summary
    o = deltas["overall"]
    n_reg = len(deltas["regressions"])
    n_imp = len(deltas["improvements"])
    n_unchanged = len(deltas["unchanged"])

    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  REGRESSION DIFF: {args.label_baseline} → {args.label_candidate}")
    print(f"{bar}")
    print(f"  Baseline:  {o['baseline_accuracy']*100:.1f}%  Gate: {'PASS' if o['gate_baseline'] else 'FAIL'}")
    print(f"  Candidate: {o['candidate_accuracy']*100:.1f}%  Gate: {'PASS' if o['gate_candidate'] else 'FAIL'}")
    delta_str = f"{o['delta']*100:+.1f}pp"
    print(f"  Delta:     {delta_str}")
    print(f"  Regressions:  {n_reg}  Improvements: {n_imp}  Unchanged: {n_unchanged}")
    print(f"{bar}")
    if args.output_dir:
        print(f"  Reports:   {args.output_dir}/")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
