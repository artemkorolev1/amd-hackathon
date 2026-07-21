#!/usr/bin/env python3
"""
Comprehensive classifier evaluation — runs Stage 2 (8-way scorer) across ALL
labeled eval datasets and reports per-file, per-category accuracy.

Usage:
    python3 eval_classifiers.py                          # run all datasets
    python3 eval_classifiers.py --verbose                # show per-question errors
    python3 eval_classifiers.py --output results.json    # save results as JSON
    python3 eval_classifiers.py --quick                  # only smaller sets (<500 Q)
"""

import json, os, re, sys, glob, time, argparse
from collections import defaultdict
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
sys.path.insert(0, _PROJECT_ROOT)

from agent.category_filter import classify, get_short_name, CATEGORIES_8WAY, PRIORITY

# ── Eval file discovery ──

KNOWN_PATHS = [
    # input/ directory
    "input/dev_40.json",
    "input/heldout_40.json",
    "input/complexity_40.json",
    "input/cx_300.json",
    # data/eval/primary
    "data/eval/primary/eval_60_medium_hard.json",
    "data/eval/primary/eval_60_docx_style.json",
    "data/eval/primary/eval_hard_218.json",
    "data/eval/primary/eval_mini_10.json",
    # data/eval/training/validation
    "data/eval/training-v1.json",
    "data/eval/training-v2.json",
    "data/eval/training-v3.json",
    "data/eval/validation-v1.json",
    "data/eval/validation-v2.json",
    "data/eval/validation-v3.json",
    # data/eval/tests
    "data/eval/tests/complexity_eval_40.json",
    "data/eval/tests/complexity_eval_candidates.json",
    "data/eval/tests/eval_longform_20.json",
    "data/eval/tests/eval_v14_test_20.json",
    "data/eval/tests/eval_v14_remaining_20.json",
    "data/eval/tests/eval_v14_timeout_stress_19.json",
    "data/eval/tests/fireworks_eval_20.json",
    # data/eval/generated
    "data/eval/generated/build-A-40.json",
    "data/eval/generated/build-B-40.json",
    "data/eval/generated/eval_from_datasets_20260712_172357.json",
    "data/eval/generated/eval_from_datasets_20260712_172426.json",
    "data/eval/generated/eval_from_datasets_20260712_172443.json",
]


def _resolve_path(p: str) -> str:
    """Resolve relative path against project root."""
    if os.path.isabs(p):
        return p
    return os.path.join(_HERE, p)


def load_questions(path: str) -> list:
    """
    Load questions from an eval JSON file.
    Returns list of dicts with normalized fields:
      task_id, prompt, category (short name), gold_answer, gold_obj
    """
    fp = _resolve_path(path)
    if not os.path.isfile(fp):
        return []

    with open(fp) as f:
        data = json.load(f)

    items = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("questions", data.get("items", [data]))
    else:
        return []

    result = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        prompt = item.get("prompt", item.get("question", ""))
        if not prompt or not prompt.strip():
            continue

        # Extract category label (try multiple field names)
        cat = (
            item.get("category")
            or item.get("category_label")
            or item.get("label")
            or item.get("label_8way")
        )
        if not cat:
            continue

        short_cat = get_short_name(cat)

        # Extract task_id
        task_id = item.get("task_id", f"{os.path.basename(path)}_q{i}")

        # Extract gold answer if available
        gold_answer = None
        gold_obj = item.get("gold", {})
        if isinstance(gold_obj, dict):
            gold_answer = gold_obj.get("answer", gold_obj.get("function"))
        elif isinstance(gold_obj, str):
            gold_answer = gold_obj
        # try expected_answer field
        if not gold_answer:
            gold_answer = item.get("expected_answer")

        result.append({
            "task_id": task_id,
            "prompt": prompt,
            "true_category": short_cat,
            "true_category_raw": cat,
            "gold_answer": gold_answer,
            "gold_obj": gold_obj,
            "_source_file": path,
        })

    return result


def evaluate_file(path: str) -> dict:
    """
    Run Stage 2 classifier on all questions in a file.
    Returns structured results dict.
    """
    questions = load_questions(path)
    if not questions:
        return {"path": path, "total": 0, "questions": [], "error": "empty or not found"}

    correct = 0
    per_cat = defaultdict(lambda: {"correct": 0, "total": 0, "errors": []})
    question_results = []

    for q in questions:
        predicted, confidence, scores = classify(q["prompt"])
        is_correct = predicted == q["true_category"]

        if is_correct:
            correct += 1

        per_cat[q["true_category"]]["total"] += 1
        if is_correct:
            per_cat[q["true_category"]]["correct"] += 1
        else:
            per_cat[q["true_category"]]["errors"].append({
                "task_id": q["task_id"],
                "true_category": q["true_category"],
                "predicted": predicted,
                "confidence": round(confidence, 3),
                "prompt_preview": q["prompt"][:120],
            })

        question_results.append({
            "task_id": q["task_id"],
            "true_category": q["true_category"],
            "predicted": predicted,
            "confidence": round(confidence, 3),
            "correct": is_correct,
            "scores": {k: round(v, 2) for k, v in sorted(scores.items())},
        })

    total = len(questions)
    return {
        "path": path,
        "file_label": os.path.basename(path),
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total * 100, 1) if total > 0 else 0.0,
        "per_category": {
            cat: {
                "correct": v["correct"],
                "total": v["total"],
                "accuracy": round(v["correct"] / v["total"] * 100, 1) if v["total"] > 0 else 0.0,
                "errors": v["errors"][:5],  # first 5 errors only
            }
            for cat, v in sorted(per_cat.items())
        },
        "questions": question_results,
    }


def compute_confusion_matrix(results_by_file: list) -> dict:
    """Build confusion matrix across all files."""
    cats = sorted(CATEGORIES_8WAY)
    matrix = {true: {pred: 0 for pred in cats} for true in cats}
    cat_totals = {c: 0 for c in cats}

    for fr in results_by_file:
        for q in fr.get("questions", []):
            tc = q["true_category"]
            pc = q["predicted"]
            if tc in matrix and pc in matrix[tc]:
                matrix[tc][pc] += 1
                cat_totals[tc] = cat_totals.get(tc, 0) + 1

    return {"matrix": matrix, "totals": cat_totals}


def print_report(results_by_file: list, verbose: bool = False):
    """Print a formatted report to stdout."""
    grand_total = sum(r["total"] for r in results_by_file)
    grand_correct = sum(r["correct"] for r in results_by_file)

    print("=" * 80)
    print(f"  STAGE 2 CLASSIFIER EVALUATION — ALL DATASETS")
    print(f"  Total questions: {grand_total}  |  Correct: {grand_correct}  |  "
          f"Overall accuracy: {round(grand_correct / grand_total * 100, 1) if grand_total else 'N/A'}%")
    print("=" * 80)

    # Per-file summary
    print(f"\n{'File':<40} {'Total':>6} {'Correct':>8} {'Acc%':>6}")
    print("-" * 60)
    for r in sorted(results_by_file, key=lambda x: -x["total"]):
        acc_str = f"{r['accuracy']:.1f}%" if r["total"] > 0 else "N/A"
        label = r.get("file_label", os.path.basename(r["path"]))
        print(f"{label:<40} {r['total']:>6} {r['correct']:>8} {acc_str:>6}")

    # Aggregate per-category across all datasets
    agg_cat = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results_by_file:
        for cat, v in r.get("per_category", {}).items():
            agg_cat[cat]["correct"] += v["correct"]
            agg_cat[cat]["total"] += v["total"]

    print(f"\n{'Category':<25} {'Total':>6} {'Correct':>8} {'Acc%':>8}")
    print("-" * 47)
    for cat in sorted(CATEGORIES_8WAY):
        v = agg_cat.get(cat, {"correct": 0, "total": 0})
        acc_str = f"{round(v['correct']/v['total']*100,1):.1f}%" if v["total"] > 0 else "N/A"
        print(f"{cat:<25} {v['total']:>6} {v['correct']:>8} {acc_str:>8}")

    # Confusion matrix
    cm = compute_confusion_matrix(results_by_file)
    matrix = cm["matrix"]
    totals = cm["totals"]
    print(f"\n{'Confusion Matrix (rows=true, cols=predicted)':^80}")
    print(f"{'':>12}", end="")
    for c in CATEGORIES_8WAY:
        print(f"{c:>12}", end="")
    print(f"{'TOTAL':>8}")
    print("-" * (12 + 12 * 9 + 8))
    for true_cat in CATEGORIES_8WAY:
        print(f"{true_cat:<12}", end="")
        for pred_cat in CATEGORIES_8WAY:
            val = matrix.get(true_cat, {}).get(pred_cat, 0)
            if true_cat == pred_cat:
                print(f"{val:>8}* ", end="")
            else:
                print(f"{val:>10}", end="")
        print(f"{totals.get(true_cat, 0):>8}")

    # Worst misclassifications (largest off-diagonal entries)
    print(f"\nTop Misclassifications (off-diagonal):")
    off_diag = []
    for true_cat in CATEGORIES_8WAY:
        for pred_cat in CATEGORIES_8WAY:
            if true_cat != pred_cat:
                val = matrix.get(true_cat, {}).get(pred_cat, 0)
                if val > 0:
                    off_diag.append((val, true_cat, pred_cat))
    off_diag.sort(reverse=True)
    for count, true_c, pred_c in off_diag[:10]:
        pct = round(count / max(totals.get(true_c, 1), 1) * 100, 1)
        print(f"  {true_c:>12} → {pred_c:<12} : {count:>4} ({pct:.1f}% of {true_c})")

    # Show error examples if verbose
    if verbose:
        for r in results_by_file:
            has_errors = any(
                v.get("errors") for v in r.get("per_category", {}).values()
            )
            if not has_errors:
                continue
            label = r.get("file_label", os.path.basename(r["path"]))
            print(f"\n  ── Errors in {label} ──")
            for cat, v in sorted(r.get("per_category", {}).items()):
                for err in v.get("errors", [])[:3]:
                    print(f"    [{cat}] predicted={err['predicted']} (conf={err['confidence']})")
                    print(f"           prompt: {err['prompt_preview']}")


def discover_eval_files(quick: bool = False) -> list:
    """Discover available eval JSON files with category labels."""
    found = []
    for path in KNOWN_PATHS:
        fp = _resolve_path(path)
        if os.path.isfile(fp):
            questions = load_questions(path)
            if questions:
                found.append({"path": path, "count": len(questions)})
                if quick and len(questions) >= 500:
                    print(f"  [quick mode] skipping {path} ({len(questions)} questions)")
                    continue

    # Also scan input/ for any new eval files
    input_dir = _resolve_path("input")
    if os.path.isdir(input_dir):
        for f in sorted(os.listdir(input_dir)):
            if f.endswith(".json") and f not in [os.path.basename(p) for p in KNOWN_PATHS]:
                fp = os.path.join(input_dir, f)
                if os.path.isfile(fp):
                    questions = load_questions(fp)
                    if questions:
                        found.append({"path": fp, "count": len(questions)})

    return found


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 classifier on all eval datasets")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-question errors")
    parser.add_argument("--output", "-o", type=str, default="", help="Save results as JSON")
    parser.add_argument("--quick", "-q", action="store_true", help="Skip datasets with >500 questions")
    args = parser.parse_args()

    # Discover
    files = discover_eval_files(quick=args.quick)
    if not files:
        print("No eval files found with category labels.")
        sys.exit(1)

    print(f"Found {len(files)} eval files with categories ({sum(f['count'] for f in files)} total questions)")
    for f in files:
        print(f"  {f['path']:<60} {f['count']:>5} questions")
    print()

    # Evaluate each file
    all_results = []
    for f in files:
        t0 = time.monotonic()
        result = evaluate_file(f["path"])
        elapsed = time.monotonic() - t0
        if result["total"] > 0:
            all_results.append(result)
            acc_str = f"{result['accuracy']:.1f}%"
            print(f"  {result['file_label']:<40} {result['total']:>5} Q  acc={acc_str:>6}  ({elapsed:.1f}s)")

    # Grand report
    print()
    print_report(all_results, verbose=args.verbose)

    # Save if requested
    if args.output:
        with open(args.output, "w") as f:
            # Strip full question list for compactness
            output_data = []
            for r in all_results:
                out = {k: v for k, v in r.items() if k != "questions"}
                # Keep summary stats only
                output_data.append(out)
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_files": len(all_results),
                "total_questions": sum(r["total"] for r in all_results),
                "total_correct": sum(r["correct"] for r in all_results),
                "overall_accuracy": round(
                    sum(r["correct"] for r in all_results)
                    / max(sum(r["total"] for r in all_results), 1) * 100, 1
                ),
                "files": output_data,
            }, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
