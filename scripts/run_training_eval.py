#!/usr/bin/env python3
"""
Run the full Pipeline against training-v3.json, grade each answer,
report per-category + overall accuracy with timing.
"""
import json, os, sys, time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["N_GPU_LAYERS"] = os.environ.get("N_GPU_LAYERS", "0")
os.environ["N_THREADS"] = os.environ.get("N_THREADS", "2")
os.environ["INFERENCE_TIMEOUT_S"] = "30.0"

from agent.pipeline import Pipeline, PipelineConfig
from scripts.grade_answer import grade_answer

EVAL_PATH = ROOT / "data/eval/training-v3.json"
RESULTS_DIR = ROOT / "eval_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_eval(path):
    with open(path) as f:
        data = json.load(f)
    # Normalise: ensure task_id on every entry
    for i, q in enumerate(data):
        if "task_id" not in q or not q["task_id"]:
            q["task_id"] = f"idx_{i}"
    return data

def main():
    questions = load_eval(EVAL_PATH)
    print(f"[eval] Loaded {len(questions)} questions", file=sys.stderr)

    cfg = PipelineConfig()
    cfg.deadline_s = 9999.0  # no deadline for local eval
    pipe = Pipeline(config=cfg)

    results = []
    cat_results = defaultdict(list)
    start_ts = time.monotonic()

    for i, q in enumerate(questions):
        tid = q["task_id"]
        prompt = q["prompt"]
        expected = q["expected_answer"]
        cat = q["category"]
        t0 = time.monotonic()

        answer = pipe.process(prompt)
        elapsed = time.monotonic() - t0

        passed, reason = grade_answer(answer, expected)

        results.append({
            "task_id": tid,
            "category": cat,
            "prompt_preview": prompt[:80].replace("\n", " "),
            "answer": answer,
            "expected": expected,
            "passed": passed,
            "reason": reason,
            "elapsed_s": round(elapsed, 2),
        })
        cat_results[cat].append(passed)
        status = "PASS" if passed else "FAIL"
        print(f"  [{i+1:3d}/{len(questions)}] {cat:14s} | {status:4s} | "
              f"{elapsed:5.1f}s | {tid}", file=sys.stderr)

    total_elapsed = time.monotonic() - start_ts
    pipe.close()

    # ── Report ──
    print("\n" + "=" * 70)
    print("  TRAINING EVAL RESULTS — pipeline.py vs training-v3.json")
    print("=" * 70)

    overall_pass = sum(1 for r in results if r["passed"])
    overall = len(results)
    print(f"\n  Overall: {overall_pass}/{overall} = "
          f"{overall_pass/overall*100:.1f}%  ({total_elapsed:.0f}s total)")

    print(f"\n  {'Category':16s} {'Passed':>6s} {'Total':>6s} {'Acc%':>6s}  {'Avg s':>6s}")
    print(f"  {'-'*16} {'-'*6} {'-'*6} {'-'*6}  {'-'*6}")
    for cat in sorted(cat_results):
        passes = sum(cat_results[cat])
        total = len(cat_results[cat])
        cat_times = [r["elapsed_s"] for r in results if r["category"] == cat]
        avg_t = sum(cat_times) / len(cat_times) if cat_times else 0
        print(f"  {cat:16s} {passes:6d} {total:6d} {passes/total*100:5.1f}%  {avg_t:5.1f}s")

    # ── Failure detail ──
    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for r in failures:
            a_short = r["answer"][:120].replace("\n", "\\n")
            e_short = r["expected"][:120].replace("\n", "\\n")
            print(f"  [{r['category']:14s}] {r['task_id']}")
            print(f"    Q: {r['prompt_preview']}")
            print(f"    A: {a_short}")
            print(f"    E: {e_short}")
            print()

    # ── Save ──
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = RESULTS_DIR / f"training_eval_{ts}.json"
    with open(detail_path, "w") as f:
        json.dump({
            "timestamp": ts,
            "eval_set": str(EVAL_PATH),
            "total_questions": overall,
            "passed": overall_pass,
            "accuracy_pct": round(overall_pass / overall * 100, 1),
            "total_elapsed_s": round(total_elapsed, 1),
            "per_category": {
                cat: {
                    "passed": sum(cat_results[cat]),
                    "total": len(cat_results[cat]),
                    "accuracy_pct": round(sum(cat_results[cat]) / len(cat_results[cat]) * 100, 1),
                }
                for cat in sorted(cat_results)
            },
            "results": results,
        }, f, indent=2, default=str)
    print(f"\n  Results saved to {detail_path}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
