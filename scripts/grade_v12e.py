#!/usr/bin/env python3
"""Grade answers using official evaluate.py pipeline but reading from JSON dataset.
Usage: python3 run_v12e.py | python3 grade_v12e.py"""
import json, sys, os
sys.path.insert(0, "/home/artem/dev/amd-hackathon")

from evaluate import fuzzy_match, extract_numbers, grade_answer
from collections import Counter

EVAL_PATH = sys.argv[1] if len(sys.argv) > 1 else "/home/artem/dev/amd-hackathon-shared/eval_all_300.json"

with open(EVAL_PATH) as f:
    data = json.load(f)
questions = data.get("questions", data if isinstance(data, list) else [])

expected = [q["expected_answer"] for q in questions]
answers_raw = [l.strip() for l in sys.stdin if l.strip()]
# Un-flatten newlines (escaped as literal \n in output)
answers = [a.replace("\\n", "\n").replace("\\r", "\r") for a in answers_raw][:len(expected)]

if len(answers) != len(expected):
    print(f"[ERROR] Got {len(answers)} answers, expected {len(expected)}", file=sys.stderr)
    # If fewer, pad with empty
    while len(answers) < len(expected):
        answers.append("")

results = []
cat_stats = {}
for i, (q, ans) in enumerate(zip(questions, answers)):
    exp = q["expected_answer"]
    passed, reason = grade_answer(ans, exp)
    results.append((i, q["prompt"][:80], ans[:80], passed, reason))
    cat = q.get("category", "unknown").split("_")[0]
    cat_stats.setdefault(cat, {"n":0,"pass":0})
    cat_stats[cat]["n"] += 1
    if passed:
        cat_stats[cat]["pass"] += 1

total = len(results)
passed_total = sum(1 for r in results if r[3])
acc = 100 * passed_total / total

print()
print("=" * 80)
print("  v12e EVALUATION RESULTS — OFFICIAL GRADER")
print("=" * 80)

for idx, task, answer, passed, reason in results:
    icon = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  {icon}  Q{idx}: {task[:60]}")
    if answer:
        print(f"       Answer: {answer[:120]}")
    if not passed:
        print(f"       Reason: {reason}")

print()
print("=" * 80)
print(f"  SUMMARY")
print(f"    Accuracy:        {passed_total}/{total} ({acc:.1f}%)")
print()
print(f"  {'Category':<18s} {'Acc':>8s} {'Count':>6s}")
for cat, s in sorted(cat_stats.items()):
    a = 100 * s["pass"] / s["n"] if s["n"] else 0
    print(f"  {cat:<18s} {a:>6.1f}% ({s['pass']}/{s['n']})")
print("=" * 80)

# Save
ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
outpath = os.path.join(OUTPUT_DIR, f"eval_v12e_{ts}.json")
with open(outpath, "w") as f:
    json.dump({
        "total": total, "passed": passed_total, "accuracy": round(acc, 1),
        "per_category": {c: s for c, s in sorted(cat_stats.items())},
        "results": results,
    }, f, indent=2, default=str)
print(f"\n  Saved: {outpath}")
