"""Extract all training data from eval_results/ into a unified JSONL for judge training."""
import json
import os
import re
import glob
from collections import defaultdict

HERE = "/home/artem/dev/amd-hackathon"
EVAL_DIR = os.path.join(HERE, "eval_results")
OUTPUT = os.path.join(HERE, "data/judge_training.jsonl")
STATS_OUT = os.path.join(HERE, "data/judge_training_stats.json")

os.makedirs(os.path.join(HERE, "data"), exist_ok=True)

def iter_eval_results(path):
    """Yield (source_file, record) for every per-question record in an eval result JSON."""
    if not os.path.isfile(path):
        return
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return
    
    # Find per-question records
    records = []
    if "results" in data and isinstance(data["results"], list):
        records = data["results"]
    elif "per_task" in data and isinstance(data["per_task"], list):
        records = data["per_task"]
    elif "questions" in data and isinstance(data["questions"], list):
        records = data["questions"]
    elif isinstance(data, list):
        records = data
    else:
        # Try common keys
        for key in ("items", "samples", "evaluations", "graded"):
            if key in data and isinstance(data[key], list):
                records = data[key]
                break
    
    model = data.get("model", os.path.basename(path).replace(".json", "")) if isinstance(data, dict) else os.path.basename(path).replace(".json", "")
    
    for rec in records:
        if not isinstance(rec, dict):
            continue
        # Must have at least a prompt and a way to determine correctness
        prompt = rec.get("prompt", rec.get("question", ""))
        if not prompt:
            continue
        answer = rec.get("answer", rec.get("output", rec.get("response", "")))
        expected = rec.get("expected", rec.get("expected_answer", 
                   rec.get("gold", rec.get("target", ""))))
        
        # Correctness label: try various field names
        correct = rec.get("correct", rec.get("passed", rec.get("grade", 
                   rec.get("is_correct", None))))
        if correct is None:
            # Try reason field
            reason = rec.get("reason", "")
            correct = "passed" in str(reason).lower() or "Passed" in str(reason)
        
        category = rec.get("category", rec.get("category_label", 
                   rec.get("label", rec.get("task_type", ""))))
        difficulty = rec.get("difficulty", "")
        task_id = rec.get("task_id", rec.get("id", ""))
        timing = rec.get("timing_ms", rec.get("time_ms", 0))
        source = rec.get("source", os.path.basename(path))
        
        yield {
            "task_id": task_id,
            "prompt": prompt,
            "answer": answer,
            "expected": expected,
            "category": category,
            "difficulty": difficulty,
            "correct": bool(correct) if isinstance(correct, (bool, int)) else False,
            "timing_ms": timing,
            "source": source,
            "_model": model,
        }


all_records = []
by_source = defaultdict(int)
by_category = defaultdict(int)
correct_total = 0
incorrect_total = 0

# Scan all JSON files in eval_results recursively
for root, dirs, files in os.walk(EVAL_DIR):
    for fname in files:
        if not fname.endswith(".json"):
            continue
        path = os.path.join(root, fname)
        count = 0
        for rec in iter_eval_results(path):
            all_records.append(rec)
            by_source[fname] += 1
            cat = rec.get("category", "unknown")
            if cat:
                by_category[cat] += 1
            if rec["correct"]:
                correct_total += 1
            else:
                incorrect_total += 1
            count += 1
        if count > 0:
            print(f"  {fname}: {count} records")

# Write unified JSONL
with open(OUTPUT, "w") as f:
    for rec in all_records:
        f.write(json.dumps(rec) + "\n")

stats = {
    "total_records": len(all_records),
    "correct": correct_total,
    "incorrect": incorrect_total,
    "accuracy": round(correct_total / max(correct_total + incorrect_total, 1), 4),
    "by_source": dict(by_source),
    "by_category": dict(by_category),
}

with open(STATS_OUT, "w") as f:
    json.dump(stats, f, indent=2)

print(f"\nTotal: {len(all_records)} records ({correct_total} correct, {incorrect_total} incorrect)")
print(f"Accuracy: {stats['accuracy']*100:.1f}%")
print(f"Saved to {OUTPUT}")
print(f"Stats to {STATS_OUT}")
