#!/usr/bin/env python3
"""Analyze all sentiment questions across eval datasets."""
import json, os, sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _PROJECT_ROOT)

from agent.category_filter import get_short_name

KNOWN_PATHS = [
    "input/dev_40.json",
    "input/heldout_40.json",
    "input/complexity_40.json",
    "input/cx_300.json",
    "data/eval/primary/eval_60_medium_hard.json",
    "data/eval/primary/eval_60_docx_style.json",
    "data/eval/primary/eval_hard_218.json",
    "data/eval/primary/eval_mini_10.json",
    "data/eval/training-v1.json",
    "data/eval/training-v2.json",
    "data/eval/training-v3.json",
    "data/eval/validation-v1.json",
    "data/eval/validation-v2.json",
    "data/eval/validation-v3.json",
    "data/eval/tests/complexity_eval_40.json",
    "data/eval/tests/complexity_eval_candidates.json",
    "data/eval/tests/eval_longform_20.json",
    "data/eval/tests/eval_v14_test_20.json",
    "data/eval/tests/eval_v14_remaining_20.json",
    "data/eval/tests/eval_v14_timeout_stress_19.json",
    "data/eval/tests/fireworks_eval_20.json",
    "data/eval/generated/build-A-40.json",
    "data/eval/generated/build-B-40.json",
    "data/eval/generated/eval_from_datasets_20260712_172357.json",
    "data/eval/generated/eval_from_datasets_20260712_172426.json",
    "data/eval/generated/eval_from_datasets_20260712_172443.json",
]

def resolve_path(p):
    if os.path.isabs(p): return p
    return os.path.join(_HERE, p)

all_sentiment = []
for path in KNOWN_PATHS:
    fp = resolve_path(path)
    if not os.path.isfile(fp): continue
    with open(fp) as f:
        data = json.load(f)
    items = data if isinstance(data, list) else data.get("questions", data.get("items", [data]))
    for idx, item in enumerate(items):
        if not isinstance(item, dict): continue
        cat = item.get("category") or item.get("category_label") or item.get("label") or item.get("label_8way") or ""
        short = get_short_name(cat)
        if short == "sentiment":
            prompt = item.get("prompt", item.get("question", ""))
            gold = item.get("gold", {})
            gold_answer = gold.get("answer", "") if isinstance(gold, dict) else str(gold) if gold else ""
            if not gold_answer:
                gold_answer = item.get("expected_answer", "")
            all_sentiment.append({
                "task_id": item.get("task_id", f"{os.path.basename(path)}_q{idx}"),
                "prompt": prompt,
                "gold_answer": gold_answer,
                "source": path,
            })

print(f"Found {len(all_sentiment)} total sentiment questions across all datasets")

# Show breakdown by source
source_counts = defaultdict(int)
has_gold = 0
for q in all_sentiment:
    source_counts[q["source"]] += 1
    if q["gold_answer"]:
        has_gold += 1

print(f"With gold answers: {has_gold}")
for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
    print(f"  {src}: {cnt}")

# Show all questions with gold answers
print("\n\nQuestions with gold answers:")
for q in all_sentiment:
    if q["gold_answer"]:
        print(f"  [{q['task_id']}] gold={q['gold_answer']}")
        print(f"    prompt: {q['prompt'][:150]}")
        print()
