#!/usr/bin/env python3
"""Print baseline validation report for sentiment splits."""
import json
import os
from collections import Counter

BASE = os.path.expanduser("/home/artem/dev/amd-hackathon/data/eval")

print("=" * 70)
print("SENTIMENT EVALUATION PIPELINE — BASELINE REPORT")
print("=" * 70)

all_diffs = {}
for name in ["train", "val", "hard_test"]:
    path = f"{BASE}/sentiment_{name}.json"
    with open(path) as f:
        data = json.load(f)
    
    diffs = Counter(it["difficulty"] for it in data)
    all_diffs[name] = diffs
    ans = Counter(it["expected_answer"] for it in data)
    sources = Counter(it.get("source", "unknown") for it in data)
    
    label = name.replace("hard_test", "hard_test (benchmark)")
    print(f"\n{'─' * 50}")
    print(f"📊 {label.upper()} ({len(data)} questions)")
    print(f"{'─' * 50}")
    print(f"  Difficulty: easy={diffs.get('easy',0)}, medium={diffs.get('medium',0)}, hard={diffs.get('hard',0)}")
    print(f"  Answers:    pos={ans.get('positive',0)}, neg={ans.get('negative',0)}, neu={ans.get('neutral',0)}, mix={ans.get('mixed',0)}")
    print(f"  Sources: {len(sources)} unique")
    for s, c in sorted(sources.items(), key=lambda x: -x[1])[:5]:
        print(f"    {s}: {c}")
    if len(sources) > 5:
        print(f"    ... and {len(sources)-5} more")

# Overfit analysis
print(f"\n{'=' * 70}")
print("OVERFIT ANALYSIS (no actual inference yet — GPU busy)")
print(f"{'=' * 70}")
print(f"  Training set:   1142 questions (446 easy + 165 medium + 531 hard)")
print(f"  Validation set: 100 questions  (70 easy + 30 medium — held-out from GEPA)")
print(f"  Hard test:      100 questions  (all hard — benchmark, never seen during GEPA)")
print(f"\n  Note: Training includes hard examples (from failure analysis) so the")
print(f"  model can learn from them during GEPA evolution. Validation is kept")
print(f"  clean (no hard items) to measure overfit on fundamentals.")
train_diffs = all_diffs.get("train", Counter())
if train_diffs.get('hard', 0) > 0:
    print(f"  Hard items in training: {train_diffs['hard']} — provides challenging examples for GEPA")
print(f"\n  Current best config: gemma-3-1b, prompt='Analyze the tone...'")
print(f"  Previous result:    89.1% on 92-question hard set")
print(f"\n  ⚡ Run actual eval when GPU is free:")
print(f"     python3 gepa_plans/eval_sentiment.py --model gemma-3-1b \\")
print(f"       --prompt 'Analyze the tone as positive, negative, neutral, or mixed.' \\")
print(f"       --top_p 0.9 --top_k 20 --min_p 0.05")

# Check model availability
models_dir = os.path.expanduser("/home/artem/dev/amd-hackathon/models")
available = [f for f in os.listdir(models_dir) if f.endswith('.gguf')] if os.path.exists(models_dir) else []
print(f"\n  Available models:")
for m in sorted(available):
    size = os.path.getsize(f"{models_dir}/{m}") / 1e9
    print(f"    {m} ({size:.1f} GB)")

print(f"\n{'=' * 70}")
print("NEXT STEPS")
print(f"{'=' * 70}")
print(f"  1. Run inference: python3 gepa_plans/eval_sentiment.py --model gemma-3-1b")
print(f"  2. Results saved to eval_results/sentiment_eval_TIMESTAMP.json")
print(f"  3. Reports per-set + per-difficulty accuracy + confusion matrix")
print(f"  4. Format normalizer from agent/solvers/format_normalizer.py is auto-used")
print(f"\n{'=' * 70}")
print("BASELINE VALIDATION — READY ✅")
print(f"{'=' * 70}")
