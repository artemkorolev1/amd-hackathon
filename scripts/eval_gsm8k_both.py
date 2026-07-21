#!/usr/bin/env python3
"""GSM8K eval — tests the full pipeline + ToRA integration on N problems."""

import os, sys, json, time, re, gc
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)

DATA_PATH = "/tmp/gsm8k_test.parquet"
N = 50  # validation batch
RUN_LABEL = "math_plus_consensus"  # experiment label for output file
OUTPUT_PATH = f"/tmp/gsm8k_eval_{RUN_LABEL}.json"

from agent.pipeline import Pipeline, PipelineConfig

# Same config as pipeline_gepa.py uses
cfg = PipelineConfig(
    model_path="/home/artem/models/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=2048,
    n_threads=4,
    consensus_samples=3,
    consensus_categories={"math"},
    category_model_map={},  # single model — no per-category overrides
)

from scripts.grade_answer import fuzzy_match
import pandas as pd

df = pd.read_parquet(DATA_PATH)

def extract_expected(answer_str):
    """Extract the numeric answer from GSM8K format: '#### 42'"""
    if "####" in answer_str:
        return answer_str.split("####")[-1].strip()
    return answer_str.strip()

print(f"GSM8K test set: {len(df)} problems", flush=True)
print(f"Testing first {N} with Pipeline + ToRA", flush=True)

pipe = Pipeline(config=cfg)

correct = 0
total = 0
results = []

for i in range(N):
    row = df.iloc[i]
    question = row["question"]
    expected = extract_expected(row["answer"])
    
    t_start = time.time()
    try:
        answer = pipe.process(question)
    except Exception as e:
        answer = ""
        print(f"  PIPE ERROR: {e}", flush=True)
    
    elapsed = time.time() - t_start
    is_correct = fuzzy_match(answer, expected)
    
    if is_correct:
        correct += 1
    total += 1
    
    results.append({
        "q": question[:80],
        "expected": expected,
        "got": answer[:50] if answer else "",
        "correct": is_correct,
        "time": round(elapsed, 1),
    })
    
    status = "✓" if is_correct else "✗"
    print(f"  [{i+1}/{N}] {status} got={answer[:40]!r} expected={expected!r} ({elapsed:.1f}s)", flush=True)
    
    # Force GC every 10 to avoid memory buildup
    if i > 0 and i % 10 == 0:
        gc.collect()

print(f"\n{'='*60}", flush=True)
print(f"Result: {correct}/{total} correct ({correct/total*100:.1f}%)", flush=True)
print(f"Avg time: {sum(r['time'] for r in results)/total:.1f}s", flush=True)
print(f"{'='*60}", flush=True)

pipe.close()

# Save results to JSON
with open(OUTPUT_PATH, "w") as f:
    json.dump({"label": RUN_LABEL, "correct": correct, "total": total, "results": results}, f, indent=2)
print(f"Results saved to {OUTPUT_PATH}", flush=True)

# Quick analysis of failures
print("\n### Failure analysis:", flush=True)
for r in results:
    if not r["correct"]:
        print(f"  Q: {r['q']}...", flush=True)
        print(f"  Expected: {r['expected']}  Got: {r['got']}", flush=True)
        print(f"  Time: {r['time']}s", flush=True)
        print(flush=True)
