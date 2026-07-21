#!/usr/bin/env python3
"""Quick host-based test of the staging pipeline with GPU.

Runs the full pipeline: load config → classify → enqueue → dispatch → judge.
Uses GPU and the host's local model. No Docker needed.
"""

import json
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from staging.ready_config import ReadyConfig
from staging.ready_queue import ReadyQueue, ReadyTask
from staging.ready_classifier import classify_batch

# Test tasks
tasks = [
    {"task_id": "t1", "prompt": "What is 15% of 240? Answer with just the number."},
    {"task_id": "t2", "prompt": "Classify the sentiment of this review as Positive, Negative, or Neutral: 'The product arrived two days late and the packaging was damaged, but the item worked perfectly and customer support resolved my complaint within an hour.'"},
    {"task_id": "t3", "prompt": "Extract all named entities from: On March 15 2023, Sundar Pichai announced that Google would open a new AI research lab in Zurich, partnering with ETH Zurich. Label each as PERSON, ORGANIZATION, LOCATION, or DATE."},
    {"task_id": "t4", "prompt": "Which country has the largest population? Answer with just the country name."},
    {"task_id": "t5", "prompt": "If all A are B, and some B are C, can we conclude that some A are C? Answer with Yes or No only."},
]

# Need Fireworks key for the FW worker
fw_key = os.environ.get("FIREWORKS_API_KEY", "")

# Override config for GPU test
import os
os.environ.setdefault("MODEL_PATH", os.path.join(os.path.dirname(__file__), "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"))

config = ReadyConfig.from_env()
config.det_workers = 0  # Zero det workers for GPU test — let local LLM handle all tasks
config.fw_workers = 0   # No FW worker for now
config.fw_api_key = fw_key
config.judgment_votes = 3  # 3 tries from the local LLM
config.deadline_s = 120
config.worker_timeout_s = 30.0
config.loc_n_gpu_layers = -1  # Use GPU

print(f"=== GPU Pipeline Test ===")
print(f"Config: {config.total_workers} workers ({config.fw_workers} FW + {config.loc_workers} Loc + {config.det_workers} Det)")
print(f"GPU: n_gpu_layers={config.loc_n_gpu_layers}, threads={config.loc_n_threads}")
print(f"Votes per task: {config.judgment_votes}")
print(f"Deadline: {config.deadline_s}s")
print()

# 1. Classify
print("1. Bulk classifying tasks...")
prompts = [t["prompt"] for t in tasks]
classified = classify_batch(prompts)
for t, c in zip(tasks, classified):
    print(f"   {t['task_id']:5s} -> {c['category']:15s} (conf={c['confidence']:.2f})")
print()

# 2. Build queue
queue = ReadyQueue()
for i, (task, cls) in enumerate(zip(tasks, classified)):
    ready = ReadyTask(
        task_id=task["task_id"],
        prompt=task["prompt"],
        category=cls["category"],
        category_4way=cls["category_4way"],
        raw_scores=cls["raw_scores"],
        confidence=cls["confidence"],
        score_delta=cls["score_delta"],
    )
    queue.enqueue(ready)
print(f"2. Queue built with {queue.total_pending} tasks")
print(f"   By category: {queue.task_counts_by_category()}")
print()

# 3. Start pool + judge
print("3. Starting pool and judge...")
from staging.ready_judge import ReadyJudge
from staging.ready_pool import ReadyPool

judge = ReadyJudge(config)
pool = ReadyPool(config)

deadline = time.monotonic() + config.deadline_s

t_start = time.monotonic()
try:
    pool.dispatch_loop(queue, judge, deadline)
    judge.ingest_results(pool._results_queue)
except Exception as e:
    print(f"   ERROR in dispatch: {e}")
finally:
    print("4. Judging remaining tasks...")
    final_results = judge.judge_all()
    elapsed = time.monotonic() - t_start
    
    print(f"\n=== Results ({len(final_results)} tasks, {elapsed:.1f}s) ===")
    strategy_counts = {}
    for r in final_results:
        strat = r.get("_judgment", {}).get("strategy", "unknown")
        strategy_counts[strat] = strategy_counts.get(strat, 0) + 1
        ans = r.get("answer", "")
        print(f"   {r['task_id']:5s} | {ans[:80]:80s} | {strat}")
    
    print(f"\nStrategy distribution: {strategy_counts}")
    print(f"Total time: {elapsed:.1f}s")
    
    # Shutdown
    pool.shutdown()
