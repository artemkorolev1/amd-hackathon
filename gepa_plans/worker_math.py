#!/usr/bin/env python3
"""Worker script for math eval. Takes model_path as argv[1].
Imports fuzzy_match from eval_common in gepa_plans/."""

import json
import os
import sys
import time
import re

# Add gepa_plans to path so we can import eval_common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_common import fuzzy_match

os.environ['PYTHONUNBUFFERED'] = '1'

DATA_PATH = "/home/artem/dev/amd-hackathon/data/eval/math_combined_80.json"
RESULTS_PATH = sys.argv[2] if len(sys.argv) > 2 else "/tmp/math_worker_results.json"
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"

PROMPT_STRATEGIES = [
    {"index": 0, "label": "empty",                        "system_prompt": ""},
    {"index": 1, "label": "Math:",                        "system_prompt": "Math:"},
    {"index": 2, "label": "Answer only with a number.",   "system_prompt": "Answer only with a number."},
    {"index": 3, "label": "Let's think step by step.",    "system_prompt": "Let's think step by step."},
    {"index": 4, "label": "Calculate:",                   "system_prompt": "Calculate:"},
    {"index": 5, "label": "Answer: ",                     "system_prompt": "Answer: "},
]

with open(DATA_PATH) as f:
    questions = json.load(f)
print(f"Worker: loaded {len(questions)} questions", flush=True)

from llama_cpp import Llama
print(f"Worker: loading model on GPU...", flush=True)
t0 = time.time()
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_gpu_layers=-1, verbose=False)
print(f"Worker: loaded in {time.time()-t0:.1f}s", flush=True)

results = {}
for strat in PROMPT_STRATEGIES:
    pi = strat["index"]
    pl = strat["label"]
    sp = strat["system_prompt"]
    pk = f"prompt_{pi}"
    print(f"Worker: prompt {pi} [{pl}]", flush=True)
    correct = 0
    total = 0
    for qi, q in enumerate(questions):
        messages = [{"role": "system", "content": sp}, {"role": "user", "content": q["prompt"]}]
        try:
            resp = llm.create_chat_completion(messages=messages, max_tokens=256, temperature=0.0)
            got = resp["choices"][0]["message"]["content"]
        except Exception as e:
            got = f"<ERROR: {e}>"
        # Extract \boxed{} content if present (qwen2.5-math wraps answers in \boxed{})
        clean_got = got
        box_match = re.search(r'\\boxed\{([^}]+)\}', got)
        if box_match:
            clean_got = box_match.group(1)
        is_correct = fuzzy_match(clean_got, q.get("expected_answer", ""))
        if is_correct:
            correct += 1
        total += 1
    acc = correct / total if total else 0
    results[pk] = {"accuracy": round(acc, 3), "correct": correct, "total": total}
    print(f"Worker: prompt {pi} done: {results[pk]}", flush=True)

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f)
print(f"Worker done: results written to {RESULTS_PATH}", flush=True)
