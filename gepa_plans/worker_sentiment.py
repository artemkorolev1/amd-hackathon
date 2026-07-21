#!/usr/bin/env python3
"""Worker script for sentiment eval. Takes model_path as argv[1].
Imports fuzzy_match from eval_common in gepa_plans/."""

import json
import os
import sys
import time

# Add gepa_plans to path so we can import eval_common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_common import fuzzy_match

os.environ['PYTHONUNBUFFERED'] = '1'

DATA_PATH = "/home/artem/dev/amd-hackathon/data/eval/sentiment_combined_25.json"
RESULTS_PATH = sys.argv[2] if len(sys.argv) > 2 else "/tmp/sentiment_worker_results.json"
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"

PROMPT_STRATEGIES = [
    {"index": 0, "label": "empty", "system_prompt": ""},
    {"index": 1, "label": "explicit_instruction",
     "system_prompt": "Classify the sentiment. Output EXACTLY one word: positive, negative, neutral, or mixed."},
    {"index": 2, "label": "label_prefix", "system_prompt": "Sentiment:"},
    {"index": 3, "label": "verbose_instruction",
     "system_prompt": "Analyze the emotional tone of the following text. Determine whether the sentiment expressed is positive, negative, neutral, or mixed. Consider word choice, context, and emotional cues. Output only a single word: positive, negative, neutral, or mixed."},
]

with open(DATA_PATH) as f:
    questions = json.load(f)
print(f"Worker: loaded {len(questions)} questions", flush=True)

from llama_cpp import Llama
print(f"Worker: loading model...", flush=True)
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
    latencies = []
    for qi, q in enumerate(questions):
        messages = [
            {"role": "system", "content": sp},
            {"role": "user", "content": q["prompt"]}
        ]
        t_req = time.time()
        try:
            resp = llm.create_chat_completion(messages=messages, max_tokens=64, temperature=0.0)
            got = resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            got = f"<ERROR: {e}>"
        latencies.append(time.time() - t_req)

        # Extract the first word (sentiment should be single word)
        first_word = got.strip().split()[0].strip(",.!;:?") if got.strip() else ""
        is_correct = fuzzy_match(first_word, q.get("expected_answer", ""))
        if is_correct:
            correct += 1
        total += 1

    acc = correct / total if total else 0
    avg_latency_ms = (sum(latencies) / len(latencies) * 1000) if latencies else 0
    results[pk] = {
        "accuracy": round(acc, 4),
        "correct": correct,
        "total": total,
        "avg_latency_ms": round(avg_latency_ms, 1)
    }
    print(f"Worker: prompt {pi} done: {results[pk]}", flush=True)

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f)
print(f"Worker done: results written to {RESULTS_PATH}", flush=True)
