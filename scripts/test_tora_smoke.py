#!/usr/bin/env python3
"""Smoke test: ToRA solver on 5 GSM8K problems using qwen2.5-math-1.5b."""

import json
import os
import sys
import time

os.environ["PYTHONUNBUFFERED"] = "1"

MODEL_PATH = "/home/artem/models/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf"
DATA_PATH = "/tmp/gsm8k_test.parquet"

from agent.solvers.tora_solver import solve_with_tora

# Test problems: 5 diverse GSM8K sample problems
TEST_PROBLEMS = [
    {
        "prompt": "Janet has 5 apples. She buys 3 more packages of apples, each containing 6 apples. Then she gives 2 apples to Tom and eats 1. How many apples does Janet have now?",
        "expected": "30"
    },
    {
        "prompt": "A baker bakes 24 cookies. He puts them into boxes. Each box holds 8 cookies. How many boxes does he need?",
        "expected": "3"
    },
    {
        "prompt": "Sarah has $50. She buys a book for $12 and a pen for $3.50. How much money does she have left?",
        "expected": "34.50"
    },
    {
        "prompt": "There are 15 students in a class. 7 of them are boys. How many are girls?",
        "expected": "8"
    },
    {
        "prompt": "A train travels at 60 miles per hour for 2.5 hours. How far does it travel?",
        "expected": "150"
    }
]

# Also load 5 from GSM8K test set
import pandas as pd
df = pd.read_parquet(DATA_PATH)
gsm8k_samples = []
for i in range(5):
    row = df.iloc[i]
    # The answer format is "#### 42" in GSM8K
    answer_str = row["answer"]
    expected_num = answer_str.split("####")[-1].strip() if "####" in answer_str else answer_str.strip()
    gsm8k_samples.append({
        "prompt": row["question"],
        "expected": expected_num
    })

ALL_TEST_PROBLEMS = TEST_PROBLEMS + gsm8k_samples

def fuzzy_match(answer, expected):
    a = str(answer).strip().lower()
    e = str(expected).strip().lower()
    if not a or not e:
        return False
    if a == e:
        return True
    if len(e) <= 20 and e in a:
        return True
    if len(a) <= 20 and a in e:
        return True
    import re
    na = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", a)]
    ne = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", e)]
    if na and ne:
        an, en = na[-1], ne[-1]
        if en != 0 and abs((an - en) / en) <= 0.01:
            return True
        if an == en:
            return True
    ta = set(t for t in re.split(r"[^a-zA-Z0-9.]+", a) if t)
    te = set(t for t in re.split(r"[^a-zA-Z0-9.]+", e) if t)
    if len(te) > 0 and len(ta & te) / len(te) >= 0.8:
        return True
    return False

print(f"Loading model from {MODEL_PATH}...", flush=True)
t0 = time.time()
from llama_cpp import Llama
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_gpu_layers=-1, verbose=False)
print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)

def infer_fn(messages, max_tokens, stop_seq, category=""):
    try:
        resp = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=40,
            min_p=0.0,
            repeat_penalty=1.0,
            stop=stop_seq,
        )
        return resp["choices"][0]["message"]["content"] or ""
    except Exception as e:
        print(f"  INFER ERROR: {e}", flush=True)
        return ""

print("\n" + "=" * 70, flush=True)
print("ToRA Smoke Test — 10 Problems", flush=True)
print("=" * 70, flush=True)

correct = 0
total = 0

for i, prob in enumerate(ALL_TEST_PROBLEMS):
    print(f"\n--- Problem {i+1} ---", flush=True)
    print(f"Q: {prob['prompt'][:100]}...", flush=True)
    print(f"Expected: {prob['expected']}", flush=True)
    
    t_start = time.time()
    try:
        ans = solve_with_tora(prob["prompt"], llm, infer_fn, max_tokens=512, timeout=10)
    except Exception as e:
        ans = None
        print(f"  TORA ERROR: {e}", flush=True)
    
    elapsed = time.time() - t_start
    is_correct = fuzzy_match(ans or "", prob["expected"])
    
    print(f"Raw answer: {ans!r}", flush=True)
    print(f"Correct: {is_correct} ({elapsed:.1f}s)", flush=True)
    
    if is_correct:
        correct += 1
    total += 1

print(f"\n{'='*70}", flush=True)
print(f"Result: {correct}/{total} correct ({correct/total*100:.1f}%)", flush=True)
print(f"{'='*70}", flush=True)

del llm
