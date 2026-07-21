#!/usr/bin/env python3
"""Quick debug: measure token length of qwen2.5-math responses."""
import json, sys, os
os.environ['PYTHONUNBUFFERED'] = '1'

from llama_cpp import Llama
MODEL_PATH = "/home/artem/models/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf"
DATA_PATH = "/home/artem/dev/amd-hackathon/data/eval/math_combined_80.json"

with open(DATA_PATH) as f:
    questions = json.load(f)

print("Loading qwen2.5-math on GPU...", flush=True)
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_gpu_layers=-1, verbose=False)

# Test 5 questions with max_tokens=512, track output length
for qi in range(5):
    q = questions[qi]
    messages = [{"role": "system", "content": ""}, {"role": "user", "content": q["prompt"]}]
    resp = llm.create_chat_completion(messages=messages, max_tokens=512, temperature=0.0)
    got = resp["choices"][0]["message"]["content"]
    token_count = resp["usage"]["completion_tokens"] if "usage" in resp else "?"
    has_box = "\\boxed{" in got
    # Find where boxed appears
    box_pos = got.find("\\boxed{") if has_box else -1
    
    print(f"\nQ{qi}: expected={q['expected_answer']}", flush=True)
    print(f"  tokens={token_count}, has_box={has_box}, box_pos={box_pos}", flush=True)
    print(f"  output[:200]: {got[:200]}", flush=True)
    if len(got) > 200:
        print(f"  ...({len(got)-200} more chars)", flush=True)
    
    # Check at what token count the answer appears
    # The fuzzy_match uses last number from \d+ pattern
    import re
    nums = re.findall(r"-?\d+(?:\.\d+)?", got)
    print(f"  numbers found: {nums[-3:] if nums else 'none'}", flush=True)

print("\nDone", flush=True)
