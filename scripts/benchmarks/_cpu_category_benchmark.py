#!/usr/bin/env python3
"""Category-by-category LLM timing analysis."""
import time, json, os, sys
os.environ["N_GPU_LAYERS"] = "0"
os.environ["N_THREADS"] = "2"

import contextlib, io

# Load model
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from llama_cpp import Llama
    llm = Llama(
        model_path="/home/artem/dev/amd-hackathon/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        n_ctx=2048, n_gpu_layers=0, n_threads=2, flash_attn=True, verbose=False
    )

# Load eval
with open("/home/artem/dev/amd-hackathon/data/eval/validation-v3.json") as f:
    data = json.load(f)

# Categorize
cats = {}
for q in data:
    c = q.get("category", "unknown")
    cats[c] = cats.get(c, 0) + 1
print(f"48 questions by category: {json.dumps(cats, indent=2)}")
print()

# Warmup
llm.create_chat_completion(messages=[{"role":"user","content":"warmup"}], max_tokens=5, temperature=0.0)

# Sample 2 per category
sampled = {}
for q in data:
    c = q.get("category", "unknown")
    if c not in sampled:
        sampled[c] = []
    if len(sampled[c]) < 3:
        sampled[c].append(q)

print(f"{'Category':15s} {'Q':5s} {'Time':8s} {'PromptT':8s} {'CompT':8s}")
print("-"*50)
total_time = 0.0
total_q = 0

for cat in sorted(sampled.keys()):
    qs = sampled[cat]
    for q in qs:
        prompt = q["prompt"]
        t0 = time.time()
        resp = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150, temperature=0.0, stop=["\n\n"]
        )
        elapsed = time.time() - t0
        usage = resp.get("usage", {})
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        print(f"{cat:15s} {q.get('task_id','?')[:4]:5s} {elapsed:7.3f}s {pt:8d} {ct:8d}")
        total_time += elapsed
        total_q += 1

print("-"*50)
print(f"{'TOTAL':15s} {total_q:5d} {total_time:7.3f}s")
print(f"{'AVG':15s} {'':5s} {total_time/total_q:7.3f}s")
print()
print("EXTRAPOLATION (raw LLM inference only):")
extrap = total_time/total_q*300
print(f"  300q x {total_time/total_q:.3f}s = {extrap:.1f}s")
print(f"  Plus model load ~1s: {extrap + 1:.1f}s")
print(f"  Deadline: 600s")
if extrap + 1 <= 600:
    print("  ✅ Achievable with raw LLM on every question")
else:
    shortfall = extrap + 1 - 600
    print(f"  ❌ NOT achievable on raw LLM ({shortfall:.0f}s over)")
    det_pct = 100 * (1 - 600 / (extrap + 1))
    print(f"  Need ~{det_pct:.0f}% deterministic bypass rate")
