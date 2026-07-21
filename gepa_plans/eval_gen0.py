#!/usr/bin/env python3
"""Evaluate all 8 prompt variants from generation_0.json on smollm2-1.7b
using the 19 factual questions from training-v3.json."""

import json
import re
import time
import sys
import os

os.environ['PYTHONUNBUFFERED'] = '1'

# ── fuzzy_match (verbatim) ────────────────────────────────────────────────
def fuzzy_match(answer: str, expected: str) -> bool:
    a, e = answer.strip().lower(), expected.strip().lower()
    if not a or not e: return False
    if a == e: return True
    if len(e) <= 20 and e in a: return True
    if len(a) <= 20 and a in e: return True
    na = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", a)]
    ne = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", e)]
    if na and ne:
        an, en = na[-1], ne[-1]
        if en != 0 and abs((an-en)/en) <= 0.01: return True
        if an == en: return True
    ta = set(t for t in re.split(r"[^a-zA-Z0-9.]+", a) if t)
    te = set(t for t in re.split(r"[^a-zA-Z0-9.]+", e) if t)
    if len(te) > 0 and len(ta & te) / len(te) >= 0.8: return True
    return False


# ── load data ──────────────────────────────────────────────────────────────
gen_path = "/home/artem/dev/amd-hackathon/gepa_plans/generation_0.json"
data_path = "/home/artem/dev/amd-hackathon/data/eval/training-v3.json"
results_path = "/home/artem/dev/amd-hackathon/gepa_plans/generation_0_results.json"

with open(gen_path) as f:
    gen = json.load(f)

with open(data_path) as f:
    all_data = json.load(f)

# Filter factual questions
questions = [q for q in all_data if q.get("category") == "factual"]
print(f"Loaded {len(questions)} factual questions out of {len(all_data)} total", flush=True)

variants = gen["variants"]
print(f"Loaded {len(variants)} prompt variants", flush=True)
for v in variants:
    print(f"  [{v['name']}] temp={v['temperature']}, max_tok={v['max_tokens']}, sys_prompt={repr(v['system_prompt'][:60])}", flush=True)

# ── load model ─────────────────────────────────────────────────────────────
print("\nLoading smollm2-1.7b model (CPU only)...", flush=True)
t0 = time.time()
from llama_cpp import Llama
llm = Llama(
    model_path="/home/artem/models/smollm2-1.7b-instruct-q4_k_m.gguf",
    n_ctx=2048,
    n_gpu_layers=0,
    n_threads=4,
    verbose=False,
)
print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)


# ── evaluate ───────────────────────────────────────────────────────────────
all_results = []

for vi, variant in enumerate(variants):
    name = variant["name"]
    system_prompt = variant["system_prompt"]
    temperature = variant["temperature"]
    max_tokens = variant["max_tokens"]

    print(f"\n{'='*60}", flush=True)
    print(f"Variant {vi+1}/{len(variants)}: [{name}]", flush=True)
    print(f"  sys_prompt: {repr(system_prompt[:80])}", flush=True)
    print(f"  temp={temperature}, max_tokens={max_tokens}", flush=True)

    details = []
    correct = 0
    total_time = 0.0

    for qi, q in enumerate(questions):
        prompt_text = q["prompt"]
        expected = q["expected_answer"]
        task_id = q.get("task_id", f"q{qi}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ]

        t_start = time.time()
        try:
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            got = response["choices"][0]["message"]["content"]
        except Exception as e:
            got = f"<ERROR: {e}>"
        elapsed = time.time() - t_start
        total_time += elapsed

        is_correct = fuzzy_match(got, expected)
        if is_correct:
            correct += 1

        details.append({
            "task_id": task_id,
            "question": prompt_text,
            "expected": expected,
            "got": got,
            "correct": is_correct,
        })

        # Debug output for first variant, dots for rest
        if vi == 0:
            mark = "✓" if is_correct else "✗"
            print(f"  [{mark}] Q{qi+1}: {prompt_text[:60]}...", flush=True)
            print(f"        expected: {expected}")
            print(f"        got:      {got[:80]}")
            print(f"        ({elapsed*1000:.0f}ms)")
        else:
            if qi == 0:
                print(f"  Progress: ", end="", flush=True)
            mark = "." if is_correct else "x"
            print(mark, end="", flush=True)

    if vi > 0:
        print()  # end progress dots line

    accuracy = correct / len(questions) if questions else 0
    avg_latency_ms = (total_time / len(questions) * 1000) if questions else 0

    print(f"  Result: {correct}/{len(questions)} correct = {accuracy:.3f} accuracy", flush=True)
    print(f"  Avg latency: {avg_latency_ms:.0f}ms", flush=True)

    all_results.append({
        "name": name,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "accuracy": round(accuracy, 3),
        "correct": correct,
        "total": len(questions),
        "avg_latency_ms": round(avg_latency_ms, 1),
        "details": details,
    })

    # Save after each variant in case of interruption
    best = max(all_results, key=lambda r: r["accuracy"])
    output = {
        "generation": gen.get("generation", 0),
        "model": gen.get("model", "smollm2-1.7b"),
        "total_questions": len(questions),
        "results": all_results,
        "best_variant": {
            "name": best["name"],
            "accuracy": best["accuracy"],
        },
        "previous_best_accuracy": gen.get("previous_best_accuracy", 0.421),
        "improvement": best["accuracy"] > gen.get("previous_best_accuracy", 0.421),
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  (intermediate save to {results_path})", flush=True)

# ── final report ───────────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("EVALUATION COMPLETE", flush=True)
print(f"{'='*60}", flush=True)

best = max(all_results, key=lambda r: r["accuracy"])
print(f"\nBest variant: [{best['name']}] with accuracy {best['accuracy']:.3f}", flush=True)
prev_best = gen.get("previous_best_accuracy", 0.421)
print(f"Previous best: {prev_best:.3f}")
print(f"Improvement: {'YES ✓' if best['accuracy'] > prev_best else 'No'}", flush=True)

print(f"\nResults saved to: {results_path}", flush=True)
