#!/usr/bin/env python3
"""
Evaluate 3 GGUF models on training-v3.json (152 questions).
Models: Qwen2.5-1.5B, Qwen2.5-Math-1.5B, Gemma 3 1B
"""
import json
import re
import sys
import os
import time
import gc
import math
from typing import List

# ---------------------------------------------------------------------------
# Fuzzy matching (from evaluate.py)
# ---------------------------------------------------------------------------
def extract_numbers(s: str) -> List[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", s)]

def tokenize(s: str) -> set:
    return set(tok for tok in re.split(r"[^a-zA-Z0-9.]+", s.lower()) if tok)

def fuzzy_match(answer: str, expected: str) -> bool:
    a, e = answer.strip().lower(), expected.strip().lower()
    if not a or not e:
        return False
    # 1. Exact
    if a == e:
        return True
    # 2. Short answer substring
    if len(e) <= 20 and e in a:
        return True
    if len(a) <= 20 and a in e:
        return True
    # 3. Numeric (1% tolerance)
    nums_a = extract_numbers(a)
    nums_e = extract_numbers(e)
    if nums_a and nums_e:
        a_num, e_num = nums_a[-1], nums_e[-1]
        if e_num != 0 and abs((a_num - e_num) / e_num) <= 0.01:
            return True
        if a_num == e_num:
            return True
    # 4. Token overlap
    ta, te = tokenize(a), tokenize(e)
    if len(te) > 0 and len(ta & te) / len(te) >= 0.8:
        return True
    return False


MODELS = [
    {
        "name": "qwen2.5-1.5b",
        "path": "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "desc": "Generalist (Qwen2.5-1.5B)",
    },
    {
        "name": "qwen2.5-math-1.5b",
        "path": "/home/artem/models/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf",
        "desc": "Math/Logic (Qwen2.5-Math-1.5B)",
    },
    {
        "name": "gemma-3-1b",
        "path": "/home/artem/models/gemma-3-1b-it-Q4_K_M.gguf",
        "desc": "Format/Summarization (Gemma 3 1B)",
    },
]

# System prompt variants per model
SYSTEM_PROMPTS = {
    "qwen2.5-1.5b": "You are a helpful assistant. Answer the question concisely and directly. Output ONLY the answer.",
    "qwen2.5-math-1.5b": "You are a math assistant. Solve the problem step by step, then output the final answer as a single number or short phrase.",
    "gemma-3-1b": "You are a helpful assistant. Answer concisely.",
}


def load_model(model_cfg):
    """Load a GGUF model with GPU offloading."""
    from llama_cpp import Llama
    print(f"  Loading {model_cfg['name']}...", file=sys.stderr)
    t0 = time.time()
    llm = Llama(
        model_path=model_cfg["path"],
        n_gpu_layers=-1,       # Offload all layers to GPU
        n_ctx=2048,            # Context window
        verbose=False,
    )
    elapsed = time.time() - t0
    print(f"  Loaded in {elapsed:.1f}s", file=sys.stderr)
    return llm


def run_model(llm, model_name, question, max_tokens=512):
    """Run a single question through the model."""
    system_prompt = SYSTEM_PROMPTS.get(model_name, "Answer concisely.")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question["prompt"]},
    ]

    t0 = time.time()
    try:
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,        # Deterministic
            stop=None,
        )
        elapsed = time.time() - t0
        answer = response["choices"][0]["message"]["content"].strip()
        tokens_in = response.get("usage", {}).get("prompt_tokens", 0)
        tokens_out = response.get("usage", {}).get("completion_tokens", 0)
    except Exception as e:
        elapsed = time.time() - t0
        answer = f"[ERROR: {e}]"
        tokens_in, tokens_out = 0, 0

    return answer, elapsed, tokens_in, tokens_out


def evaluate_model(model_cfg, questions):
    """Run all questions through one model and return results."""
    import gc
    llm = load_model(model_cfg)
    results = []

    total_ok = 0
    total_time = 0
    total_tokens_in = 0
    total_tokens_out = 0
    by_category = {}
    by_difficulty = {}

    n = len(questions)
    for i, q in enumerate(questions):
        cat = q["category"]
        diff = q.get("difficulty", "unknown")
        expected = q["expected_answer"]

        answer, elapsed, tok_in, tok_out = run_model(llm, model_cfg["name"], q)
        ok = fuzzy_match(answer, expected)

        results.append({
            "task_id": q.get("task_id", f"q_{i}"),
            "category": cat,
            "difficulty": diff,
            "expected": expected,
            "got": answer,
            "correct": ok,
            "time_s": round(elapsed, 2),
            "tokens_in": tok_in,
            "tokens_out": tok_out,
        })

        total_ok += 1 if ok else 0
        total_time += elapsed
        total_tokens_in += tok_in
        total_tokens_out += tok_out

        by_category.setdefault(cat, {"correct": 0, "total": 0})
        by_category[cat]["total"] += 1
        by_category[cat]["correct"] += 1 if ok else 0

        by_difficulty.setdefault(diff, {"correct": 0, "total": 0})
        by_difficulty[diff]["total"] += 1
        by_difficulty[diff]["correct"] += 1 if ok else 0

        progress = f"[{i+1}/{n}] {cat:20s} {'✓' if ok else '✗'} {elapsed:.1f}s".ljust(65)
        if not ok:
            progress += f"  GOT: {answer[:60]}"
        print(progress, file=sys.stderr)

    # Free GPU memory
    del llm
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    summary = {
        "model": model_cfg["name"],
        "desc": model_cfg["desc"],
        "total": n,
        "correct": total_ok,
        "accuracy": round(total_ok / n * 100, 1),
        "total_time_s": round(total_time, 1),
        "avg_time_s": round(total_time / n, 2),
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "avg_tokens_out": round(total_tokens_out / n, 1),
        "by_category": {k: {**v, "acc": round(v["correct"] / v["total"] * 100, 1)} for k, v in sorted(by_category.items())},
        "by_difficulty": {k: {**v, "acc": round(v["correct"] / v["total"] * 100, 1)} for k, v in sorted(by_difficulty.items())},
    }

    return summary, results


def main():
    # Load dataset
    data_path = "/home/artem/dev/amd-hackathon/data/eval/training-v3.json"
    with open(data_path) as f:
        questions = json.load(f)

    print(f"Dataset: {len(questions)} questions", file=sys.stderr)

    all_summaries = []
    all_results = {}

    for model_cfg in MODELS:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Model: {model_cfg['desc']} ({model_cfg['name']})", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        summary, results = evaluate_model(model_cfg, questions)
        all_summaries.append(summary)
        all_results[model_cfg["name"]] = {"summary": summary, "results": results}

        print(f"\n  Accuracy: {summary['correct']}/{summary['total']} = {summary['accuracy']}%", file=sys.stderr)
        print(f"  Time: {summary['total_time_s']:.0f}s total, {summary['avg_time_s']:.2f}s avg/question", file=sys.stderr)
        print(f"  Tokens: {summary['total_tokens_out']} out ({summary['avg_tokens_out']:.0f}/q)", file=sys.stderr)

    # Print comparison table
    print(f"\n\n{'='*60}")
    print(f"FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Accuracy':<10} {'Time':<10} {'Tok/q':<10}")
    print(f"{'-'*55}")
    for s in all_summaries:
        print(f"{s['desc']:<25} {s['accuracy']:<10}% {s['avg_time_s']:<10.2f}s {s['avg_tokens_out']:<10.0f}")
    print()

    # Category comparison
    categories = sorted(all_summaries[0]["by_category"].keys())
    print(f"{'Category':<20}", end="")
    for s in all_summaries:
        print(f" {s['model']:<18}", end="")
    print()
    print(f"{'-'*20}", end="")
    for _ in all_summaries:
        print(f" {'-'*18}", end="")
    print()
    for cat in categories:
        print(f"{cat:<20}", end="")
        for s in all_summaries:
            acc = s["by_category"].get(cat, {}).get("acc", 0)
            print(f" {acc:<18}%", end="")
        print()

    # Difficulty comparison
    difficulties = sorted(all_summaries[0]["by_difficulty"].keys())
    print(f"\n{'Difficulty':<15}", end="")
    for s in all_summaries:
        print(f" {s['model']:<18}", end="")
    print()
    print(f"{'-'*15}", end="")
    for _ in all_summaries:
        print(f" {'-'*18}", end="")
    print()
    for diff in difficulties:
        print(f"{diff:<15}", end="")
        for s in all_summaries:
            acc = s["by_difficulty"].get(diff, {}).get("acc", 0)
            print(f" {acc:<18}%", end="")
        print()

    # Save detailed results
    output = {
        "dataset": "training-v3.json",
        "total_questions": len(questions),
        "models": all_summaries,
        "detailed": {k: v["results"] for k, v in all_results.items()},
    }
    
    out_path = "/home/artem/dev/amd-hackathon/data/eval/three_model_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to {out_path}")

    # Print sample failures for each model
    for s, (mn, rdata) in zip(all_summaries, all_results.items()):
        failures = [r for r in rdata["results"] if not r["correct"]]
        print(f"\n{'='*60}")
        print(f"Sample failures: {s['model']} ({len(failures)} total)")
        print(f"{'='*60}")
        for f in failures[:8]:
            print(f"  [{f['category']:15s}] exp: {f['expected'][:50]}")
            print(f"  {'':19s} got: {f['got'][:80]}")
            print()


if __name__ == "__main__":
    main()
