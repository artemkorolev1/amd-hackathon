#!/usr/bin/env python3
"""
Quick re-evaluation of best cells to capture per-question failure details.
Reuses the saved gen0/gen1 results, re-evaluates only the top cells per model.
"""
import json, os, sys, time, gc, re
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agent.cell import Cell, DecodingConfig
from agent.evaluation_agent import EvaluationAgent

MODEL_PATHS = {
    "qwen2.5-1.5b": "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "qwen2.5-coder-1.5b": "/home/artem/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    "gemma-3-1b": "/home/artem/models/gemma-3-1b-it-Q4_K_M.gguf",
}
MODEL_KEYS = list(MODEL_PATHS.keys())
EVAL_SET_PATH = os.path.join(PROJECT_ROOT, "data", "eval", "generated", "sentiment_comprehensive_hard.json")
LOG_DIR = os.path.join(PROJECT_ROOT, "gepa_logs", "sentiment_gepa_20260713_102044")

# Load eval set
with open(EVAL_SET_PATH) as f:
    questions = json.load(f)
for q in questions:
    q["category"] = "sentiment"

# Also load from results to get exact question mapping  
with open(os.path.join(LOG_DIR, "eval_questions.json")) as f:
    eval_questions = json.load(f)

print(f"Loaded {len(questions)} questions")

# Load gen0 and gen1 results
with open(os.path.join(LOG_DIR, "gen0_results.json")) as f:
    gen0_data = json.load(f)
with open(os.path.join(LOG_DIR, "gen1_results.json")) as f:
    gen1_data = json.load(f)

# Find best cell per model per generation
best_cells_info = {}
all_data = gen0_data + gen1_data
for d in all_data:
    mk = d["model_key"]
    acc = d.get("metadata", {}).get("accuracy", 0.0)
    key = (mk, d["name"])
    if key not in best_cells_info or acc > best_cells_info[key]["metadata"]["accuracy"]:
        best_cells_info[key] = d

# Select top cells for failure analysis - one per model (best overall)
model_best = {}
for d in all_data:
    mk = d["model_key"]
    acc = d.get("metadata", {}).get("accuracy", 0.0)
    if mk not in model_best or acc > model_best[mk]["metadata"]["accuracy"]:
        model_best[mk] = d

print(f"\nWill re-evaluate {len(model_best)} best cells (one per model):")
for mk, d in model_best.items():
    print(f"  {mk}: {d['name']} (acc={d['metadata']['accuracy']:.3f})")

# Build Cell objects
cells_to_eval = []
for mk, d in model_best.items():
    dec = d["decoding"]
    cell = Cell(
        task_id="T03",
        model_key=mk,
        system_prompt=d["system_prompt"],
        decoding=DecodingConfig(
            temperature=dec["temperature"],
            max_tokens=dec["max_tokens"],
            top_p=dec["top_p"],
            top_k=dec["top_k"],
            min_p=dec["min_p"],
            repeat_penalty=dec["repeat_penalty"],
            seed=dec.get("seed"),
        ),
        aggregation="single",
        name=f"reval_{mk}",
        generation=0,
    )
    cells_to_eval.append(cell)

# Evaluate one model at a time
from llama_cpp import Llama

all_details = {}
for cell in cells_to_eval:
    mk = cell.model_key
    print(f"\n── Loading {mk} ──")
    llm = Llama(
        model_path=MODEL_PATHS[mk],
        n_ctx=2048,
        n_gpu_layers=-1,
        n_threads=4,
        verbose=False,
    )
    
    # Evaluate this single cell
    from agent.evaluation_agent import fuzzy_match, _compute_extra_metrics
    
    cat = "sentiment"
    task_questions = [q for q in questions if q.get("category") == cat]
    correct = 0
    details = []
    
    for q_idx, q in enumerate(task_questions):
        prompt_text = q.get("prompt", "")
        expected = q.get("expected_answer", "")
        
        messages = [
            {"role": "system", "content": cell.system_prompt},
            {"role": "user", "content": prompt_text},
        ]
        start = time.time()
        try:
            dec = cell.decoding
            resp = llm.create_chat_completion(
                messages=messages,
                max_tokens=dec.max_tokens,
                temperature=dec.temperature,
                top_p=dec.top_p,
                top_k=dec.top_k,
                min_p=dec.min_p,
                repeat_penalty=dec.repeat_penalty,
                seed=dec.seed,
            )
            elapsed = (time.time() - start) * 1000
            answer = resp["choices"][0]["message"]["content"].strip()
            usage = resp.get("usage", {})
            tok_count = usage.get("completion_tokens", len(answer.split()))
            
            is_correct = fuzzy_match(answer, expected)
            if is_correct:
                correct += 1
            
            extra = _compute_extra_metrics(cat, answer, expected)
            
            details.append({
                "idx": q_idx,
                "question": prompt_text[:80],
                "expected": expected,
                "got": answer[:120],
                "correct": is_correct,
                "latency_ms": round(elapsed, 1),
                "tokens": tok_count,
                "difficulty": q.get("difficulty", "unknown"),
                "source": q.get("source", ""),
            })
            
            if (q_idx + 1) % 20 == 0:
                print(f"  [{mk}] {q_idx+1}/{len(task_questions)} questions, acc so far: {correct/(q_idx+1):.3f}")
                
        except Exception as e:
            print(f"  [{mk}] Error on question {q_idx}: {e}")
            details.append({
                "idx": q_idx,
                "question": prompt_text[:80],
                "expected": expected,
                "error": str(e),
                "correct": False,
                "difficulty": q.get("difficulty", "unknown"),
            })
    
    n = len(task_questions)
    acc = correct / n
    print(f"  [{mk}] Final acc: {acc:.4f} ({correct}/{n})")
    
    all_details[mk] = {
        "cell_name": cell.name,
        "original_name": model_best[mk]["name"],
        "system_prompt": cell.system_prompt,
        "decoding": cell.decoding.to_dict(),
        "accuracy": acc,
        "correct": correct,
        "total": n,
        "details": details,
    }
    
    # Unload model
    del llm
    gc.collect()
    time.sleep(0.5)

# Save details
with open(os.path.join(LOG_DIR, "failure_details.json"), "w") as f:
    # Prepare serializable version
    serializable = {}
    for mk, data in all_details.items():
        serializable[mk] = {
            "original_name": data["original_name"],
            "system_prompt": data["system_prompt"],
            "decoding": data["decoding"],
            "accuracy": data["accuracy"],
            "correct": data["correct"],
            "total": data["total"],
            "details": data["details"],
        }
    json.dump(serializable, f, indent=2)

print(f"\n\nFailure details saved to {LOG_DIR}/failure_details.json")

# ── Generate failure analysis ────────────────────────────────────────────────

print("\n\n=== FAILURE ANALYSIS ===")
for mk, data in all_details.items():
    print(f"\n── {mk} ({data['original_name']}) ──")
    print(f"  Overall accuracy: {data['accuracy']:.4f} ({data['correct']}/{data['total']})")
    
    # Per difficulty
    by_diff = defaultdict(list)
    for d in data["details"]:
        diff = d.get("difficulty", "unknown")
        by_diff[diff].append(d)
    
    for diff, items in sorted(by_diff.items()):
        n_correct = sum(1 for d in items if d.get("correct", False))
        n_total = len(items)
        print(f"  {diff:8s}: {n_correct}/{n_total} = {n_correct/max(n_total,1):.3f}")
    
    # Hard question failures
    hard_fails = [d for d in data["details"] 
                  if d.get("difficulty") == "hard" and not d.get("correct", True)]
    print(f"\n  Hard question failures: {len(hard_fails)}")
    
    # Pattern analysis
    false_pos = 0
    neutral_err = 0
    mixed_err = 0
    for f in hard_fails:
        got_lower = f.get("got", "").lower().strip().rstrip(".")
        exp_lower = f.get("expected", "").lower().strip()
        
        # Check if got opposite sentiment
        if got_lower in ["positive", "negative"] and exp_lower in ["positive", "negative"]:
            if got_lower != exp_lower:
                false_pos += 1
        elif "neutral" in got_lower and "neutral" not in exp_lower:
            neutral_err += 1
        elif "mixed" in got_lower and "mixed" not in exp_lower:
            mixed_err += 1
    
    print(f"    False positives/negatives: {false_pos}")
    print(f"    Neutral misclassifications: {neutral_err}")
    print(f"    Mixed misclassifications: {mixed_err}")
    print(f"    Other: {len(hard_fails) - false_pos - neutral_err - mixed_err}")
    
    # Top 10 failure examples
    print(f"\n  Top 10 failures:")
    for i, f in enumerate(hard_fails[:10]):
        q_short = f.get("question", "")[:60]
        exp = f.get("expected", "")
        got = f.get("got", "")[:40]
        print(f"    {i+1}. E:{exp} G:{got} | {q_short}")

print("\nDone!")
