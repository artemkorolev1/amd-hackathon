#!/usr/bin/env python3
"""Evaluate 5 prompt strategies across 4 GGUF models on 58 factual questions on GPU.
Uses same fuzzy_match, chat format, temperature=0, max_tokens=64.
Saves results to: /home/artem/dev/amd-hackathon/gepa_plans/factual_58q_results.json"""

import json
import re
import time
import sys
import os
import gc

os.environ['PYTHONUNBUFFERED'] = '1'

# ── fuzzy_match (verbatim from eval_gen0.py) ──────────────────────────────
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


# ── Config ──────────────────────────────────────────────────────────────────
RESULTS_PATH = "/home/artem/dev/amd-hackathon/gepa_plans/factual_58q_results.json"
DATA_PATH = "/home/artem/dev/amd-hackathon/data/eval/factual_combined_80.json"

MODELS = [
    {"name": "smollm2-1.7b",       "path": "/home/artem/models/smollm2-1.7b-instruct-q4_k_m.gguf"},
    {"name": "qwen2.5-1.5b",       "path": "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"},
    {"name": "llama-3.2-1b",       "path": "/home/artem/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"},
    {"name": "gemma-3-1b",         "path": "/home/artem/models/gemma-3-1b-it-Q4_K_M.gguf"},
]

PROMPT_STRATEGIES = [
    {"index": 0, "label": "empty",                     "system_prompt": ""},
    {"index": 1, "label": "Answer:",                   "system_prompt": "Answer:"},
    {"index": 2, "label": "Fact:",                     "system_prompt": "Fact:"},
    {"index": 3, "label": "Answer directly.",          "system_prompt": "Answer directly."},
    {"index": 4, "label": "Be precise.",               "system_prompt": "Be precise. Use exact names, dates, and numbers."},
]

PREVIOUS_BEST = {
    "smollm2-1.7b": {"prompt": "empty", "accuracy": 0.474},
    "qwen2.5-1.5b": {"prompt": "Be precise.", "accuracy": 0.368},
    "llama-3.2-1b": {"prompt": "Fact:", "accuracy": 0.421},
    "gemma-3-1b": {"prompt": "Answer:", "accuracy": 0.263},
}

# ── Load eval data ──────────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    questions = json.load(f)

print(f"Loaded {len(questions)} factual questions from {DATA_PATH}", flush=True)

# ── Results accumulator ─────────────────────────────────────────────────────
results = {}
best_per_model = {}

# Load existing results if available (recovery)
if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH) as f:
        existing = json.load(f)
    results = existing.get("results", {})
    print(f"Loaded existing results for models: {list(results.keys())}", flush=True)

def save_results():
    """Write results to file."""
    output = {
        "date": "2026-07-13",
        "dataset": "factual_combined_58",
        "models_tested": [m["name"] for m in MODELS],
        "prompts": PROMPT_STRATEGIES,
        "results": results,
        "best_per_model": best_per_model,
        "overall_best": {},
        "previous_best": PREVIOUS_BEST,
    }
    # Compute best_per_model
    for mname in results:
        prompts = results[mname]
        best_key = None
        best_acc = -1.0
        best_label = ""
        for pkey, pdata in prompts.items():
            if pdata["accuracy"] > best_acc:
                best_acc = pdata["accuracy"]
                best_key = pkey
                best_label = next(s["label"] for s in PROMPT_STRATEGIES if f"prompt_{s['index']}" == pkey)
        best_per_model[mname] = {"prompt_index": int(best_key.split("_")[1]), "accuracy": best_acc, "label": best_label}
    # Compute overall best
    all_entries = []
    for mname, prompts in results.items():
        for pkey, pdata in prompts.items():
            all_entries.append((mname, pkey, pdata["accuracy"]))
    if all_entries:
        best_m, best_pk, best_a = max(all_entries, key=lambda x: x[2])
        best_pi = int(best_pk.split("_")[1])
        best_label = next(s["label"] for s in PROMPT_STRATEGIES if s["index"] == best_pi)
        output["overall_best"] = {
            "model": best_m,
            "prompt_index": best_pi,
            "accuracy": best_a,
            "label": best_label,
        }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  [save] Results saved to {RESULTS_PATH}", flush=True)


# ── Eval function ───────────────────────────────────────────────────────────
def evaluate_model(model_info, existing_results=None):
    """Load model, evaluate all 5 prompt strategies, unload, return results dict."""
    mname = model_info["name"]
    mpath = model_info["path"]
    existing_results = existing_results or {}

    print(f"\n{'='*70}", flush=True)
    print(f"MODEL: {mname}", flush=True)
    print(f"{'='*70}", flush=True)

    # Start with existing results (per-prompt recovery)
    model_results = dict(existing_results)
    completed_prompts = set(model_results.keys())
    if completed_prompts:
        print(f"  Already have {len(completed_prompts)} prompts completed: {sorted(completed_prompts)}", flush=True)

    # Check if we need to do any work
    all_pkeys = set(f"prompt_{s['index']}" for s in PROMPT_STRATEGIES)
    if completed_prompts == all_pkeys:
        print(f"  All prompts already completed for {mname}, skipping.", flush=True)
        return model_results

    # Load model
    from llama_cpp import Llama
    print(f"Loading {mname} on GPU (n_gpu_layers=-1)...", flush=True)
    t0 = time.time()
    llm = Llama(
        model_path=mpath,
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False,
    )
    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

    for si, strat in enumerate(PROMPT_STRATEGIES):
        pidx = strat["index"]
        plabel = strat["label"]
        system_prompt = strat["system_prompt"]
        pkey = f"prompt_{pidx}"

        # Skip already-completed prompts
        if pkey in model_results:
            print(f"\n  ── Skipping Prompt {pidx}: [{plabel}] (already completed) ──", flush=True)
            continue

        print(f"\n  ── Prompt {pidx}: [{plabel}] system_prompt={repr(system_prompt[:80])} ──", flush=True)

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
                    max_tokens=64,
                    temperature=0.0,
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

            # Progress indicator
            mark = "✓" if is_correct else "✗"
            if qi == 0:
                print(f"    ", end="", flush=True)
            print(mark, end="", flush=True)

        print()  # end progress line

        accuracy = correct / len(questions) if questions else 0
        avg_latency_ms = (total_time / len(questions) * 1000) if questions else 0

        print(f"    Result: {correct}/{len(questions)} correct = {accuracy:.3f}  avg_lat={avg_latency_ms:.0f}ms", flush=True)

        model_results[pkey] = {
            "accuracy": round(accuracy, 3),
            "correct": correct,
            "total": len(questions),
            "avg_latency_ms": round(avg_latency_ms, 1),
        }

        # Incremental save after each prompt
        results[mname] = model_results
        save_results()

    # Unload model
    print(f"\n  Unloading {mname}...", flush=True)
    del llm
    gc.collect()
    print(f"  Done.", flush=True)

    return model_results


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    total_start = time.time()
    for mi, model_info in enumerate(MODELS):
        mname = model_info["name"]
        existing_mresults = results.get(mname, {})
        # Check if all prompts already done
        all_pkeys = set(f"prompt_{s['index']}" for s in PROMPT_STRATEGIES)
        if set(existing_mresults.keys()) == all_pkeys:
            print(f"\nSkipping {mname} (all {len(all_pkeys)} prompts already completed)", flush=True)
            continue
        if existing_mresults:
            print(f"\nResuming {mname} ({len(existing_mresults)}/{len(all_pkeys)} prompts completed)", flush=True)
        model_results = evaluate_model(model_info, existing_results=existing_mresults)
        results[mname] = model_results
        save_results()
        elapsed = time.time() - total_start
        print(f"\n  [{mname}] finished. Total elapsed: {elapsed/60:.1f}min", flush=True)

    # Final summary
    print(f"\n{'='*70}", flush=True)
    print(f"FINAL SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)

    # Re-save with final best computation
    save_results()

    # Print summary
    with open(RESULTS_PATH) as f:
        final = json.load(f)
    print(f"\nBest per model:")
    for mname, bp in final.get("best_per_model", {}).items():
        print(f"  {mname}: prompt [{bp['label']}] accuracy={bp['accuracy']:.3f}")
    ob = final.get("overall_best", {})
    print(f"\nOverall best: {ob.get('model','?')} with prompt [{ob.get('label','?')}] accuracy={ob.get('accuracy',0):.3f}")

    # Compare with previous best
    print(f"\nComparison to previous best:")
    for mname, prev in PREVIOUS_BEST.items():
        bp = final.get("best_per_model", {}).get(mname, {})
        if bp:
            delta = bp.get("accuracy", 0) - prev.get("accuracy", 0)
            sign = "+" if delta >= 0 else ""
            print(f"  {mname}: {bp.get('label','?')} {bp.get('accuracy',0):.3f} (was {prev.get('accuracy',0):.3f} with '{prev.get('prompt','?')}', Δ={sign}{delta:.3f})")

    print(f"\nTotal runtime: {(time.time()-total_start)/60:.1f} minutes", flush=True)
    print(f"\nResults saved to: {RESULTS_PATH}", flush=True)
