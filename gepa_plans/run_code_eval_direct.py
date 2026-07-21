#!/usr/bin/env python3
"""Direct code eval for code_debug + code_gen. Single process per model, no subprocess workers."""
import json, sys, os, time, re

sys.path.insert(0, "/home/artem/dev/amd-hackathon/gepa_plans")
from eval_common import fuzzy_match

os.environ["PYTHONUNBUFFERED"] = "1"

DATA_DIR = "/home/artem/dev/amd-hackathon/data/eval"
RESULTS_PATH = "/home/artem/dev/amd-hackathon/gepa_plans/code_eval_results.json"

MODELS = [
    ("qwen2.5-coder-1.5b-instruct", "/home/artem/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
    ("qwen2.5-1.5b-instruct", "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"),
]

DEBUG_PROMPTS = [
    ("empty", ""),
    ("Fix the bug:", "Fix the bug:"),
    ("Debug:", "Debug:"),
    ("Fix:", "Fix:"),
]

GEN_PROMPTS = [
    ("empty", ""),
    ("Implement:", "Implement:"),
    ("Generate code:", "Generate code:"),
    ("Code:", "Code:"),
]

def load_data():
    with open(f"{DATA_DIR}/training-v3.json") as f:
        train = json.load(f)
    with open(f"{DATA_DIR}/validation-v3.json") as f:
        val = json.load(f)
    combined = train + val
    debug = [q for q in combined if q["category"] == "code_debug"]
    gen = [q for q in combined if q["category"] == "code_gen"]
    return debug, gen

def build_prompt(q, prefix):
    raw = q["prompt"]
    if prefix:
        # Replace "Fix this Python function:" or "Write a Python function:" with prefix
        for old in ["Fix this Python function:", "Write a Python function:"]:
            if raw.startswith(old):
                return prefix + " " + raw[len(old):].strip()
    return raw

def run_model(model_path, model_name, eval_data, prompt_presets):
    from llama_cpp import Llama as LlamaCpp
    print(f"\n### Loading {model_name}...", flush=True)
    t0 = time.time()
    llm = LlamaCpp(model_path=model_path, n_ctx=2048, n_gpu_layers=-1, verbose=False)
    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

    results = {}
    for label, prefix in prompt_presets:
        correct = 0
        total = 0
        latencies = []
        for q in eval_data:
            prompt = build_prompt(q, prefix)
            expected = q.get("expected_answer", q.get("answer", ""))
            t1 = time.time()
            try:
                resp = llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=256, temperature=0.0,
                )
                output = resp["choices"][0]["message"]["content"] or ""
            except:
                output = ""
            latency = (time.time() - t1) * 1000
            latencies.append(latency)
            if fuzzy_match(output, expected):
                correct += 1
            total += 1

        acc = correct / total if total else 0
        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        results[label] = {"accuracy": round(acc, 4), "correct": correct, "total": total, "avg_latency_ms": round(avg_lat, 1)}
        print(f"  [{label:20s}] acc={acc:.4f} ({correct}/{total}) lat={avg_lat:.0f}ms", flush=True)

    return results

def main():
    print("Code Eval — Direct (no subprocess workers)", flush=True)
    debug_data, gen_data = load_data()
    print(f"  code_debug: {len(debug_data)} questions", flush=True)
    print(f"  code_gen: {len(gen_data)} questions", flush=True)

    all_results = {}
    for model_name, model_path in MODELS:
        print(f"\n{'='*60}", flush=True)
        print(f"Model: {model_name}", flush=True)
        print(f"{'='*60}", flush=True)

        print("\n--- Code Debug ---", flush=True)
        debug_r = run_model(model_path, model_name, debug_data, DEBUG_PROMPTS)
        print("\n--- Code Gen ---", flush=True)
        gen_r = run_model(model_path, model_name, gen_data, GEN_PROMPTS)

        all_results[model_name] = {"code_debug": debug_r, "code_gen": gen_r}

    # Summary
    best_per_model = {}
    for mn in all_results:
        best_d = max(all_results[mn]["code_debug"].items(), key=lambda x: x[1]["accuracy"])
        best_g = max(all_results[mn]["code_gen"].items(), key=lambda x: x[1]["accuracy"])
        best_per_model[mn] = {
            "code_debug_best": best_d[0],
            "code_debug_acc": best_d[1]["accuracy"],
            "code_gen_best": best_g[0],
            "code_gen_acc": best_g[1]["accuracy"],
        }

    output = {
        "task": "code_gepa_eval",
        "date": "2026-07-13",
        "dataset": "training-v3 + validation-v3",
        "code_debug_questions": len(debug_data),
        "code_gen_questions": len(gen_data),
        "models": [m[0] for m in MODELS],
        "results": all_results,
        "best_per_model": best_per_model,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}", flush=True)

    # TL;DR
    print(f"\n{'='*60}", flush=True)
    print("CODE EVAL SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for mn, bp in best_per_model.items():
        print(f"{mn:35s} Debug: [{bp['code_debug_best']:20s}] {bp['code_debug_acc']:.4f}  Gen: [{bp['code_gen_best']:20s}] {bp['code_gen_acc']:.4f}", flush=True)

if __name__ == "__main__":
    main()
