#!/usr/bin/env python3
"""Orchestrator for sentiment eval. Launches worker_sentiment.py as subprocess per model."""

import json
import subprocess
import sys
import os
import time

os.environ['PYTHONUNBUFFERED'] = '1'

RESULTS_PATH = "/home/artem/dev/amd-hackathon/gepa_plans/sentiment_eval_results.json"
WORKER = "/home/artem/dev/amd-hackathon/gepa_plans/worker_sentiment.py"

PROMPT_STRATEGIES = [
    {"index": 0, "label": "empty", "system_prompt": ""},
    {"index": 1, "label": "explicit_instruction",
     "system_prompt": "Classify the sentiment. Output EXACTLY one word: positive, negative, neutral, or mixed."},
    {"index": 2, "label": "label_prefix", "system_prompt": "Sentiment:"},
    {"index": 3, "label": "verbose_instruction",
     "system_prompt": "Analyze the emotional tone of the following text. Determine whether the sentiment expressed is positive, negative, neutral, or mixed. Consider word choice, context, and emotional cues. Output only a single word: positive, negative, neutral, or mixed."},
]

MODELS = [
    {"name": "qwen2.5-1.5b-instruct",     "path": "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"},
    {"name": "qwen2.5-coder-1.5b-instruct","path": "/home/artem/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"},
    {"name": "smollm2-1.7b-instruct",      "path": "/home/artem/models/smollm2-1.7b-instruct-q4_k_m.gguf"},
    {"name": "llama-3.2-1b-instruct",      "path": "/home/artem/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"},
]

if __name__ == "__main__":
    total_start = time.time()
    results = {}

    for mi, model_info in enumerate(MODELS):
        mname = model_info["name"]
        mpath = model_info["path"]
        tmp_results = f"/tmp/sentiment_results_{mname}.json"

        print(f"\n{'='*70}", flush=True)
        print(f"Model {mi+1}/{len(MODELS)}: {mname}", flush=True)
        print(f"{'='*70}", flush=True)

        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, WORKER, mpath, tmp_results],
            capture_output=True, text=True, timeout=600,
        )
        elapsed = time.time() - t0
        print(f"  Runtime: {elapsed:.1f}s", flush=True)

        if proc.stdout:
            for line in proc.stdout.strip().split('\n'):
                print(f"  {line}", flush=True)

        if proc.returncode != 0:
            print(f"  [ERROR] exit code {proc.returncode}", flush=True)
            if proc.stderr:
                for line in proc.stderr.strip().split('\n')[-5:]:
                    print(f"  STDERR: {line}", flush=True)
            continue

        with open(tmp_results) as f:
            results[mname] = json.load(f)
        os.unlink(tmp_results)

        # Report for this model
        all_prompts = results[mname]
        best_key = max(all_prompts, key=lambda k: all_prompts[k]["accuracy"])
        best_pi = int(best_key.split("_")[1])
        best_label = next(s["label"] for s in PROMPT_STRATEGIES if s["index"] == best_pi)
        print(f"  Best: [{best_label}] acc={all_prompts[best_key]['accuracy']:.4f} "
              f"({all_prompts[best_key]['correct']}/{all_prompts[best_key]['total']}) "
              f"latency={all_prompts[best_key]['avg_latency_ms']:.1f}ms", flush=True)

    # Build final summary
    best_per_model = {}
    for mn in results:
        bk = max(results[mn], key=lambda k: results[mn][k]["accuracy"])
        bi = int(bk.split("_")[1])
        bl = next(s["label"] for s in PROMPT_STRATEGIES if s["index"] == bi)
        best_per_model[mn] = {
            "prompt_index": bi,
            "accuracy": results[mn][bk]["accuracy"],
            "label": bl,
            "correct": results[mn][bk]["correct"],
            "total": results[mn][bk]["total"],
            "avg_latency_ms": results[mn][bk]["avg_latency_ms"],
        }

    all_entries = [(mn, pk, results[mn][pk]["accuracy"]) for mn in results for pk in results[mn]]
    overall_best = {}
    if all_entries:
        best_m, best_pk, best_a = max(all_entries, key=lambda x: x[2])
        best_pi = int(best_pk.split("_")[1])
        best_label = next(s["label"] for s in PROMPT_STRATEGIES if s["index"] == best_pi)
        overall_best = {
            "model": best_m,
            "prompt_index": best_pi,
            "accuracy": best_a,
            "label": best_label,
            "correct": results[best_m][best_pk]["correct"],
            "total": results[best_m][best_pk]["total"],
            "avg_latency_ms": results[best_m][best_pk]["avg_latency_ms"],
        }

    output = {
        "task": "sentiment_gepa_optimization",
        "date": "2026-07-13",
        "dataset": "sentiment_combined_25",
        "num_questions": 25,
        "models_tested": [m["name"] for m in MODELS],
        "prompts": PROMPT_STRATEGIES,
        "results": results,
        "best_per_model": best_per_model,
        "overall_best": overall_best,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  [save] {RESULTS_PATH}", flush=True)

    # Final summary
    print(f"\n{'='*70}", flush=True)
    print(f"SENTIMENT EVAL DONE", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Dataset: sentiment_combined_25 ({25} questions)")
    print(f"Questions drawn from: training-v3 (19) + validation-v3 (6)")
    print()
    for mn, bp in sorted(best_per_model.items()):
        print(f"  {mn:30s}: [{bp['label']:25s}] acc={bp['accuracy']:.4f} ({bp['correct']}/{bp['total']}) "
              f"lat={bp['avg_latency_ms']:.1f}ms")
    if overall_best:
        print()
        print(f"  Overall best: {overall_best['model']} + [{overall_best['label']}] = "
              f"{overall_best['accuracy']:.4f} ({overall_best['correct']}/{overall_best['total']})")
    print(f"  Total time: {(time.time()-total_start)/60:.1f}min", flush=True)
