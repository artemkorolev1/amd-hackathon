#!/usr/bin/env python3
"""Orchestrator for math eval. Launches worker_math.py as a subprocess per model.
Uses shared eval_common.fuzzy_match — no string interpolation issues."""

import json
import subprocess
import sys
import os
import time

os.environ['PYTHONUNBUFFERED'] = '1'

RESULTS_PATH = "/home/artem/dev/amd-hackathon/gepa_plans/math_94q_results.json"
WORKER = "/home/artem/dev/amd-hackathon/gepa_plans/worker_math.py"

PROMPT_STRATEGIES = [
    {"index": 0, "label": "empty",                        "system_prompt": ""},
    {"index": 1, "label": "Math:",                        "system_prompt": "Math:"},
    {"index": 2, "label": "Answer only with a number.",   "system_prompt": "Answer only with a number."},
    {"index": 3, "label": "Let's think step by step.",    "system_prompt": "Let's think step by step."},
    {"index": 4, "label": "Calculate:",                   "system_prompt": "Calculate:"},
    {"index": 5, "label": "Answer: ",                     "system_prompt": "Answer: "},
]

MODELS = [
    {"name": "qwen2.5-math-1.5b", "path": "/home/artem/models/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf"},
    {"name": "smollm2-1.7b",      "path": "/home/artem/models/smollm2-1.7b-instruct-q4_k_m.gguf"},
    {"name": "qwen2.5-1.5b",      "path": "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"},
]

if __name__ == "__main__":
    total_start = time.time()
    results = {}

    for mi, model_info in enumerate(MODELS):
        mname = model_info["name"]
        mpath = model_info["path"]
        tmp_results = f"/tmp/math_results_{mname}.json"

        print(f"\n{'='*70}", flush=True)
        print(f"Model {mi+1}/{len(MODELS)}: {mname}", flush=True)
        print(f"{'='*70}", flush=True)

        # Launch as fresh subprocess — each model in its own process
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

        # Read results
        with open(tmp_results) as f:
            results[mname] = json.load(f)
        os.unlink(tmp_results)

        # Build report
        all_prompts = results[mname]
        best_key = max(all_prompts, key=lambda k: all_prompts[k]["accuracy"])
        best_pi = int(best_key.split("_")[1])
        best_label = next(s["label"] for s in PROMPT_STRATEGIES if s["index"] == best_pi)
        print(f"  Best: [{best_label}] acc={all_prompts[best_key]['accuracy']:.3f}", flush=True)

        # Save incremental
        best_per_model = {}
        for mn in results:
            bk = max(results[mn], key=lambda k: results[mn][k]["accuracy"])
            bi = int(bk.split("_")[1])
            bl = next(s["label"] for s in PROMPT_STRATEGIES if s["index"] == bi)
            best_per_model[mn] = {"prompt_index": bi, "accuracy": results[mn][bk]["accuracy"], "label": bl}

        output = {
            "date": "2026-07-13",
            "dataset": "math_combined_94",
            "models_tested": [m["name"] for m in MODELS],
            "prompts": PROMPT_STRATEGIES,
            "results": results,
            "best_per_model": best_per_model,
            "overall_best": {},
        }
        all_entries = [(mn, pk, results[mn][pk]["accuracy"]) for mn in results for pk in results[mn]]
        if all_entries:
            best_m, best_pk, best_a = max(all_entries, key=lambda x: x[2])
            best_pi = int(best_pk.split("_")[1])
            best_label = next(s["label"] for s in PROMPT_STRATEGIES if s["index"] == best_pi)
            output["overall_best"] = {"model": best_m, "prompt_index": best_pi, "accuracy": best_a, "label": best_label}

        with open(RESULTS_PATH, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  [save] {RESULTS_PATH}", flush=True)

    # Final
    print(f"\n{'='*70}", flush=True)
    print(f"MATH EVAL DONE", flush=True)
    print(f"{'='*70}", flush=True)
    with open(RESULTS_PATH) as f:
        final = json.load(f)
    for mn, bp in final.get("best_per_model", {}).items():
        print(f"  {mn}: [{bp['label']}] acc={bp['accuracy']:.3f}")
    ob = final.get("overall_best", {})
    if ob:
        print(f"\nOverall: {ob['model']} + [{ob['label']}] = {ob['accuracy']:.3f}")
    print(f"Total: {(time.time()-total_start)/60:.1f}min", flush=True)
