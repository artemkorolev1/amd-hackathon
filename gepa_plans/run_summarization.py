#!/usr/bin/env python3
"""Orchestrator for summarization eval. Launches subprocess per model."""
import json, subprocess, sys, os, time

os.environ["PYTHONUNBUFFERED"] = "1"

RESULTS_PATH = "/home/artem/dev/amd-hackathon/gepa_plans/summarization_eval_results.json"
WORKER = "/home/artem/dev/amd-hackathon/gepa_plans/worker_summarization.py"

MODELS = [
    {"name": "qwen2.5-1.5b-instruct",     "path": "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"},
    {"name": "qwen2.5-coder-1.5b-instruct","path": "/home/artem/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"},
    {"name": "llama-3.2-1b-instruct",      "path": "/home/artem/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"},
]

if __name__ == "__main__":
    total_start = time.time()
    results = {}

    for mi, model_info in enumerate(MODELS):
        mname = model_info["name"]
        mpath = model_info["path"]
        tmp_results = f"/tmp/summarization_results_{mname}.json"

        print(f"\n{'='*70}", flush=True)
        print(f"Model {mi+1}/{len(MODELS)}: {mname}", flush=True)
        print(f"{'='*70}", flush=True)

        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, WORKER, mpath, tmp_results],
            capture_output=True, text=True, timeout=600,
        )
        elapsed = time.time() - t0

        if proc.stdout:
            for line in proc.stdout.strip().split("\n"):
                print(f"  {line}", flush=True)

        if proc.returncode != 0:
            print(f"  [ERROR] exit code {proc.returncode}", flush=True)
            if proc.stderr:
                for line in proc.stderr.strip().split("\n")[-5:]:
                    print(f"  STDERR: {line}", flush=True)
            continue

        with open(tmp_results) as f:
            results[mname] = json.load(f)["results"]
        os.unlink(tmp_results)

        print(f"  Runtime: {elapsed:.1f}s", flush=True)
        best_key = max(results[mname], key=lambda k: results[mname][k]["accuracy"])
        best_info = results[mname][best_key]
        print(f"  Best: [{best_info['label']}] acc={best_info['accuracy']:.4f} ({best_info['correct']}/{best_info['total']}) lat={best_info['avg_latency_ms']:.1f}ms", flush=True)

    # Summary
    best_per_model = {}
    for mn in results:
        raw = results[mn]
        per_prompt = raw.get("results", raw) if isinstance(raw, dict) else raw
        prompt_keys = [k for k in per_prompt if k.startswith("prompt_")]
        if not prompt_keys:
            continue
        bk = max(prompt_keys, key=lambda k: per_prompt[k]["accuracy"])
        best_per_model[mn] = {
            "prompt_key": bk,
            "accuracy": per_prompt[bk]["accuracy"],
            "label": per_prompt[bk]["label"],
            "correct": per_prompt[bk]["correct"],
            "total": per_prompt[bk]["total"],
            "avg_latency_ms": per_prompt[bk]["avg_latency_ms"],
        }

    all_entries = [(mn, pk, per_prompt[pk]["accuracy"]) for mn in results for pk in (results[mn].get("results", results[mn]) if isinstance(results[mn], dict) else results[mn]) if str(pk).startswith("prompt_")]
    overall_best = {}
    if all_entries:
        best_m, best_pk, best_a = max(all_entries, key=lambda x: x[2])
        overall_best = {
            "model": best_m,
            "prompt_key": best_pk,
            "accuracy": best_a,
            "label": results[best_m][best_pk]["label"],
            "correct": results[best_m][best_pk]["correct"],
            "total": results[best_m][best_pk]["total"],
            "avg_latency_ms": results[best_m][best_pk]["avg_latency_ms"],
        }

    output = {
        "task": "summarization_gepa_eval",
        "date": "2026-07-13",
        "dataset": "summarization_combined_25",
        "num_questions": 25,
        "models_tested": [m["name"] for m in MODELS],
        "results": results,
        "best_per_model": best_per_model,
        "overall_best": overall_best,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  [save] {RESULTS_PATH}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("SUMMARIZATION EVAL SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    for mn, bp in sorted(best_per_model.items()):
        print(f"  {mn:30s}: [{bp['label']:25s}] acc={bp['accuracy']:.4f} ({bp['correct']}/{bp['total']}) lat={bp['avg_latency_ms']:.1f}ms", flush=True)
    if overall_best:
        print(f"\n  Overall: {overall_best['model']} + [{overall_best['label']}] = {overall_best['accuracy']:.4f}", flush=True)
    print(f"  Total time: {(time.time()-total_start)/60:.1f}min", flush=True)
