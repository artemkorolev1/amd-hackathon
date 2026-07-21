#!/usr/bin/env python3
"""Evaluate 6 prompt strategies across 4 GGUF models on 94 math questions on GPU.
Each model runs as a SEPARATE subprocess. Fixed regex escaping for worker code."""

import json
import re
import time
import sys
import os
import subprocess

os.environ['PYTHONUNBUFFERED'] = '1'

RESULTS_PATH = "/home/artem/dev/amd-hackathon/gepa_plans/math_94q_results.json"
DATA_PATH = "/home/artem/dev/amd-hackathon/data/eval/math_combined_80.json"

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
    {"name": "llama-3.2-1b",      "path": "/home/artem/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"},
]

# The worker template uses {model_path} and {tmp_path} as .format() placeholders
# Regex is written DIRECTLY (no string-interpolation mangling)
WORKER_TEMPLATE = r"""import json, re, time, os, gc, sys
os.environ['PYTHONUNBUFFERED'] = '1'

DATA_PATH = "/home/artem/dev/amd-hackathon/data/eval/math_combined_80.json"

PROMPT_STRATEGIES = [
    {"index": 0, "label": "empty",                        "system_prompt": ""},
    {"index": 1, "label": "Math:",                        "system_prompt": "Math:"},
    {"index": 2, "label": "Answer only with a number.",   "system_prompt": "Answer only with a number."},
    {"index": 3, "label": "Let's think step by step.",    "system_prompt": "Let's think step by step."},
    {"index": 4, "label": "Calculate:",                   "system_prompt": "Calculate:"},
    {"index": 5, "label": "Answer: ",                     "system_prompt": "Answer: "},
]

def fm(a, e):
    a, e = a.strip().lower(), e.strip().lower()
    if not a or not e: return False
    if a == e: return True
    if len(e) <= 20 and e in a: return True
    if len(a) <= 20 and a in e: return True
    na = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", a)]
    ne = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", e)]
    if na and ne:
        an, en = na[-1], ne[-1]
        if en != 0 and abs((an - en) / en) <= 0.01: return True
        if an == en: return True
    ta = set(t for t in re.split(r"[^a-zA-Z0-9.]+", a) if t)
    te = set(t for t in re.split(r"[^a-zA-Z0-9.]+", e) if t)
    if len(te) > 0 and len(ta & te) / len(te) >= 0.8: return True
    return False

with open(DATA_PATH) as f:
    questions = json.load(f)
print(f"Worker: loaded {len(questions)} questions", flush=True)

from llama_cpp import Llama
print(f"Worker: loading model on GPU...", flush=True)
t0 = time.time()
llm = Llama(model_path="{model_path}", n_ctx=2048, n_gpu_layers=-1, verbose=False)
print(f"Worker: loaded in {time.time()-t0:.1f}s", flush=True)

results = {{}}
for strat in PROMPT_STRATEGIES:
    pi = strat["index"]
    pl = strat["label"]
    sp = strat["system_prompt"]
    pk = f"prompt_{{pi}}"
    print(f"Worker: prompt {{pi}} [{{pl}}]", flush=True)
    correct = 0
    total = 0
    for qi, q in enumerate(questions):
        messages = [{{"role": "system", "content": sp}}, {{"role": "user", "content": q["prompt"]}}]
        try:
            resp = llm.create_chat_completion(messages=messages, max_tokens=64, temperature=0.0)
            got = resp["choices"][0]["message"]["content"]
        except Exception as e:
            got = f"<ERROR: {{e}}>"
        is_correct = fm(got, q.get("expected_answer", ""))
        if is_correct:
            correct += 1
        total += 1
    acc = correct / total if total else 0
    results[pk] = {{"accuracy": round(acc, 3), "correct": correct, "total": total}}
    print(f"Worker: prompt {{pi}} done: {{results[pk]}}", flush=True)

with open("{tmp_path}", "w") as f:
    json.dump(results, f)
print(f"Worker done: results written to {{tmp_path}}", flush=True)
"""


def eval_model_subprocess(model_name, model_path):
    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, dir='/tmp')
    tmp_path = tmp.name
    tmp.close()

    worker_code = WORKER_TEMPLATE.replace("{model_path}", model_path).replace("{tmp_path}", tmp_path)
    worker_file = f'/tmp/math_worker_{model_name}.py'
    with open(worker_file, 'w') as f:
        f.write(worker_code)

    print(f"  [subprocess] Launching {model_name}...", flush=True)
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, worker_file],
        capture_output=True, text=True, timeout=600,
    )
    elapsed = time.time() - t0
    print(f"  [subprocess] {model_name} finished in {elapsed:.1f}s", flush=True)
    if result.stdout:
        for line in result.stdout.strip().split('\n'):
            print(f"    {line}", flush=True)
    if result.returncode != 0:
        print(f"  [ERROR] {model_name} exit code {result.returncode}", flush=True)
        if result.stderr:
            for line in result.stderr.strip().split('\n')[-10:]:
                print(f"    STDERR: {line}", flush=True)
        os.unlink(tmp_path)
        os.unlink(worker_file)
        return None

    with open(tmp_path) as f:
        model_results = json.load(f)
    os.unlink(tmp_path)
    os.unlink(worker_file)
    return model_results


if __name__ == "__main__":
    with open(DATA_PATH) as f:
        questions = json.load(f)
    print(f"Master: loaded {len(questions)} math questions", flush=True)

    total_start = time.time()
    results = {}

    for mi, model_info in enumerate(MODELS):
        mname = model_info["name"]
        print(f"\n{'='*70}", flush=True)
        print(f"Model {mi+1}/{len(MODELS)}: {mname}", flush=True)
        print(f"{'='*70}", flush=True)

        model_results = eval_model_subprocess(mname, model_info["path"])
        if model_results is None:
            print(f"  [SKIP] {mname} failed", flush=True)
            continue

        results[mname] = model_results

        # Build and save report
        best_per_model = {}
        for mn in results:
            best_key = max(results[mn], key=lambda k: results[mn][k]["accuracy"])
            best_pi = int(best_key.split("_")[1])
            best_label = next(s["label"] for s in PROMPT_STRATEGIES if s["index"] == best_pi)
            best_per_model[mn] = {
                "prompt_index": best_pi, "accuracy": results[mn][best_key]["accuracy"], "label": best_label,
            }

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
        print(f"  [save] Results saved to {RESULTS_PATH}", flush=True)

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
