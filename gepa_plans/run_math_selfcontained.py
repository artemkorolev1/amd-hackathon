#!/usr/bin/env python3
"""Self-contained math eval — no subprocesses, no bash wrapper.
Runs all 3 models sequentially in one process with aggressive memory cleanup.
Single file, correct regex, max_tokens=256, \boxed extraction."""

import json
import re
import time
import os
import gc

os.environ['PYTHONUNBUFFERED'] = '1'

DATA_PATH = "/home/artem/dev/amd-hackathon/data/eval/math_combined_80.json"
RESULTS_PATH = "/home/artem/dev/amd-hackathon/gepa_plans/math_94q_results.json"

PROMPT_STRATEGIES = [
    {"index": 0, "label": "empty", "system_prompt": ""},
    {"index": 1, "label": "Math:", "system_prompt": "Math:"},
    {"index": 2, "label": "Answer only with a number.", "system_prompt": "Answer only with a number."},
    {"index": 3, "label": "Let's think step by step.", "system_prompt": "Let's think step by step."},
    {"index": 4, "label": "Calculate:", "system_prompt": "Calculate:"},
    {"index": 5, "label": "Answer: ", "system_prompt": "Answer: "},
]

MODELS = [
    {"name": "qwen2.5-math-1.5b", "path": "/home/artem/models/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf"},
    {"name": "smollm2-1.7b",      "path": "/home/artem/models/smollm2-1.7b-instruct-q4_k_m.gguf"},
    {"name": "qwen2.5-1.5b",      "path": "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"},
]


def fuzzy_match(answer, expected):
    a, e = str(answer).strip().lower(), str(expected).strip().lower()
    if not a or not e:
        return False
    if a == e:
        return True
    if len(e) <= 20 and e in a:
        return True
    if len(a) <= 20 and a in e:
        return True
    na = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", a)]
    ne = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", e)]
    if na and ne:
        an, en = na[-1], ne[-1]
        if en != 0 and abs((an - en) / en) <= 0.01:
            return True
        if an == en:
            return True
    ta = set(t for t in re.split(r"[^a-zA-Z0-9.]+", a) if t)
    te = set(t for t in re.split(r"[^a-zA-Z0-9.]+", e) if t)
    if len(te) > 0 and len(ta & te) / len(te) >= 0.8:
        return True
    return False


def extract_from_box(raw):
    """Extract content from \boxed{...} if present."""
    if not isinstance(raw, str):
        return raw
    m = re.search(r'\\boxed\{([^}]+)\}', raw)
    return m.group(1) if m else raw


def save_results(results, partial=False):
    bp = {}
    for mn in results:
        bk = max(results[mn], key=lambda k: results[mn][k]["accuracy"])
        bi = int(bk.split("_")[1])
        bl = next(s["label"] for s in PROMPT_STRATEGIES if s["index"] == bi)
        bp[mn] = {"prompt_index": bi, "accuracy": results[mn][bk]["accuracy"], "label": bl}
    ob = {}
    ae = [(mn, pk, results[mn][pk]["accuracy"]) for mn in results for pk in results[mn]]
    if ae:
        bm, bpk, ba = max(ae, key=lambda x: x[2])
        bi = int(bpk.split("_")[1])
        bl = next(s["label"] for s in PROMPT_STRATEGIES if s["index"] == bi)
        ob = {"model": bm, "prompt_index": bi, "accuracy": ba, "label": bl}
    out = {
        "date": "2026-07-13",
        "dataset": "math_combined_94",
        "models_tested": [m["name"] for m in MODELS],
        "prompts": PROMPT_STRATEGIES,
        "results": results,
        "best_per_model": bp,
        "overall_best": ob,
        "note": "max_tokens=256, boxed_extraction=True, correct_regex=True" if not partial else "PARTIAL - in progress",
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  [save] {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    with open(DATA_PATH) as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} math questions", flush=True)

    total_start = time.time()
    results = {}

    for mi, model_info in enumerate(MODELS):
        mname = model_info["name"]
        mpath = model_info["path"]

        print(f"\n{'='*70}", flush=True)
        print(f"Model {mi+1}/{len(MODELS)}: {mname}", flush=True)
        print(f"{'='*70}", flush=True)

        from llama_cpp import Llama
        print(f"  Loading {mname} on GPU...", flush=True)
        t0 = time.time()
        llm = Llama(model_path=mpath, n_ctx=2048, n_gpu_layers=-1, verbose=False)
        print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

        model_results = {}

        for strat in PROMPT_STRATEGIES:
            pi = strat["index"]
            pl = strat["label"]
            sp = strat["system_prompt"]
            pk = f"prompt_{pi}"

            print(f"  Prompt {pi}: [{pl}] sys={repr(sp[:40])}", end=" ", flush=True)

            correct = 0
            total = 0
            t_prompt = time.time()

            for qi, q in enumerate(questions):
                messages = [{"role": "system", "content": sp}, {"role": "user", "content": q["prompt"]}]
                try:
                    resp = llm.create_chat_completion(messages=messages, max_tokens=256, temperature=0.0)
                    got = resp["choices"][0]["message"]["content"]
                except Exception as e:
                    got = f"<ERROR: {e}>"

                clean = extract_from_box(got)
                is_correct = fuzzy_match(clean, q.get("expected_answer", ""))
                if is_correct:
                    correct += 1
                total += 1

            acc = correct / total if total else 0
            lat = (time.time() - t_prompt) / total * 1000
            model_results[pk] = {"accuracy": round(acc, 3), "correct": correct, "total": total, "avg_latency_ms": round(lat, 1)}
            print(f"={acc:.3f} ({correct}/{total}) {lat:.0f}ms/q", flush=True)

        results[mname] = model_results
        save_results(results)

        # Unload
        print(f"  Unloading {mname}...", flush=True)
        del llm
        gc.collect()
        gc.collect()
        print(f"  Done. Elapsed: {(time.time()-total_start)/60:.1f}min total", flush=True)

    # Final
    print(f"\n{'='*70}", flush=True)
    print(f"MATH EVAL DONE", flush=True)
    print(f"{'='*70}", flush=True)
    save_results(results)
    with open(RESULTS_PATH) as f:
        final = json.load(f)
    for mn, bp in final.get("best_per_model", {}).items():
        print(f"  {mn}: [{bp['label']}] acc={bp['accuracy']:.3f}")
    ob = final.get("overall_best", {})
    if ob:
        print(f"\nOverall: {ob['model']} + [{ob['label']}] = {ob['accuracy']:.3f}")
    print(f"Total: {(time.time()-total_start)/60:.1f}min", flush=True)
