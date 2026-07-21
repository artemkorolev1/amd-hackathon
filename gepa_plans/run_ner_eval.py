#!/usr/bin/env python3
"""
NER (T05) GEPA Optimisation — Full Evaluation Pipeline.

Builds combined eval set, tests 4 prompt strategies x 5 models + deterministic solver,
saves results to gepa_plans/ner_eval_results.json.
"""
import json, sys, os, subprocess, time, tempfile, copy

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = "/home/artem/dev/amd-hackathon"
MODEL_PATHS = {
    "qwen2.5-coder-1.5b":     os.path.join(os.path.expanduser("~"), "models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
    "qwen2.5-1.5b":           os.path.join(os.path.expanduser("~"), "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    "llama-3.2-1b":           os.path.join(os.path.expanduser("~"), "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
    "qwen2.5-math-1.5b":      os.path.join(os.path.expanduser("~"), "models/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf"),
    "smollm2-1.7b":           os.path.join(os.path.expanduser("~"), "models/smollm2-1.7b-instruct-q4_k_m.gguf"),
}

# ── Prompt strategies ─────────────────────────────────────────────────────────
PROMPT_STRATEGIES = [
    {"name": "blank",         "system_prompt": "",                      "user_prefix": ""},
    {"name": "verbose_extract", "system_prompt": "Extract named entities (PERSON, ORG, LOC, DATE, MISC):", "user_prefix": ""},
    {"name": "label_only",    "system_prompt": "Entities:",             "user_prefix": ""},
    {"name": "format_entity_type", "system_prompt": "Identify all named entities in the text. Format: Entity (TYPE)", "user_prefix": ""},
]

# ── Build combined NER eval set ────────────────────────────────────────────────
def build_ner_eval_set():
    with open(os.path.join(BASE, "data/eval/training-v3.json")) as f:
        train = json.load(f)
    with open(os.path.join(BASE, "data/eval/validation-v3.json")) as f:
        valid = json.load(f)
    ner_train = [q for q in train if q.get("category") == "ner"]
    ner_valid = [q for q in valid if q.get("category") == "ner"]
    combined = ner_train + ner_valid
    print(f"NER eval set: {len(ner_train)} training + {len(ner_valid)} validation = {len(combined)} questions", file=sys.stderr)
    return combined

# ── Run deterministic solver (prototype_ner_v3) ────────────────────────────────
def run_deterministic_solver(questions):
    """Test the deterministic regex-based NER solver from prototype_ner_v3.py."""
    sys.path.insert(0, os.path.join(BASE))
    from agent.solvers.prototype_ner_v3 import solve_ner
    from gepa_plans.eval_common import fuzzy_match

    results = []
    total_f1 = 0.0
    total_exact = 0
    total_fuzzy = 0

    for q in questions:
        result = solve_ner(q["prompt"], q["category"])
        result_str = result.strip() if result else ""
        expected_str = q["expected_answer"].strip()

        # Entity-level F1
        def norm(s):
            return s.replace('\u2019', "'").replace('\u2018', "'").replace('\u201c', '"').replace('\u201d', '"')
        rn, en = norm(result_str), norm(expected_str)
        exp_lines = set()
        for line in en.split('\n'):
            line = line.strip()
            if ':' in line: exp_lines.add(line)
        res_lines = set()
        for line in rn.split('\n'):
            line = line.strip()
            if ':' in line: res_lines.add(line)
        intersection = exp_lines & res_lines
        precision = len(intersection) / len(res_lines) if res_lines else 0.0
        recall = len(intersection) / len(exp_lines) if exp_lines else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        exact_ok = rn == en
        fuzzy_ok = fuzzy_match(result_str, expected_str)

        total_f1 += f1
        total_exact += 1 if exact_ok else 0
        total_fuzzy += 1 if fuzzy_ok else 0

        results.append({
            "task_id": q["task_id"],
            "expected": expected_str,
            "got": result_str,
            "exact_match": exact_ok,
            "fuzzy_match": fuzzy_ok,
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "correct_entities": len(intersection),
            "expected_entities": len(exp_lines),
            "result_entities": len(res_lines),
        })

    n = len(questions)
    return {
        "model": "prototype_ner_v3 (deterministic)",
        "prompt_name": "none (rule-based)",
        "system_prompt": "",
        "user_prefix": "",
        "workflow": False,
        "questions": n,
        "exact_match_accuracy": round(total_exact / n * 100, 1) if n else 0,
        "fuzzy_match_accuracy": round(total_fuzzy / n * 100, 1) if n else 0,
        "avg_f1": round(total_f1 / n, 4) if n else 0.0,
        "total_time_s": 0.0,
        "results": results,
    }

# ── Spawn subprocess worker for a single model+prompt ──────────────────────────
def spawn_worker(model_name, prompt_cfg, eval_file, workflow=False):
    """Spawn ner_worker.py as subprocess. Returns parsed JSON result."""
    cmd = [
        sys.executable,
        os.path.join(BASE, "gepa_plans/ner_worker.py"),
        "--model-path", MODEL_PATHS[model_name],
        "--model-name", model_name,
        "--prompt-name", prompt_cfg["name"],
        "--eval-file", eval_file,
        "--max-tokens", "128",
    ]
    if prompt_cfg["system_prompt"]:
        cmd += ["--system-prompt", prompt_cfg["system_prompt"]]
    if prompt_cfg["user_prefix"]:
        cmd += ["--user-prefix", prompt_cfg["user_prefix"]]
    if workflow:
        cmd += ["--workflow"]

    print(f"  Spawning: {model_name} / {prompt_cfg['name']}" + (" (2-step workflow)" if workflow else ""), file=sys.stderr)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  ERROR (exit={result.returncode}): {result.stderr[:500]}", file=sys.stderr)
        return None

    try:
        data = json.loads(result.stdout)
        data["total_time_s"] = round(elapsed, 1)
        return data
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}", file=sys.stderr)
        print(f"  stdout[:300]: {result.stdout[:300]}", file=sys.stderr)
        print(f"  stderr[:300]: {result.stderr[:300]}", file=sys.stderr)
        return None

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70, file=sys.stderr)
    print("NER (T05) — GEPA Optimization Evaluation", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    # 1. Build combined eval set
    questions = build_ner_eval_set()
    eval_file = os.path.join(BASE, "gepa_plans/_ner_eval_set_temp.json")
    with open(eval_file, "w") as f:
        json.dump(questions, f, indent=2)

    all_results = []
    total_runs = len(MODEL_PATHS) * (len(PROMPT_STRATEGIES) + 1) + 1  # models * (prompts + workflow) + deterministic
    run_count = 0

    # 2. Run deterministic solver first (no GPU needed)
    print(f"\n{'='*70}", file=sys.stderr)
    print("Deterministic Solver (prototype_ner_v3)", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    det_result = run_deterministic_solver(questions)
    all_results.append(det_result)
    print(f"  → Exact: {det_result['exact_match_accuracy']}%  Fuzzy: {det_result['fuzzy_match_accuracy']}%  AvgF1: {det_result['avg_f1']:.3f}", file=sys.stderr)
    run_count += 1

    # 3. Run LLM models — each model + each prompt strategy in subprocess
    for model_name in MODEL_PATHS:
        print(f"\n{'='*70}", file=sys.stderr)
        print(f"Model: {model_name}", file=sys.stderr)
        print(f"{'='*70}", file=sys.stderr)

        for prompt_cfg in PROMPT_STRATEGIES:
            run_count += 1
            print(f"\n  --- Prompt: \"{prompt_cfg['name']}\" ({run_count}/{total_runs}) ---", file=sys.stderr)
            result = spawn_worker(model_name, prompt_cfg, eval_file, workflow=False)
            if result:
                all_results.append(result)
                print(f"  → Exact: {result['exact_match_accuracy']}%  Fuzzy: {result['fuzzy_match_accuracy']}%  AvgF1: {result['avg_f1']:.3f}  Time: {result['total_time_s']:.0f}s", file=sys.stderr)

        # 4. Try NER_2STEP_WORKFLOW
        run_count += 1
        print(f"\n  --- Workflow: NER_2STEP ({run_count}/{total_runs}) ---", file=sys.stderr)
        workflow_cfg = {"name": "ner_2step_workflow", "system_prompt": "", "user_prefix": ""}
        result = spawn_worker(model_name, workflow_cfg, eval_file, workflow=True)
        if result:
            all_results.append(result)
            print(f"  → Exact: {result['exact_match_accuracy']}%  Fuzzy: {result['fuzzy_match_accuracy']}%  AvgF1: {result['avg_f1']:.3f}  Time: {result['total_time_s']:.0f}s", file=sys.stderr)

    # 5. Cleanup temp
    try:
        os.remove(eval_file)
    except OSError:
        pass

    # 6. Summary table
    print(f"\n\n{'='*70}", file=sys.stderr)
    print("SUMMARY — NER EVALUATION", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    print(f"{'Model':<28} {'Prompt':<22} {'Exact%':<8} {'Fuzzy%':<8} {'AvgF1':<8} {'Time':<8}", file=sys.stderr)
    print(f"{'-'*80}", file=sys.stderr)
    for r in all_results:
        model_short = r["model"].split("/")[-1].replace(".gguf", "")
        if len(model_short) > 27:
            model_short = model_short[:24] + "..."
        print(f"{model_short:<28} {r['prompt_name']:<22} {r['exact_match_accuracy']:<8} {r['fuzzy_match_accuracy']:<8} {r['avg_f1']:<8} {r['total_time_s']:.0f}s", file=sys.stderr)

    # 7. Find best per model
    print(f"\n{'='*70}", file=sys.stderr)
    print("BEST PER MODEL (by Exact Match %)", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    best_per_model = {}
    for r in all_results:
        key = r["model"].split("/")[-1].replace(".gguf", "")
        if key not in best_per_model or r["exact_match_accuracy"] > best_per_model[key]["exact_match_accuracy"]:
            best_per_model[key] = r
    for model, r in sorted(best_per_model.items()):
        print(f"  {model:<25} → {r['prompt_name']:<22} (Exact: {r['exact_match_accuracy']}%  F1: {r['avg_f1']:.3f})", file=sys.stderr)

    # 8. Save results
    out_path = os.path.join(BASE, "gepa_plans/ner_eval_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "pipeline": "NER (T05) GEPA optimization",
            "eval_size": len(questions),
            "models_tested": len(MODEL_PATHS) + 1,  # + deterministic
            "prompt_strategies": len(PROMPT_STRATEGIES),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}", file=sys.stderr)
    print("Done.", file=sys.stderr)

if __name__ == "__main__":
    main()
