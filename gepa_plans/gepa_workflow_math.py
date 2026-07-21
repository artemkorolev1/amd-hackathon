#!/usr/bin/env python3
"""
GEPA prompt evolution for 3-step math workflow.

Tests 75 prompt combinations (5 plan × 5 solve × 3 compose) on a 10-question
subset to find the best, then verifies the top combo(s) on all 94 questions.

Models tested:
  - qwen2.5-coder-1.5b-instruct  (/home/artem/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf)
  - qwen2.5-1.5b-instruct        (/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf)

Usage:
    python gepa_workflow_math.py [--model coder|instruct] [--quick]
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import re
import argparse
from dataclasses import dataclass, field
from typing import Any, Optional

sys.path.insert(0, '/home/artem/dev/amd-hackathon')

DATA_PATH = "/home/artem/dev/amd-hackathon/data/eval/math_combined_80.json"
RESULTS_PATH = "/home/artem/dev/amd-hackathon/gepa_plans/gepa_math_results.json"

MODEL_CODER = "/home/artem/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
MODEL_INSTRUCT = "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"

# ── Prompt variants ───────────────────────────────────────────────────────────

PLAN_PROMPTS = [
    "Analyze this math problem. List the steps needed to solve it. Output ONLY the numbered plan, nothing else.",
    "Create a concise step-by-step plan for solving this math problem. List operations and expected calculations. DO NOT solve it — just plan. Output only the numbered plan.",
    "Break down this problem into clear sequential steps. For each step, state what calculation or operation is needed. Format as a numbered list. No solution text.",
    "Identify the known values, the unknown, and the math operations needed to find the answer. Present as a short plan. Output ONLY the plan.",
    "Plan how to solve this. What are the intermediate calculations? List them in order. Be specific about numbers and operations. Plan only.",
]

SOLVE_PROMPTS = [
    "Execute the plan step by step. Show each calculation clearly. Put the final numeric answer in \\boxed{}.",
    "Follow the plan above. Work through each step showing your work. End with \\boxed{answer} containing just the number.",
    "Now solve using the plan. Show every calculation. Verify each step. Final answer in \\boxed{}.",
    "Implement the plan. Step by step computation shown. Final answer must be in \\boxed{} with no extra text inside.",
    "Solve step by step following the provided plan. Double-check your arithmetic. Conclude with \\boxed{number}.",
]

COMPOSE_PROMPTS = [
    "Present the final answer: The answer is \\boxed{number}.",
    "Format: \\boxed{answer}. Only output the boxed answer.",
    "Final answer only: \\boxed{number}.",
]

# ── Baseline combo (matching what Llama eval used) ──────────────────────────
BASELINE_COMBO = {
    "plan": "Create a concise step-by-step plan for solving this math problem. List operations and expected calculations. DO NOT solve it — just plan. Output only the numbered plan.",
    "solve": "Follow the plan above. Work through each step showing your work. End with \\boxed{answer} containing just the number.",
    "compose": "Format: \\boxed{answer}. Only output the boxed answer.",
    "label": "baseline",
}


# ── Answer matching ──────────────────────────────────────────────────────────

def extract_boxed(text: str) -> Optional[str]:
    """Extract content from \boxed{...} or \\(...\\)."""
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m:
        return m.group(1).strip()
    m = re.search(r'\\\(([^)]+)\\\)', text)
    if m:
        return m.group(1).strip()
    return None


def extract_final_number(text: str) -> Optional[str]:
    """Try to extract a number from the last line or sentence."""
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        # Look for patterns like "answer is X" or "= X" or just a number
        m = re.search(r'(?:answer|is|=|:)\s*(\d+(?:\.\d+)?)', line, re.IGNORECASE)
        if m:
            return m.group(1)
        # Look for a standalone number at end
        m = re.search(r'(\d+(?:\.\d+)?)\s*$', line)
        if m:
            return m.group(1)
    return None


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    s = s.strip().lower()
    # Remove leading zeros
    s = re.sub(r'^0+(\d)', r'\1', s)
    # Remove trailing .0
    s = re.sub(r'\.0$', '', s)
    return s


def is_correct(got: str, expected: str) -> bool:
    """Check if got matches expected."""
    if not got or not expected:
        return False
    got_n = normalize_answer(got)
    exp_n = normalize_answer(expected)
    if got_n == exp_n:
        return True
    # Check if expected is contained in got (handles multi-answer cases)
    if exp_n in got_n:
        return True
    return False


# ── Workflow runner ──────────────────────────────────────────────────────────

class WorkflowRunner:
    """Runs 3-step workflows with arbitrary prompt combos."""

    def __init__(self, model_path: str, model_label: str):
        from llama_cpp import Llama
        self.model_label = model_label
        print(f"  Loading {model_label}...", flush=True)
        t0 = time.time()
        self.llm = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=-1, verbose=False)
        print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

    def unload(self):
        del self.llm
        gc.collect()
        if 'torch' in sys.modules:
            import torch
            torch.cuda.empty_cache()
        time.sleep(0.5)

    def infer(self, messages: list[dict], max_tokens: int = 256, temperature: float = 0.0) -> str:
        try:
            resp = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp["choices"][0]["message"]["content"]
        except Exception as e:
            return f"<ERROR: {e}>"

    def run_workflow(self, prompt: str, plan_prompt: str, solve_prompt: str,
                     compose_prompt: str) -> dict[str, Any]:
        """Run a 3-step workflow and return results."""
        artifacts: dict[str, str] = {"_input": prompt}
        step_results = []
        t0 = time.time()

        steps = [
            ("plan", plan_prompt),
            ("solve", solve_prompt),
            ("compose", compose_prompt),
        ]

        for step_name, sys_prompt in steps:
            prior_keys = [k for k in artifacts if k != "_input"]
            if prior_keys:
                last_key = prior_keys[-1]
                last_out = artifacts[last_key]
                if len(last_out) > 400:
                    last_out = last_out[:400] + "\n...[truncated]"
                user_msg = (
                    f"Original problem:\n{prompt}\n\n=== [{last_key}] output ===\n"
                    f"{last_out}\n\nNow: {sys_prompt}"
                )
            else:
                user_msg = prompt

            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ]

            t_step = time.time()
            result = self.infer(messages)
            step_time = time.time() - t_step

            artifacts[step_name] = result
            step_results.append({
                "step": step_name,
                "latency_ms": round(step_time * 1000, 1),
                "output_truncated": result[:150] + "..." if len(result) > 150 else result,
            })

        total_time = time.time() - t0
        final_output = artifacts.get("compose", artifacts.get("solve", ""))
        boxed = extract_boxed(final_output)
        if boxed:
            final_answer = boxed
        else:
            # Try solve step for boxed answer
            solve_output = artifacts.get("solve", "")
            boxed = extract_boxed(solve_output)
            if boxed:
                final_answer = boxed
            else:
                final_answer = extract_final_number(final_output) or final_output.strip()

        return {
            "final_answer": final_answer,
            "artifacts": artifacts,
            "step_results": step_results,
            "total_latency_ms": round(total_time * 1000, 1),
        }


# ── Grid search ──────────────────────────────────────────────────────────────

def grid_search(questions: list[dict], runner: WorkflowRunner,
                save_label: str, max_combos: Optional[int] = None) -> list[dict]:
    """Run all 75 prompt combos on questions, return ranked results."""
    all_results = []
    total_combos = len(PLAN_PROMPTS) * len(SOLVE_PROMPTS) * len(COMPOSE_PROMPTS)
    combo_idx = 0

    for pi, plan_p in enumerate(PLAN_PROMPTS):
        for si, solve_p in enumerate(SOLVE_PROMPTS):
            for ci, compose_p in enumerate(COMPOSE_PROMPTS):
                combo_idx += 1
                if max_combos and combo_idx > max_combos:
                    break

                combo_label = f"P{pi}S{si}C{ci}"
                correct = 0
                total_latency = 0
                details = []

                for qi, q in enumerate(questions):
                    q_prompt = q["prompt"]
                    expected = str(q.get("expected_answer", ""))

                    result = runner.run_workflow(q_prompt, plan_p, solve_p, compose_p)
                    got = result["final_answer"]
                    correct_flag = is_correct(got, expected)
                    if correct_flag:
                        correct += 1
                    total_latency += result["total_latency_ms"]

                    details.append({
                        "q_idx": qi,
                        "expected": expected,
                        "got": got[:100],
                        "correct": correct_flag,
                        "latency_ms": result["total_latency_ms"],
                    })

                accuracy = correct / len(questions)
                avg_latency = total_latency / len(questions)

                entry = {
                    "combo_index": combo_idx,
                    "label": combo_label,
                    "plan_idx": pi,
                    "solve_idx": si,
                    "compose_idx": ci,
                    "plan_prompt": plan_p[:80] + "...",
                    "solve_prompt": solve_p[:80] + "...",
                    "compose_prompt": compose_p[:40] + "...",
                    "accuracy": round(accuracy, 4),
                    "correct": correct,
                    "total": len(questions),
                    "avg_latency_ms": round(avg_latency, 1),
                    "details": details,
                }
                all_results.append(entry)

                mark = "✓" if accuracy >= 0.5 else " "
                print(f"  {mark} [{combo_label}] acc={accuracy:.3f}  "
                      f"({correct}/{len(questions)})  lat={avg_latency:.0f}ms",
                      flush=True)

                # Force break if combo_idx reached max
            if max_combos and combo_idx >= max_combos:
                break
        if max_combos and combo_idx >= max_combos:
            break

    # Sort by accuracy descending, then latency ascending
    all_results.sort(key=lambda r: (-r["accuracy"], r["avg_latency_ms"]))
    return all_results


# ── Full evaluation ──────────────────────────────────────────────────────────

def full_eval(questions: list[dict], runner: WorkflowRunner,
              plan_prompt: str, solve_prompt: str, compose_prompt: str,
              label: str) -> dict[str, Any]:
    """Run a single prompt combo on ALL questions."""
    correct = 0
    total_latency = 0
    details = []
    t0 = time.time()

    print(f"  Full eval with combo '{label}' on {len(questions)} questions...", flush=True)
    for qi, q in enumerate(questions):
        q_prompt = q["prompt"]
        expected = str(q.get("expected_answer", ""))

        result = runner.run_workflow(q_prompt, plan_p, solve_p, compose_p)
        got = result["final_answer"]
        correct_flag = is_correct(got, expected)
        if correct_flag:
            correct += 1
        total_latency += result["total_latency_ms"]

        details.append({
            "q_idx": qi,
            "task_id": q.get("task_id", f"q{qi}"),
            "expected": expected,
            "got": got[:100],
            "correct": correct_flag,
            "latency_ms": result["total_latency_ms"],
        })

        if (qi + 1) % 20 == 0:
            print(f"    ... {qi+1}/{len(questions)} correct={correct}/{qi+1} "
                  f"({correct/(qi+1):.3f})", flush=True)

    total_time = time.time() - t0
    accuracy = correct / len(questions)

    return {
        "label": label,
        "model": runner.model_label,
        "questions": len(questions),
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "total_time_s": round(total_time, 1),
        "avg_latency_ms": round(total_latency / len(questions), 1),
        "plan_prompt": plan_prompt,
        "solve_prompt": solve_prompt,
        "compose_prompt": compose_prompt,
        "details": details,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GEPA workflow math optimizer")
    parser.add_argument("--model", choices=["coder", "instruct", "both"],
                        default="both", help="Model(s) to test")
    parser.add_argument("--quick", action="store_true",
                        help="Only test first 5 plan × 3 solve × 2 compose = 30 combos")
    parser.add_argument("--skip-grid", action="store_true",
                        help="Skip grid search, only run full eval on best found")
    parser.add_argument("--full-only", action="store_true",
                        help="Only run full 94-question eval (no grid search)")
    args = parser.parse_args()

    # Load data
    with open(DATA_PATH) as f:
        all_questions = json.load(f)
    subset = all_questions[:10]  # First 10 for grid search
    print(f"Loaded {len(all_questions)} questions (subset: {len(subset)})", flush=True)

    # Results accumulator
    all_data = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "math_combined_80.json",
        "questions_total": len(all_questions),
        "subset_size": len(subset),
        "prompt_variants": {
            "plan": len(PLAN_PROMPTS),
            "solve": len(SOLVE_PROMPTS),
            "compose": len(COMPOSE_PROMPTS),
            "total_combos": len(PLAN_PROMPTS) * len(SOLVE_PROMPTS) * len(COMPOSE_PROMPTS),
        },
        "models": {},
    }

    models_to_test = []
    if args.model in ("coder", "both"):
        models_to_test.append(("coder", MODEL_CODER))
    if args.model in ("instruct", "both"):
        models_to_test.append(("instruct", MODEL_INSTRUCT))

    for model_label, model_path in models_to_test:
        print(f"\n{'='*70}", flush=True)
        print(f" MODEL: {model_label}", flush=True)
        print(f"{'='*70}", flush=True)

        runner = WorkflowRunner(model_path, model_label)
        model_data = {}

        # ── Grid search ─────────────────────────────────────────────────────
        grid_results = []
        if not args.full_only:
            max_combos = 30 if args.quick else None
            print(f"\n--- Grid search: testing prompt combos ---", flush=True)
            grid_results = grid_search(subset, runner, model_label,
                                       max_combos=max_combos)
            print(f"\nTop 5 combos for {model_label}:", flush=True)
            for i, r in enumerate(grid_results[:5]):
                print(f"  #{i+1}: [{r['label']}] acc={r['accuracy']:.4f} "
                      f"lat={r['avg_latency_ms']:.0f}ms", flush=True)
                print(f"       plan={r['plan_prompt']}", flush=True)
                print(f"       solve={r['solve_prompt']}", flush=True)
                print(f"       compose={r['compose_prompt']}", flush=True)

        model_data["grid_search"] = grid_results

        # ── Full eval on top 2 combos ──────────────────────────────────────
        full_evals = []
        combos_to_eval = []

        if grid_results:
            # Top 2 from grid search
            for i in range(min(2, len(grid_results))):
                r = grid_results[i]
                combos_to_eval.append({
                    "label": f"top{i+1}_{r['label']}",
                    "plan": PLAN_PROMPTS[r["plan_idx"]],
                    "solve": SOLVE_PROMPTS[r["solve_idx"]],
                    "compose": COMPOSE_PROMPTS[r["compose_idx"]],
                })
            # Also add the baseline
            combos_to_eval.append(BASELINE_COMBO)
        elif args.full_only:
            # Use sensible defaults
            combos_to_eval = [BASELINE_COMBO]

        print(f"\n--- Full eval on {len(all_questions)} questions ---", flush=True)
        for combo in combos_to_eval:
            print(f"\n  Combo: {combo['label']}", flush=True)
            result = full_eval(all_questions, runner,
                               combo["plan"], combo["solve"], combo["compose"],
                               combo["label"])
            full_evals.append(result)
            print(f"  Result: {result['correct']}/{result['questions']} = "
                  f"{result['accuracy']:.4f}  ({result['total_time_s']:.1f}s)",
                  flush=True)

        model_data["full_evals"] = full_evals
        all_data["models"][model_label] = model_data

        runner.unload()
        print(f"\nFinished {model_label}", flush=True)

    # ── Save results ────────────────────────────────────────────────────────
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_data, f, indent=2, default=str)
    print(f"\n{'='*70}", flush=True)
    print(f"RESULTS SAVED TO: {RESULTS_PATH}", flush=True)
    print(f"{'='*70}", flush=True)

    # Print summary
    print(f"\n{'='*70}", flush=True)
    print(f" SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    for model_label in all_data["models"]:
        md = all_data["models"][model_label]
        print(f"\n{model_label}:", flush=True)
        if md.get("full_evals"):
            for fe in md["full_evals"]:
                print(f"  {fe['label']}: {fe['correct']}/{fe['questions']} = "
                      f"{fe['accuracy']:.4f}  ({fe['total_time_s']:.1f}s)",
                      flush=True)
        if md.get("grid_search"):
            print(f"  Grid search top: acc={md['grid_search'][0]['accuracy']:.4f} "
                  f"({md['grid_search'][0]['label']})", flush=True)

    # Compare with Llama baseline
    print(f"\n  Llama-3.2-1B baseline: 53/94 = 0.564", flush=True)


if __name__ == "__main__":
    main()
