#!/usr/bin/env python3
"""
Evaluation pipeline for AMD ACT II hackathon.

MODES:
  model    — Run an LLM against a question set
  stage    — Run a single pipeline stage/filter
  e2e      — Run the full pipeline end-to-end, logging all steps
  compare  — Compare all saved results

OUTPUT (always persistent):
  eval_results/{run_id}_detail.json   — Per-item raw output + metrics
  eval_results/{run_id}_summary.json  — Aggregated stats
  eval_results/{run_id}_log.json      — (e2e only) Per-step traces

USAGE:
  # Run any HF model (no registry needed)
  python eval_pipeline.py --mode model --model Qwen/Qwen2.5-7B-Instruct --qs <file>

  # With a short label for results
  python eval_pipeline.py --mode model --model Qwen/Qwen2.5-7B-Instruct --label Qwen2.5-7B

  # Structured prompt mode
  python eval_pipeline.py --mode model --model <id> --prompt-mode structured

  # Stage eval (after implementing the stage function)
  python eval_pipeline.py --mode stage --stage stage2_category

  # End-to-end pipeline eval
  python eval_pipeline.py --mode e2e

  # Compare all results in eval_results/
  python eval_pipeline.py --mode compare
"""

import argparse
import importlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SHARED_DIR = Path(os.environ.get("SHARED_DIR", "/home/artem/dev/amd-hackathon-shared"))
RESULTS_DIR = PROJECT_ROOT / "eval_results"

from agent.pre_filter import stage0 as stage0_run
from agent.pre_filter import (
    RE_GREETING, RE_PURE_FENCE, RE_PURE_ARITH, RE_SINGLE_DEF,
    RE_CODE_FENCE, RE_CODE_HEADER, RE_ARITH_EXPR, RE_SUMMARIZE,
)
from agent.category_filter import classify as stage2_classify
from agent.complexity_filter import score as stage3_score
from agent.solvers import verify as verify_output

# ---------------------------------------------------------------------------
# Stage registry — each entry is (function_reference, inputs, outputs_desc)
# All stages functional and LLM-free — pure deterministic/heuristic.
# Covers: Stage 0 (16 filters across 3 tiers), Stage 1 (4 axes),
# Stage 2 (8-way scorer), Stage 3 (22-signal complexity),
# Stage 4 (decision table), Quality gates.
# ---------------------------------------------------------------------------
STAGES: Dict[str, Dict[str, Any]] = {
    # ── Stage 0 Tier 0 — Immediate bypass ──
    "tier0_greeting": {
        "fn": lambda prompt: {"matched": bool(RE_GREETING.match(prompt.strip()))},
        "inputs": ["prompt"], "outputs_desc": {"matched": "bool"},
    },
    "tier0_pure_fence": {
        "fn": lambda prompt: {"matched": bool(RE_PURE_FENCE.match(prompt.strip()))},
        "inputs": ["prompt"], "outputs_desc": {"matched": "bool"},
    },
    "tier0_pure_arith": {
        "fn": lambda prompt: {"matched": bool(RE_PURE_ARITH.match(prompt.strip()))},
        "inputs": ["prompt"], "outputs_desc": {"matched": "bool"},
    },
    "tier0_single_def": {
        "fn": lambda prompt: {"matched": bool(RE_SINGLE_DEF.match(prompt.strip()))},
        "inputs": ["prompt"], "outputs_desc": {"matched": "bool"},
    },

    # ── Stage 0 Tier 1 — Route to category ──
    "tier1_fenced_code": {
        "fn": lambda prompt: {"matched": bool(RE_CODE_FENCE.search(prompt))},
        "inputs": ["prompt"], "outputs_desc": {"matched": "bool"},
    },
    "tier1_implicit_code": {
        "fn": lambda prompt: {"matched": bool(RE_CODE_HEADER.search(prompt))},
        "inputs": ["prompt"], "outputs_desc": {"matched": "bool"},
    },
    "tier1_arithmetic_expr": {
        "fn": lambda prompt: {"matched": bool(RE_ARITH_EXPR.search(prompt)) and len(prompt.split()) < 15},
        "inputs": ["prompt"], "outputs_desc": {"matched": "bool"},
    },
    "tier1_explicit_summarize": {
        "fn": lambda prompt: {"matched": bool(RE_SUMMARIZE.match(prompt.strip()))},
        "inputs": ["prompt"], "outputs_desc": {"matched": "bool"},
    },

    # ── Stage 0 — Aggregate result ──
    "stage0_aggregate": {
        "fn": lambda prompt: {
            "action": stage0_run(prompt).action,
            "category_hint": stage0_run(prompt).category or "",
        },
        "inputs": ["prompt"], "outputs_desc": {"action": "bypass|route_to_stage3|continue",
                                                "category_hint": "str"},
    },

    # ── Stage 2 — 8-way category ──
    "stage2_category": {
        "fn": lambda prompt: dict(zip(["category", "confidence", "scores"], stage2_classify(prompt))),
        "inputs": ["prompt"], "outputs_desc": {"category": "str", "confidence": "float", "scores": "dict"},
    },

    # ── Stage 3 — Per-category complexity ──
    "stage3_complexity": {
        "fn": lambda prompt, category: {"complexity_score": stage3_score(prompt, category)},
        "inputs": ["prompt", "category"], "outputs_desc": {"complexity_score": "float 0..1"},
    },

    # ── Stage 4 — Decision table / pipeline ──
    "stage4_decision_table": {
        "fn": lambda category, complexity_score: {
            "needs_api": str(complexity_score >= 0.3 or category not in {"math", "logic", "sentiment", "ner", "factual", "code_debug"}),
            "simple_det": str(complexity_score < 0.3 and category in {"math", "logic", "sentiment", "ner", "factual", "code_debug"}),
        },
        "inputs": ["category", "complexity_score"],
        "outputs_desc": {"needs_api": "bool", "simple_det": "bool"},
    },

    # ── Quality gates (hedge / degenerate / length checks) ──
    "qc_hedge_detect": {
        "fn": lambda prompt: {
            "has_hedge": verify_output._has_hedge(prompt),
            "is_degenerate": verify_output._is_degenerate(prompt),
            "tooshort": verify_output._is_too_short(prompt),
        },
        "inputs": ["prompt"], "outputs_desc": {"has_hedge": "bool", "is_degenerate": "bool", "tooshort": "bool"},
    },
}


# ---------------------------------------------------------------------------
# Fuzzy matching (from evaluate.py)
# ---------------------------------------------------------------------------

def tokenize(s: str) -> set:
    return set(tok for tok in re.split(r"[^a-zA-Z0-9.]+", s.lower()) if tok)


def extract_numbers(s: str) -> List[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", s)]


def fuzzy_match(answer: str, expected: str) -> Dict[str, Any]:
    result = {"matched": False, "strategy": None, "details": {}}
    a_clean = answer.strip().lower()
    e_clean = expected.strip().lower()

    if a_clean == e_clean:
        return {"matched": True, "strategy": "exact"}
    if len(e_clean) >= 3 and e_clean in a_clean:
        return {"matched": True, "strategy": "substring_expected_in_answer"}
    if len(a_clean) >= 3 and len(e_clean) <= 20 and a_clean in e_clean:
        return {"matched": True, "strategy": "substring_answer_in_expected"}

    a_tokens = tokenize(answer)
    e_tokens = tokenize(expected)
    overlap = len(a_tokens & e_tokens) / len(e_tokens) if e_tokens else 0
    result["details"]["token_overlap"] = overlap
    if overlap >= 0.7:
        return {"matched": True, "strategy": "token_overlap_70pct"}

    a_nums = extract_numbers(answer)
    e_nums = extract_numbers(expected)
    if a_nums and e_nums and a_nums == e_nums:
        return {"matched": True, "strategy": "number_match"}
    if a_nums and e_nums and len(a_nums) == 1 and len(e_nums) == 1:
        if abs(a_nums[0] - e_nums[0]) / max(abs(e_nums[0]), 1) < 0.05:
            return {"matched": True, "strategy": "numeric_tolerance_5pct"}

    return result


# ---------------------------------------------------------------------------
# Question loading
# ---------------------------------------------------------------------------

def load_questions(path_spec: str) -> List[Dict[str, Any]]:
    path = Path(path_spec)
    if not path.is_absolute():
        resolved = PROJECT_ROOT / path
        if not resolved.exists():
            resolved = SHARED_DIR / path.name
        path = resolved

    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    if isinstance(data, dict) and all(k in data for k in ("simple", "medium", "hard")):
        items = []
        for diff, qs in data.items():
            for q in qs:
                q["difficulty"] = diff
                items.append(q)
        return items
    if isinstance(data, list):
        return data
    raise ValueError(f"Unknown question format in {path}")


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def load_model(hf_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"  Loading {hf_id}...")
    t0 = time.time()

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"  Loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


def run_model(
    model, tokenizer, prompt: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    import torch

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        formatted = f"### User:\n{prompt}\n\n### Assistant:\n"

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_tokens = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    latency = time.time() - t0

    output_ids = outputs[0][input_tokens:]
    output_tokens = output_ids.shape[0]
    raw = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    return {
        "raw_output": raw,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_s": round(latency, 3),
    }


# ---------------------------------------------------------------------------
# Eval: model mode
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    prompt: str
    expected_answer: str
    category: str
    difficulty: str
    raw_output: str
    match: bool
    match_details: dict
    input_tokens: int
    output_tokens: int
    latency_s: float


def eval_model(
    hf_id: str,
    label: str,
    questions: List[Dict],
    prompt_mode: str = "zero-shot",
    max_q: int = 0,
) -> List[ModelResult]:
    system_prompt = None
    if prompt_mode == "structured":
        system_prompt = (
            "You are an AI assistant that answers classification and reasoning questions accurately. "
            "Analyze the query carefully and respond with the correct answer. "
            "Give your final answer as a concise response."
        )

    model, tokenizer = load_model(hf_id)
    items = questions[:max_q] if max_q else questions
    results = []

    print(f"  Running {len(items)} questions ({prompt_mode})...")
    for i, q in enumerate(items):
        prompt_text = q.get("prompt", q.get("question", ""))
        expected = q.get("expected_answer", "")
        cat = q.get("category", "unknown")
        diff = q.get("difficulty", "unknown")

        if i % 10 == 0:
            print(f"    [{i}/{len(items)}] {label}")

        try:
            r = run_model(model, tokenizer, prompt_text, system_prompt)
            fm = fuzzy_match(r["raw_output"], expected)
            results.append(ModelResult(
                prompt=prompt_text, expected_answer=expected,
                category=cat, difficulty=diff,
                raw_output=r["raw_output"], match=fm["matched"],
                match_details=fm, input_tokens=r["input_tokens"],
                output_tokens=r["output_tokens"], latency_s=r["latency_s"],
            ))
        except Exception as e:
            results.append(ModelResult(
                prompt=prompt_text, expected_answer=expected,
                category=cat, difficulty=diff,
                raw_output=f"<ERROR: {e}>", match=False,
                match_details={"error": str(e)},
                input_tokens=0, output_tokens=0, latency_s=0,
            ))

    del model
    import torch
    torch.cuda.empty_cache()
    return results


def summarize_model(results: List[ModelResult], label: str, prompt_mode: str) -> Dict:
    total = len(results)
    if not total:
        return {"label": label, "mode": prompt_mode, "total": 0}

    matched = sum(1 for r in results if r.match)
    accuracy = round(matched / total * 100, 1)

    by_diff = {}
    for r in results:
        d = r.difficulty
        by_diff.setdefault(d, {"total": 0, "matched": 0})
        by_diff[d]["total"] += 1
        if r.match:
            by_diff[d]["matched"] += 1
    for d, v in by_diff.items():
        v["accuracy_pct"] = round(v["matched"] / v["total"] * 100, 1)

    by_cat = {}
    for r in results:
        c = r.category
        by_cat.setdefault(c, {"total": 0, "matched": 0})
        by_cat[c]["total"] += 1
        if r.match:
            by_cat[c]["matched"] += 1
    for c, v in by_cat.items():
        v["accuracy_pct"] = round(v["matched"] / v["total"] * 100, 1)

    in_toks = [r.input_tokens for r in results if r.input_tokens > 0]
    out_toks = [r.output_tokens for r in results if r.output_tokens > 0]
    lats = [r.latency_s for r in results if r.latency_s > 0]

    return {
        "label": label,
        "mode": prompt_mode,
        "model_id": label,
        "total": total,
        "matched": matched,
        "accuracy_pct": accuracy,
        "by_difficulty": by_diff,
        "by_category": by_cat,
        "avg_input_tokens": round(sum(in_toks)/len(in_toks), 1) if in_toks else 0,
        "avg_output_tokens": round(sum(out_toks)/len(out_toks), 1) if out_toks else 0,
        "avg_latency_s": round(sum(lats)/len(lats), 3) if lats else 0,
        "total_tokens_consumed": sum(out_toks) if out_toks else 0,
    }


# ---------------------------------------------------------------------------
# Eval: stage mode
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    prompt: str
    expected_answer: str
    category: str
    difficulty: str
    stage: str
    inputs: dict
    outputs: dict
    latency_s: float


def eval_stage(stage_name: str, questions: List[Dict], max_q: int = 0) -> List[StageResult]:
    cfg = STAGES.get(stage_name)
    if not cfg:
        available = ", ".join(STAGES.keys())
        raise ValueError(f"Unknown stage '{stage_name}'. Available: {available}")

    fn = cfg["fn"]
    items = questions[:max_q] if max_q else questions
    results = []

    for q in items:
        t0 = time.time()
        prompt_text = q.get("prompt", "")
        inputs = {k: prompt_text if k == "prompt" else q.get(k) for k in cfg["inputs"]}
        if fn is None:
            outputs = {"_error": "Not implemented", "_inputs": inputs}
        else:
            outputs = fn(**inputs)
        latency = time.time() - t0

        results.append(StageResult(
            prompt=prompt_text,
            expected_answer=q.get("expected_answer", ""),
            category=q.get("category", "unknown"),
            difficulty=q.get("difficulty", "unknown"),
            stage=stage_name,
            inputs=inputs,
            outputs=outputs,
            latency_s=round(latency, 3),
        ))

    return results


# ---------------------------------------------------------------------------
# Eval: e2e mode
# ---------------------------------------------------------------------------

@dataclass
class E2EResult:
    prompt: str
    expected_answer: str
    category: str
    difficulty: str
    stage_traces: List[dict]
    final_output: Any
    final_match: bool
    total_latency_s: float


def eval_e2e(questions: List[Dict], max_q: int = 0) -> List[E2EResult]:
    items = questions[:max_q] if max_q else questions
    results = []

    for q in items:
        prompt_text = q.get("prompt", "")
        state = {"prompt": prompt_text}
        traces = []

        for stage_name, cfg in STAGES.items():
            fn = cfg["fn"]
            if fn is None:
                traces.append({"stage": stage_name, "status": "not_implemented"})
                continue

            t0 = time.time()
            try:
                inputs = {}
                for k in cfg["inputs"]:
                    inputs[k] = state.get(k, prompt_text)
                outputs = fn(**inputs)
                state.update(outputs)
                traces.append({
                    "stage": stage_name,
                    "status": "ok",
                    "inputs_snapshot": {k: v for k, v in inputs.items()
                                        if isinstance(v, (str, int, float, bool))},
                    "outputs_snapshot": outputs,
                    "latency_s": round(time.time() - t0, 3),
                })
            except Exception as e:
                traces.append({
                    "stage": stage_name,
                    "status": f"error: {e}",
                    "latency_s": round(time.time() - t0, 3),
                })

        final_match = fuzzy_match(str(state.get("solver", "")), q.get("expected_answer", ""))
        results.append(E2EResult(
            prompt=prompt_text,
            expected_answer=q.get("expected_answer", ""),
            category=q.get("category", "unknown"),
            difficulty=q.get("difficulty", "unknown"),
            stage_traces=traces,
            final_output=state,
            final_match=final_match["matched"],
            total_latency_s=sum(t.get("latency_s", 0) for t in traces),
        ))

    return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save(run_id: str, detail: Any, summary: dict, log: Optional[list] = None):
    RESULTS_DIR.mkdir(exist_ok=True)

    if isinstance(detail, list) and detail and hasattr(detail[0], "__dataclass_fields__"):
        detail = [asdict(r) for r in detail]

    with open(RESULTS_DIR / f"{run_id}_detail.json", "w") as f:
        json.dump(detail, f, indent=2, default=str)
    with open(RESULTS_DIR / f"{run_id}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    if log:
        with open(RESULTS_DIR / f"{run_id}_log.json", "w") as f:
            json.dump(log, f, indent=2, default=str)

    print(f"  → {RESULTS_DIR}/{run_id}_detail.json")
    print(f"  → {RESULTS_DIR}/{run_id}_summary.json")


def load_all() -> List[Dict]:
    out = []
    for f in sorted(RESULTS_DIR.glob("*_summary.json")):
        with open(f) as fh:
            out.append(json.load(fh))
    return out


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_compare(summaries: List[Dict]):
    if not summaries:
        print("No results in eval_results/")
        return

    print("\n" + "=" * 110)
    print("MODEL COMPARISON")
    print("=" * 110)
    h = f"{'Label':<20} {'Mode':<12} {'Acc%':<7} {'Simple':<8} {'Medium':<8} {'Hard':<8} {'InTok':<7} {'OutTok':<7} {'Lat(s)':<7}"
    print(h)
    print("-" * 110)
    for s in summaries:
        if "accuracy_pct" not in s:
            continue
        d = s.get("by_difficulty", {})
        row = (
            f"{s.get('label', '?'):<20} "
            f"{s.get('mode', '?'):<12} "
            f"{s.get('accuracy_pct', 0):<7} "
            f"{d.get('simple', {}).get('accuracy_pct', '-'):<8} "
            f"{d.get('medium', {}).get('accuracy_pct', '-'):<8} "
            f"{d.get('hard', {}).get('accuracy_pct', '-'):<8} "
            f"{s.get('avg_input_tokens', 0):<7} "
            f"{s.get('avg_output_tokens', 0):<7} "
            f"{s.get('avg_latency_s', 0):<7}"
        )
        print(row)
    print("=" * 110)

    # Category breakdown
    models = [s for s in summaries if "by_category" in s]
    if not models:
        return
    cats = sorted({c for m in models for c in m["by_category"]})
    print("\n" + "=" * 110)
    print("PER-CATEGORY ACCURACY")
    print("=" * 110)
    h = f"{'Label':<20} {'Mode':<12}"
    for c in cats:
        h += f" {c[:12]:<14}"
    print(h)
    print("-" * 110)
    for m in models:
        row = f"{m.get('label', '?'):<20} {m.get('mode', '?'):<12}"
        for c in cats:
            row += f" {str(m['by_category'].get(c, {}).get('accuracy_pct', '-')):<14}"
        print(row)
    print("=" * 110)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AMD ACT II Eval Pipeline")
    parser.add_argument("--mode", choices=["model", "stage", "e2e", "compare"],
                        default="compare", help="Eval mode")
    parser.add_argument("--model", type=str, default="",
                        help="HuggingFace model ID (e.g. Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--label", type=str, default="",
                        help="Short label for results (defaults to model ID)")
    parser.add_argument("--prompt-mode", choices=["zero-shot", "structured"],
                        default="zero-shot", help="Prompting mode (model mode only)")
    parser.add_argument("--stage", type=str, default="",
                        help="Stage name for stage eval")
    parser.add_argument("--qs", type=str, default="",
                        help="Question set JSON file")
    parser.add_argument("--max-q", type=int, default=0,
                        help="Max questions (0 = all)")
    parser.add_argument("--out", type=str, default="",
                        help="Output directory (default: eval_results/)")
    args = parser.parse_args()

    if args.out:
        global RESULTS_DIR
        RESULTS_DIR = Path(args.out)
        RESULTS_DIR.mkdir(exist_ok=True)

    # --- compare ---
    if args.mode == "compare":
        print_compare(load_all())
        return

    # --- load questions ---
    if not args.qs:
        candidates = ["eval_60_balanced.json", "eval_all_300.json",
                      "eval_simple_100.json", "eval_hard_100.json"]
        for c in candidates:
            p = SHARED_DIR / c
            if p.exists():
                args.qs = str(p)
                break
    questions = load_questions(args.qs)
    print(f"Loaded {len(questions)} questions from {args.qs}")

    # --- model ---
    if args.mode == "model":
        if not args.model:
            print("ERROR: --model <hf_id> required (e.g. Qwen/Qwen2.5-7B-Instruct)")
            return
        label = args.label or args.model.replace("/", "_")
        results = eval_model(args.model, label, questions, args.prompt_mode, args.max_q)
        summary = summarize_model(results, label, args.prompt_mode)
        run_id = f"model_{label}_{args.prompt_mode}"
        save(run_id, results, summary)
        print_compare([summary])

    # --- stage ---
    elif args.mode == "stage":
        if not args.stage:
            print("ERROR: --stage <name> required. Available stages:")
            for s in STAGES:
                print(f"  {s}")
            return
        results = eval_stage(args.stage, questions, args.max_q)
        implemented = sum(1 for r in results if "_error" not in r.outputs)
        total = len(results)
        run_id = f"stage_{args.stage}"
        summary = {
            "stage": args.stage,
            "total": total,
            "implemented": implemented,
            "not_implemented": total - implemented,
        }
        save(run_id, results, summary)

    # --- e2e ---
    elif args.mode == "e2e":
        results = eval_e2e(questions, args.max_q)
        matched = sum(1 for r in results if r.final_match)
        total = len(results)
        summary = {
            "mode": "e2e",
            "total": total,
            "matched": matched,
            "accuracy_pct": round(matched / total * 100, 1) if total else 0,
        }
        if results:
            summary["avg_total_latency_s"] = round(
                sum(r.total_latency_s for r in results) / total, 3
            )
        log = [asdict(r) for r in results]
        save("e2e_full_pipeline", results, summary, log)


if __name__ == "__main__":
    main()
