#!/usr/bin/env python3
"""
Compare all 6 ensemble candidate GGUF models on the same question set.

Usage:
    python compare_models.py [--qs eval_mini_10.json] [--max-q 40]

Outputs a comparison table with accuracy and latency per model.
"""
import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
sys.path.insert(0, str(HERE))

# Ensure complexity model is found
os.environ.setdefault("COMPLEXITY_MODEL_DIR", str(HERE / "shared" / "classifiers" / "best_complexity_model"))

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

# ── Pipeline imports (deterministic solvers, categories, prompts) ──────
from agent.pre_filter import stage0
from agent.category_filter import classify_with_detail as _stage2_detail
from agent.complexity import score as mlm_complexity
from agent.solvers.deterministic import (
    solve_arithmetic, solve_logic, solve_sentiment, solve_ner,
    solve_factual_qa, solve_code_debugging,
)
from agent.dynamic_prompts import build_system_prompt, build_merged_prompt, get_max_tokens, get_stop_sequences

# ── Model registry ─────────────────────────────────────────────────────
MODELS = {
    "Qwen2.5-1.5B-Instruct": os.path.expanduser("~/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    "Qwen2.5-Coder-1.5B":    os.path.expanduser("~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
    "Qwen2.5-Math-1.5B":     os.path.expanduser("~/models/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf"),
    "Gemma-3-1B":            os.path.expanduser("~/models/gemma-3-1b-it-Q4_K_M.gguf"),
    "SmolLM2-1.7B":          os.path.expanduser("~/models/smollm2-1.7b-instruct-q4_k_m.gguf"),
    "Llama-3.2-1B":          os.path.expanduser("~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
}

N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "-1"))
N_CTX = 4096

# ── Reasoning-headroom detection ───────────────────────────────────────
_REASONING_MODELS = {"nemotron", "deepseek", "qwq", "qwen3"}

def _has_reasoning_headroom(model_label: str) -> bool:
    return any(m in model_label.lower() for m in _REASONING_MODELS)

# ── Category overrides (same as harness.py) ─────────────────────────────
_DOC_HEADER_RE = re.compile(
    r'^(HEADLINE:|LEGAL BRIEF|STATEMENT BY|ARTICLE:|TRANSCRIPT:|MEMO:|PRESS RELEASE:)',
    re.IGNORECASE | re.MULTILINE,
)
_HARD_MATH_RE = re.compile(
    r"\b(law of sines|law of cosines|geometric series|cofactor|determinant"
    r"|inclusion.exclusion|bayes(?:ian)?|conditional probability"
    r"|permutations?|combinations?|integral|derivative|matrix|eigenvalu"
    r"|logarithm|log base|\\bmod\\b|modular arithmetic|chinese remainder"
    r"|ratio|proportion|in the ratio|lcm|gcd|least common multiple"
    r"|greatest common divisor|rate.*time|time.*rate|work rate"
    r"|how many (?:different |distinct |possible )?ways"
    r"|distinct.*\\bdigits?|distinct.*\\bnumbers?"
    r"|nCr|nPr)\b",
    re.IGNORECASE,
)

# Reasoning-safe system prompts (for models with <think> blocks)
_REASONING_PROMPTS = {
    "sentiment": (
        "Classify the sentiment. Output EXACTLY one word: positive, negative, neutral, or mixed."
    ),
    "ner": (
        "Extract all named entities grouped by type: "
        "PERSON: ...; ORGANIZATION: ...; LOCATION: ..."
    ),
    "math": (
        "Solve the math problem. After thinking, output ONLY: Answer: <value>."
    ),
    "logic": (
        "Solve the logic puzzle. Output ONLY the final answer — NO preamble."
    ),
    "factual": (
        "Answer the question directly and concisely. Under 100 words."
    ),
    "summarization": (
        "Summarize the text. Max 2 sentences."
    ),
    "code_gen": (
        "Write the requested function inside ```python...```. No explanation."
    ),
    "code_debug": (
        "Fix the bug. Output the corrected function inside ```python...```."
    ),
}

# ── Model loader ───────────────────────────────────────────────────────
def load_model(path: str):
    from llama_cpp import Llama
    print(f"  Loading {Path(path).name}  (n_gpu_layers={N_GPU_LAYERS})", file=sys.stderr)
    t0 = time.time()
    llm = Llama(
        model_path=path,
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        n_threads=4 if N_GPU_LAYERS == 0 else 2,
        flash_attn=True,
        verbose=False,
    )
    elapsed = time.time() - t0
    print(f"  Loaded in {elapsed:.1f}s", file=sys.stderr)
    return llm

def infer(llm, messages, max_tok, stop_seq, timeout=60):
    """Single inference call with timeout via thread pool."""
    import concurrent.futures
    def _call():
        return llm.create_chat_completion(
            messages=messages, max_tokens=max_tok,
            temperature=0.0, stop=stop_seq,
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        try:
            fut = ex.submit(_call)
            resp = fut.result(timeout=timeout)
            raw = resp["choices"][0]["message"]["content"] or ""
            # Strip <think> blocks
            raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
            if "</think>" in raw:
                raw = raw.split("</think>", 1)[1].strip()
            return raw
        except concurrent.futures.TimeoutError:
            print(f"  Inference timed out ({timeout}s)", file=sys.stderr)
            return ""
        except Exception as e:
            print(f"  Inference error: {e}", file=sys.stderr)
            return ""

# ── Fuzzy matching (from evaluate.py) ──────────────────────────────────
def tokenize(s):
    return set(tok for tok in re.split(r"[^a-zA-Z0-9.]+", s.lower()) if tok)

def extract_numbers(s):
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", s)]

def fuzzy_match(answer, expected):
    a_clean = answer.strip().lower()
    e_clean = expected.strip().lower()
    if a_clean == e_clean:
        return True
    if len(e_clean) >= 3 and e_clean in a_clean:
        return True
    if len(a_clean) >= 3 and len(e_clean) <= 20 and a_clean in e_clean:
        return True
    a_tokens = tokenize(answer)
    e_tokens = tokenize(expected)
    if e_tokens and len(a_tokens & e_tokens) / len(e_tokens) >= 0.7:
        return True
    a_nums = extract_numbers(answer)
    e_nums = extract_numbers(expected)
    if a_nums and e_nums and a_nums == e_nums:
        return True
    return False

# ── Per-model evaluation ───────────────────────────────────────────────
def evaluate_model(llm, label, questions, max_q=0):
    """Run all questions through the pipeline using the given model."""
    items = questions[:max_q] if max_q else questions
    results = []

    for i, q in enumerate(items):
        prompt = q.get("prompt", q.get("question", ""))
        expected = q.get("expected_answer", "")
        cat_name = q.get("category", "unknown")

        t0 = time.time()

        # ── 1. Stage 0 bypass ──
        s0 = stage0(prompt)
        if s0.action == "bypass" and s0.direct_answer:
            results.append({
                "prompt": prompt, "expected": expected, "category": cat_name,
                "answer": s0.direct_answer, "match": fuzzy_match(s0.direct_answer, expected),
                "latency": time.time() - t0, "method": "stage0_bypass",
            })
            continue

        # ── 2. Category classification ──
        detail = _stage2_detail(prompt)
        category = detail["category"]
        score_delta = detail["score_delta"]
        raw_scores = detail["raw_scores"]

        # Category overrides (trimmed — same logic as harness.py)
        if category not in ("summarization",) and _DOC_HEADER_RE.search(prompt):
            category = "summarization"
        if category != "summarization":
            if re.search(r'\bsummariz[ei]', prompt.lower()):
                category = "summarization"
        if category == "math" and len(prompt) > 600:
            if not re.search(r'[=×÷]|\\frac', prompt):
                category = "summarization"
        if re.search(r'\b(extract|identify)\b.{0,40}\b(named entity|entities)\b', prompt.lower()):
            category = "ner"

        # ── 3. Complexity ──
        complexity = mlm_complexity(prompt)

        # ── 4. System prompt ──
        reasoning_headroom = _has_reasoning_headroom(label)
        if reasoning_headroom:
            sys_prompt = _REASONING_PROMPTS.get(category, _REASONING_PROMPTS["factual"])
        else:
            secondary = ""
            sorted_cats = sorted(raw_scores.items(), key=lambda x: -x[1])
            if len(sorted_cats) >= 2:
                scat = sorted_cats[0][0] if sorted_cats[0][0] != category else sorted_cats[1][0]
                if score_delta < 0.5 and scat:
                    secondary = scat
            if secondary:
                sys_prompt = build_merged_prompt(category, secondary, complexity)
            else:
                sys_prompt = build_system_prompt(category, complexity)

        max_tok = get_max_tokens(category, complexity)
        stop_seq = get_stop_sequences(category)

        # ── 5. Deterministic solvers (try before LLM) ──
        det_cat_map = {"math": "math_arithmetic", "sentiment": "sentiment",
                       "factual": "other_complex", "code_debug": "code_debugging"}
        answer = None
        if category in det_cat_map:
            solver_cat = det_cat_map[category]
            for solver_fn in (solve_arithmetic, solve_logic, solve_sentiment, solve_ner,
                              solve_factual_qa, solve_code_debugging):
                try:
                    ans = solver_fn(prompt, solver_cat)
                    if ans:
                        answer = ans
                        break
                except Exception:
                    pass

        if not answer:
            messages = [{"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt}]
            answer = infer(llm, messages, max_tok, stop_seq)
            # Retry without system prompt
            if not answer:
                answer = infer(llm, [{"role": "user", "content": prompt}], max_tok, stop_seq)

        latency = time.time() - t0
        results.append({
            "prompt": prompt, "expected": expected, "category": cat_name,
            "answer": answer or "", "match": fuzzy_match(answer or "", expected),
            "latency": latency, "method": "local_llm" if answer else "empty",
        })

        if (i + 1) % 5 == 0:
            print(f"    [{i+1}/{len(items)}] {label}", file=sys.stderr)

    return results


def summarize(results, label):
    total = len(results)
    matched = sum(1 for r in results if r["match"])
    avg_latency = sum(r["latency"] for r in results) / max(total, 1)
    by_cat = {}
    for r in results:
        c = r["category"]
        by_cat.setdefault(c, {"total": 0, "matched": 0})
        by_cat[c]["total"] += 1
        if r["match"]:
            by_cat[c]["matched"] += 1

    summary = {
        "label": label,
        "total": total,
        "matched": matched,
        "accuracy": f"{matched/total*100:.1f}%" if total else "N/A",
        "avg_latency_s": round(avg_latency, 3),
        "by_category": {c: f"{v['matched']}/{v['total']}" for c, v in sorted(by_cat.items())},
    }
    return summary

# ── Main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Compare GGUF ensemble candidates")
    parser.add_argument("--qs", default="data/eval/primary/eval_mini_10.json",
                        help="Question set JSON file (default: data/eval/primary/eval_mini_10.json)")
    parser.add_argument("--max-q", type=int, default=0,
                        help="Max questions per model (0 = all)")
    parser.add_argument("--models", nargs="+",
                        default=list(MODELS.keys()),
                        help=f"Models to compare (default: all). Choices: {', '.join(MODELS.keys())}")
    parser.add_argument("--output", default="",
                        help="Write comparison JSON to this path")
    args = parser.parse_args()

    qs_path = Path(args.qs)
    if not qs_path.is_absolute():
        qs_path = HERE.parent / args.qs
    if not qs_path.exists():
        qs_path = HERE.parent / args.qs

    with open(qs_path) as f:
        data = json.load(f)
    questions = data if isinstance(data, list) else data.get("questions", data)

    print(f"\nQuestion set: {qs_path.name} ({len(questions)} questions)", file=sys.stderr)
    print(f"Models: {len(args.models)}", file=sys.stderr)
    print(f"\n{'='*90}", file=sys.stderr)

    all_summaries = []
    all_details = {}

    for label in args.models:
        if label not in MODELS:
            print(f"  Unknown model '{label}', skipping", file=sys.stderr)
            continue
        model_path = MODELS[label]
        if not os.path.exists(model_path):
            print(f"  Model not found: {model_path}, skipping", file=sys.stderr)
            continue

        print(f"\n── {label} ──", file=sys.stderr)
        llm = load_model(model_path)
        t0 = time.time()
        details = evaluate_model(llm, label, questions, args.max_q)
        elapsed = time.time() - t0
        summary = summarize(details, label)
        summary["total_time_s"] = round(elapsed, 1)
        all_summaries.append(summary)
        all_details[label] = details
        print(f"  Accuracy: {summary['accuracy']}  |  Avg: {summary['avg_latency_s']}s/q  |  Total: {elapsed:.0f}s", file=sys.stderr)

        # Free GPU memory
        del llm
        import gc
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    # ── Print comparison table ──
    print(f"\n{'='*90}")
    print(f"{'Model':<24} {'Accuracy':>10} {'Avg Lat':>10} {'Total':>10} {'By Category':<40}")
    print(f"{'-'*24} {'-'*10} {'-'*10} {'-'*10} {'-'*40}")
    for s in sorted(all_summaries, key=lambda x: -float(x["accuracy"].rstrip("%"))):
        cat_str = "; ".join(f"{k}:{v}" for k, v in s["by_category"].items())
        print(f"{s['label']:<24} {s['accuracy']:>10} {s['avg_latency_s']:>8.2f}s {s['total_time_s']:>8.0f}s  {cat_str}")

    print(f"\n{'='*90}")

    # ── Save results ──
    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = HERE / "eval_results" / out_path
        out_path.parent.mkdir(exist_ok=True)
        output = {"summaries": all_summaries, "details": all_details}
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"Saved to {out_path}", file=sys.stderr)

    # ── Print final recommendation ──
    print()
    best = max(all_summaries, key=lambda s: float(s["accuracy"].rstrip("%")))
    fastest = min(all_summaries, key=lambda s: s["avg_latency_s"])
    print(f"  Best accuracy:  {best['label']}  ({best['accuracy']})")
    print(f"  Fastest:        {fastest['label']}  ({fastest['avg_latency_s']:.2f}s/q)")


if __name__ == "__main__":
    main()
