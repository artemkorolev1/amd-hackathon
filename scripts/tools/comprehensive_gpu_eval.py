#!/usr/bin/env python3
"""
Comprehensive per-category GPU eval runner - SPECIFIC DATASETS ONLY.
Loads only the designated eval datasets (not training bundles).
"""
import argparse
import gc
import json
import os
import re
import sys
import time
import logging
from pathlib import Path
from collections import defaultdict, OrderedDict

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("comprehensive_eval")

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from scripts.grade_answer import fuzzy_match, grade_answer, extract_numbers, summarization_grade

# Per-category inference parameters (overrides for specific categories)
# All other categories use the defaults from run_eval() parameters.
CATEGORY_INFER_PARAMS = {
    "summarization": {"temperature": 0.2, "repeat_penalty": 1.1, "top_p": 0.95},
    "sentiment":     {"temperature": 0.0, "top_p": 0.9, "top_k": 20},
}

# The specific eval datasets to load (relative to eval dir)
TARGET_DATASETS = [
    "primary/eval_hard_218.json",     # 218 hard questions
    "training-v3.json",               # 152 items (19 per category)
    "math_combined_80.json",          # 94 math items
    "sentiment_combined_25.json",     # 25 sentiment items
    "summarization_combined_25.json", # 25 summarization items
    "factual_combined_80.json",       # 58 factual items
]

# Category-specific prompts
CATEGORY_PROMPTS = {
    "code_debug": "Fix the bug in the following code. Only return the corrected code, nothing else.",
    "code_gen": "Write a Python function for the following task. Only return the function definition, nothing else.",
    "factual": "Answer the following question accurately and concisely.",
    "general": "Answer the following question accurately and concisely.",
    "logic": "Solve the following logic puzzle step by step. Provide your final answer clearly.",
    "math": "Solve the following math problem step by step. Provide your final answer as a number.",
    "ner": "Extract all named entities (PERSON, ORG, LOC, DATE, etc.) from the text. List each entity on a new line with its type.",
    "sentiment": "What is the sentiment of the following text? Answer with exactly one word: positive, negative, or neutral.",
    "summarization": "Summarize the following text in 1-2 sentences. Be concise.",
}

SYSTEM_PROMPTS = {
    "code_debug": "You are an expert Python programmer. Fix bugs in code. Return only the corrected code, no explanations.",
    "code_gen": "You are an expert Python programmer. Write clean, correct Python functions. Return only the function definition.",
    "factual": "You are a helpful assistant that answers factual questions accurately and concisely.",
    "general": "You are a helpful assistant that answers questions accurately and concisely.",
    "logic": "You are a logic puzzle solver. Think step by step and provide the final answer clearly.",
    "math": "You are a math problem solver. Solve step by step and provide the final answer clearly.",
    "ner": "You are an NER (Named Entity Recognition) system. Extract all named entities with their types.",
    "sentiment": "You are a sentiment classifier. Respond with exactly one word: positive, negative, or neutral.",
    "summarization": "You are a summarization system. Summarize concisely in 1-2 sentences.",
}


def load_target_datasets(eval_dir: str) -> list[dict]:
    """Load only the target datasets, deduplicating by prompt."""
    eval_path = Path(eval_dir).resolve()
    seen_prompts = OrderedDict()

    for rel_path in TARGET_DATASETS:
        fpath = eval_path / rel_path
        if not fpath.exists():
            logger.warning("Dataset not found: %s", fpath)
            continue
        try:
            with open(fpath) as f:
                raw = json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Error loading %s: %s", fpath, e)
            continue

        entries = raw if isinstance(raw, list) else raw.get("tasks", raw.get("questions", [raw]))
        src_name = os.path.basename(rel_path)

        for item in entries:
            if not isinstance(item, dict):
                continue
            prompt = item.get("prompt", item.get("question", item.get("text", "")))
            if not prompt:
                continue
            if prompt not in seen_prompts:
                norm_item = dict(item)
                if "task_id" not in norm_item:
                    norm_item["task_id"] = f"q-{abs(hash(prompt)) % 10**8:08x}"
                if "category" not in norm_item:
                    norm_item["category"] = "general"
                if "expected_answer" not in norm_item:
                    gold = norm_item.get("gold", {})
                    if isinstance(gold, dict):
                        norm_item["expected_answer"] = str(gold.get("answer", gold.get("entities", "")))
                    else:
                        norm_item["expected_answer"] = norm_item.get("answer", norm_item.get("expected", ""))
                if "difficulty" not in norm_item:
                    norm_item["difficulty"] = ""
                if "source" not in norm_item:
                    norm_item["source"] = src_name

                seen_prompts[prompt] = norm_item

    items = list(seen_prompts.values())
    logger.info("Loaded %d unique items from %d datasets", len(items), len(TARGET_DATASETS))

    cats = defaultdict(int)
    for item in items:
        cats[item.get("category", "unknown")] += 1
    logger.info("Category breakdown:")
    for c, n in sorted(cats.items()):
        logger.info("  %s: %d", c, n)

    return items


def run_eval(
    model_path: str,
    items: list[dict],
    n_gpu_layers: int = -1,
    n_ctx: int = 4096,
    n_threads: int = 4,
    max_tokens: int = 512,
) -> list[dict]:
    """Run evaluation on all items using the given GGUF model on GPU."""
    logger.info("Loading model: %s", model_path)
    logger.info("GPU layers: %d, Context: %d, Threads: %d", n_gpu_layers, n_ctx, n_threads)

    from llama_cpp import Llama

    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        flash_attn=True,
        verbose=False,
    )
    logger.info("Model loaded successfully")

    results = []
    total = len(items)
    start_time = time.time()
    cat_times = defaultdict(list)

    for idx, item in enumerate(items):
        prompt = item.get("prompt", "")
        category = item.get("category", "general")
        tid = item.get("task_id", f"q-{idx:04d}")
        difficulty = item.get("difficulty", "")
        expected = item.get("expected_answer", "")

        if len(prompt) > 8000:
            logger.info("[%d/%d] SKIP (too long: %d chars) %s", idx + 1, total, len(prompt), tid)
            results.append({
                "task_id": tid, "category": category, "difficulty": difficulty,
                "prompt": prompt[:200], "expected": expected,
                "answer": "", "timing_ms": 0, "correct": False,
                "reason": "Skipped (prompt too long)", "source": item.get("source", ""),
            })
            continue

        sys_prompt = SYSTEM_PROMPTS.get(category, "You are a helpful assistant.")
        user_prompt = CATEGORY_PROMPTS.get(category, "")
        full_prompt = f"{user_prompt}\n\n{prompt}" if user_prompt else prompt

        # Build inference params with per-category overrides
        infer_kwargs = {
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stop": None,
        }
        cat_params = CATEGORY_INFER_PARAMS.get(category, {})
        infer_kwargs.update(cat_params)

        t0 = time.time()
        try:
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": full_prompt},
                ],
                **infer_kwargs,
            )
            answer = response["choices"][0]["message"]["content"] or ""
        except Exception as e:
            logger.warning("Error on %s: %s", tid, e)
            answer = ""

        elapsed_ms = (time.time() - t0) * 1000
        cat_times[category].append(elapsed_ms)
        answer = answer.strip()

        if not answer:
            correct, reason = False, "Empty answer"
        else:
            # Use summarization-specific grading for summarization category
            if category == "summarization":
                correct = summarization_grade(answer, expected)
                reason = "Passed" if correct else f"expected: {expected[:120]}, got: {answer[:120]}"
            else:
                correct, reason = grade_answer(answer, expected)
            accept_list = item.get("accept", [])
            if not correct and accept_list:
                for alias in accept_list:
                    alias_str = str(alias) if not isinstance(alias, str) else alias
                    if alias_str:
                        p, r = grade_answer(answer, alias_str)
                        if p:
                            correct = True
                            reason = "Passed (accepted alias)"
                            break

        results.append({
            "task_id": tid, "category": category, "difficulty": difficulty,
            "prompt": prompt[:200], "expected": expected,
            "answer": answer[:500], "timing_ms": round(elapsed_ms, 1),
            "correct": correct, "reason": reason, "source": item.get("source", ""),
        })

        if (idx + 1) % 25 == 0 or idx == 0:
            logger.info(
                "[%d/%d] %s | %s | acc=%.1f%% | time=%.1fs",
                idx + 1, total, tid, category,
                sum(1 for r in results if r["correct"]) / max(len(results), 1) * 100,
                time.time() - start_time,
            )

    total_elapsed = time.time() - start_time
    llm.close()
    del llm
    gc.collect()

    logger.info("Completed %d items in %.1fs (%.1f ms/item)",
                len(results), total_elapsed,
                total_elapsed / max(len(results), 1) * 1000)

    return results


def build_report(results: list[dict], model_name: str, model_path: str) -> dict:
    """Build a comprehensive report."""
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total > 0 else 0.0

    by_category = defaultdict(lambda: {"total": 0, "correct": 0, "timings": []})
    for r in results:
        cat = r.get("category", "unknown")
        by_category[cat]["total"] += 1
        if r["correct"]:
            by_category[cat]["correct"] += 1
        by_category[cat]["timings"].append(r["timing_ms"])
    for cat in by_category:
        t = by_category[cat]["total"]
        c = by_category[cat]["correct"]
        by_category[cat]["accuracy"] = c / t if t > 0 else 0.0
        timings = by_category[cat]["timings"]
        by_category[cat]["avg_time_ms"] = sum(timings) / len(timings) if timings else 0.0

    by_difficulty = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        diff = r.get("difficulty") or "unspecified"
        by_difficulty[diff]["total"] += 1
        if r["correct"]:
            by_difficulty[diff]["correct"] += 1
    for diff in by_difficulty:
        t = by_difficulty[diff]["total"]
        c = by_difficulty[diff]["correct"]
        by_difficulty[diff]["accuracy"] = c / t if t > 0 else 0.0

    by_source = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        src = r.get("source", "unknown")
        by_source[src]["total"] += 1
        if r["correct"]:
            by_source[src]["correct"] += 1
    for src in by_source:
        t = by_source[src]["total"]
        c = by_source[src]["correct"]
        by_source[src]["accuracy"] = c / t if t > 0 else 0.0

    failures = [r for r in results if not r["correct"]]
    timings = [r["timing_ms"] for r in results if r["timing_ms"] > 0]
    avg_time = sum(timings) / len(timings) if timings else 0
    max_time = max(timings) if timings else 0
    total_time = sum(timings)

    return {
        "model_name": model_name,
        "model_path": model_path,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "by_category": dict(by_category),
        "by_difficulty": dict(by_difficulty),
        "by_source": dict(by_source),
        "failures": failures,
        "timing": {
            "avg_ms": round(avg_time, 1),
            "max_ms": round(max_time, 1),
            "total_sec": round(total_time / 1000, 1),
        },
    }


def write_markdown_report(report: dict, output_path: str):
    """Write comprehensive markdown report."""
    lines = []
    lines.append("# Comprehensive Per-Category GPU Eval Report")
    lines.append("")
    lines.append(f"- **Model**: {report['model_name']}")
    lines.append(f"- **Model path**: {report['model_path']}")
    lines.append(f"- **Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **GPU**: RTX A4000 8GB, N_GPU_LAYERS=-1")
    lines.append("")

    lines.append("## Overall Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Total Questions | {report['total']} |")
    lines.append(f"| Correct | {report['correct']} |")
    lines.append(f"| **Accuracy** | **{report['accuracy']*100:.2f}%** |")
    lines.append(f"| 84.2% Gate | {'✅ PASS' if report['accuracy'] >= 0.842 else '❌ FAIL'} |")
    lines.append(f"| Avg Time/Question | {report['timing']['avg_ms']:.1f} ms |")
    lines.append(f"| Max Time/Question | {report['timing']['max_ms']:.1f} ms |")
    lines.append(f"| Total Time | {report['timing']['total_sec']:.1f} s |")
    lines.append("")

    lines.append("## Per-Category Breakdown")
    lines.append("")
    lines.append("| Category | Total | Correct | Accuracy | Avg Time (ms) |")
    lines.append("|---|---|---|---|---|")
    for cat in sorted(report["by_category"].keys()):
        s = report["by_category"][cat]
        acc_str = f"{s['accuracy']*100:.1f}%"
        pass_icon = "✅" if s["accuracy"] >= 0.842 else "❌"
        lines.append(f"| {pass_icon} {cat} | {s['total']} | {s['correct']} | {acc_str} | {s['avg_time_ms']:.0f} |")
    lines.append("")

    lines.append("## Per-Source Breakdown")
    lines.append("")
    lines.append("| Source | Total | Correct | Accuracy |")
    lines.append("|---|---|---|---|")
    for src in sorted(report["by_source"].keys()):
        s = report["by_source"][src]
        lines.append(f"| {src} | {s['total']} | {s['correct']} | {s['accuracy']*100:.1f}% |")
    lines.append("")

    lines.append("## Per-Difficulty Breakdown")
    lines.append("")
    lines.append("| Difficulty | Total | Correct | Accuracy |")
    lines.append("|---|---|---|---|")
    for diff in sorted(report["by_difficulty"].keys()):
        s = report["by_difficulty"][diff]
        lines.append(f"| {diff} | {s['total']} | {s['correct']} | {s['accuracy']*100:.1f}% |")
    lines.append("")

    failures = report["failures"]
    lines.append(f"## Failures ({len(failures)})")
    lines.append("")
    if failures:
        lines.append("| # | Task ID | Category | Expected | Got (snippet) | Reason |")
        lines.append("|---|---|---|---|---|---|")
        for i, f in enumerate(failures[:100]):
            exp_snippet = f["expected"][:80] if f["expected"] else "(empty)"
            ans_snippet = f["answer"][:80] if f["answer"] else "(empty)"
            lines.append(f"| {i+1} | {f['task_id']} | {f['category']} | {exp_snippet} | {ans_snippet} | {f['reason'][:60]} |")
        if len(failures) > 100:
            lines.append(f"| ... | ({len(failures) - 100} more) | | | | |")
    lines.append("")

    for cat in sorted(report["by_category"].keys()):
        cat_results = [r for r in report["failures"] if r["category"] == cat]
        if cat_results:
            lines.append(f"### {cat} Failures ({len(cat_results)})")
            lines.append("")
            lines.append("| Task ID | Expected | Got | Reason |")
            lines.append("|---|---|---|---|")
            for f in cat_results[:30]:
                exp_snippet = f["expected"][:60] if f["expected"] else "(empty)"
                ans_snippet = f["answer"][:60] if f["answer"] else "(empty)"
                lines.append(f"| {f['task_id']} | {exp_snippet} | {ans_snippet} | {f['reason'][:50]} |")
            lines.append("")

    report_str = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report_str)

    logger.info("Report written to %s", output_path)
    return output_path


def print_summary(report: dict):
    """Print concise summary to stdout."""
    print()
    print("=" * 80)
    print(f"  COMPREHENSIVE GPU EVAL SUMMARY")
    print(f"  Model: {report['model_name']}")
    print(f"  Total: {report['total']}  Correct: {report['correct']}  Accuracy: {report['accuracy']*100:.2f}%")
    print(f"  Gate (84.2%): {'PASSED' if report['accuracy'] >= 0.842 else 'FAILED'}")
    print(f"  Avg time: {report['timing']['avg_ms']:.1f}ms  Total: {report['timing']['total_sec']:.1f}s")
    print("=" * 80)
    print()
    print("  Per-Category:")
    for cat in sorted(report["by_category"].keys()):
        s = report["by_category"][cat]
        print(f"    {cat:20s}  {s['correct']:3d}/{s['total']:<3d}  {s['accuracy']*100:5.1f}%  ({s['avg_time_ms']:.0f}ms)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Comprehensive GPU eval runner")
    parser.add_argument("--model", "-m", required=True, help="Path to GGUF model")
    parser.add_argument("--model-name", "-n", default=None, help="Model display name")
    parser.add_argument("--eval-dir", default=str(_HERE / "data/eval"), help="Eval data directory")
    parser.add_argument("--output-dir", default=str(_HERE / "eval_results"), help="Output directory")
    parser.add_argument("--gpu-layers", type=int, default=-1, help="GPU layers (-1 = all)")
    parser.add_argument("--ctx", type=int, default=4096, help="Context size")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads")
    parser.add_argument("--max-tokens", type=int, default=768, help="Max tokens per answer")
    args = parser.parse_args()

    model_name = args.model_name or os.path.basename(args.model).replace(".gguf", "")

    items = load_target_datasets(args.eval_dir)
    logger.info("Running evaluation on %d items with model %s", len(items), model_name)

    os.makedirs(args.output_dir, exist_ok=True)
    results = run_eval(
        model_path=args.model,
        items=items,
        n_gpu_layers=args.gpu_layers,
        n_ctx=args.ctx,
        n_threads=args.threads,
        max_tokens=args.max_tokens,
    )

    report = build_report(results, model_name, args.model)

    # Save detailed JSON
    ts = time.strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(args.output_dir, f"comprehensive_eval_{model_name}_{ts}.json")
    with open(results_path, "w") as f:
        json.dump({
            "model": model_name,
            "model_path": args.model,
            "total": report["total"],
            "correct": report["correct"],
            "accuracy": report["accuracy"],
            "by_category": report["by_category"],
            "by_source": report["by_source"],
            "results": results,
        }, f, indent=2, default=str)
    logger.info("Results saved to %s", results_path)

    # Write markdown report to the requested location
    dated_report = _HERE / f"per_category_gpu_evals_{ts}.md"
    write_markdown_report(report, str(dated_report))

    print_summary(report)
    print(f"  Report: {dated_report}")
    print(f"  JSON:   {results_path}")
    print("=" * 80)

    return 0 if report["accuracy"] >= 0.842 else 1


if __name__ == "__main__":
    sys.exit(main())
