#!/usr/bin/env python3
"""
GEPA Sentiment Optimization Runner
====================================
- 92-question comprehensive hard eval set (40 hard, 26 medium, 26 easy)
- 3 models: qwen2.5-1.5b, qwen2.5-coder-1.5b, gemma-3-1b
- 2 generations (gen 0 seed + gen 1 evolved)
- Compares default (top_p=1.0, top_k=40, min_p=0.0) vs optimized (top_p=0.9, top_k=20, min_p=0.05)
- Uses newly-enabled parameters (top_p, top_k, min_p, repeat_penalty) that were previously
  NOT passed to inference calls (bug fix applied to evaluation_agent.py)

Usage:
    python3 gepa_plans/run_sentiment_gepa.py
"""

import json
import os
import sys
import time
import gc
import re
import random
import copy
from pathlib import Path
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agent.cell import Cell, DecodingConfig
from agent.evaluation_agent import EvaluationAgent
from agent.mutation_agent import MutationAgent

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_PATHS = {
    "qwen2.5-1.5b": "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "qwen2.5-coder-1.5b": "/home/artem/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    "gemma-3-1b": "/home/artem/models/gemma-3-1b-it-Q4_K_M.gguf",
}

MODEL_KEYS = list(MODEL_PATHS.keys())

EVAL_SET_PATH = os.path.join(
    PROJECT_ROOT, "data", "eval", "generated", "sentiment_comprehensive_hard.json"
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "research")
OUTPUT_REPORT = os.path.join(RESULTS_DIR, "sentiment_gepa_results.md")
LOG_DIR = os.path.join(PROJECT_ROOT, "gepa_logs")

SEED = 42
GENERATIONS = 2  # gen 0 + gen 1 (keep it fast)

# ── Default vs Optimized params ──────────────────────────────────────────────

DEFAULT_PARAMS = {
    "temperature": 0.0,
    "max_tokens": 64,
    "top_p": 1.0,
    "top_k": 40,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "seed": None,
}

OPTIMIZED_PARAMS = {
    "temperature": 0.0,
    "max_tokens": 64,
    "top_p": 0.9,
    "top_k": 20,
    "min_p": 0.05,
    "repeat_penalty": 1.0,
    "seed": SEED,
}

# ── 8 seed prompt configurations (per model) ─────────────────────────────────

SEED_PROMPTS = [
    # (name, prompt, params_dict)
    (
        "classify_default",
        "Classify the sentiment. Output EXACTLY one word: positive, negative, neutral, or mixed.",
        DEFAULT_PARAMS,
    ),
    (
        "classify_optimized",
        "Classify the sentiment. Output EXACTLY one word: positive, negative, neutral, or mixed.",
        OPTIMIZED_PARAMS,
    ),
    (
        "analyze_tone_optimized",
        "Analyze the tone as positive, negative, neutral, or mixed.",
        OPTIMIZED_PARAMS,
    ),
    (
        "pick_one_optimized",
        "Pick one: positive/negative/neutral/mixed.",
        OPTIMIZED_PARAMS,
    ),
    (
        "watch_sarcasm_optimized",
        "Classify this text's sentiment. Watch for sarcasm and hedging.",
        OPTIMIZED_PARAMS,
    ),
    (
        "sentiment_default",
        "Sentiment: positive, negative, neutral, or mixed.",
        DEFAULT_PARAMS,
    ),
    (
        "empty_optimized",
        "",
        OPTIMIZED_PARAMS,
    ),
    (
        "tone_question_optimized",
        "Is the tone positive, negative, neutral, or mixed? Answer with one word.",
        OPTIMIZED_PARAMS,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Model Cache (loads one model at a time)
# ═══════════════════════════════════════════════════════════════════════════════

class SingleModelCache:
    """ModelCache wrapper that loads exactly one model, then unloads on clear."""

    def __init__(self):
        self._model = None
        self._loaded_key = None

    def get(self, model_key: str):
        if self._loaded_key != model_key:
            self.clear()
            from llama_cpp import Llama
            path = MODEL_PATHS.get(model_key)
            if not path:
                raise ValueError(f"Unknown model key: {model_key}")
            print(f"\n  ── Loading {model_key} ({os.path.basename(path)}) ──")
            t0 = time.time()
            self._model = Llama(
                model_path=path,
                n_ctx=2048,
                n_gpu_layers=-1,
                n_threads=4,
                verbose=False,
            )
            elapsed = time.time() - t0
            print(f"  Loaded in {elapsed:.1f}s")
            self._loaded_key = model_key
        return self._model

    def clear(self):
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded_key = None
            gc.collect()
            # Small delay for GPU memory to be reclaimed
            time.sleep(0.5)

    def get_loaded_key(self):
        return self._loaded_key


# ═══════════════════════════════════════════════════════════════════════════════
#  Eval set loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_eval_set(path=None):
    """Load the 92-question comprehensive hard eval set."""
    if path is None:
        path = EVAL_SET_PATH
    with open(path) as f:
        questions = json.load(f)
    # Tag each question with category="sentiment" for EvaluationAgent filtering
    for q in questions:
        q["category"] = "sentiment"
        if "task_id" not in q:
            q["task_id"] = ""
    print(f"\nLoaded {len(questions)} evaluation questions")
    # Count by difficulty
    diffs = defaultdict(int)
    for q in questions:
        diffs[q.get("difficulty", "unknown")] += 1
    for d, c in sorted(diffs.items()):
        print(f"  {d}: {c}")
    return questions


# ═══════════════════════════════════════════════════════════════════════════════
#  Seed population builder
# ═══════════════════════════════════════════════════════════════════════════════

def create_seed_population():
    """Create 24 cells (3 models × 8 prompt/param variants)."""
    population = []
    for model_key in MODEL_KEYS:
        for i, (name, prompt_text, params) in enumerate(SEED_PROMPTS):
            cell = Cell(
                task_id="T03",  # sentiment
                model_key=model_key,
                system_prompt=prompt_text,
                decoding=DecodingConfig(
                    temperature=params["temperature"],
                    max_tokens=params["max_tokens"],
                    top_p=params["top_p"],
                    top_k=params["top_k"],
                    min_p=params["min_p"],
                    repeat_penalty=params["repeat_penalty"],
                    seed=params.get("seed"),
                ),
                aggregation="single",
                name=f"seed_{model_key}_{name}",
                generation=0,
            )
            population.append(cell)
    print(f"\nCreated seed population: {len(population)} cells")
    for c in population:
        print(f"  {c.name:45s} | top_p={c.decoding.top_p} top_k={c.decoding.top_k} min_p={c.decoding.min_p} | "
              f"prompt={c.system_prompt[:50]!r}")
    return population


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluation (model-by-model, unloading between models)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_generation(population, questions, label="Generation"):
    """Evaluate all cells, loading one model at a time to respect 8GB VRAM."""
    print(f"\n{'='*60}")
    print(f"  EVALUATING {label}: {len(population)} cells")
    print(f"{'='*60}")

    # Group cells by model
    by_model = defaultdict(list)
    for idx, c in enumerate(population):
        by_model[c.model_key].append((idx, c))

    cache = SingleModelCache()
    eval_agent = EvaluationAgent(cache)

    for model_key, cell_list in by_model.items():
        print(f"\n  [{model_key}] Evaluating {len(cell_list)} cells...")
        # Build sub-population for this model
        sub_cells = [c for _, c in cell_list]
        eval_agent.evaluate(sub_cells, questions)
        # Report per-cell results
        for c in sub_cells:
            acc = c.metadata.get("accuracy", 0.0)
            correct = c.metadata.get("correct", 0)
            total = c.metadata.get("total", 0)
            fmt = c.metadata.get("format_compliance", 0.0)
            lat = c.metadata.get("avg_latency_ms", 0.0)
            print(f"    {c.name:45s} | acc={acc:.3f} ({correct}/{total}) "
                  f"fmt={fmt:.2f} lat={lat:.0f}ms")
        # Unload model
        cache.clear()

    # Update the original population list — cells were modified in-place via eval_agent
    print(f"\n  Evaluation complete.")
    return population  # cells modified in-place


def summarize_generation(population, gen_label, questions_list=None):
    """Print detailed summary of a generation's results."""
    print(f"\n{'─'*60}")
    print(f"  {gen_label} SUMMARY")
    print(f"{'─'*60}")

    # Best per model
    by_model = defaultdict(list)
    for c in population:
        by_model[c.model_key].append(c)

    best_overall = None
    best_overall_acc = -1

    for mk in MODEL_KEYS:
        cells = by_model.get(mk, [])
        if not cells:
            continue
        cells_sorted = sorted(cells, key=lambda c: c.metadata.get("accuracy", 0.0), reverse=True)
        best = cells_sorted[0]
        acc = best.metadata.get("accuracy", 0.0)
        correct = best.metadata.get("correct", 0)
        total = best.metadata.get("total", 0)
        print(f"\n  [{mk}] Best cell:")
        print(f"    Name:    {best.name}")
        print(f"    Prompt:  {best.system_prompt!r}")
        print(f"    Params:  temp={best.decoding.temperature}, top_p={best.decoding.top_p}, "
              f"top_k={best.decoding.top_k}, min_p={best.decoding.min_p}, "
              f"repeat_penalty={best.decoding.repeat_penalty}, seed={best.decoding.seed}")
        print(f"    Acc:     {acc:.3f} ({correct}/{total})")
        print(f"    Latency: {best.metadata.get('avg_latency_ms', 0):.0f}ms avg")

        # Top 3
        print(f"    Top 3:")
        for j, c in enumerate(cells_sorted[:3]):
            a = c.metadata.get("accuracy", 0.0)
            co = c.metadata.get("correct", 0)
            to = c.metadata.get("total", 0)
            print(f"      {j+1}. {c.name:45s} acc={a:.3f} ({co}/{to}) prompt={c.system_prompt[:40]!r}")

        if acc > best_overall_acc:
            best_overall_acc = acc
            best_overall = best

    # By difficulty
    if best_overall and "details" in best_overall.metadata and questions_list:
        print(f"\n  Per-difficulty breakdown (best overall {best_overall.name}):")
        details = best_overall.metadata["details"]
        by_diff = defaultdict(list)
        for q_idx, d in enumerate(details):
            # Find the question difficulty
            q = questions_list[q_idx] if q_idx < len(questions_list) else {}
            diff = q.get("difficulty", "unknown")
            by_diff[diff].append(d.get("correct", False))

        for diff, results in sorted(by_diff.items()):
            n_correct = sum(1 for r in results if r)
            n_total = len(results)
            print(f"    {diff:8s}: {n_correct}/{n_total} = {n_correct/max(n_total,1):.3f}")

    print(f"\n  {'─'*60}")
    return best_overall


# ═══════════════════════════════════════════════════════════════════════════════
#  Generation comparison
# ═══════════════════════════════════════════════════════════════════════════════

def compare_generations(gen0_pop, gen1_pop):
    """Compare gen 0 vs gen 1 results."""
    print(f"\n{'='*60}")
    print(f"  GENERATION COMPARISON")
    print(f"{'='*60}")

    report_sections = []
    comparison_data = {}

    for mk in MODEL_KEYS:
        gen0_cells = [c for c in gen0_pop if c.model_key == mk]
        gen1_cells = [c for c in gen1_pop if c.model_key == mk]

        if not gen0_cells or not gen1_cells:
            continue

        best0 = max(gen0_cells, key=lambda c: c.metadata.get("accuracy", 0.0))
        best1 = max(gen1_cells, key=lambda c: c.metadata.get("accuracy", 0.0))

        acc0 = best0.metadata.get("accuracy", 0.0)
        acc1 = best1.metadata.get("accuracy", 0.0)
        delta = acc1 - acc0

        print(f"\n  [{mk}]")
        print(f"    Gen 0 best: {best0.name:45s} acc={acc0:.3f}")
        print(f"    Gen 1 best: {best1.name:45s} acc={acc1:.3f}")
        print(f"    Δ: {delta:+.4f} ({delta*100:+.2f}pp)")

        comparison_data[mk] = {
            "best0": best0,
            "best1": best1,
            "acc0": acc0,
            "acc1": acc1,
            "delta": delta,
        }

    return comparison_data


# ═══════════════════════════════════════════════════════════════════════════════
#  Report generator
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(gen0_pop, gen1_pop, questions, comparison_data, run_log_dir):
    """Write structured markdown report to research/sentiment_gepa_results.md."""
    print(f"\n  Writing report to {OUTPUT_REPORT}")

    lines = []
    lines.append("# Sentiment GEPA Optimization Results\n")
    lines.append(f"**Run date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**Eval set:** 92 questions (40 hard, 26 medium, 26 easy)\n")
    lines.append(f"**Models:** {', '.join(MODEL_KEYS)}\n")
    lines.append(f"**Generations:** {GENERATIONS} (gen 0 → gen 1)\n")
    lines.append(f"**Log directory:** `{run_log_dir}`\n")
    lines.append("---\n")

    # ── 1. Best cell per model ──────────────────────────────────────────────
    lines.append("## 1. Best Cell Per Model\n")
    lines.append("| Model | Gen | Cell | Prompt | Params | Accuracy |")
    lines.append("|-------|-----|------|--------|--------|----------|")

    for mk in MODEL_KEYS:
        for gen_label, pop in [("0", gen0_pop), ("1", gen1_pop)]:
            cells = [c for c in pop if c.model_key == mk]
            if not cells:
                continue
            best = max(cells, key=lambda c: c.metadata.get("accuracy", 0.0))
            acc = best.metadata.get("accuracy", 0.0)
            correct = best.metadata.get("correct", 0)
            total = best.metadata.get("total", 0)
            params_str = (
                f"t={best.decoding.temperature}, "
                f"tp={best.decoding.top_p}, "
                f"tk={best.decoding.top_k}, "
                f"mp={best.decoding.min_p}"
            )
            prompt_short = best.system_prompt[:60] if best.system_prompt else "(empty)"
            lines.append(
                f"| {mk} | {gen_label} | {best.name} | {prompt_short} | {params_str} | "
                f"{acc:.3f} ({correct}/{total}) |"
            )

    lines.append("")

    # ── 2. Parameter Pareto Front ───────────────────────────────────────────
    lines.append("## 2. Parameter Pareto Front\n")
    lines.append("Which `(top_p, top_k, min_p)` combinations dominate across all cells.\n")

    # Collect all unique param combos with accuracies
    all_cells = gen0_pop + gen1_pop
    combo_map = defaultdict(list)
    for c in all_cells:
        key = (c.decoding.top_p, c.decoding.top_k, c.decoding.min_p, c.decoding.repeat_penalty)
        acc = c.metadata.get("accuracy", 0.0)
        combo_map[key].append(acc)

    # Sort by mean accuracy
    combo_stats = []
    for combo, accs in combo_map.items():
        mean_acc = sum(accs) / len(accs)
        max_acc = max(accs)
        combo_stats.append((combo, mean_acc, max_acc, len(accs)))

    combo_stats.sort(key=lambda x: x[2], reverse=True)  # sort by max acc

    lines.append("| top_p | top_k | min_p | repeat_penalty | Mean Acc | Max Acc | Cells |")
    lines.append("|-------|-------|-------|----------------|----------|---------|-------|")
    for combo, mean_acc, max_acc, count in combo_stats:
        lines.append(
            f"| {combo[0]} | {combo[1]} | {combo[2]} | {combo[3]} | "
            f"{mean_acc:.3f} | {max_acc:.3f} | {count} |"
        )

    # Pareto dominance: a combo dominates if no other combo has both higher max_acc
    # and a higher or equal count
    pareto_combos = []
    for i, (c1, m1, x1, n1) in enumerate(combo_stats):
        dominated = False
        for j, (c2, m2, x2, n2) in enumerate(combo_stats):
            if i == j:
                continue
            if x2 >= x1 and n2 >= n1 and (x2 > x1 or n2 > n1):
                dominated = True
                break
        if not dominated:
            pareto_combos.append((c1, m1, x1, n1))

    lines.append("")
    lines.append("**Pareto-optimal parameter combinations:**\n")
    lines.append("| top_p | top_k | min_p | repeat_penalty | Mean Acc | Max Acc |")
    lines.append("|-------|-------|-------|----------------|----------|---------|")
    for combo, mean_acc, max_acc, count in pareto_combos[:10]:
        lines.append(
            f"| {combo[0]} | {combo[1]} | {combo[2]} | {combo[3]} | "
            f"{mean_acc:.3f} | {max_acc:.3f} |"
        )
    lines.append("")

    # ── 3. Default vs Optimized comparison ──────────────────────────────────
    lines.append("## 3. Default vs Optimized Parameters\n")
    lines.append(
        "Comparison: **Default** `(top_p=1.0, top_k=40, min_p=0.0)` vs "
        "**Optimized** `(top_p=0.9, top_k=20, min_p=0.05, seed=42)`\n"
    )

    # Find cells that use default params vs optimized params
    for mk in MODEL_KEYS:
        lines.append(f"### {mk}\n")
        lines.append("| Type | Cell | Prompt | Accuracy |")
        lines.append("|------|------|--------|----------|")

        default_cells = [c for c in all_cells if c.model_key == mk
                         and c.decoding.top_p == 1.0
                         and c.decoding.top_k == 40
                         and c.decoding.min_p == 0.0]
        opt_cells = [c for c in all_cells if c.model_key == mk
                     and c.decoding.top_p == 0.9
                     and c.decoding.top_k == 20
                     and c.decoding.min_p == 0.05]

        for cells, label in [(default_cells, "Default"), (opt_cells, "Optimized")]:
            for c in cells:
                acc = c.metadata.get("accuracy", 0.0)
                correct = c.metadata.get("correct", 0)
                total = c.metadata.get("total", 0)
                prompt_short = c.system_prompt[:50] if c.system_prompt else "(empty)"
                lines.append(
                    f"| {label} | {c.name} | {prompt_short} | "
                    f"{acc:.3f} ({correct}/{total}) |"
                )
        lines.append("")

    # ── 4. Failure Analysis (hard questions) ────────────────────────────────
    lines.append("## 4. Failure Analysis on Hard Questions\n")
    hard_questions = [q for q in questions if q.get("difficulty") == "hard"]
    lines.append(f"Analyzing {len(hard_questions)} hard questions across all cells.\n")

    # Find the single best cell overall
    all_best = max(all_cells, key=lambda c: c.metadata.get("accuracy", 0.0))

    # If details available, analyze failures
    if "details" in all_best.metadata:
        details = all_best.metadata["details"]
        # Map details back to questions
        failures = []
        for q_idx, d in enumerate(details):
            q = questions[q_idx] if q_idx < len(questions) else {}
            if q.get("difficulty") == "hard" and not d.get("correct", True):
                failures.append({
                    "question": q.get("prompt", "")[:80],
                    "expected": q.get("expected_answer", ""),
                    "got": d.get("got", ""),
                    "reasoning": q.get("reasoning", ""),
                })

        lines.append(f"**Best cell:** {all_best.name} (acc={all_best.metadata.get('accuracy', 0):.3f})\n")
        lines.append(f"**Hard question failures:** {len(failures)}/{len(hard_questions)}\n")

        if failures:
            lines.append("| # | Expected | Got | Question |")
            lines.append("|---|----------|-----|----------|")
            for i, f in enumerate(failures[:20]):  # top 20
                lines.append(
                    f"| {i+1} | {f['expected']} | {f['got'][:30]} | {f['question'][:60]} |"
                )
            lines.append("")

            # Failure patterns
            false_pos = sum(1 for f in failures if f["got"].lower() in ["positive", "negative"]
                            and f["got"].lower() != f["expected"])
            neutral_misclass = sum(1 for f in failures if "neutral" in f["got"].lower()
                                   and "neutral" not in f["expected"].lower())
            mixed_misclass = sum(1 for f in failures if "mixed" in f["got"].lower()
                                 and "mixed" not in f["expected"].lower())
            lines.append("**Failure patterns:**\n")
            lines.append(f"- False positives/negatives (got opposite sentiment): {false_pos}")
            lines.append(f"- Neutral misclassifications: {neutral_misclass}")
            lines.append(f"- Mixed misclassifications: {mixed_misclass}")
            lines.append(f"- Other failures: {len(failures) - false_pos - neutral_misclass - mixed_misclass}")
            lines.append("")

    # ── 5. Cross-generation improvement ─────────────────────────────────────
    lines.append("## 5. Cross-Generation Improvement\n")
    lines.append("| Model | Gen 0 Best | Gen 1 Best | Δ |")
    lines.append("|-------|------------|------------|-----|")

    for mk in MODEL_KEYS:
        if mk in comparison_data:
            d = comparison_data[mk]
            delta_str = f"{d['delta']:+.4f}"
            if d['delta'] > 0:
                delta_str += " ✅"
            elif d['delta'] < 0:
                delta_str += " ❌"
            else:
                delta_str += " ➖"
            lines.append(
                f"| {mk} | {d['acc0']:.3f} | {d['acc1']:.3f} | {delta_str} |"
            )

    lines.append("")

    # Save
    with open(OUTPUT_REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  Report saved.")
    return lines


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global questions

    print("=" * 60)
    print("  GEPA Sentiment Optimization Run")
    print("=" * 60)

    # Create log dir
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_log_dir = os.path.join(LOG_DIR, f"sentiment_gepa_{timestamp}")
    os.makedirs(run_log_dir, exist_ok=True)

    # 1. Load eval set
    questions = load_eval_set()
    print(f"  Questions: {len(questions)}")

    # Save questions to log dir
    with open(os.path.join(run_log_dir, "eval_questions.json"), "w") as f:
        json.dump(questions, f, indent=2)

    # 2. Create seed population
    gen0_population = create_seed_population()

    # 3. Evaluate gen 0
    print(f"\n{'='*60}")
    print(f"  GENERATION 0 — EVALUATING SEED POPULATION")
    print(f"{'='*60}")

    evaluate_generation(gen0_population, questions, "Gen 0")
    print("\n")
    summarize_generation(gen0_population, "GENERATION 0", questions)

    # Save gen 0 results
    gen0_data = []
    for c in gen0_population:
        gen0_data.append({
            "name": c.name,
            "model_key": c.model_key,
            "system_prompt": c.system_prompt,
            "decoding": c.decoding.to_dict(),
            "metadata": c.metadata,
        })
    with open(os.path.join(run_log_dir, "gen0_results.json"), "w") as f:
        json.dump(gen0_data, f, indent=2, default=str)

    # 4. Evolve to gen 1
    print(f"\n{'='*60}")
    print(f"  EVOLVING TO GENERATION 1")
    print(f"{'='*60}")

    mutation_agent = MutationAgent(model_keys=MODEL_KEYS, seed=SEED)
    gen1_population = mutation_agent.evolve(
        gen0_population,
        tags=["params"],  # bias toward parameter mutations (ops 10-14)
        target_size=max(10, len(gen0_population)),
        elite_count=3,
        crossover_fraction=0.5,
    )

    print(f"\n  Gen 1 population: {len(gen1_population)} cells")
    for c in gen1_population:
        print(f"    {c.name:45s} | gen={c.generation} | model={c.model_key} | "
              f"prompt={c.system_prompt[:50]!r}")

    # 5. Evaluate gen 1
    print(f"\n{'='*60}")
    print(f"  GENERATION 1 — EVALUATING EVOLVED POPULATION")
    print(f"{'='*60}")

    evaluate_generation(gen1_population, questions, "Gen 1")
    print("\n")
    best_gen1 = summarize_generation(gen1_population, "GENERATION 1", questions)

    # Save gen 1 results
    gen1_data = []
    for c in gen1_population:
        gen1_data.append({
            "name": c.name,
            "model_key": c.model_key,
            "system_prompt": c.system_prompt,
            "decoding": c.decoding.to_dict(),
            "metadata": c.metadata,
        })
    with open(os.path.join(run_log_dir, "gen1_results.json"), "w") as f:
        json.dump(gen1_data, f, indent=2, default=str)

    # 6. Compare generations
    comparison = compare_generations(gen0_population, gen1_population)

    # 7. Generate report
    report_lines = generate_report(
        gen0_population, gen1_population, questions, comparison, run_log_dir
    )

    # 8. Final summary
    print(f"\n{'='*60}")
    print(f"  RUN COMPLETE")
    print(f"{'='*60}")
    print(f"  Log directory: {run_log_dir}")
    print(f"  Report:        {OUTPUT_REPORT}")

    # Find best overall
    all_cells = gen0_population + gen1_population
    best_overall = max(all_cells, key=lambda c: c.metadata.get("accuracy", 0.0))
    print(f"\n  Best cell overall:")
    print(f"    Name:    {best_overall.name}")
    print(f"    Model:   {best_overall.model_key}")
    print(f"    Prompt:  {best_overall.system_prompt!r}")
    print(f"    Params:  {best_overall.decoding.to_dict()}")
    print(f"    Acc:     {best_overall.metadata.get('accuracy', 0):.4f}")

    return {
        "gen0": gen0_population,
        "gen1": gen1_population,
        "best": best_overall,
        "comparison": comparison,
        "log_dir": run_log_dir,
    }


if __name__ == "__main__":
    main()
