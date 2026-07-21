#!/usr/bin/env python3
"""
gepa_plans/sentiment_gepa_v2.py — GEPA evolution for sentiment with honest exact-label eval.

Uses:
  - Format normalizer on ALL LLM outputs → exact label match
  - Hybrid VADER+LLM router (optimized)
  - Training set subset for evolution, val set for validation
  - 3 models x 8 prompt variants + param variations = 24+ seed cells

Usage:
    python3 gepa_plans/sentiment_gepa_v2.py --generations 2 --questions 100
"""

import argparse
import copy
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime

BASE = os.path.expanduser("/home/artem/dev/amd-hackathon")
sys.path.insert(0, BASE)

DATA_DIR = f"{BASE}/data/eval"
RESULTS_DIR = f"{BASE}/research"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATHS = {
    "gemma-3-1b": f"{BASE}/models/gemma-3-1b-it-Q4_K_M.gguf",
    "qwen2.5-1.5b": f"{BASE}/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "qwen2.5-coder-1.5b": f"{BASE}/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
}

# ── Import format normalizer ──────────────────────────────────────────────────

try:
    from agent.solvers.format_normalizer import normalize_sentiment_output
except ImportError:
    def normalize_sentiment_output(text):
        """Fallback simple normalizer."""
        if not text:
            return "unknown", "low"
        t = text.strip().lower()
        labels = ["positive", "negative", "neutral", "mixed"]
        for label in labels:
            if re.search(r'(?:^|[\s,;.:!?"\'(])' + re.escape(label) + r'(?:[\s,;.:!?"\')]|$)', t):
                return label, "high"
        # Substring fallback
        for label in labels:
            if label in t:
                return label, "low"
        return "unknown", "low"

# Import hybrid
try:
    from agent.solvers.sentiment_hybrid import classify_sentiment_hybrid
    HAS_HYBRID = True
except ImportError:
    HAS_HYBRID = False

# ── Data loading ──────────────────────────────────────────────────────────────

def load_split(name):
    path = f"{DATA_DIR}/sentiment_{name}.json"
    with open(path) as f:
        return json.load(f)

def sample_questions(data, n=None, seed=42):
    """Sample n questions from data, stratified by difficulty if possible."""
    if n is None or n >= len(data):
        return data
    rng = random.Random(seed)
    # Stratified by difficulty
    by_diff = defaultdict(list)
    for item in data:
        by_diff[item.get("difficulty", "unknown")].append(item)
    sampled = []
    for diff, items in by_diff.items():
        count = max(1, int(n * len(items) / len(data)))
        rng.shuffle(items)
        sampled.extend(items[:count])
    # If we have too many, trim
    rng.shuffle(sampled)
    return sampled[:n]

# ── Model cache ───────────────────────────────────────────────────────────────

_model_cache = {}

def get_model(model_key):
    if model_key not in _model_cache:
        path = MODEL_PATHS.get(model_key)
        if not path or not os.path.exists(path):
            raise ValueError(f"Model not found: {path}")
        from llama_cpp import Llama
        print(f"  Loading {model_key} ...")
        _model_cache[model_key] = Llama(
            model_path=path, n_ctx=2048, n_gpu_layers=-1,
            n_threads=4, verbose=False,
        )
    return _model_cache[model_key]

# ── Prompt variants ──────────────────────────────────────────────────────────

PROMPT_VARIANTS = [
    # Current default
    "Analyze the tone as positive, negative, neutral, or mixed.",
    # Very explicit
    "Classify the sentiment. Output EXACTLY one word: positive, negative, neutral, or mixed.",
    # Minimal
    "Positive, negative, neutral, or mixed? Answer with one word.",
    # Label-focused
    "Sentiment: positive, negative, neutral, or mixed.",
    # Direct question
    "What is the sentiment? Reply with one word.",
    # Simple binary (no neutral/mixed)
    "Is the sentiment positive or negative? Reply with one word.",
    # Blank/empty (proved best for factual)
    "",
    # With format instruction
    "Analyze this text's sentiment. Reply with ONLY one word: positive, negative, neutral, or mixed.",
    # JSON format
    'Classify the sentiment. Output JSON: {"sentiment": "positive|negative|neutral|mixed"}',
    # Concise
    "Sentiment?",
]

PARAM_VARIANTS = [
    {"top_p": 0.9, "top_k": 20, "min_p": 0.05},
    {"top_p": 0.85, "top_k": 30, "min_p": 0.03},
    {"top_p": 0.95, "top_k": 15, "min_p": 0.07},
    {"top_p": 0.9, "top_k": 40, "min_p": 0.0},
]

# ── Honest evaluation ─────────────────────────────────────────────────────────

def evaluate_cell(model_key, system_prompt, params, questions, use_hybrid=True):
    """
    Evaluate a single cell (model + prompt + params) on questions.
    Uses format_normalizer for honest exact-label match.
    Returns accuracy metrics.
    """
    llm = get_model(model_key)
    temperature = 0.0  # Always use temperature=0 for eval
    repeat_penalty = params.get("repeat_penalty", 1.0)
    max_tokens = 128
    seed = params.get("seed", 42)

    correct = 0
    total = 0
    total_latency = 0.0
    source_counts = defaultdict(int)
    source_correct = defaultdict(int)

    for item in questions:
        text = item["prompt"]
        expected = item["expected_answer"].strip().lower()

        if use_hybrid and HAS_HYBRID:
            # Hybrid routing
            def llm_infer_fn(sys_msg, user_msg):
                full = f"{sys_msg}\n\n{user_msg}"
                start_t = time.time()
                response = llm(
                    full, max_tokens=max_tokens, temperature=temperature,
                    top_p=params.get("top_p", 0.9), top_k=params.get("top_k", 20),
                    min_p=params.get("min_p", 0.05),
                    repeat_penalty=repeat_penalty, seed=seed,
                    echo=False, stop=["\n\n", "---"],
                )
                return response["choices"][0]["text"].strip() if response.get("choices") else ""

            start_t = time.time()
            hybrid_result = classify_sentiment_hybrid(
                text=text, llm_infer_fn=llm_infer_fn,
                system_prompt=system_prompt,
            )
            elapsed = time.time() - start_t
            predicted = hybrid_result["label"]
            source = hybrid_result["source"]
        else:
            # Pure LLM
            full = f"{system_prompt}\n\n{text}" if system_prompt else text
            start_t = time.time()
            response = llm(
                full, max_tokens=max_tokens, temperature=temperature,
                top_p=params.get("top_p", 0.9), top_k=params.get("top_k", 20),
                min_p=params.get("min_p", 0.05),
                repeat_penalty=repeat_penalty, seed=seed,
                echo=False, stop=["\n\n", "---"],
            )
            elapsed = time.time() - start_t
            raw = response["choices"][0]["text"].strip() if response.get("choices") else ""
            predicted, _ = normalize_sentiment_output(raw)
            source = "llm"

        is_correct = (predicted == expected)
        if is_correct:
            correct += 1
        total += 1
        total_latency += elapsed

    accuracy = correct / total * 100 if total > 0 else 0.0
    avg_latency = total_latency / total if total > 0 else 0.0

    return {
        "accuracy": round(accuracy, 2),
        "correct": correct,
        "total": total,
        "avg_latency_ms": round(avg_latency * 1000, 1),
    }

# ── Cell creation ────────────────────────────────────────────────────────────

def create_seed_cells(models, prompts, params_list):
    """Create seed cells: model x prompt x param (at least 24)."""
    cells = []
    cell_id = 0
    for model_key in models:
        for prompt in prompts[:8]:  # Use first 8 prompts
            for params in params_list[:1]:  # Start with 1 param set per cell
                cells.append({
                    "id": f"seed_{cell_id}",
                    "model": model_key,
                    "system_prompt": prompt,
                    "params": params,
                    "temperature": 0.0,
                    "generation": 0,
                    "parent": None,
                })
                cell_id += 1
    return cells

# ── Mutation ──────────────────────────────────────────────────────────────────

def mutate_prompt(prompt, mutation_rate=0.3):
    """Apply random mutations to a prompt."""
    if random.random() > mutation_rate:
        return prompt

    ops = [
        # Add instruction
        lambda p: p + " Be precise." if not p.endswith(".") else p + " Be precise.",
        # Add constraint
        lambda p: p + " One word only." if "one word" not in p.lower() else p,
        # Shorten
        lambda p: p[:len(p)//2].rsplit(' ', 1)[0] + '.' if len(p) > 20 else p,
        # Swap label order
        lambda p: p.replace("positive, negative", "negative, positive") if "positive, negative" in p else p,
        # Make shorter
        lambda p: "Classify the sentiment." if len(p) > 30 else p + " Be concise.",
        # Add format emphasis
        lambda p: p + " ONLY one word." if "only" not in p.lower() else p,
    ]
    op = random.choice(ops)
    try:
        return op(prompt)
    except:
        return prompt

def mutate_params(params):
    """Mutate generation parameters slightly."""
    new = dict(params)
    new["top_p"] = min(1.0, max(0.5, params["top_p"] + random.uniform(-0.1, 0.1)))
    new["top_k"] = max(1, min(100, params["top_k"] + random.randint(-10, 10)))
    new["min_p"] = max(0.0, min(0.2, params["min_p"] + random.uniform(-0.02, 0.02)))
    return new

# ── Selection & Evolution ────────────────────────────────────────────────────

def select_best(cells, model_key, top_n=3):
    """Select top N cells for a given model by accuracy."""
    model_cells = [c for c in cells if c["model"] == model_key and c.get("scores")]
    sorted_cells = sorted(model_cells, key=lambda c: c["scores"].get("accuracy", 0), reverse=True)
    return sorted_cells[:top_n]

def next_generation(population, model_keys, pop_size=8):
    """Create next generation by selecting best and mutating."""
    next_pop = []
    for mk in model_keys:
        best = select_best(population, mk, top_n=3)
        if not best:
            # No cells for this model — create random ones
            for _ in range(pop_size):
                from random import choice
                prompt = choice(PROMPT_VARIANTS[:8])
                params = choice(PARAM_VARIANTS)
                next_pop.append({
                    "id": f"{mk}_rand_{len(next_pop)}",
                    "model": mk,
                    "system_prompt": prompt,
                    "params": params,
                    "temperature": 0.0,
                    "generation": population[0].get("generation", 0) + 1 if population else 1,
                    "parent": None,
                })
            continue

        # Keep elites
        for cell in best:
            clone = copy.deepcopy(cell)
            clone["id"] = f"{mk}_elite_{cell['id']}"
            clone["generation"] = cell.get("generation", 0) + 1
            clone["parent"] = cell["id"]
            next_pop.append(clone)

        # Generate mutants from elites
        existing = len([c for c in next_pop if c["model"] == mk])
        while existing < pop_size:
            parent = random.choice(best)
            child = copy.deepcopy(parent)
            child["id"] = f"{mk}_mut_{len(next_pop)}"
            child["system_prompt"] = mutate_prompt(parent["system_prompt"])
            child["params"] = mutate_params(parent["params"])
            child["generation"] = parent.get("generation", 0) + 1
            child["parent"] = parent["id"]
            child.pop("scores", None)
            next_pop.append(child)
            existing += 1

    return next_pop

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sentiment GEPA v2 - Honest Eval")
    parser.add_argument("--generations", type=int, default=2, help="Number of generations")
    parser.add_argument("--questions", type=int, default=100, help="Questions per eval subset")
    parser.add_argument("--no-hybrid", action="store_false", dest="hybrid", default=True)
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer cells")
    args = parser.parse_args()

    MODELS = ["gemma-3-1b", "qwen2.5-1.5b", "qwen2.5-coder-1.5b"]

    print("=" * 70)
    print("SENTIMENT GEPA v2 - Honest Exact-Label Evaluation")
    print("=" * 70)
    print(f"  Models: {MODELS}")
    print(f"  Generations: {args.generations}")
    print(f"  Questions per eval: {args.questions}")
    print(f"  Hybrid: {args.hybrid}")

    # Load data
    train_raw = load_split("train")
    val_raw = load_split("val")

    print(f"\n  Loaded {len(train_raw)} train, {len(val_raw)} val questions")

    # Sample subset for evolution
    rng = random.Random(42)
    train_sample = sample_questions(train_raw, args.questions)
    print(f"  Training subset: {len(train_sample)} questions (stratified)")

    # Create seed cells
    seed_cells = create_seed_cells(MODELS, PROMPT_VARIANTS, PARAM_VARIANTS)
    print(f"\n  Created {len(seed_cells)} seed cells")

    if args.quick:
        seed_cells = seed_cells[:12]  # Quick test with fewer cells

    # Evolution loop
    population = seed_cells
    all_generations = []
    best_overall = None

    for gen in range(args.generations + 1):
        print(f"\n{'='*70}")
        print(f"GENERATION {gen} ({len(population)} cells)")
        print(f"{'='*70}")

        # Evaluate each cell on training subset
        for i, cell in enumerate(population):
            if cell.get("scores"):
                continue  # Already evaluated

            print(f"  Cell {i+1}/{len(population)}: {cell['model']} | prompt='{cell['system_prompt'][:50]}...' | params={cell['params']}")

            try:
                scores = evaluate_cell(
                    cell["model"], cell["system_prompt"], cell["params"],
                    train_sample, use_hybrid=args.hybrid,
                )
                cell["scores"] = scores
                print(f"    acc={scores['accuracy']:.1f}% ({scores['correct']}/{scores['total']})")
            except Exception as e:
                print(f"    ERROR: {e}")
                cell["scores"] = {"accuracy": 0.0, "correct": 0, "total": 0, "avg_latency_ms": 999}

        # Per-model results
        print(f"\n  --- Generation {gen} Results ---")
        gen_results = {"generation": gen, "cells": []}
        for mk in MODELS:
            mk_cells = [c for c in population if c["model"] == mk and c.get("scores")]
            if not mk_cells:
                continue
            best_cell = max(mk_cells, key=lambda c: c["scores"]["accuracy"])
            print(f"\n  Best for {mk}:")
            print(f"    Prompt: {best_cell['system_prompt'][:80]}")
            print(f"    Params: {best_cell['params']}")
            print(f"    Train acc: {best_cell['scores']['accuracy']:.1f}%")

            # Validate on val set
            print(f"    Validating on val set ({len(val_raw)} questions)...")
            try:
                val_scores = evaluate_cell(
                    best_cell["model"], best_cell["system_prompt"],
                    best_cell["params"], val_raw, use_hybrid=args.hybrid,
                )
                best_cell["val_scores"] = val_scores
                print(f"    Val acc: {val_scores['accuracy']:.1f}% ({val_scores['correct']}/{val_scores['total']})")
            except Exception as e:
                print(f"    Val ERROR: {e}")

            gen_results["cells"].append({
                "id": best_cell["id"],
                "model": mk,
                "system_prompt": best_cell["system_prompt"],
                "params": best_cell["params"],
                "train_accuracy": best_cell["scores"]["accuracy"],
                "val_accuracy": best_cell.get("val_scores", {}).get("accuracy", 0.0),
            })

            # Track best overall
            val_acc = best_cell.get("val_scores", {}).get("accuracy", 0)
            if best_overall is None or val_acc > best_overall.get("val_accuracy", 0):
                best_overall = {
                    "generation": gen,
                    "model": mk,
                    "system_prompt": best_cell["system_prompt"],
                    "params": best_cell["params"],
                    "train_accuracy": best_cell["scores"]["accuracy"],
                    "val_accuracy": val_acc,
                }

        all_generations.append(gen_results)

        # Create next generation (unless last)
        if gen < args.generations:
            population = next_generation(population, MODELS, pop_size=8)
            print(f"\n  Created {len(population)} cells for next generation")

    # ═════════════════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"\nBest overall configuration:")
    if best_overall:
        print(f"  Generation: {best_overall['generation']}")
        print(f"  Model: {best_overall['model']}")
        print(f"  Prompt: '{best_overall['system_prompt']}'")
        print(f"  Params: {best_overall['params']}")
        print(f"  Train accuracy: {best_overall['train_accuracy']:.1f}%")
        print(f"  Val accuracy: {best_overall['val_accuracy']:.1f}%")

    print(f"\nPer-model best on val set:")
    for mk in MODELS:
        best_for_model = None
        for gen_res in all_generations:
            for c in gen_res["cells"]:
                if c["model"] == mk:
                    if best_for_model is None or c.get("val_accuracy", 0) > best_for_model.get("val_accuracy", 0):
                        best_for_model = c
        if best_for_model:
            print(f"\n  {mk}:")
            print(f"    Prompt: '{best_for_model['system_prompt']}'")
            print(f"    Params: {best_for_model['params']}")
            print(f"    Train: {best_for_model['train_accuracy']:.1f}%  Val: {best_for_model.get('val_accuracy', 0):.1f}%")

    # Save results
    output = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "config": {
            "models": MODELS,
            "generations": args.generations,
            "questions_per_eval": args.questions,
            "hybrid": args.hybrid,
        },
        "all_generations": all_generations,
        "best_overall": best_overall,
    }
    output_path = f"{RESULTS_DIR}/sentiment_gepa_v2_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Write markdown report
    report_path = f"{RESULTS_DIR}/sentiment_gepa_v2_results.md"
    with open(report_path, "w") as f:
        f.write("# Sentiment GEPA v2 Results\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"## Config\n\n")
        f.write(f"- Models: {MODELS}\n")
        f.write(f"- Generations: {args.generations}\n")
        f.write(f"- Questions per eval: {args.questions}\n")
        f.write(f"- Hybrid mode: {args.hybrid}\n")
        f.write(f"- Temperature: 0.0\n\n")

        f.write("## Results per Generation\n\n")
        for gen_res in all_generations:
            f.write(f"### Generation {gen_res['generation']}\n\n")
            f.write("| Model | Prompt | Params | Train Acc | Val Acc |\n")
            f.write("|-------|--------|--------|-----------|--------|\n")
            for c in gen_res["cells"]:
                prompt_short = c["system_prompt"][:50].replace("|", "\\|") if c["system_prompt"] else "(empty)"
                params_str = str(c["params"])
                f.write(f"| {c['model']} | {prompt_short} | {params_str} | {c['train_accuracy']:.1f}% | {c.get('val_accuracy', 0):.1f}% |\n")
            f.write("\n")

        if best_overall:
            f.write("## Best Overall\n\n")
            f.write(f"- **Model**: {best_overall['model']}\n")
            f.write(f"- **Prompt**: `{best_overall['system_prompt']}`\n")
            f.write(f"- **Params**: `{best_overall['params']}`\n")
            f.write(f"- **Train accuracy**: {best_overall['train_accuracy']:.1f}%\n")
            f.write(f"- **Val accuracy**: {best_overall['val_accuracy']:.1f}%\n\n")

        f.write("## Per-Model Best\n\n")
        for mk in MODELS:
            best_for_model = None
            for gen_res in all_generations:
                for c in gen_res["cells"]:
                    if c["model"] == mk:
                        if best_for_model is None or c.get("val_accuracy", 0) > best_for_model.get("val_accuracy", 0):
                            best_for_model = c
            if best_for_model:
                f.write(f"### {mk}\n\n")
                f.write(f"- **Prompt**: `{best_for_model['system_prompt']}`\n")
                f.write(f"- **Params**: `{best_for_model['params']}`\n")
                f.write(f"- **Train**: {best_for_model['train_accuracy']:.1f}%\n")
                f.write(f"- **Val**: {best_for_model.get('val_accuracy', 0):.1f}%\n\n")

    print(f"\n  Results saved to {output_path}")
    print(f"  Report saved to {report_path}")


if __name__ == "__main__":
    main()
