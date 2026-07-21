#!/usr/bin/env python3
"""
Pipeline-Integrated GEPA — wraps Pipeline.process() so candidate prompts
are tested through the full system (classifier → deterministic solvers → LLM →
post-processing → Fireworks). Uses the existing routing_table injection point
in Pipeline.__init__ (no Pipeline changes needed).

Implements NSGA-II Pareto evolution across three objectives:
  - accuracy (maximize)   → acc_norm
  - avg latency (minimize) → lat_norm (1 - normalized)
  - avg tokens (minimize)  → tokens_norm (1 - normalized)

Usage:
    python3 agent/pipeline_gepa.py --category factual --generations 3 --questions 19
    python3 agent/pipeline_gepa.py --category factual --generations 1 --questions 5  # smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional

import agent.dynamic_prompts as _dp
from scripts.grade_answer import fuzzy_match

# ── Constants ────────────────────────────────────────────────────────────────

RESULTS_DIR = "/home/artem/dev/amd-hackathon/gepa_plans"
TRAINING_DATA = "/home/artem/dev/amd-hackathon/data/eval/training-v3.json"
DEFAULT_MODEL_PATH = "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"

POPULATION_SIZE = 6
NUM_GENERATIONS = 3
ELITE_COUNT = 2
TOURNAMENT_K = 3
MAX_TOKENS_DEFAULT = 128
TEMPERATURE_DEFAULT = 0.0
MAX_PROMPT_CHARS = 300
TEMP_OPTIONS = [0.0, 0.1, 0.2]
CONVERGENCE_THRESHOLD_PP = 5.0

# ── Mutation Pools (same patterns as agent/gepa_category_runner.py) ──────────

CONSTRAINT_POOL = [
    "No preamble.",
    "No explanation.",
    "Be specific.",
    "Don't hedge.",
    "Keep under 15 words.",
    "Keep under 5 words.",
    "Output only the answer.",
    "Address all parts of the question.",
]

PREFIXES: dict[str, list[str]] = {
    "factual": ["", "Fact:", "Answer:", "Q:", "Answer directly:"],
    "math": ["", "Math:", "Solve:", "Answer:", "Calculate:"],
    "logic": ["", "Logic:", "Solve:", "Answer:", "Reason:"],
    "sentiment": ["", "Sentiment:", "Classify:", "Label:"],
    "ner": ["", "Entities:", "NER:", "Extract:"],
    "summarization": ["", "Summary:", "TL;DR:", "Brief:", "Concisely:"],
    "code_gen": ["", "Code:", "Implement:", "Python:"],
    "code_debug": ["", "Debug:", "Fix:", "Code:", "Corrected:"],
}

TASK_INSTRUCTIONS: dict[str, list[str]] = {
    "factual": [
        "Answer with facts only.",
        "Use exact names and numbers.",
        "Be concise and direct.",
        "If unsure, state what you know.",
        "No opinions, just facts.",
    ],
    "math": [
        "Solve step by step.",
        "Output only the numeric answer.",
        "Double-check your arithmetic.",
        "Use proper mathematical notation.",
        "Show your work briefly.",
    ],
    "logic": [
        "Use logical deduction.",
        "Consider all possibilities.",
        "Eliminate impossible options.",
        "Output only the solution.",
        "Be systematic in your reasoning.",
    ],
    "sentiment": [
        "Classify as POSITIVE or NEGATIVE only.",
        "Output exactly one word.",
        "Watch for sarcasm and hedging.",
        "Consider the overall tone.",
        "No explanation. Just the label.",
    ],
    "ner": [
        "Extract all named entities.",
        "Label as PERSON, ORG, LOC, or DATE.",
        "Only include entities explicitly in the text.",
        "Format as TYPE: value, one per line.",
        "If no entities found, output 'None'.",
    ],
    "summarization": [
        "Summarize in 1-2 sentences.",
        "Use exact names and numbers.",
        "Capture the core event only.",
        "Be concise and specific.",
        "Include who, what, when.",
    ],
    "code_gen": [
        "Write clean, working Python code.",
        "Include necessary imports.",
        "Use appropriate data structures.",
        "Handle edge cases.",
        "Return a function definition.",
    ],
    "code_debug": [
        "Find the exact bug.",
        "Fix without changing the function signature.",
        "Preserve original functionality.",
        "Only output the corrected code.",
        "Check for off-by-one errors.",
    ],
}

SHORT_STYLES = ["Be concise.", "Be brief.", "Keep it short."]
MEDIUM_STYLES = ["Answer clearly.", "Provide a clear answer.", "Respond with the answer."]

THESAURUS: dict[str, list[str]] = {
    "Answer the question directly.": [
        "Respond directly to the question.",
        "Give a direct answer.",
        "Answer straight.",
        "Answer clearly.",
    ],
    "Be concise.": ["Keep it brief.", "Be brief.", "Answer concisely."],
    "Provide the exact answer.": [
        "Give the exact answer.",
        "State the precise answer.",
        "Exact answer only.",
    ],
}

logger = logging.getLogger("pipeline_gepa")


# ═════════════════════════════════════════════════════════════════════════════
# NSGA-II Pareto Functions
# ═════════════════════════════════════════════════════════════════════════════


def pareto_dominates(a: dict, b: dict) -> bool:
    """Return True if a Pareto-dominates b across acc_norm, tokens_norm, lat_norm."""
    objs = ["acc_norm", "tokens_norm", "lat_norm"]
    better_or_eq = all(a[o] >= b[o] - 1e-10 for o in objs)
    strictly_better = any(a[o] > b[o] + 1e-10 for o in objs)
    return better_or_eq and strictly_better


def fast_non_dominated_sort(population: list) -> list[list[int]]:
    """Return list of front indices, front[0] = Pareto-optimal set."""
    n = len(population)
    S = [set() for _ in range(n)]
    n_count = [0] * n
    fronts: list[list[int]] = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if pareto_dominates(population[i], population[j]):
                S[i].add(j)
            elif pareto_dominates(population[j], population[i]):
                n_count[i] += 1

    current = [i for i in range(n) if n_count[i] == 0]
    if not current:
        return [list(range(n))]
    fronts.append(current)

    while current:
        nxt = []
        for i in current:
            for j in S[i]:
                n_count[j] -= 1
                if n_count[j] == 0:
                    nxt.append(j)
        if not nxt:
            break
        fronts.append(nxt)
        current = nxt

    return fronts


def crowding_distance(
    population: list,
    front: list[int],
    objectives: tuple[str, ...] = ("acc_norm", "tokens_norm", "lat_norm"),
) -> list[float]:
    """Compute crowding distance for individuals in a front."""
    if len(front) <= 2:
        return [float("inf")] * len(front)
    dist = [0.0] * len(front)
    for obj in objectives:
        fs = sorted(range(len(front)), key=lambda idx: population[front[idx]][obj])
        mn = population[front[fs[0]]][obj]
        mx = population[front[fs[-1]]][obj]
        denom = mx - mn + 1e-9
        dist[fs[0]] = float("inf")
        dist[fs[-1]] = float("inf")
        for pos in range(1, len(fs) - 1):
            i_prev = front[fs[pos - 1]]
            i_next = front[fs[pos + 1]]
            dist[fs[pos]] += (population[i_next][obj] - population[i_prev][obj]) / denom
    return dist


def normalize_objectives(population: list) -> None:
    """Normalize accuracy (higher=better), tokens (lower=better), latency (lower=better)."""
    accs = [v["accuracy"] for v in population]
    tokens = [v["avg_output_tokens"] for v in population]
    lats = [v["avg_latency_ms"] for v in population]
    min_a, max_a = min(accs), max(accs)
    min_t, max_t = min(tokens), max(tokens)
    min_l, max_l = min(lats), max(lats)

    for v in population:
        v["acc_norm"] = (v["accuracy"] - min_a) / (max_a - min_a + 1e-9)
        v["tokens_norm"] = 1.0 - (v["avg_output_tokens"] - min_t) / (max_t - min_t + 1e-9)
        v["lat_norm"] = 1.0 - (v["avg_latency_ms"] - min_l) / (max_l - min_l + 1e-9)


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation — full Pipeline through routing_table injection
# ═════════════════════════════════════════════════════════════════════════════


def evaluate_candidate(
    candidate: dict, category: str, questions: list[dict]
) -> dict:
    """Evaluate a single candidate prompt through the full Pipeline.

    Temporarily patches ``agent.dynamic_prompts._CATEGORY_PROMPTS`` so the
    candidate's system_prompt is used at every complexity level, then runs
    all questions through ``pipe.process()`` and grades with ``fuzzy_match``.

    Returns dict with accuracy, avg_output_tokens, avg_latency_ms,
    and per_question results.
    """
    from agent.pipeline import Pipeline, PipelineConfig

    system_prompt = candidate.get("system_prompt", "")
    temperature = candidate.get("temperature", TEMPERATURE_DEFAULT)
    max_tokens = candidate.get("max_tokens", MAX_TOKENS_DEFAULT)

    # Build config with GPU layers
    cfg = PipelineConfig(
        model_path=DEFAULT_MODEL_PATH,
        n_gpu_layers=-1,
        n_ctx=2048,
        n_threads=4,
        consensus_samples=1,  # no consensus voting for GEPA evals
    )

    # Patch dynamic_prompts so the candidate prompt is used at every level
    import agent.dynamic_prompts as _dp
    orig_prompts = _dp._CATEGORY_PROMPTS.get(category, {}).copy()
    for level in orig_prompts:
        _dp._CATEGORY_PROMPTS[category][level] = system_prompt

    pipe: Optional[Pipeline] = None
    try:
        pipe = Pipeline(config=cfg)

        correct = 0
        total_latency_ms = 0.0
        total_tokens = 0
        per_question: list[dict] = []

        for i, q in enumerate(questions):
            prompt_text = q["prompt"]
            expected = q.get("expected_answer", "")

            start = time.time()
            answer = pipe.process(prompt_text)
            elapsed_ms = (time.time() - start) * 1000

            passed = fuzzy_match(answer, expected)
            if passed:
                correct += 1

            total_latency_ms += elapsed_ms
            tok_est = max(1, len(answer) // 4)  # rough token estimate
            total_tokens += tok_est

            per_question.append({
                "task_id": q.get("task_id", f"q{i}"),
                "answer": answer,
                "expected": expected,
                "passed": passed,
                "latency_ms": elapsed_ms,
                "tokens_est": tok_est,
            })

        n = len(questions)
        return {
            "accuracy": correct / n if n else 0.0,
            "avg_output_tokens": total_tokens / n if n else 0,
            "avg_latency_ms": total_latency_ms / n if n else 0.0,
            "per_question": per_question,
            "total_correct": correct,
            "total_questions": n,
        }

    except Exception as exc:
        logger.error("Candidate evaluation failed: %s", exc)
        n = len(questions)
        return {
            "accuracy": 0.0,
            "avg_output_tokens": 999,
            "avg_latency_ms": 99999.0,
            "per_question": [
                {
                    "task_id": q.get("task_id", f"q{i}"),
                    "answer": "",
                    "expected": q.get("expected_answer", ""),
                    "passed": False,
                    "latency_ms": 0,
                    "tokens_est": 0,
                }
                for i, q in enumerate(questions)
            ],
            "total_correct": 0,
            "total_questions": n if n else 0,
        }

    finally:
        # Restore original dynamic_prompts
        _dp._CATEGORY_PROMPTS[category] = orig_prompts
        if pipe is not None:
            try:
                pipe.close()
            except Exception:
                pass


# ═════════════════════════════════════════════════════════════════════════════
# Population Generation
# ═════════════════════════════════════════════════════════════════════════════


def build_seed_population(category: str, population_size: int = POPULATION_SIZE) -> list[dict]:
    """Seed from dynamic_prompts._CATEGORY_PROMPTS + empty prompt + random variants.

    Each seed is a dict with name, system_prompt, temperature, max_tokens.
    """
    population: list[dict] = []
    cat_prompts = _dp._CATEGORY_PROMPTS.get(category, {})

    # Add all complexity levels from dynamic_prompts as seed candidates
    for level_name, prompt_text in cat_prompts.items():
        population.append({
            "name": f"seed_{level_name}_{category}",
            "system_prompt": prompt_text,
            "temperature": 0.0,
            "max_tokens": MAX_TOKENS_DEFAULT,
        })

    # Add an empty prompt baseline (raw model behaviour)
    population.append({
        "name": f"seed_empty_{category}",
        "system_prompt": "",
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS_DEFAULT,
    })

    # Fill remaining slots with random variants
    while len(population) < population_size:
        population.append(
            _generate_random_variant(f"rand_{len(population)}_{category}", category)
        )

    return population[:population_size]


def _generate_random_variant(name: str, category: str) -> dict:
    """Create a random prompt from prefix + instruction + constraint pools."""
    prefixes = PREFIXES.get(category, [""])
    instructions = TASK_INSTRUCTIONS.get(category, [])
    prefix = random.choice(prefixes) if prefixes else ""

    parts: list[str] = []
    if prefix:
        parts.append(prefix)
    if instructions and random.random() < 0.7:
        parts.append(random.choice(instructions))

    num_c = random.randint(0, 2)
    c_text = ". ".join(
        random.sample(CONSTRAINT_POOL, min(num_c, len(CONSTRAINT_POOL)))
    )
    if c_text:
        parts.append(c_text)

    text = " ".join(parts).strip()
    if len(text) > MAX_PROMPT_CHARS:
        text = text[:MAX_PROMPT_CHARS].rsplit(" ", 1)[0]
        if not text.endswith("."):
            text += "."

    return {
        "name": name,
        "system_prompt": text,
        "temperature": random.choice(TEMP_OPTIONS),
        "max_tokens": MAX_TOKENS_DEFAULT,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Genetic Operators
# ═════════════════════════════════════════════════════════════════════════════


def _truncate_prompt(text: str, max_chars: int = MAX_PROMPT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    if not truncated.endswith("."):
        truncated += "."
    return truncated


def mutate(variant: dict, category: str) -> dict:
    """Apply one of 8 mutation operators to create a child variant."""
    child = {
        "name": variant["name"] + "_mut",
        "system_prompt": variant["system_prompt"],
        "temperature": variant["temperature"],
        "max_tokens": variant["max_tokens"],
    }

    op = random.randint(0, 7)
    text = child["system_prompt"]
    instructions = TASK_INSTRUCTIONS.get(category, [])
    task_prefixes = PREFIXES.get(category, [""])

    if op == 0:
        # (a) Rephrase using thesaurus
        for orig, replacements in THESAURUS.items():
            if orig in text:
                text = text.replace(orig, random.choice(replacements), 1)
                break

    elif op == 1:
        # (b) Add a task-specific instruction
        existing_inst = [c for c in instructions if c.lower() in text.lower()]
        available = [c for c in instructions if c not in existing_inst]
        if not available:
            available = CONSTRAINT_POOL
        if available:
            constraint = random.choice(available)
            if text and not text.endswith("."):
                text += "."
            text = text + " " + constraint

    elif op == 2:
        # (c) Remove the last sentence
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) > 1:
            sentences = sentences[:-1]
            text = " ".join(sentences)

    elif op == 3:
        # (d) Swap verbosity style
        swapped = False
        for s in SHORT_STYLES:
            if s in text:
                text = text.replace(s, random.choice(MEDIUM_STYLES), 1)
                swapped = True
                break
        if not swapped:
            for s in MEDIUM_STYLES:
                if s in text:
                    text = text.replace(s, random.choice(SHORT_STYLES), 1)
                    break

    elif op == 4:
        # (e) Change prefix
        current_prefix = None
        for p in task_prefixes:
            if p and text.startswith(p):
                current_prefix = p
                break
        options = [p for p in task_prefixes if p != current_prefix]
        if options:
            new_prefix = random.choice(options)
            if current_prefix:
                text = text.replace(current_prefix, new_prefix, 1)
            else:
                text = new_prefix + " " + text if new_prefix else text

    elif op == 5:
        # (f) Add/change word limit constraint
        n = random.choice([5, 10, 15, 20, 30])
        for old_n in [5, 10, 15, 20, 30]:
            old = f"Keep under {old_n} words."
            if old in text:
                text = text.replace(old, "").strip()
        if text and not text.endswith("."):
            text += "."
        text = text + f" Keep under {n} words."

    elif op == 6:
        # (g) Add preamble guard
        if "No preamble" not in text and "No commentary" not in text:
            if text and not text.endswith("."):
                text += "."
            text = text + " No preamble. No commentary."

    elif op == 7:
        # (h) Change temperature
        options = [t for t in TEMP_OPTIONS if t != child["temperature"]]
        if options:
            child["temperature"] = random.choice(options)

    child["system_prompt"] = re.sub(r"\s+", " ", text).strip()
    child["system_prompt"] = _truncate_prompt(child["system_prompt"])
    child["name"] = child["name"] + f"_op{op}"
    return child


def crossover_prompts(p1: dict, p2: dict) -> dict:
    """Interleave sentences from two parent prompts."""
    s1 = [s.strip() for s in re.split(r"(?<=[.!?])\s+", p1["system_prompt"]) if s.strip()]
    s2 = [s.strip() for s in re.split(r"(?<=[.!?])\s+", p2["system_prompt"]) if s.strip()]

    if not s1 and not s2:
        child_text = ""
    elif not s1:
        child_text = " ".join(s2)
    elif not s2:
        child_text = " ".join(s1)
    else:
        child: list[str] = []
        i = j = 0
        while i < len(s1) and j < len(s2):
            if random.random() < 0.5:
                child.append(s1[i])
                i += 1
            else:
                child.append(s2[j])
                j += 1
        remaining = s1[i:] if random.random() < 0.5 else s2[j:]
        child.extend(remaining)
        cleaned = [s.rstrip(".!?") for s in child]
        child_text = ". ".join(cleaned)
        if child and not child_text.endswith("."):
            child_text += "."

    return {
        "name": f"xover_{p1['name']}_{p2['name']}",
        "system_prompt": child_text,
        "temperature": p1["temperature"] if random.random() < 0.5 else p2["temperature"],
        "max_tokens": p1["max_tokens"] if random.random() < 0.5 else p2["max_tokens"],
    }


def tournament_select(
    population: list, rank_map: dict[int, int], k: int = TOURNAMENT_K
) -> dict:
    """Select individual via tournament (rank first, then crowding distance)."""
    if not population:
        raise ValueError("tournament_select called on empty population")
    best_idx = random.randrange(len(population))
    best = population[best_idx]
    best_rank = rank_map.get(best_idx, 999)
    best_crowd = best.get("crowding_dist", 0.0)

    for _ in range(k - 1):
        idx = random.randrange(len(population))
        ind = population[idx]
        r = rank_map.get(idx, 999)
        c = ind.get("crowding_dist", 0.0)
        if r < best_rank or (r == best_rank and c > best_crowd):
            best = ind
            best_rank = r
            best_crowd = c
    return best


def check_convergence(history: list, threshold: float = CONVERGENCE_THRESHOLD_PP) -> bool:
    """Return True if top-3 mean accuracy delta < threshold (in pp)."""
    if len(history) < 2:
        return False
    prev = history[-2].get("top3_mean", 0)
    curr = history[-1].get("top3_mean", 0)
    delta = abs(curr - prev) * 100
    logger.info("  Top-3 mean accuracy delta: %.2f pp (threshold: %.1f)", delta, threshold)
    return delta < threshold


# ═════════════════════════════════════════════════════════════════════════════
# Main GEPA Loop
# ═════════════════════════════════════════════════════════════════════════════


def run_pipeline_gepa(
    category: str = "factual",
    num_generations: int = NUM_GENERATIONS,
    population_size: int = POPULATION_SIZE,
    num_questions: Optional[int] = None,
) -> Optional[dict]:
    """Run pipeline-integrated GEPA evolution for *category*.

    Args:
        category: Task category (e.g., "factual", "math", "logic").
        num_generations: Number of evolutionary generations.
        population_size: Number of candidates per generation.
        num_questions: If set, use a random subset of this many questions.

    Returns:
        Summary dict, or None on fatal error.
    """
    print("=" * 70)
    print("Pipeline-Integrated GEPA Evolution")
    print(f"  Category:     {category}")
    print(f"  Model:        {DEFAULT_MODEL_PATH}")
    print(f"  Generations:  {num_generations}")
    print(f"  Population:   {population_size}")
    print("=" * 70)

    # ── 1. Load dataset ────────────────────────────────────────────────────
    print(f"\n[1] Loading training data …")
    with open(TRAINING_DATA) as f:
        all_data = json.load(f)
    questions = [q for q in all_data if q.get("category") == category]
    print(f"    Found {len(questions)} questions for '{category}'")

    if num_questions is not None and 0 < num_questions < len(questions):
        random.shuffle(questions)
        questions = questions[:num_questions]
        print(f"    Using subset: {num_questions} questions")

    if not questions:
        print(f"    ✗ No questions found for category '{category}'.")
        return None

    # ── 2. Build seed population ──────────────────────────────────────────
    print(f"\n[2] Building seed population ({population_size} variants) …")
    population = build_seed_population(category, population_size)
    for i, v in enumerate(population):
        d = v["system_prompt"][:80] if v["system_prompt"] else "(empty)"
        print(f"    [{i}] {v['name']}: {d!r}  (t={v['temperature']})")

    global_best: dict = {"variant": None, "accuracy": -1.0, "generation": 0}
    history: list[dict] = []
    all_generations_data: list[dict] = []
    fronts: list[list[int]] = []  # defined here so it's bound for the final summary

    # ── 3. Generation loop ────────────────────────────────────────────────
    for gen in range(num_generations):
        print(f"\n{'=' * 70}")
        print(f"Generation {gen + 1} / {num_generations}")
        print(f"{'=' * 70}")

        # ── 3a. Evaluate all candidates through the full Pipeline ─────────
        print(f"\n[3a] Evaluating {len(population)} candidates through Pipeline …")

        # Strip previous fitness keys
        for v in population:
            for key in (
                "accuracy",
                "avg_output_tokens",
                "avg_latency_ms",
                "acc_norm",
                "tokens_norm",
                "lat_norm",
                "crowding_dist",
                "rank",
                "per_question",
                "total_correct",
                "total_questions",
            ):
                v.pop(key, None)

        for i, v in enumerate(population):
            d = v["system_prompt"][:60] if v["system_prompt"] else "(empty)"
            t0 = time.time()
            result = evaluate_candidate(v, category, questions)
            elapsed = time.time() - t0

            v["accuracy"] = result["accuracy"]
            v["avg_output_tokens"] = result["avg_output_tokens"]
            v["avg_latency_ms"] = result["avg_latency_ms"]
            v["per_question"] = result["per_question"]
            v["total_correct"] = result["total_correct"]
            v["total_questions"] = result["total_questions"]

            print(
                f"    [{i+1}/{len(population)}] {v['name']}: "
                f"acc={result['accuracy']:.3f}  "
                f"lat={result['avg_latency_ms']:.0f}ms  "
                f"tok={result['avg_output_tokens']:.0f}  "
                f"({elapsed:.1f}s)"
            )

        # ── 3b. Normalize & Pareto sort ──────────────────────────────────
        print(f"\n[3b] Computing Pareto fronts …")
        normalize_objectives(population)
        fronts = fast_non_dominated_sort(population)
        print(f"    Front sizes: {[len(f) for f in fronts]}")

        # Crowding distance on front 0
        if fronts:
            dists = crowding_distance(population, fronts[0])
            for pos, idx in enumerate(fronts[0]):
                population[idx]["crowding_dist"] = dists[pos]
        for idx in range(len(population)):
            if "crowding_dist" not in population[idx]:
                population[idx]["crowding_dist"] = 0.0

        # Build rank map
        rank_map: dict[int, int] = {}
        for rank_idx, front in enumerate(fronts):
            for idx in front:
                rank_map[idx] = rank_idx

        # ── 3c. Track best ───────────────────────────────────────────────
        print(f"\n[3c] Best accuracy:")
        top3_accs: list[float] = []
        for v in population:
            acc = v["accuracy"]
            top3_accs.append(acc)
            if acc > global_best["accuracy"]:
                global_best = {
                    "variant": dict(v),
                    "accuracy": acc,
                    "generation": gen + 1,
                }

        top3_accs.sort(reverse=True)
        top3_mean = sum(top3_accs[:3]) / min(3, len(top3_accs))
        best_v = max(population, key=lambda x: x["accuracy"])
        print(
            f"    Best: acc={best_v['accuracy']:.3f}  "
            f"\"{best_v['system_prompt'][:80]}\""
        )
        print(f"    Top-3 mean: {top3_mean:.4f}")

        gen_record = {
            "generation": gen + 1,
            "top3_mean": top3_mean,
            "best_accuracy": best_v["accuracy"],
            "best_prompt": best_v["system_prompt"],
            "pareto_front_0_size": len(fronts[0]) if fronts else 0,
        }
        history.append(gen_record)

        # ── 3d. Save generation snapshot ──────────────────────────────────
        gen_data: dict = {
            "generation": gen + 1,
            "population": [
                {
                    "name": v["name"],
                    "system_prompt": v["system_prompt"],
                    "temperature": v["temperature"],
                    "accuracy": v["accuracy"],
                    "avg_output_tokens": v["avg_output_tokens"],
                    "avg_latency_ms": v["avg_latency_ms"],
                    "acc_norm": v["acc_norm"],
                    "tokens_norm": v["tokens_norm"],
                    "lat_norm": v["lat_norm"],
                    "crowding_dist": v.get("crowding_dist", 0.0),
                }
                for v in population
            ],
            "pareto_front_0_indices": fronts[0] if fronts else [],
        }
        all_generations_data.append(gen_data)

        # ── 3e. Check convergence ─────────────────────────────────────────
        if len(history) >= 2:
            converged = check_convergence(history)
            if converged:
                print(f"\n    >>> Converged at generation {gen + 1}. Stopping early.")
                break

        # ── 3f. Create next generation ────────────────────────────────────
        if gen == num_generations - 1:
            print(f"\n    Final generation reached.")
            break

        print(f"\n[3f] Creating next generation …")
        next_population: list[dict] = []

        # Elitism: keep top-ranked (lowest rank number → best front)
        ranked = sorted(
            [
                (
                    idx,
                    rank_map.get(idx, 999),
                    -population[idx].get("crowding_dist", 0.0),
                )
                for idx in range(len(population))
            ],
            key=lambda x: (x[1], x[2]),
        )
        for idx, _, _ in ranked[:ELITE_COUNT]:
            elite = dict(population[idx])
            elite["name"] = f"elite_g{gen}_{idx}_{category}"
            next_population.append(elite)
        print(f"    Elites: {[e['name'] for e in next_population]}")

        while len(next_population) < population_size - 1:
            if len(next_population) < population_size * 0.6:
                # Crossover + mutation
                p1 = tournament_select(population, rank_map, TOURNAMENT_K)
                p2 = tournament_select(population, rank_map, TOURNAMENT_K)
                child = crossover_prompts(p1, p2)
                child = mutate(child, category)
                child["name"] = f"gen{gen}_c{len(next_population)}_{category}"
            else:
                # Pure mutation
                parent = tournament_select(population, rank_map, TOURNAMENT_K)
                child = mutate(parent, category)
                child["name"] = f"gen{gen}_m{len(next_population)}_{category}"
            next_population.append(child)

        # Fresh random variant for diversity
        fresh = _generate_random_variant(f"fresh_gen{gen}_{category}", category)
        next_population.append(fresh)
        print(f"    Fresh variant: {fresh['name']}")

        population = next_population
        print(f"    Next population size: {len(population)}")

    # ── 4. Final Summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"FINAL SUMMARY — Category: {category}")
    print(f"{'=' * 70}")

    final_results: dict = {}
    if global_best["variant"] is not None:
        bv = global_best["variant"]
        print(f"\n  Global best (gen {global_best['generation']}):")
        print(f"    Accuracy:      {global_best['accuracy']:.3f}")
        print(f"    Prompt:        {bv['system_prompt']!r}")
        print(f"    Temperature:   {bv['temperature']}")
        print(f"    Avg latency:   {bv.get('avg_latency_ms', 0):.0f} ms")
        print(f"    Avg tokens:    {bv.get('avg_output_tokens', 0):.0f}")
        final_results = {
            "best_accuracy": global_best["accuracy"],
            "best_prompt": bv["system_prompt"],
            "temperature": bv["temperature"],
            "max_tokens": bv["max_tokens"],
            "avg_output_tokens": bv.get("avg_output_tokens", 0),
            "avg_latency_ms": bv.get("avg_latency_ms", 0.0),
            "variant_name": bv["name"],
            "best_generation": global_best["generation"],
        }
    else:
        # Fallback: current population best
        bv = max(population, key=lambda v: v["accuracy"])
        print(f"\n  Best in final population (fallback):")
        print(f"    Accuracy:      {bv['accuracy']:.3f}")
        print(f"    Prompt:        {bv['system_prompt']!r}")
        final_results = {
            "best_accuracy": bv["accuracy"],
            "best_prompt": bv["system_prompt"],
            "temperature": bv["temperature"],
            "max_tokens": bv["max_tokens"],
            "avg_output_tokens": bv.get("avg_output_tokens", 0),
            "avg_latency_ms": bv.get("avg_latency_ms", 0.0),
            "variant_name": bv["name"],
            "best_generation": num_generations,
        }

    # Pareto-optimal prompts from final population
    pareto_indices = fronts[0] if fronts else list(range(len(population)))
    pareto_prompts = []
    for idx in pareto_indices:
        v = population[idx]
        pareto_prompts.append({
            "name": v["name"],
            "system_prompt": v["system_prompt"],
            "temperature": v["temperature"],
            "max_tokens": v["max_tokens"],
            "accuracy": v["accuracy"],
            "avg_output_tokens": v["avg_output_tokens"],
            "avg_latency_ms": v["avg_latency_ms"],
            "acc_norm": v["acc_norm"],
            "tokens_norm": v["tokens_norm"],
            "lat_norm": v["lat_norm"],
        })

    # ── 5. Save results ─────────────────────────────────────────────────────
    output: dict = {
        "category": category,
        "model_path": DEFAULT_MODEL_PATH,
        "num_questions": len(questions),
        "num_generations": len(all_generations_data),
        "population_size": population_size,
        "seed_prompts": {
            level: text
            for level, text in _dp._CATEGORY_PROMPTS.get(category, {}).items()
        },
        "history": history,
        "final_results": final_results,
        "pareto_optimal_prompts": pareto_prompts,
        "generations": all_generations_data,
    }

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    out_path = Path(RESULTS_DIR) / f"pipeline_{category}_gepa_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")

    return output


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline-Integrated GEPA Prompt Evolution"
    )
    parser.add_argument(
        "--category",
        default="factual",
        choices=[
            "factual", "math", "logic", "sentiment",
            "ner", "summarization", "code_gen", "code_debug",
        ],
        help="Task category (default: factual)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=NUM_GENERATIONS,
        help=f"Number of evolutionary generations (default: {NUM_GENERATIONS})",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=POPULATION_SIZE,
        help=f"Population size per generation (default: {POPULATION_SIZE})",
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=None,
        help="Number of training questions to use (default: all for category)",
    )
    args = parser.parse_args()

    print(f"\n{'#' * 70}")
    print(f"# Pipeline GEPA: {args.category.upper()}")
    print(f"# Generations: {args.generations}  Population: {args.population}")
    print(f"{'#' * 70}")

    result = run_pipeline_gepa(
        category=args.category,
        num_generations=args.generations,
        population_size=args.population,
        num_questions=args.questions,
    )

    if result:
        fr = result.get("final_results", {})
        print(f"\n{'=' * 70}")
        print(f"Pipeline GEPA Complete — {args.category}")
        print(f"  Best accuracy: {fr.get('best_accuracy', '?'):.3f}")
        print(f"  Prompt: {fr.get('best_prompt', '?')!r}")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
