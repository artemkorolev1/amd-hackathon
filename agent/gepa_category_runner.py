#!/usr/bin/env python3
"""
GEPA Category Runner — runs Pareto-based prompt evolution for any task category.
Usage: python3 agent/gepa_category_runner.py --category sentiment --model qwen2.5-1.5b [--questions 10] [--generations 3]
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from collections import defaultdict

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_PATHS = {
    "qwen2.5-1.5b": "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "qwen2.5-math-1.5b": "/home/artem/models/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf",
    "qwen2.5-coder-1.5b": "/home/artem/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    "smollm2-1.7b": "/home/artem/models/smollm2-1.7b-instruct-q4_k_m.gguf",
    "qwen2.5-base": "/home/artem/models/qwen2.5-1.5b-base-q4_k_m.gguf",
}

TRAINING_DATA = "/home/artem/dev/amd-hackathon/data/eval/training-v3.json"
RESULTS_DIR = "/home/artem/dev/amd-hackathon/gepa_plans"

POPULATION_SIZE = 8
NUM_GENERATIONS = 3
NUM_QUESTIONS = 19
ELITE_COUNT = 2
TOURNAMENT_K = 3
MAX_TOKENS_DEFAULT = 128
TEMPERATURE_DEFAULT = 0.0
MAX_PROMPT_CHARS = 300
CONVERGENCE_THRESHOLD_PP = 5.0
TEMP_OPTIONS = [0.0, 0.1, 0.2]

# ── Task-specific seed prompts ───────────────────────────────────────────────

SEED_PROMPTS = {
    "sentiment": [
        "",
        "Classify the sentiment. Output exactly one word: POSITIVE or NEGATIVE.",
        "Sentiment:",
        "Classify the sentiment as positive or negative. Output only the label.",
        "Analyze the emotional tone. Output exactly: POSITIVE or NEGATIVE.",
    ],
    "ner": [
        "",
        "Extract entities from the text. Format: category: entity",
        "Entities:",
        "Extract named entities with labels: person, org, loc, date.",
        "List all named entities in the format: TYPE: value",
    ],
    "code_gen": [
        "",
        "Write Python code for the task. Output only the code.",
        "Implement the following requirement in Python.",
        "Write a Python function that solves the problem. Return only the code.",
        "Complete the Python code as requested.",
    ],
    "summarization": [
        "",
        "Summarize the text in 1-2 sentences. Use exact names and numbers.",
        "Summary:",
        "Provide a concise summary capturing the core event. Be specific.",
        "TL;DR: Give a brief summary with key details.",
    ],
    "factual": [
        "",
        "Answer the question. Use exact names, dates, and numbers.",
        "Fact:",
        "Answer directly with precise factual information.",
        "Provide a concise factual answer to the question.",
    ],
    "math": [
        "",
        "Solve the math problem. Output only the answer.",
        "Math:",
        "Calculate the answer. Be precise with numbers.",
        "Compute the result step by step, then output the answer.",
    ],
    "logic": [
        "",
        "Solve the logic puzzle. Output only the final answer.",
        "Logic:",
        "Think through the reasoning step by step, then output the solution.",
        "Apply logical deduction to find the answer. Be precise.",
    ],
    "code_debug": [
        "",
        "Find and fix the bug in the Python code. Output only the corrected code.",
        "Debug:",
        "Identify the error in the code and provide the corrected version.",
        "Fix the bug. Output only the corrected function. Preserve the original signature.",
    ],
}

# ── Task-specific instructions/mutation pools ────────────────────────────────

TASK_INSTRUCTIONS = {
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
    "code_gen": [
        "Write clean, working Python code.",
        "Include necessary imports.",
        "Use appropriate data structures.",
        "Handle edge cases.",
        "Return a function definition.",
        "Comment the code briefly.",
    ],
    "summarization": [
        "Summarize in 1-2 sentences.",
        "Use exact names and numbers.",
        "Capture the core event only.",
        "Be concise and specific.",
        "Include who, what, when.",
    ],
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
    "code_debug": [
        "Find the exact bug.",
        "Fix without changing the function signature.",
        "Preserve original functionality.",
        "Only output the corrected code.",
        "Check for off-by-one errors.",
    ],
}

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

PREFIXES = {
    "sentiment": ["", "Sentiment:", "Classify:", "Label:"],
    "ner": ["", "Entities:", "NER:", "Extract:"],
    "code_gen": ["", "Code:", "Implement:", "Python:"],
    "summarization": ["", "Summary:", "TL;DR:", "Brief:", "Concisely:"],
    "factual": ["", "Fact:", "Answer:", "Q:", "Answer directly:"],
    "math": ["", "Math:", "Solve:", "Answer:", "Calculate:"],
    "logic": ["", "Logic:", "Solve:", "Answer:", "Reason:"],
    "code_debug": ["", "Debug:", "Fix:", "Code:", "Corrected:"],
}

THESAURUS = {
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

SHORT_STYLES = ["Be concise.", "Be brief.", "Keep it short."]
MEDIUM_STYLES = ["Answer clearly.", "Provide a clear answer.", "Respond with the answer."]


# ── Fuzzy Matching ───────────────────────────────────────────────────────────

def fuzzy_match(answer: str, expected: str) -> bool:
    a, e = answer.strip().lower(), expected.strip().lower()
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


# ── Model Cache ──────────────────────────────────────────────────────────────

class ModelCache:
    def __init__(self):
        self._models = {}

    def get(self, model_key: str):
        if model_key not in self._models:
            path = MODEL_PATHS.get(model_key)
            if not path:
                raise ValueError(f"Unknown model key: {model_key}")
            from llama_cpp import Llama
            print(f"  [model] Loading {model_key} from {path} ...")
            self._models[model_key] = Llama(
                model_path=path,
                n_ctx=2048,
                n_gpu_layers=-1,
                n_threads=4,
                verbose=False,
            )
            print(f"  [model] {model_key} loaded.")
        return self._models[model_key]

    def clear(self):
        self._models.clear()


_model_cache = ModelCache()


# ── Pareto Functions ─────────────────────────────────────────────────────────

def pareto_dominates(a: dict, b: dict) -> bool:
    objs = ["acc_norm", "tokens_norm", "lat_norm"]
    better_or_eq = all(a[o] >= b[o] - 1e-10 for o in objs)
    strictly_better = any(a[o] > b[o] + 1e-10 for o in objs)
    return better_or_eq and strictly_better


def fast_non_dominated_sort(population: list) -> list:
    n = len(population)
    S = [set() for _ in range(n)]
    n_count = [0] * n
    fronts = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if pareto_dominates(population[i], population[j]):
                S[i].add(j)
            elif pareto_dominates(population[j], population[i]):
                n_count[i] += 1
    current_front = [i for i in range(n) if n_count[i] == 0]
    if not current_front:
        return [list(range(n))]
    fronts.append(current_front)
    while current_front:
        next_front = []
        for i in current_front:
            for j in S[i]:
                n_count[j] -= 1
                if n_count[j] == 0:
                    next_front.append(j)
        if not next_front:
            break
        fronts.append(next_front)
        current_front = next_front
    return fronts


def crowding_distance(population: list, front: list,
                      objectives=("acc_norm", "tokens_norm", "lat_norm")) -> list:
    if len(front) <= 2:
        return [float("inf")] * len(front)
    dist = [0.0] * len(front)
    for obj in objectives:
        front_sorted = sorted(range(len(front)), key=lambda idx: population[front[idx]][obj])
        obj_min = population[front[front_sorted[0]]][obj]
        obj_max = population[front[front_sorted[-1]]][obj]
        denom = obj_max - obj_min + 1e-9
        dist[front_sorted[0]] = float("inf")
        dist[front_sorted[-1]] = float("inf")
        for pos in range(1, len(front_sorted) - 1):
            i_prev = front[front_sorted[pos - 1]]
            i_next = front[front_sorted[pos + 1]]
            delta = (population[i_next][obj] - population[i_prev][obj]) / denom
            dist[front_sorted[pos]] += delta
    return dist


def normalize_objectives(variants: list, model_key: str):
    accs = [v.get("accuracy", {}).get(model_key, 0.0) for v in variants]
    tokens = [v.get("avg_output_tokens", {}).get(model_key, 0.0) for v in variants]
    lats = [v.get("avg_latency_ms", {}).get(model_key, 0.0) for v in variants]
    min_acc, max_acc = min(accs), max(accs)
    min_tok, max_tok = min(tokens), max(tokens)
    min_lat, max_lat = min(lats), max(lats)
    for v in variants:
        acc = v.get("accuracy", {}).get(model_key, 0.0)
        tok = v.get("avg_output_tokens", {}).get(model_key, 0.0)
        lat = v.get("avg_latency_ms", {}).get(model_key, 0.0)
        if "acc_norm" not in v:
            v["acc_norm"] = {}
        if "tokens_norm" not in v:
            v["tokens_norm"] = {}
        if "lat_norm" not in v:
            v["lat_norm"] = {}
        v["acc_norm"][model_key] = (acc - min_acc) / (max_acc - min_acc + 1e-9)
        v["tokens_norm"][model_key] = 1.0 - (tok - min_tok) / (max_tok - min_tok + 1e-9)
        v["lat_norm"][model_key] = 1.0 - (lat - min_lat) / (max_lat - min_lat + 1e-9)


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_variant_on_model(variant: dict, model_key: str, questions: list) -> dict:
    try:
        llm = _model_cache.get(model_key)
    except Exception as e:
        print(f"    [ERROR] Failed to load model {model_key}: {e}")
        return {"accuracy": 0.0, "avg_output_tokens": 999, "avg_latency_ms": 99999.0}

    correct = 0
    total_tokens = 0
    total_latency = 0.0
    system_prompt = variant.get("system_prompt", "")
    temperature = variant.get("temperature", TEMPERATURE_DEFAULT)
    max_tokens = variant.get("max_tokens", MAX_TOKENS_DEFAULT)
    top_p = variant.get("top_p", 1.0)
    top_k = variant.get("top_k", 40)
    min_p = variant.get("min_p", 0.0)
    repeat_penalty = variant.get("repeat_penalty", 1.0)

    for idx, q in enumerate(questions):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q["prompt"]},
        ]
        start = time.time()
        try:
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repeat_penalty=repeat_penalty,
            )
            elapsed = (time.time() - start) * 1000
            answer = response["choices"][0]["message"]["content"].strip()
            usage = response.get("usage", {})
            tok_count = usage.get("completion_tokens", len(answer.split()))

            if fuzzy_match(answer, q["expected_answer"]):
                correct += 1

            total_tokens += tok_count
            total_latency += elapsed

            if idx % 10 == 0 and idx > 0:
                print(f"      [{model_key}] {idx}/{len(questions)} questions done ...")

        except Exception as e:
            print(f"      [{model_key}] Question {idx} failed: {e}")
            total_tokens += 0
            total_latency += 100

    accuracy = correct / len(questions) if questions else 0.0
    avg_tok = total_tokens / len(questions) if questions else 0
    avg_lat = total_latency / len(questions) if questions else 0.0

    return {"accuracy": accuracy, "avg_output_tokens": avg_tok, "avg_latency_ms": avg_lat}


def evaluate_population(population: list, questions: list, model_keys: list):
    for i, variant in enumerate(population):
        print(f"  Evaluating variant {i+1}/{len(population)}: '{variant['name']}'")
        for mk in model_keys:
            result = evaluate_variant_on_model(variant, mk, questions)
            if "accuracy" not in variant:
                variant["accuracy"] = {}
            if "avg_output_tokens" not in variant:
                variant["avg_output_tokens"] = {}
            if "avg_latency_ms" not in variant:
                variant["avg_latency_ms"] = {}
            variant["accuracy"][mk] = result["accuracy"]
            variant["avg_output_tokens"][mk] = result["avg_output_tokens"]
            variant["avg_latency_ms"][mk] = result["avg_latency_ms"]
    for mk in model_keys:
        normalize_objectives(population, mk)


# ── Population Generation ────────────────────────────────────────────────────

def generate_random_variant(name: str, category: str) -> dict:
    prefixes = PREFIXES.get(category, [""])
    instructions = TASK_INSTRUCTIONS.get(category, [])
    prefix = random.choice(prefixes) if prefixes else ""

    parts = []
    if prefix:
        parts.append(prefix)

    if instructions and random.random() < 0.7:
        parts.append(random.choice(instructions))

    num_constraints = random.randint(0, 2)
    constraint_text = ". ".join(random.sample(CONSTRAINT_POOL, min(num_constraints, len(CONSTRAINT_POOL))))
    if constraint_text:
        parts.append(constraint_text)

    system_prompt = " ".join(parts).strip()
    if len(system_prompt) > MAX_PROMPT_CHARS:
        system_prompt = system_prompt[:MAX_PROMPT_CHARS].rsplit(" ", 1)[0]
        if not system_prompt.endswith("."):
            system_prompt += "."

    return {
        "name": name,
        "system_prompt": system_prompt,
        "temperature": random.choice(TEMP_OPTIONS),
        "max_tokens": MAX_TOKENS_DEFAULT,
    }


def build_seed_population(category: str) -> list:
    seeds = SEED_PROMPTS.get(category, [""])
    population = []
    for i, prompt_text in enumerate(seeds):
        population.append({
            "name": f"seed_{i}_{category}",
            "system_prompt": prompt_text,
            "temperature": 0.0,
            "max_tokens": MAX_TOKENS_DEFAULT,
        })
    while len(population) < POPULATION_SIZE:
        population.append(generate_random_variant(f"rand_{len(population)}_{category}", category))
    return population


# ── Genetic Operators ────────────────────────────────────────────────────────

def _truncate_prompt(text: str, max_chars: int = MAX_PROMPT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    if not truncated.endswith("."):
        truncated += "."
    return truncated


def mutate(variant: dict, category: str) -> dict:
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
        # (a) Rephrase
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
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) > 1:
            sentences = sentences[:-1]
            text = " ".join(sentences)

    elif op == 3:
        # (d) Swap verbosity
        for s in SHORT_STYLES:
            if s in text:
                text = text.replace(s, random.choice(MEDIUM_STYLES), 1)
                break
        else:
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
        # (f) Add "Keep under N words"
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

    text = re.sub(r'\s+', ' ', text).strip()
    text = _truncate_prompt(text)
    child["system_prompt"] = text
    child["name"] = child["name"] + f"_op{op}"

    return child


def crossover_prompts(p1: dict, p2: dict) -> dict:
    s1 = [s.strip() for s in re.split(r'(?<=[.!?])\s+', p1["system_prompt"]) if s.strip()]
    s2 = [s.strip() for s in re.split(r'(?<=[.!?])\s+', p2["system_prompt"]) if s.strip()]

    if not s1 and not s2:
        child_text = ""
    elif not s1:
        child_text = " ".join(s2)
    elif not s2:
        child_text = " ".join(s1)
    else:
        child = []
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

    temp = p1["temperature"] if random.random() < 0.5 else p2["temperature"]
    mt = p1["max_tokens"] if random.random() < 0.5 else p2["max_tokens"]

    return {
        "name": f"xover_{p1['name']}_{p2['name']}",
        "system_prompt": child_text,
        "temperature": temp,
        "max_tokens": mt,
    }


def tournament_select(population: list, model_rank_map: dict, model_keys: list, k: int = 3) -> dict:
    if not population:
        raise ValueError("tournament_select called on empty population")
    best_idx = random.randrange(len(population))
    best = population[best_idx]
    best_rank_sum = 0.0
    best_crowd = 0.0
    for mk in model_keys:
        rm = model_rank_map.get(mk, {})
        best_rank_sum += rm.get(best_idx, 999)
        crowd = best.get("crowding_dist", {}).get(mk, 0.0)
        best_crowd += crowd

    for _ in range(k - 1):
        idx = random.randrange(len(population))
        ind = population[idx]
        rank_sum = 0.0
        crowd_sum = 0.0
        for mk in model_keys:
            rm = model_rank_map.get(mk, {})
            rank_sum += rm.get(idx, 999)
            crowd = ind.get("crowding_dist", {}).get(mk, 0.0)
            crowd_sum += crowd

        if rank_sum < best_rank_sum or (rank_sum == best_rank_sum and crowd_sum > best_crowd):
            best = ind
            best_rank_sum = rank_sum
            best_crowd = crowd_sum

    return best


def check_convergence(history: list, threshold: float = CONVERGENCE_THRESHOLD_PP) -> bool:
    if len(history) < 2:
        return False
    prev = history[-2].get("top3_mean", 0)
    curr = history[-1].get("top3_mean", 0)
    delta = abs(curr - prev) * 100
    print(f"  Top-3 mean accuracy delta: {delta:.2f} pp (threshold: {threshold})")
    return delta < threshold


# ── Main GEPA Loop ───────────────────────────────────────────────────────────

def run_gepa(category: str, model_keys: list, num_generations: int = NUM_GENERATIONS,
             num_questions: int = None):
    print("=" * 70)
    print(f"GEPA: Genetic Pareto Algorithm Prompt Optimizer")
    print(f"Category: {category}")
    print("=" * 70)

    # 1. Load dataset
    print(f"\n[1] Loading dataset ...")
    with open(TRAINING_DATA, "r") as f:
        all_data = json.load(f)

    questions = [q for q in all_data if q.get("category") == category]
    if not questions:
        # Try from combined files
        combined_map = {
            "sentiment": "/home/artem/dev/amd-hackathon/data/eval/sentiment_combined_25.json",
            "summarization": "/home/artem/dev/amd-hackathon/data/eval/summarization_combined_25.json",
        }
        if category in combined_map:
            with open(combined_map[category]) as f:
                questions = json.load(f)

    print(f"    Loaded {len(questions)} questions for category '{category}'.")

    if num_questions and num_questions < len(questions):
        random.shuffle(questions)
        questions = questions[:num_questions]
        print(f"    Using subset: {num_questions} questions.")

    if not questions:
        print(f"    ERROR: No questions found for category '{category}'.")
        return None

    # 2. Build seed population
    print(f"\n[2] Building seed population ({POPULATION_SIZE} variants) ...")
    population = build_seed_population(category)
    for i, v in enumerate(population):
        prompt_display = v["system_prompt"][:80] if v["system_prompt"] else "(empty)"
        print(f"    [{i}] {v['name']}: \"{prompt_display}\" (temp={v['temperature']})")

    global_best_per_model = {mk: {"variant": None, "accuracy": -1.0} for mk in model_keys}

    # 3. Generation loop
    history = []
    all_generations_data = []

    for gen in range(num_generations):
        print(f"\n{'=' * 70}")
        print(f"Generation {gen + 1} / {num_generations}")
        print(f"{'=' * 70}")

        # 3a. Evaluate
        print("\n[3a] Evaluating population ...")
        for v in population:
            for key in ["accuracy", "avg_output_tokens", "avg_latency_ms",
                        "acc_norm", "tokens_norm", "lat_norm", "crowding_dist", "rank"]:
                v.pop(key, None)

        evaluate_population(population, questions, model_keys)

        # 3b. Pareto fronts
        print("\n[3b] Computing Pareto fronts per model ...")
        model_rank_map = {}
        all_fronts = {}

        for mk in model_keys:
            model_pop = []
            for idx, v in enumerate(population):
                model_pop.append({
                    "acc_norm": v.get("acc_norm", {}).get(mk, 0.0),
                    "tokens_norm": v.get("tokens_norm", {}).get(mk, 0.0),
                    "lat_norm": v.get("lat_norm", {}).get(mk, 0.0),
                    "idx": idx,
                })
            fronts = fast_non_dominated_sort(model_pop)
            all_fronts[mk] = fronts
            rank_map = {}
            for rank_idx, front in enumerate(fronts):
                dists = crowding_distance(model_pop, front)
                for pos, orig_idx in enumerate(front):
                    rank_map[orig_idx] = rank_idx
                    if "crowding_dist" not in population[orig_idx]:
                        population[orig_idx]["crowding_dist"] = {}
                    population[orig_idx]["crowding_dist"][mk] = dists[pos]
            model_rank_map[mk] = rank_map
            print(f"    {mk}: Front 0 size = {len(fronts[0]) if fronts else 0}")

        # 3c. Update best tracking
        print("\n[3c] Per-model best accuracy:")
        gen_top3_accs = []
        for mk in model_keys:
            best_acc = -1.0
            best_variant = None
            for v in population:
                acc = v.get("accuracy", {}).get(mk, 0.0)
                if acc > best_acc:
                    best_acc = acc
                    best_variant = v
            if best_acc > global_best_per_model[mk]["accuracy"]:
                global_best_per_model[mk] = {
                    "variant": dict(best_variant) if best_variant else None,
                    "accuracy": best_acc,
                    "generation": gen + 1,
                }
            print(f"    {mk}: best accuracy = {best_acc:.3f}  ({best_variant['name'] if best_variant else 'N/A'})")
            if best_variant and best_variant["system_prompt"]:
                print(f"      prompt: \"{best_variant['system_prompt'][:120]}\"")
            accs_sorted = sorted(
                [v.get("accuracy", {}).get(mk, 0.0) for v in population],
                reverse=True
            )
            gen_top3_accs.extend(accs_sorted[:3])

        top3_mean = sum(gen_top3_accs) / len(gen_top3_accs) if gen_top3_accs else 0.0

        gen_record = {
            "generation": gen + 1,
            "top3_mean": top3_mean,
            "per_model_best": {mk: global_best_per_model[mk]["accuracy"] for mk in model_keys},
            "pareto_front_sizes": {mk: len(all_fronts[mk][0]) if mk in all_fronts and all_fronts[mk] else 0 for mk in model_keys},
        }
        history.append(gen_record)

        print(f"\n    Pareto front sizes: {gen_record['pareto_front_sizes']}")
        print(f"    Top-3 mean accuracy: {top3_mean:.4f}")

        # Save generation data
        gen_data = {
            "generation": gen + 1,
            "population": [
                {
                    "name": v["name"],
                    "system_prompt": v["system_prompt"],
                    "temperature": v["temperature"],
                    "accuracy": v.get("accuracy", {}),
                    "avg_output_tokens": v.get("avg_output_tokens", {}),
                    "avg_latency_ms": v.get("avg_latency_ms", {}),
                }
                for v in population
            ],
            "pareto_front_sizes": gen_record["pareto_front_sizes"],
            "per_model_best": gen_record["per_model_best"],
        }
        all_generations_data.append(gen_data)

        # 3f. Check convergence
        if len(history) >= 2:
            converged = check_convergence(history)
            if converged:
                print(f"\n  >>> Converged at generation {gen + 1}. Stopping early.")
                break

        # 3g. Create next generation
        if gen == num_generations - 1:
            print("\n  Final generation reached.")
            break

        print("\n[3g] Creating next generation ...")
        next_population = []

        ranked = sorted(
            [(idx, sum(model_rank_map.get(mk, {}).get(idx, 999) for mk in model_keys))
             for idx in range(len(population))],
            key=lambda x: (x[1], -sum(population[x[0]].get("crowding_dist", {}).get(mk, 0.0) for mk in model_keys))
        )
        for idx, _ in ranked[:ELITE_COUNT]:
            elite = dict(population[idx])
            elite["name"] = f"elite_g{gen}_{idx}_{category}"
            next_population.append(elite)
        print(f"    Elites: {[e['name'] for e in next_population]}")

        while len(next_population) < POPULATION_SIZE - 1:
            if len(next_population) < POPULATION_SIZE * 0.6:
                p1 = tournament_select(population, model_rank_map, model_keys, TOURNAMENT_K)
                p2 = tournament_select(population, model_rank_map, model_keys, TOURNAMENT_K)
                child = crossover_prompts(p1, p2)
                child = mutate(child, category)
                child["name"] = f"gen{gen}_c{len(next_population)}_{category}"
            else:
                parent = tournament_select(population, model_rank_map, model_keys, TOURNAMENT_K)
                child = mutate(parent, category)
                child["name"] = f"gen{gen}_m{len(next_population)}_{category}"
            next_population.append(child)

        fresh = generate_random_variant(f"fresh_gen{gen}_{category}", category)
        next_population.append(fresh)
        print(f"    Fresh variant: {fresh['name']}")

        population = next_population
        print(f"    Next population size: {len(population)}")

    # 4. Final Summary
    print(f"\n{'=' * 70}")
    print(f"FINAL SUMMARY — Category: {category}")
    print(f"{'=' * 70}")

    final_results = {}
    for mk in model_keys:
        best_info = global_best_per_model[mk]
        best_v = best_info["variant"]
        best_acc = best_info["accuracy"]

        if best_v:
            avg_tok = best_v.get("avg_output_tokens", {}).get(mk, 0)
            avg_lat = best_v.get("avg_latency_ms", {}).get(mk, 0.0)
            gen_found = best_info.get("generation", "?")
            print(f"\n  Model: {mk} (best from gen {gen_found})")
            print(f"    Best accuracy:    {best_acc:.3f}")
            print(f"    Best prompt:      \"{best_v['system_prompt']}\"")
            print(f"    Temperature:      {best_v['temperature']}")
            print(f"    Avg output tokens: {avg_tok:.1f}")
            print(f"    Avg latency (ms): {avg_lat:.1f}")
            final_results[mk] = {
                "best_accuracy": best_acc,
                "best_prompt": best_v["system_prompt"],
                "temperature": best_v["temperature"],
                "max_tokens": best_v["max_tokens"],
                "avg_output_tokens": avg_tok,
                "avg_latency_ms": avg_lat,
                "variant_name": best_v["name"],
                "best_generation": gen_found,
            }
        else:
            best_v_final = None
            best_acc_final = -1.0
            for v in population:
                acc = v.get("accuracy", {}).get(mk, 0.0)
                if acc > best_acc_final:
                    best_acc_final = acc
                    best_v_final = v
            if best_v_final:
                avg_tok = best_v_final.get("avg_output_tokens", {}).get(mk, 0)
                avg_lat = best_v_final.get("avg_latency_ms", {}).get(mk, 0.0)
                print(f"\n  Model: {mk} (fallback)")
                print(f"    Best accuracy:    {best_acc_final:.3f}")
                print(f"    Best prompt:      \"{best_v_final['system_prompt']}\"")
                final_results[mk] = {
                    "best_accuracy": best_acc_final,
                    "best_prompt": best_v_final["system_prompt"],
                    "temperature": best_v_final["temperature"],
                    "max_tokens": best_v_final["max_tokens"],
                    "avg_output_tokens": avg_tok,
                    "avg_latency_ms": avg_lat,
                    "variant_name": best_v_final["name"],
                    "best_generation": num_generations,
                }

    # 5. Save results
    output = {
        "category": category,
        "num_questions": len(questions),
        "num_generations": len(all_generations_data),
        "model_keys": model_keys,
        "history": history,
        "final_results": final_results,
        "generations": all_generations_data,
    }

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    out_path = Path(RESULTS_DIR) / f"{category}_gepa_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return output


def get_baseline_accuracy(category: str, model_key: str) -> float:
    """Get the baseline (empty prompt) accuracy for a category."""
    with open(TRAINING_DATA, "r") as f:
        all_data = json.load(f)
    questions = [q for q in all_data if q.get("category") == category]
    if not questions:
        combined_map = {
            "sentiment": "/home/artem/dev/amd-hackathon/data/eval/sentiment_combined_25.json",
            "summarization": "/home/artem/dev/amd-hackathon/data/eval/summarization_combined_25.json",
        }
        if category in combined_map:
            with open(combined_map[category]) as f:
                questions = json.load(f)
    if not questions:
        return 0.0

    variant = {"name": "empty", "system_prompt": "", "temperature": 0.0, "max_tokens": MAX_TOKENS_DEFAULT}
    result = evaluate_variant_on_model(variant, model_key, questions)
    return result["accuracy"]


def main():
    parser = argparse.ArgumentParser(description="GEPA Category Prompt Evolution")
    parser.add_argument("--category", required=True, choices=["sentiment", "ner", "code_gen", "summarization", "code_debug", "math", "factual", "logic"],
                        help="Task category to evolve prompts for")
    parser.add_argument("--model", default="qwen2.5-1.5b",
                        help="Primary model key (default: qwen2.5-1.5b)")
    parser.add_argument("--questions", type=int, default=None,
                        help="Number of questions to use (subset)")
    parser.add_argument("--generations", type=int, default=NUM_GENERATIONS,
                        help=f"Number of generations (default: {NUM_GENERATIONS})")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Only compute baseline accuracy, don't run GEPA")
    parser.add_argument("--models", nargs="+",
                        default=["qwen2.5-1.5b", "qwen2.5-coder-1.5b"],
                        help="Model keys to use (default: qwen2.5-1.5b qwen2.5-coder-1.5b)")

    args = parser.parse_args()

    if args.baseline_only:
        print(f"\nComputing baseline (empty prompt) accuracy for '{args.category}' on {args.model}...")
        acc = get_baseline_accuracy(args.category, args.model)
        print(f"  Baseline accuracy: {acc:.3f}")
        return

    print(f"\n{'#' * 70}")
    print(f"# GEPA Evolution: {args.category.upper()}")
    print(f"# Model(s): {', '.join(args.models)}")
    print(f"# Generations: {args.generations}")
    print(f"{'#' * 70}")

    result = run_gepa(
        category=args.category,
        model_keys=args.models,
        num_generations=args.generations,
        num_questions=args.questions,
    )

    if result:
        print(f"\n{'=' * 70}")
        print(f"GEPA Evolution Complete: {args.category}")
        print(f"{'=' * 70}")
        for mk, info in result["final_results"].items():
            print(f"  {mk}: accuracy={info['best_accuracy']:.3f}, prompt=\"{info['best_prompt'][:100]}\"")
    else:
        print(f"\nERROR: GEPA evolution failed for category '{args.category}'.")


if __name__ == "__main__":
    main()
