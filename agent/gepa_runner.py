#!/usr/bin/env python3
"""
GEPA: Genetic Pareto Algorithm prompt optimizer for factual QA on small LLMs.

Synthesizes two research methodologies:
  1. genetic_prompt_evolution_methodology.md (genome representation, mutation operators)
  2. ParetoMethodology.md (Pareto dominance, NSGA-II sorting, crowding distance)

Evolves prompts for 4 local GGUF models on 19 factual questions from training-v3.json.
Run directly:  python agent/gepa_runner.py
"""

import json
import re
import random
import time
from pathlib import Path

# ---- Constants ----

MODEL_PATHS = {
    "qwen2.5-1.5b": "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "qwen2.5-math-1.5b": "/home/artem/models/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf",
    "qwen2.5-coder-1.5b": "/home/artem/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    "smollm2-1.7b": "/home/artem/models/smollm2-1.7b-instruct-q4_k_m.gguf",
    "qwen2.5-base": "/home/artem/models/qwen2.5-1.5b-base-q4_k_m.gguf",
}

DATASET_PATH = "/home/artem/dev/amd-hackathon/data/eval/training-v3.json"
RESULTS_PATH = "/home/artem/dev/amd-hackathon/eval_results/gepa_results.json"

POPULATION_SIZE = 8        # 8 variants total (evaluated on all 4 models)
NUM_GENERATIONS = 3
NUM_QUESTIONS = 19
ELITE_COUNT = 2
TOURNAMENT_K = 3
MAX_TOKENS_DEFAULT = 64
TEMPERATURE_DEFAULT = 0.0
MAX_PROMPT_CHARS = 200     # methodology recommendation: prevent prompt-smuggling
CONVERGENCE_THRESHOLD_PP = 5.0  # percentage points (methodology: 5.0)

# 12 curated factual-QA constraints (from genetic_prompt_evolution_methodology.md)
CONSTRAINT_POOL = [
    "No preamble.",
    "No explanation.",
    "Use exact names, dates, and numbers.",
    "Keep under 15 words.",
    "Keep under 5 words.",
    "Be precise.",
    "If unsure, give your best guess.",
    "Don't hedge.",
    "Output only the answer.",
    "Be specific.",
    "Use complete sentences if needed.",
    "Address all parts of the question.",
]

# Prefix options
PREFIXES = ["", "Answer:", "Fact:", "Q:", "Answer directly:"]

# Rephrase thesaurus for FORMAT_INSTRUCTION mutation
THESAURUS = {
    "Answer the question directly.": [
        "Respond directly to the question.",
        "Give a direct answer.",
        "Answer straight.",
        "Answer clearly.",
    ],
    "Be concise.": [
        "Keep it brief.",
        "Be brief.",
        "Answer concisely.",
    ],
    "Provide the exact answer.": [
        "Give the exact answer.",
        "State the precise answer.",
        "Exact answer only.",
    ],
}

# Verbosity style pools
SHORT_STYLES = ["Be concise.", "Be brief.", "Keep it short."]
MEDIUM_STYLES = ["Answer clearly.", "Provide a clear answer.", "Respond with the answer."]

# Temperature options for mutation (h)
TEMP_OPTIONS = [0.0, 0.1, 0.2]

# 4 known-good factual prompts (seeds for generation 0)
KNOWN_GOOD_PROMPTS = [
    "Fact:",
    "Answer the question directly. Use exact names, dates, and numbers. Keep under 15 words. No preamble.",
    "Answer:",
    "Answer directly.",
]

# ---- Grading Function ----

def fuzzy_match(answer: str, expected: str) -> bool:
    """4-cascade fuzzy match: exact -> substring -> numeric 1% -> token overlap >= 80%."""
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


# ---- Model Cache (lazy-loaded, shared across all evaluations) ----

class ModelCache:
    """Lazy-loads and caches llama_cpp.Llama instances by model key."""

    def __init__(self):
        self._models = {}

    def get(self, model_key: str):
        """Return cached Llama instance, loading on first use."""
        if model_key not in self._models:
            path = MODEL_PATHS.get(model_key)
            if not path:
                raise ValueError(f"Unknown model key: {model_key}")
            from llama_cpp import Llama
            print(f"  [model] Loading {model_key} from {path} ...")
            self._models[model_key] = Llama(
                model_path=path,
                n_ctx=2048,
                n_gpu_layers=-1,  # offload all layers to GPU (local eval)
                n_threads=4,
                verbose=False,
            )
            print(f"  [model] {model_key} loaded.")
        return self._models[model_key]

    def clear(self):
        self._models.clear()


# Shared global cache
_model_cache = ModelCache()


# ---- Pareto Objective Functions ----

def pareto_dominates(a: dict, b: dict) -> bool:
    """
    Returns True if a Pareto-dominates b.
    Objectives: maximize accuracy, minimize avg_output_tokens, minimize avg_latency_ms.
    Uses normalized values stored under acc_norm, tokens_norm, lat_norm (all higher=better).
    """
    objs = ["acc_norm", "tokens_norm", "lat_norm"]
    better_or_eq = all(a[o] >= b[o] - 1e-10 for o in objs)
    strictly_better = any(a[o] > b[o] + 1e-10 for o in objs)
    return better_or_eq and strictly_better


def fast_non_dominated_sort(population: list) -> list:
    """
    NSGA-II fast non-dominated sort.
    Returns list of fronts, each front is a list of indices into population.
    Front[0] = Pareto-optimal set.
    """
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

    # Front 0
    current_front = [i for i in range(n) if n_count[i] == 0]
    if not current_front:
        return [list(range(n))]
    fronts.append(current_front)

    # Subsequent fronts
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
    """
    Compute crowding distance for each individual in a front.
    Returns list of distances aligned with front list order.
    Higher distance = more isolated (prefer for diversity).
    """
    if len(front) <= 2:
        return [float("inf")] * len(front)

    dist = [0.0] * len(front)
    for obj in objectives:
        front_sorted = sorted(range(len(front)), key=lambda idx: population[front[idx]][obj])
        obj_min = population[front[front_sorted[0]]][obj]
        obj_max = population[front[front_sorted[-1]]][obj]
        denom = obj_max - obj_min + 1e-9
        # Boundaries get infinite distance
        dist[front_sorted[0]] = float("inf")
        dist[front_sorted[-1]] = float("inf")
        for pos in range(1, len(front_sorted) - 1):
            i_prev = front[front_sorted[pos - 1]]
            i_next = front[front_sorted[pos + 1]]
            delta = (population[i_next][obj] - population[i_prev][obj]) / denom
            dist[front_sorted[pos]] += delta

    return dist


# ---- Normalization ----

def normalize_objectives(variants: list, model_key: str):
    """Normalize accuracy, avg_output_tokens, avg_latency_ms to [0,1] for a specific model's eval data."""
    # Collect values for this model
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


# ---- Evaluation ----

def evaluate_variant_on_model(variant: dict, model_key: str, questions: list) -> dict:
    """Run a single prompt variant on all questions for a given model.
    Returns dict with accuracy, avg_output_tokens, avg_latency_ms."""
    try:
        llm = _model_cache.get(model_key)
    except Exception as e:
        print(f"    [ERROR] Failed to load model {model_key}: {e}")
        return {"accuracy": 0.0, "avg_output_tokens": 999, "avg_latency_ms": 99999.0}

    correct = 0
    total_tokens = 0
    total_latency = 0.0
    system_prompt = variant["system_prompt"]
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
            elapsed = (time.time() - start) * 1000  # ms
            answer = response["choices"][0]["message"]["content"].strip()
            usage = response.get("usage", {})
            tok_count = usage.get("completion_tokens", len(answer.split()))

            if fuzzy_match(answer, q["expected_answer"]):
                correct += 1

            total_tokens += tok_count
            total_latency += elapsed

            if idx % 5 == 0 and idx > 0:
                print(f"      [{model_key}] {idx}/{len(questions)} questions done ...")

        except Exception as e:
            print(f"      [{model_key}] Question {idx} failed: {e}")
            total_tokens += 0
            total_latency += 100  # penalize failures

    accuracy = correct / len(questions) if questions else 0.0
    avg_tok = total_tokens / len(questions) if questions else 0
    avg_lat = total_latency / len(questions) if questions else 0.0

    return {"accuracy": accuracy, "avg_output_tokens": avg_tok, "avg_latency_ms": avg_lat}


def evaluate_population(population: list, questions: list):
    """Evaluate all variants on all 4 models. Mutates variants in-place with results.
    Normalization happens once after ALL variants are evaluated."""
    model_keys = list(MODEL_PATHS.keys())
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
    # Normalize once after all variants have data
    for mk in model_keys:
        normalize_objectives(population, mk)


# ---- Seed Population ----

def generate_random_variant(name: str) -> dict:
    """Create a random prompt variant."""
    prefixes = PREFIXES
    prefix = random.choice(prefixes)

    # Build a prompt from random components
    parts = []
    if prefix:
        parts.append(prefix)

    # Random instruction
    if random.random() < 0.7:
        parts.append(random.choice([
            "Answer the question directly.",
            "Be concise.",
            "Provide the exact answer.",
            "Answer briefly.",
            "Respond with the answer.",
        ]))

    # Random constraints (0-2)
    num_constraints = random.randint(0, 2)
    constraint_text = ". ".join(random.sample(CONSTRAINT_POOL, min(num_constraints, len(CONSTRAINT_POOL))))
    if constraint_text:
        parts.append(constraint_text)

    system_prompt = " ".join(parts)
    system_prompt = system_prompt.strip()

    # Enforce max length
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


def build_seed_population() -> list:
    """Generation 0: 4 known-good prompts + 4 random variants."""
    population = []
    for i, prompt_text in enumerate(KNOWN_GOOD_PROMPTS):
        population.append({
            "name": f"seed_{i}",
            "system_prompt": prompt_text,
            "temperature": 0.0,
            "max_tokens": MAX_TOKENS_DEFAULT,
        })
    for i in range(POPULATION_SIZE - len(KNOWN_GOOD_PROMPTS)):
        population.append(generate_random_variant(f"rand_{i}"))
    return population


# ---- Genetic Operators ----

def _truncate_prompt(text: str, max_chars: int = MAX_PROMPT_CHARS) -> str:
    """Truncate prompt at word boundary if over max_chars."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    if not truncated.endswith("."):
        truncated += "."
    return truncated


def mutate(variant: dict) -> dict:
    """Apply one of 8 mutation operators to a prompt variant. Returns a new dict."""
    child = {
        "name": variant["name"] + "_mut",
        "system_prompt": variant["system_prompt"],
        "temperature": variant["temperature"],
        "max_tokens": variant["max_tokens"],
    }

    op = random.randint(0, 7)
    text = child["system_prompt"]

    if op == 0:
        # (a) Rephrase: replace a phrase with synonym from thesaurus
        for orig, replacements in THESAURUS.items():
            if orig in text:
                text = text.replace(orig, random.choice(replacements), 1)
                break
        # No fallback partial match — the exact-phrase check is sufficient
        # and the old partial-match fallback created broken text.

    elif op == 1:
        # (b) Add a constraint from pool
        existing = [c for c in CONSTRAINT_POOL if c.lower() in text.lower()]
        available = [c for c in CONSTRAINT_POOL if c not in existing]
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
        # (d) Swap verbosity: short -> medium or medium -> short
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
        for p in PREFIXES:
            if p and text.startswith(p):
                current_prefix = p
                break
        options = [p for p in PREFIXES if p != current_prefix]
        if options:
            new_prefix = random.choice(options)
            if current_prefix:
                text = text.replace(current_prefix, new_prefix, 1)
            else:
                text = new_prefix + " " + text if new_prefix else text

    elif op == 5:
        # (f) Add "Keep under N words"
        n = random.choice([5, 10, 15, 20])
        # Remove existing word limit constraints first
        for old_n in [5, 10, 15, 20]:
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

    # Clean up double spaces, leading/trailing space, and enforce max length
    text = re.sub(r'\s+', ' ', text).strip()
    text = _truncate_prompt(text)
    child["system_prompt"] = text
    child["name"] = child["name"] + f"_op{op}"

    return child


def crossover_prompts(p1: dict, p2: dict) -> dict:
    """Sentence-level uniform crossover between two parent prompts."""
    s1 = [s.strip() for s in re.split(r'(?<=[.!?])\s+', p1["system_prompt"]) if s.strip()]
    s2 = [s.strip() for s in re.split(r'(?<=[.!?])\s+', p2["system_prompt"]) if s.strip()]

    if not s1 and not s2:
        child_text = ""
    elif not s1:
        child_text = " ".join(s2)
        child_text = child_text[:-1] + "." if child_text.endswith(".") else child_text
    elif not s2:
        child_text = " ".join(s1)
        child_text = child_text[:-1] + "." if child_text.endswith(".") else child_text
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
        # Append remaining from random parent
        remaining = s1[i:] if random.random() < 0.5 else s2[j:]
        child.extend(remaining)
        # Remove trailing period from each sentence before joining to avoid double periods
        cleaned = [s.rstrip(".!?") for s in child]
        child_text = ". ".join(cleaned)
        if child and not child_text.endswith("."):
            child_text += "."

    # Also crossover temperature and max_tokens
    temp = p1["temperature"] if random.random() < 0.5 else p2["temperature"]
    mt = p1["max_tokens"] if random.random() < 0.5 else p2["max_tokens"]

    return {
        "name": f"xover_{p1['name']}_{p2['name']}",
        "system_prompt": child_text,
        "temperature": temp,
        "max_tokens": mt,
    }


def tournament_select(population: list, model_rank_map: dict, model_keys: list, k: int = 3) -> dict:
    """Tournament selection using aggregate Pareto rank across models as primary fitness,
    crowding distance as tiebreaker (higher=better).
    Returns the selected individual dict."""
    if not population:
        raise ValueError("tournament_select called on empty population")

    # Pick first candidate and compute its rank sum
    best_idx = random.randrange(len(population))
    best = population[best_idx]
    best_rank_sum = 0.0
    best_crowd = 0.0
    for mk in model_keys:
        rm = model_rank_map.get(mk, {})
        best_rank_sum += rm.get(best_idx, 999)
        crowd = best.get("crowding_dist", {}).get(mk, 0.0)
        best_crowd += crowd

    # Compete against k-1 additional random candidates
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


# ---- Convergence Check ----

def check_convergence(history: list, threshold: float = CONVERGENCE_THRESHOLD_PP) -> bool:
    """Check if top-3 mean accuracy delta < threshold (in percentage points) from previous generation."""
    if len(history) < 2:
        return False
    prev = history[-2]["top3_mean"]
    curr = history[-1]["top3_mean"]
    delta = abs(curr - prev) * 100  # convert to percentage points
    print(f"  Top-3 mean accuracy delta: {delta:.2f} pp (threshold: {threshold})")
    return delta < threshold



# ---- Main GEPA Loop ----

def run_gepa():
    """Main GEPA optimization loop: 3 generations of Pareto-based prompt evolution."""

    # 1. Load dataset
    print("=" * 70)
    print("GEPA: Genetic Pareto Algorithm Prompt Optimizer")
    print("=" * 70)

    print("\n[1] Loading dataset ...")
    with open(DATASET_PATH, "r") as f:
        all_data = json.load(f)
    factual_questions = [q for q in all_data if q.get("category") == "factual"]
    print(f"    Loaded {len(factual_questions)} factual questions (expected 19).")
    if len(factual_questions) != 19:
        print(f"    WARNING: Expected 19 factual questions, got {len(factual_questions)}.")

    model_keys = list(MODEL_PATHS.keys())

    # 2. Build seed population
    print("\n[2] Building seed population (4 known-good + 4 random) ...")
    population = build_seed_population()
    for i, v in enumerate(population):
        print(f"    [{i}] {v['name']}: \"{v['system_prompt'][:80]}...\" (temp={v['temperature']})")

    # Track the globally best variant per model across ALL generations
    global_best_per_model = {mk: {"variant": None, "accuracy": -1.0} for mk in model_keys}

    # 3. Generation loop
    history = []

    for gen in range(NUM_GENERATIONS):
        print(f"\n{'=' * 70}")
        print(f"Generation {gen + 1} / {NUM_GENERATIONS}")
        print(f"{'=' * 70}")

        # 3a. Evaluate population on all models
        print("\n[3a] Evaluating population ...")
        # Clear previous eval data to avoid stale norm values
        for v in population:
            for key in ["accuracy", "avg_output_tokens", "avg_latency_ms",
                        "acc_norm", "tokens_norm", "lat_norm", "crowding_dist", "rank"]:
                v.pop(key, None)

        evaluate_population(population, factual_questions)

        # 3b. Compute Pareto fronts per model
        print("\n[3b] Computing Pareto fronts per model ...")
        model_rank_map = {}
        all_fronts = {}

        for mk in model_keys:
            # Build a per-model population list with normalized objectives
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

            # Assign ranks and crowding distances back to population
            rank_map = {}
            for rank_idx, front in enumerate(fronts):
                # Compute crowding distance for this front
                dists = crowding_distance(model_pop, front)
                for pos, orig_idx in enumerate(front):
                    rank_map[orig_idx] = rank_idx
                    if "crowding_dist" not in population[orig_idx]:
                        population[orig_idx]["crowding_dist"] = {}
                    population[orig_idx]["crowding_dist"][mk] = dists[pos]

            model_rank_map[mk] = rank_map
            print(f"    {mk}: Front 0 size = {len(fronts[0]) if fronts else 0}")

        # 3c. Update global best tracking and report
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
            # Update global best
            if best_acc > global_best_per_model[mk]["accuracy"]:
                # Deep-copy the best variant for preservation
                global_best_per_model[mk] = {
                    "variant": dict(best_variant) if best_variant else None,
                    "accuracy": best_acc,
                    "generation": gen + 1,
                }
            print(f"    {mk}: best accuracy = {best_acc:.3f}  ({best_variant['name'] if best_variant else 'N/A'})")
            if best_variant:
                print(f"      prompt: \"{best_variant['system_prompt'][:100]}...\"")

            # Collect top-3 accuracies for this model
            accs_sorted = sorted(
                [v.get("accuracy", {}).get(mk, 0.0) for v in population],
                reverse=True
            )
            top3 = accs_sorted[:3]
            gen_top3_accs.extend(top3)

        # 3d. Compute overall top-3 mean accuracy (average of top-3 from each model)
        top3_mean = sum(gen_top3_accs) / len(gen_top3_accs) if gen_top3_accs else 0.0

        gen_record = {
            "generation": gen + 1,
            "top3_mean": top3_mean,
            "per_model_best": {mk: global_best_per_model[mk]["accuracy"]
                               for mk in model_keys},
            "pareto_front_sizes": {mk: len(all_fronts[mk][0]) if mk in all_fronts and all_fronts[mk] else 0
                                    for mk in model_keys},
        }
        history.append(gen_record)

        # 3e. Report
        print(f"\n    Pareto front sizes: {gen_record['pareto_front_sizes']}")
        print(f"    Top-3 mean accuracy: {top3_mean:.4f}")

        # 3f. Check convergence (requires at least 2 generations)
        if len(history) >= 2:
            converged = check_convergence(history)
            if converged:
                print(f"\n  >>> Converged at generation {gen + 1}. Stopping early.")
                break

        # 3g. Create next generation (skip after last generation)
        if gen == NUM_GENERATIONS - 1:
            print("\n  Final generation reached.")
            break

        print("\n[3g] Creating next generation ...")
        next_population = []

        # Elitism: carry over top ELITE_COUNT non-dominated solutions
        # Pick the best variants by rank sum (lower=better), tiebreak by crowding distance sum (higher=better)
        ranked = sorted(
            [(idx, sum(model_rank_map.get(mk, {}).get(idx, 999) for mk in model_keys))
             for idx in range(len(population))],
            key=lambda x: (x[1], -sum(population[x[0]].get("crowding_dist", {}).get(mk, 0.0) for mk in model_keys))
        )
        for idx, _ in ranked[:ELITE_COUNT]:
            elite = dict(population[idx])
            elite["name"] = f"elite_g{gen}_{idx}"
            next_population.append(elite)
        print(f"    Elites: {[e['name'] for e in next_population]}")

        # Fill rest via crossover + mutation
        while len(next_population) < POPULATION_SIZE - 1:  # -1 for fresh variant
            if len(next_population) < POPULATION_SIZE * 0.6:
                # Crossover
                p1 = tournament_select(population, model_rank_map, model_keys, TOURNAMENT_K)
                p2 = tournament_select(population, model_rank_map, model_keys, TOURNAMENT_K)
                child = crossover_prompts(p1, p2)
                # Apply mutation to crossover child
                child = mutate(child)
                child["name"] = f"gen{gen}_c{len(next_population)}"
            else:
                # Mutation of selected parent
                parent = tournament_select(population, model_rank_map, model_keys, TOURNAMENT_K)
                child = mutate(parent)
                child["name"] = f"gen{gen}_m{len(next_population)}"
            next_population.append(child)

        # Inject 1 fresh random variant
        fresh = generate_random_variant(f"fresh_gen{gen}")
        next_population.append(fresh)
        print(f"    Fresh variant: {fresh['name']}")

        population = next_population
        print(f"    Next population size: {len(population)}")

    # 4. Final summary — use GLOBALLY best variants, not just final population
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
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
            # Fallback: find best in final population
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
                print(f"\n  Model: {mk} (fallback — final population)")
                print(f"    Best accuracy:    {best_acc_final:.3f}")
                print(f"    Best prompt:      \"{best_v_final['system_prompt']}\"")
                print(f"    Temperature:      {best_v_final['temperature']}")
                print(f"    Avg output tokens: {avg_tok:.1f}")
                print(f"    Avg latency (ms): {avg_lat:.1f}")
                final_results[mk] = {
                    "best_accuracy": best_acc_final,
                    "best_prompt": best_v_final["system_prompt"],
                    "temperature": best_v_final["temperature"],
                    "max_tokens": best_v_final["max_tokens"],
                    "avg_output_tokens": avg_tok,
                    "avg_latency_ms": avg_lat,
                    "variant_name": best_v_final["name"],
                    "best_generation": NUM_GENERATIONS,
                }
            else:
                print(f"\n  Model: {mk} — No results.")
                final_results[mk] = {
                    "best_accuracy": 0.0,
                    "best_prompt": "",
                    "temperature": 0.0,
                    "max_tokens": MAX_TOKENS_DEFAULT,
                    "avg_output_tokens": 0,
                    "avg_latency_ms": 0.0,
                    "variant_name": "N/A",
                    "best_generation": None,
                }

    # 5. Save results
    output = {
        "history": history,
        "final_results": final_results,
        "global_best_per_model": {
            mk: {
                "best_accuracy": info["accuracy"],
                "best_prompt": info["variant"]["system_prompt"] if info["variant"] else "",
                "temperature": info["variant"]["temperature"] if info["variant"] else 0.0,
                "max_tokens": info["variant"]["max_tokens"] if info["variant"] else MAX_TOKENS_DEFAULT,
                "generation": info.get("generation", None),
            }
            for mk, info in global_best_per_model.items()
        },
        "population": [
            {
                "name": v["name"],
                "system_prompt": v["system_prompt"],
                "temperature": v["temperature"],
                "max_tokens": v["max_tokens"],
                "accuracy": v.get("accuracy", {}),
                "avg_output_tokens": v.get("avg_output_tokens", {}),
                "avg_latency_ms": v.get("avg_latency_ms", {}),
            }
            for v in population
        ],
    }

    out_path = Path(RESULTS_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    return output


if __name__ == "__main__":
    run_gepa()
