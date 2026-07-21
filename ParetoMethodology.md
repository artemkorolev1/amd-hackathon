# Pareto-Based Multi-Objective Prompt Optimization Methodology

## Overview

This document describes a lightweight Pareto-aware evolutionary strategy for optimizing prompts across **accuracy**, **output token efficiency**, and **latency** simultaneously. Designed for a 3-iteration budget (≈3 generations) with 4 small models (qwen2.5-1.5b, qwen2.5-math-1.5b, qwen2.5-coder-1.5b, smollm2-1.7b) on 19 factual QA questions. Each iteration evaluates ~6 prompt variants per model (total ~24–48 evaluations/iteration).

> **Target output:** Code files `pareto_prompt_optimizer.py` (optimization loop) + `genetic_prompt_ops.py` (crossover/mutation). This doc provides the spec for both.

---

## 1. Pareto Front Concepts Applied to Prompts

### 1.1 The Three Objectives

| Objective | Symbol | Direction | Rationale |
|-----------|--------|-----------|-----------|
| Accuracy | `acc` | **Maximize** | Fraction of questions passing 4-cascade fuzzy match |
| Output token count | `tokens` | **Minimize** | Short answers are more precise, less hallucination-prone, easier to substring-match |
| Inference latency | `latency` | **Minimize** | Critical for small-model deployment; correlates with output length but not perfectly (some models are slower per-token) |

### 1.2 Why Pareto Instead of Weighted Sum

- **Weighted sum** collapses trade-offs into a single score — you must tune λ₁, λ₂, λ₃, and the optimal weights change per category.
- **Pareto front** preserves the true set of non-dominated variants. The operator picks the final prompt based on available compute/reliability budget.
- Small models show **strong trade-offs**: a high-accuracy prompt may be verbose (many tokens → hurts fuzzy-match because surplus text dilutes the key answer). A terse prompt may be fast but inaccurate. The Pareto front surfaces both extremes.

### 1.3 Pareto Dominance Definition

A prompt variant **A** dominates **B** (written `A ≺ B`) iff:

1. `acc(A) ≥ acc(B)` **and** `tokens(A) ≤ tokens(B)` **and** `latency(A) ≤ latency(B)`
2. At least one inequality is **strict** (i.e., `A` is strictly better in ≥1 objective).

If neither dominates the other, they are **non-dominated** and both live on the Pareto front.

### 1.4 Normalization for Comparison

Before dominance checks, normalize each objective to [0, 1] across the current population:

```
acc_norm[i]   = (acc[i] - min_acc) / (max_acc - min_acc + 1e-9)
tokens_norm[i]= 1.0 - (tokens[i] - min_tok) / (max_tok - min_tok + 1e-9)   # invert so higher=better
lat_norm[i]   = 1.0 - (latency[i] - min_lat) / (max_lat - min_lat + 1e-9)  # invert
```

Then **dominance** uses the normalized values with **all three ≥**. This prevents one objective (e.g., latency range 0.5–2.0s vs accuracy range 0.0–1.0) from dominating the comparison due to scale.

---

## 2. Detecting Pareto Dominance Between Prompt Variants

### 2.1 Pairwise Dominance Algorithm

```
Input:  population P (list of dicts with acc_norm, tokens_norm, lat_norm)
Output: for each variant i → set of variants it dominates, set that dominate it

function dominates(a, b):
    # returns True if a dominates b
    a_better_or_equal = all(a[obj] >= b[obj] for obj in [acc_norm, tokens_norm, lat_norm])
    a_strictly_better = any(a[obj] > b[obj] for obj in [acc_norm, tokens_norm, lat_norm])
    return a_better_or_equal AND a_strictly_better

for each i in P:
    for each j in P:
        if i != j and dominates(i, j):
            mark i → dominates j
```

### 2.2 Complexity Note

O(n²) per generation for n = 12–24 variants: negligible. No need for kd-tree or sweepline for populations this small.

### 2.3 Implementation Skeleton

```python
def pareto_dominates(a: dict, b: dict) -> bool:
    """Returns True if a Pareto-dominates b on normalized objectives."""
    objs = ["acc_norm", "tokens_norm", "lat_norm"]
    better_or_eq = all(a[o] >= b[o] - 1e-10 for o in objs)
    strictly_better = any(a[o] > b[o] + 1e-10 for o in objs)
    return better_or_eq and strictly_better
```

---

## 3. NSGA-II-Style Non-Dominated Sorting (Adapted for Small Populations)

### 3.1 Front Assignment

We use the **fast-non-dominated-sort** from NSGA-II, simplified for n ≤ 24.

**Algorithm per generation:**

1. For each variant `p`, compute:
   - `S[p]` = set of variants that `p` dominates
   - `n[p]` = count of variants that dominate `p`

2. `F[0]` = all variants with `n[p] == 0` (Pareto front #1, rank 0)

3. For each `p` in `F[0]`, decrement `n[q]` for each `q` in `S[p]`. If any `n[q]` reaches 0, put `q` in `F[1]`.

4. Repeat until all variants are assigned to a front.

### 3.2 Small-Population Adaptation

With 12–24 variants and 3 objectives, expect:
- **Front 0:** 3–8 variants (the real Pareto frontier)
- **Front 1:** 4–10 variants
- **Front 2+:** remaining (largely dominated variants)

**Only keep Fronts 0 and 1** for parent selection in the next generation. This aggressive truncation is justified by the 3-generation budget — we cannot afford to propagate dominated solutions.

### 3.3 Implementation Skeleton

```python
def non_dominated_sort(population: list[dict]) -> list[list[int]]:
    """
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
```

---

## 4. Crowding Distance to Maintain Diversity in Prompt Space

### 4.1 Purpose

Without diversity pressure, the population converges to one region of the trade-off surface (e.g., all "ultra-terse" prompts). Crowding distance preferentially keeps variants in less-dense regions.

### 4.2 Algorithm

For each front `F_k`:

1. Initialize `distance[i] = 0` for all `i` in `F_k`.
2. For each objective `m` in `{acc_norm, tokens_norm, lat_norm}`:
   - Sort `F_k` by objective `m`.
   - Set boundary points: `distance[first] = distance[last] = INF` (always preserved).
   - For interior points (i = 1 to len-2):
     ```
     distance[i] += (val_m[i+1] - val_m[i-1]) / (max_m - min_m + 1e-9)
     ```
3. Higher crowding distance = more isolated → more likely to survive.

### 4.3 Integration with Selection

When selecting the **top N** variants to survive to the next generation:

1. Fill ranks in order: all of F₀, then F₁, then F₂...
2. If a front would cause overflow (> N survivors), sort that front by crowding distance (descending) and take the remainder.

### 4.4 Implementation Skeleton

```python
def crowding_distance(population: list[dict], front: list[int]) -> list[float]:
    """Returns crowding distance for each index in front. Higher = more diverse."""
    n_obj = 3
    dist = [0.0] * len(front)
    for obj in ["acc_norm", "tokens_norm", "lat_norm"]:
        front_sorted = sorted(front, key=lambda i: population[i][obj])
        obj_min = population[front_sorted[0]][obj]
        obj_max = population[front_sorted[-1]][obj]
        denom = obj_max - obj_min + 1e-9
        # Boundaries
        dist[0] = float("inf")
        dist[-1] = float("inf")
        for idx in range(1, len(front_sorted) - 1):
            i_prev = front_sorted[idx - 1]
            i_next = front_sorted[idx + 1]
            delta = (population[i_next][obj] - population[i_prev][obj]) / denom
            dist[idx] += delta
    return dist  # aligned with front list order
```

---

## 5. Combining Pareto Optimization with Genetic Operators

### 5.1 Prompt Representation

Each individual is a dictionary:

```python
{
    "id": str,                # unique variant ID
    "model": str,             # model name key
    "system_prompt": str,     # the system prompt text
    "temperature": float,     # [0.0, 1.5]
    "max_tokens": int,        # [16, 1024]
    # — evaluated metrics —
    "accuracy": float,        # 0.0–1.0
    "tokens_output": int,     # total tokens across all answers
    "latency": float,         # total wall time in seconds
    # — normalized (computed post-eval) —
    "acc_norm": float,
    "tokens_norm": float,
    "lat_norm": float,
    # — NSGA-II state —
    "rank": int,              # Pareto front rank (0 = best)
    "crowding_dist": float,
}
```

**The prompt template** has 4 mutation "hotspots":
- **Instruction verb** (e.g., "Answer", "Solve", "List", "Extract", "Classify", "Output")
- **Format constraint** (e.g., "in backticks", "as JSON", "one word", "bullet points")
- **Tone modifier** (e.g., "directly", "step by step", "brief", "terse", "precise")
- **Length guidance** (e.g., "under 15 words", "1 sentence", "2-3 lines")

### 5.2 Crossover (Prompt Blending)

For small populations, **uniform crossover** works better than single-point crossover because prompt text is short and semantic boundaries are fuzzy.

```
Parent A: system_prompt = "Answer the question directly. Keep under 15 words."
Parent B: system_prompt = "Solve step by step. End with 'Answer: ...'."

Child: picks each "sentence" from either parent, with sentence boundaries split on '.'
→ "Answer the question directly. End with 'Answer: ...'."
```

**Sentence-level uniform crossover** (sentences are the natural semantic unit for prompts):

```python
import re

def crossover_prompts(p1: str, p2: str) -> str:
    """Sentence-level uniform crossover."""
    s1 = [s.strip() for s in re.split(r'(?<=[.!?])\s+', p1) if s.strip()]
    s2 = [s.strip() for s in re.split(r'(?<=[.!?])\s+', p2) if s.strip()]
    child = []
    i = j = 0
    while i < len(s1) and j < len(s2):
        if random.random() < 0.5:
            child.append(s1[i]); i += 1
        else:
            child.append(s2[j]); j += 1
    child.extend(s1[i:] if random.random() < 0.5 else s2[j:])
    return ". ".join(child) + ("." if child else "")
```

**Temperature/MaxTokens crossover**: blend with 50/50 probability per gene.

### 5.3 Mutation (Five Operators)

Each variant undergoes **one randomly chosen mutation** (if not elite):

| Operator | Probability | Effect |
|----------|-------------|--------|
| **Instruction swap** | 25% | Replace instruction verb: "Answer"↔"List"↔"Extract"↔"Classify"↔"Output"↔"Solve"↔"Find" |
| **Format twist** | 20% | Change output format: "backticks"↔"JSON"↔"one word"↔"bullet"↔"sentence"↔"bare" |
| **Tone shift** | 20% | Swap tone: "directly"↔"step by step"↔"brief"↔"terse"↔"precise"↔"concise"↔"detailed" |
| **Length clamp** | 20% | Change max_tokens to a new value from {16, 32, 64, 128, 256, 512, 1024}; adjust "under X words" in prompt accordingly |
| **Temperature jitter** | 15% | Add Gaussian noise: temp = clip(temp + N(0, 0.15), 0.0, 1.5) |

### 5.4 Elite Preservation

The **top 3 non-dominated variants** (rank 0 with highest crowding distance) are carried over unmodified to the next generation. This guarantees monotonic improvement of the Pareto front.

### 5.5 Generation Pipeline

```
Generation g:
  ┌─ Evaluate all N individuals (run inference, collect metrics)
  ├─ Normalize objectives across population
  ├─ Non-dominated sort → rank[]
  ├─ Compute crowding distance
  ├─ Report Pareto front (rank 0)
  │
  ├─ [Selection] Elite count E = min(3, |rank0|)
  ├─            Parent pool: all rank-0 + rank-1 individuals
  │
  ├─ [Crossover] pairs weighted by crowding distance (higher = more likely parent)
  │             → N_cross = ceil((N - E) * 0.6)
  │
  ├─ [Mutation] remaining slots filled by mutating selected rank-0/1 individuals
  │             → N_mutate = N - E - N_cross
  │
  └─ Generation g+1 = elites ∪ crossover_children ∪ mutants
```

---

## 6. Fitness Sharing to Prevent Population Collapse

### 6.1 The Problem

With only 3 generations, greedy Pareto selection can collapse all variants toward one "winning" template style (e.g., all prompts become "Answer:" variants from rank 0 — short, fast, but maybe not the most accurate for hard questions).

### 6.2 Fitness Sharing Distance Metric

Define **prompt edit distance** between two variants as **Levenshtein distance on system_prompt text**, normalized by the longer prompt length:

```python
def prompt_distance(p1: str, p2: str) -> float:
    """Normalized Levenshtein distance in [0, 1]."""
    d = levenshtein(p1, p2)
    return d / max(len(p1), len(p2), 1)
```

Alternative (simpler): **word Jaccard distance** = 1 - (|words₁ ∩ words₂| / |words₁ ∪ words₂|). This captures synonym-level similarity better (e.g., "Answer:" vs "Respond:" are close).

### 6.3 Shared Fitness Calculation

After computing raw non-dominated rank:

```python
def shared_fitness(individual, population, sigma_share=0.3, alpha=1.0):
    """
    Reduce fitness of individuals that are too similar to others.
    sigma_share = niche radius (in prompt distance space)
    alpha = 1.0 (linear sharing)
    """
    niche_count = 0
    for other in population:
        d = prompt_distance(individual["system_prompt"], other["system_prompt"])
        if d < sigma_share:
            niche_count += (1 - d / sigma_share) ** alpha
    return individual["rank"] + niche_count - 1  # higher shared_fitness = worse
```

**Integration**: When sorting within a front for selection, use `shared_fitness` as the tie-breaker instead of raw crowding distance. This pushes selection toward under-populated prompt regions.

### 6.4 Niche Radius Tuning

For our prompt space (typical system prompt: 5–25 words, word-Jaccard range 0.0–1.0):
- `sigma_share = 0.4` works well (niche radius ≈ 40% word dissimilarity)
- Two variants differing by >1 keyword (e.g., "Answer..." vs "List...") are in different niches

### 6.5 Implementation: Combined Survival Score

```
survival_score = (rank, -crowding_distance + shared_penalty)
               = (lower_rank_is_better, higher_crowding_is_better)
```

When `rank` is equal, we sort by `crowding_distance` (diversity in objective space). `shared_fitness` adds a penalty in prompt-text space. Both are needed: crowding distance maintains **objective diversity** (different accuracy/token/latency trade-offs), shared fitness maintains **prompt template diversity** (different phrasing styles that might unlock new trade-offs in future generations).

---

## 7. Related Work & Simplifications for 3-Iteration Budget

### 7.1 Relevant Literature

| Source | Idea | How We Simplify |
|--------|------|----------------|
| **NSGA-II** (Deb et al., 2002) | Fast non-dominated sort + crowding distance | Use full algorithm but only for n ≤ 24 |
| **Pareto Prompting** (Wang et al., 2024) | Pareto-optimal LLM prompts for code generation | Adapted for factual QA + small models |
| **DSPy** (Khattab et al., 2023) | Programmatic prompt optimization with teleprompters | We replace their Bayesian optimizer with NSGA-II for multi-objective |
| **AutoML / HPO** (Ray Tune, Optuna) | Multi-objective hyperparameter optimization (MOHPO) | We treat `{temperature, max_tokens}` as HPs with prompt text as structured search space |
| **PRM / Process Reward Models** | Stepwise verification | Not used (no reward model available); our "fitness" = eval accuracy directly |
| **Prompt Ensembling** | Multiple prompts → vote → consensus | After final generation, we can form a Pareto ensemble: pick 2–3 non-dominated prompts and ensemble their answers |

### 7.2 Key Simplifications for 3 Generations

1. **No full NSGA-II generational loop** — we skip the tournament selection and instead do **elitist survival + crossover + mutation** directly. With 3 gens, tournament overhead isn't worth it.

2. **Static population size** = 12–16 per model per iteration. No adaptive resizing.

3. **No constraint handling** — all prompts are valid by construction (no invalid region in prompt space).

4. **Cold start** ≠ random. Generation 0 uses the existing best-performing prompts from R3 (`eval_prompt_ablation_r3.py` results) plus human-designed variations. This saves one generation.

5. **Multi-model parallelism**: Each iteration evaluates all 4 models simultaneously (llama-server --parallel N). The Pareto front is computed **per-model**, not across models, since model ID is a categorical that cannot be compared.

6. **Accuracy-first fallback**: After gen 3, if the Pareto front has >3 variants, the **operator picks the variant** that maximizes `acc_norm + 0.1 * tokens_norm + 0.05 * lat_norm` — a light weight that breaks ties toward the most accurate.

### 7.3 AutoML/HPO Parallels

| AutoML concept | Prompt optimization analog |
|----------------|---------------------------|
| Hyperparameter config | `{system_prompt, temperature, max_tokens}` |
| Trial | One full eval of 19 questions |
| Budget (wall time) | 3 iterations × ~24 variants × 19 Q = ~1368 inferences total |
| Early stopping | If `accuracy == 1.0` on a variant, freeze it (skip mutation) |
| Multi-fidelity | Not needed; every trial evaluates all 19 questions (small dataset) |

### 7.4 Prompt Ensembling from Pareto Front

After the final generation, the Pareto front contains ≥2 variants. These can be ensembled:

```
For each question:
  1. Query all Pareto-optimal prompts (or the top 3 by crowding distance).
  2. Collect answers.
  3. Return the answer with most fuzzy-match consensus
     (if tied, shortest answer wins — ties into token efficiency).
```

This ensemble approach leverages the maintained diversity and has been shown (DSPy, Prompt Ensembling literature) to outperform any single prompt.

---

## 8. Concrete Algorithm Pseudocode

```
Algorithm: ParetoPromptEvolution

Input:
  models = ["qwen2.5-1.5b", "qwen2.5-math-1.5b", "qwen2.5-coder-1.5b", "smollm2-1.7b"]
  dataset = training-v3.json (19 questions)
  N_pop = 14 per model (total 56 individuals, but eval'd per-model)
  G_max = 3
  seed_prompts = 6 hand-crafted prompts per category (from R3 results)

Output:
  Pareto front per model: [(prompt_variant, {acc, tokens, latency})]

For each model m in models:
  # Generation 0: seed from best R3 prompts + variations
  pop = seed_prompts_for_model(m)  # 6 prompts

  For gen = 0 to G_max:
    # Evaluate
    for each variant in pop:
      results = evaluate(variant, dataset)  # returns acc, total_tokens, total_time
      variant.update(results)

    # Normalize
    normalize_objectives(pop)

    # Non-dominated sort
    fronts = non_dominated_sort(pop)

    # Crowding distance
    for front in fronts:
      compute_crowding_distance(pop, front)

    # Report
    print_pareto_front(pop, fronts[0], gen)

    if gen == G_max:
      break

    # Selection & reproduction
    elites = select_elites(pop, fronts, E=3)
    parents = pop[fronts[0] + fronts[1]]  # rank 0 and rank 1

    next_pop = copy(elites)

    # Crossover
    while len(next_pop) < N_pop * 0.6:
      p1, p2 = roulette_wheel_select(parents, weight=crowding_distance)
      child = crossover(p1, p2)
      next_pop.append(child)

    # Mutation (fill remainder)
    while len(next_pop) < N_pop:
      parent = random_choice(parents, weight=crowding_distance)
      child = mutate(parent)
      next_pop.append(child)

    pop = next_pop

  # Final Pareto front
  return [(pop[i], pop[i]["accuracy"], pop[i]["tokens_output"], pop[i]["latency"])
          for i in fronts[0]]
```

---

## 9. Metrics & Logging

Each generation must log:

```
Generation 1:
  Model qwen2.5-1.5b:
    Pareto front size: 4
    Front ranges: acc [0.42, 0.68], tokens [120, 890], latency [3.2s, 12.1s]
    Best accuracy: 0.68 (v3-force-guess)
    Fastest: 3.2s (v4-direct)
    Most token-efficient: 120 tokens (v4-direct)
  Diversity (mean pairwise Jaccard): 0.35

  Model qwen2.5-math-1.5b:
    ...
```

**File output:** `pareto_evolution_log.json` — full per-generation state.

---

## 10. Pitfalls & Mitigations

| Pitfall | Mitigation |
|---------|-----------|
| 3 gens is too few for convergence | Cold-start from R3 best prompts; elitism ensures the front never gets worse |
| Accuracy noise (small n=19) | Use bootstrap confidence intervals? Not needed — 19 questions per variant is the same set, so comparisons are paired. Minor noise doesn't flip dominance. |
| Diversity collapse | Fitness sharing + crowding distance + enforce at least 2 distinct prompt templates in elite set |
| Latency varies by system load | Run all variants for one generation in a single llama-server session. Compare relative, not absolute. |
| Prompt length vs token count conflated | Track both `system_prompt_len` (words) and `output_tokens` separately. The objective is output tokens only. |
| Fuzzy match stochasticity | Fixed seed, temperature=0 for main eval. (Temperature=0 is deterministic for GGUF.) |

---

## 11. Code Interface Spec

### 11.1 `pareto_prompt_optimizer.py`

```python
class ParetoPromptOptimizer:
    def __init__(self, models: dict[str, str], dataset_path: str):
        """models = {model_name: gguf_path, ...}"""

    def run(self, n_generations=3, population_size=14) -> dict:
        """
        Returns: {
            model_name: [
                {"variant": ..., "accuracy": ..., "tokens_output": ..., "latency": ...},
                ...
            ]
        }
        """

    def evaluate_population(self, pop: list, model: str) -> list:
        """Run inference on all 19 questions. Returns acc, token_count, latency."""

    def get_pareto_front(self, pop: list) -> list:
        """Non-dominated set."""
```

### 11.2 `genetic_prompt_ops.py`

```python
def crossover(p1: dict, p2: dict) -> dict:
def mutate(variant: dict, mutation_rate: float = 0.3) -> dict:
def prompt_distance(p1: str, p2: str) -> float:
def shared_fitness(ind: dict, population: list, sigma_share=0.4) -> float:
def seed_population(model: str, n: int, r3_results: dict) -> list:
```

### 11.3 Expected Non-Dominated Sort Output (Example)

For a generation with 14 variants:

```
Front 0 (rank 0, 5 variants):
  idx=3  acc=0.74 tok=234  lat=4.2s  prompt="Answer:"
  idx=7  acc=0.68 tok=89   lat=2.1s  prompt="Answer directly."
  idx=11 acc=0.63 tok=45   lat=1.8s  prompt="Guess:"
  idx=2  acc=0.58 tok=412  lat=9.3s  prompt="Step by step. Verify. Answer:"
  idx=5  acc=0.84 tok=1023 lat=22.1s prompt="Solve carefully. Show all working..."

Front 1 (rank 1, 6 variants):
  idx=1  acc=0.63 tok=312  lat=7.1s  prompt="List entities:"
  idx=9  acc=0.42 tok=56   lat=1.8s  prompt="Answer:"
  ... (dominated by at least one front-0 variant)
```

Note that idx=5 (acc=0.84) and idx=3 (acc=0.74) are both on the same Pareto front — one is more accurate but much slower, the other is a balanced trade-off. This is exactly the information the operator needs to make the final selection.

---

## Appendix: Quick-Reference Implementation Plan

| Step | File | Function | LoC |
|------|------|----------|-----|
| Dominance check | `pareto_prompt_optimizer.py` | `pareto_dominates()` | 5 |
| Non-dominated sort | `pareto_prompt_optimizer.py` | `non_dominated_sort()` | 25 |
| Crowding distance | `pareto_prompt_optimizer.py` | `crowding_distance()` | 20 |
| Fitness sharing | `genetic_prompt_ops.py` | `shared_fitness()` | 10 |
| Prompt distance | `genetic_prompt_ops.py` | `prompt_distance()` | 8 |
| Crossover | `genetic_prompt_ops.py` | `crossover()` | 18 |
| Mutation | `genetic_prompt_ops.py` | `mutate()` | 40 |
| Seed population | `genetic_prompt_ops.py` | `seed_population()` | 20 |
| Main loop | `pareto_prompt_optimizer.py` | `ParetoPromptOptimizer.run()` | 60 |
| **Total** | | | **~206 LoC** |
