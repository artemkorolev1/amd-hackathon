# Genetic-Algorithm-Based Prompt Evolution for Small LLMs on Factual QA

## Methodology

**Target Domain:** Factual QA (19 questions from NQ-Open, training-v3.json)  
**Models:** qwen2.5-1.5b, qwen2.5-math-1.5b, qwen2.5-coder-1.5b, smollm2-1.7b (all GGUF Q4_K_M)  
**Grading:** 4-cascade fuzzy match (exact → substring → numeric 1% tolerance → token overlap ≥70%)  
**Budget:** 3 iterations (rounds) of genetic evolution  
**Goal:** Find model+prompt combo that beats or matches Llama-3.2-1B's ~80% accuracy on factuals  
**Current Baseline Best:** `"Fact:"` or `"Answer the question directly. Use exact names, dates, and numbers. Keep under 15 words. No preamble."`

---

## 1. Prompt as a Genome

A prompt for factual QA is represented as a **fixed-length genome** of 5 ordered slots. Each slot is a categorical or discrete parameter. Representing prompts as structured tuples makes crossover and mutation well-defined.

### Genome Structure

```
[PREFIX, FORMAT_INSTRUCTION, CONSTRAINT_CLAUSE, VERBOSITY_LEVEL, FORMAT_MARKER]
```

| Slot | Description | Allele Options |
|------|-------------|----------------|
| `PREFIX` | Opening word or label | `"Fact:"`, `"Answer:"`, `"Q:"`, `"Respond:"`, `""` (empty) |
| `FORMAT_INSTRUCTION` | How to structure the answer | `"Answer the question directly."`, `"Answer in 1-3 words."`, `"Be concise."`, `"Provide the exact answer."`, `""` |
| `CONSTRAINT_CLAUSE` | Quality guardrails | `"Use exact names, dates, and numbers."`, `"No preamble."`, `"No explanation."`, `"Keep under 15 words."`, `"If unsure, give your best guess."`, `""` |
| `VERBOSITY_LEVEL` | How terse | `"ultra-terse"` (1-3 words), `"terse"` (one sentence), `"moderate"` (2-3 sentences), `"verbose"` (explain briefly) |
| `FORMAT_MARKER` | Ending delimiter | `""` (empty), `":"`, `"\\n"`, `" ->"`, `"Answer:"` |

### Rendering a Genome into a System Prompt

```python
def render_genome(g: dict) -> str:
    """Convert genome tuple into a system prompt string."""
    parts = [
        g["prefix"],
        g["format_instruction"],
        g["constraint_clause"],
        _verbosity_text(g["verbosity_level"]),
        g["format_marker"],
    ]
    # Filter empties, join with spaces, strip
    return " ".join(p for p in parts if p).strip()
```

**Example genome → prompt:**

```
PREFIX="Answer:" + FORMAT="Answer the question directly." + CONSTRAINT="Use exact names." + VERB="ultra-terse" + MARKER=""
→ "Answer: Answer the question directly. Use exact names."
```

### Genome Constraints (Hard Rules)

1. If `VERBOSITY_LEVEL="ultra-terse"`, the `CONSTRAINT_CLAUSE` must contain a word-limit rule (e.g., "Keep under 15 words" or "1-3 words").
2. `FORMAT_MARKER` cannot duplicate `PREFIX` (e.g., "Answer: ... Answer:" is disallowed).
3. Total rendered prompt must be ≤ 200 characters (prevents prompt-smuggling and keeps inference cheap).
4. No two prompts in the same generation may be identical (dedup after generation).

---

## 2. Mutation Operators

Each mutation is applied with a per-slot probability. This design mimics how a human prompt engineer iterates: rephrase, tighten, relax, or restructure.

| Operator | Target Slot(s) | Probability | Effect |
|----------|---------------|-------------|--------|
| **Rephrase** | `FORMAT_INSTRUCTION` | 30% | Replace with a semantically equivalent phrase from a curated thesaurus (see below) |
| **Add Constraint** | `CONSTRAINT_CLAUSE` | 25% | Append one additional constraint from a constraint pool; if already has 3 constraints, randomly replace one |
| **Remove Constraint** | `CONSTRAINT_CLAUSE` | 15% | Remove a randomly chosen constraint; if only 1 constraint left, skip |
| **Swap Verbosity** | `VERBOSITY_LEVEL` | 20% | Move one level: ultra-terse ↔ terse ↔ moderate ↔ verbose (can't jump two levels in one mutation) |
| **Change Format Marker** | `FORMAT_MARKER` | 15% | Replace with a different marker from the allele list; can also set to empty string |
| **Change Prefix** | `PREFIX` | 15% | Pick a different prefix from the allele list |
| **Constraint Swap** | `CONSTRAINT_CLAUSE` | 10% | Replace one constraint with a different one from the pool |
| **Full Shuffle** | All | 5% (per-child) | Re-roll all slots (acts as diversity injection, similar to random restart) |

### Constraint Pool

Curated list of factual-QA-relevant constraints (drawn from ablation results):

```
No preamble.
No explanation.
Use exact names, dates, and numbers.
Keep under 15 words.
Keep under 5 words.
Be precise.
If unsure, give your best guess.
Don't hedge.
Output only the answer.
Be specific.
Use complete sentences if needed.
Address all parts of the question.
```

### Rephrase Thesaurus (for FORMAT_INSTRUCTION)

Each entry is a semantic equivalent:

| Original | Rephrase Options |
|----------|-----------------|
| "Answer the question directly." | "Respond directly to the question.", "Give a direct answer.", "Answer straight.", "Answer clearly." |
| "Answer in 1-3 words." | "Respond in a few words.", "Keep answer to 1-3 words.", "Short answer only." |
| "Be concise." | "Keep it brief.", "Be brief.", "Answer concisely." |
| "Provide the exact answer." | "Give the exact answer.", "State the precise answer.", "Exact answer only." |
| "Think step by step." | "Reason step by step.", "Explain your reasoning.", "Walk through the logic." |

### Mutation Implementation

```python
import random

def mutate(genome: dict, constraint_pool: list, thesaurus: dict) -> dict:
    """Return a mutated copy of the genome."""
    child = dict(genome)
    ops = []

    # Rephrase format instruction
    if random.random() < 0.30:
        current = child["format_instruction"]
        if current in thesaurus and thesaurus[current]:
            child["format_instruction"] = random.choice(thesaurus[current])
            ops.append("rephrase")

    # Add constraint
    if random.random() < 0.25:
        constraints = child["constraint_clause"].split(". ")
        available = [c for c in constraint_pool if c not in constraints]
        if available:
            if len(constraints) >= 3:
                constraints[random.randrange(len(constraints))] = random.choice(available)
            else:
                constraints.append(random.choice(available))
            child["constraint_clause"] = ". ".join(constraints)
            ops.append("add_constraint")

    # Remove constraint
    if random.random() < 0.15:
        constraints = child["constraint_clause"].split(". ")
        if len(constraints) > 1:
            constraints.pop(random.randrange(len(constraints)))
            child["constraint_clause"] = ". ".join(constraints)
            ops.append("remove_constraint")

    # Swap verbosity
    if random.random() < 0.20:
        levels = ["ultra-terse", "terse", "moderate", "verbose"]
        idx = levels.index(child["verbosity_level"])
        delta = random.choice([-1, 1])
        new_idx = max(0, min(len(levels) - 1, idx + delta))
        child["verbosity_level"] = levels[new_idx]
        ops.append("swap_verbosity")

    # Change format marker
    if random.random() < 0.15:
        markers = ["", ":", "\\n", " ->", "Answer:"]
        current = child["format_marker"]
        options = [m for m in markers if m != current]
        child["format_marker"] = random.choice(options)
        ops.append("change_marker")

    # Change prefix
    if random.random() < 0.15:
        prefixes = ["Fact:", "Answer:", "Q:", "Respond:", ""]
        current = child["prefix"]
        options = [p for p in prefixes if p != current]
        child["prefix"] = random.choice(options)
        ops.append("change_prefix")

    # Full shuffle (diversity injection)
    if random.random() < 0.05:
        child = generate_random_genome()
        ops.append("full_shuffle")

    return child
```

---

## 3. Crossover Between Winning Prompts

Uniform crossover at the slot level. Given two parent genomes, each slot in the child is inherited from one parent with 50% probability.

### Crossover Operator

```python
def crossover(parent_a: dict, parent_b: dict) -> dict:
    """Uniform slot-level crossover between two parent genomes."""
    child = {}
    slots = ["prefix", "format_instruction", "constraint_clause", "verbosity_level", "format_marker"]
    for slot in slots:
        child[slot] = parent_a[slot] if random.random() < 0.5 else parent_b[slot]
    return child
```

### Selection of Parents

Only the **top-N** individuals (by fitness) are eligible to be parents for crossover. This ensures convergence pressure while maintaining diversity.

- **Crossover rate:** 60% of the next generation comes from crossover.
- **Remaining 40%:** 25% mutation of top performers, 15% random new genomes (fresh blood).

### Elitism

The single best prompt from each generation is always carried over unchanged (elitism count = 1). This prevents fitness regression.

---

## 4. Tournament Selection vs. Rank-Based

### Recommendation: Tournament Selection (k=3)

For small populations (≤20 individuals) with noisy fitness evaluations (19 questions, stochastic LLM outputs), tournament selection outperforms rank-based selection because:

| Criterion | Tournament (k=3) | Rank-Based |
|-----------|-----------------|------------|
| Selection pressure | Adjustable via k | Fixed by rank distribution |
| Noise tolerance | High — single lucky win doesn't dominate | Lower — rank can be affected by outliers |
| Diversity preservation | Better — weaker individuals have non-zero chance | Worse — bottom ranks may never be selected |
| Computational cost | O(k) per selection | Requires full sort |

**Why k=3 specifically:** With population size 12-20, k=3 gives strong individuals ~70% chance of being selected while retaining ~10% chance for lower-ranked individuals. This balances exploitation vs. exploration.

```python
def tournament_select(population: list, fitness_scores: list, k: int = 3) -> dict:
    """Select one individual via k-way tournament."""
    best = None
    best_fitness = -1
    for _ in range(k):
        idx = random.randrange(len(population))
        if fitness_scores[idx] > best_fitness:
            best_fitness = fitness_scores[idx]
            best = population[idx]
    return best
```

### Why Not Rank-Based

Rank-based selection normalizes by position rather than raw fitness. With only 19 factual questions and scores being small integers (0-19), many individuals may tie. Rank-based breaks ties arbitrarily, which can lose structure (e.g., a prompt with 12/19 may rank identically to one with 11/19 if the score distribution is sparse). Tournament selection naturally handles ties because tied individuals compete in the same tournament and either may win.

---

## 5. Population Sizing for a 3-Iteration Budget

### Constraint Analysis

- **Total evaluations available:** 3 iterations × population_size evaluations per iteration
- **Each evaluation costs:** ~1-3 seconds per question × 19 questions ≈ 19-57 seconds per prompt (depending on model and hardware)
- **Practical limit:** With 4 models × population_size prompts × 3 iterations, we cannot exceed roughly 200-300 total evaluations before time becomes prohibitive.

### Recommended Population Sizes

| Strategy | Population | Generations | Total Evals | Notes |
|----------|-----------|-------------|-------------|-------|
| **Conservative** | 6 | 3 | 18 | Very small; high risk of premature convergence |
| **Balanced (recommended)** | 12 | 3 | 36 | Good diversity; run 3 parallel trials per model |
| **Aggressive** | 20 | 3 | 60 | High diversity; may not converge in 3 rounds |

### Per-Model Allocation

For each of the 4 models, run an independent GA with population_size=12:

```
Total evaluations = 4 models × 12 pop × 3 iterations = 144 prompt evaluations
Each eval = 19 questions × ~2s = ~38s → ~1.5 hours total wall time (sequential)
With parallelism (4 model servers): ~22 minutes
```

### Population Initialization

Generation 0 is seeded with carefully chosen prompts rather than fully random:

1. **The known best:** "Fact:" (ultra-minimal) and the current "Answer the question directly..." prompt (2 seeds)
2. **Variant hand-crafted prompts** that performed decently in ablation: "Answer:", "Respond:", "Q:" prefixes (3 seeds)
3. **Randomized genomes** from the allele space (remaining 7 seeds)

This warm-start avoids wasting iterations on obviously bad prompts (like no prefix + verbose + no constraints) that we already know fail.

---

## 6. Convergence Detection

### Primary Criterion: Top-3 Accuracy Delta

**Stop when the mean accuracy of the top-3 prompts differs by < 5 percentage points between consecutive rounds.**

```python
def check_convergence(history: list, round_num: int, threshold: float = 5.0) -> bool:
    """
    Check if GA has converged.
    
    history: list of dicts, one per round, each with {"top3_mean": float}
    round_num: current round (0-indexed)
    threshold: max allowed delta in percentage points
    
    Returns True if converged (should stop).
    """
    if round_num < 1:
        return False
    prev = history[round_num - 1]["top3_mean"]
    curr = history[round_num]["top3_mean"]
    delta = abs(curr - prev)
    print(f"  Top-3 accuracy delta: {delta:.1f} pp (threshold: {threshold})")
    return delta < threshold
```

### Secondary Indicators

| Indicator | Trigger | Action |
|-----------|---------|--------|
| Best prompt unchanged for 2 rounds | Elitism carries same prompt forward | Log warning, reduce mutation rate for next round |
| Population entropy (genetic diversity) < 20% | Fewer than 3 unique genome structures remain | Inject 2 random new individuals (immigration) |
| All prompts converge on same prefix + marker | Over-specialization | Apply full_shuffle mutation to bottom 3 individuals |
| Best accuracy < 30% at round 2 | Population is stuck in bad region | Re-seed with 3 new hand-crafted prompts, kill bottom 50% |

### Diversity Tracking

Compute **allele frequency** per slot each round:

```python
def diversity_score(population: list, slot: str) -> float:
    """Shannon entropy of allele distribution for a given slot."""
    from collections import Counter
    import math
    alleles = [ind[slot] for ind in population]
    counts = Counter(alleles)
    total = len(alleles)
    entropy = -sum((c/total) * math.log2(c/total) for c in counts.values())
    max_entropy = math.log2(len(alleles))
    return entropy / max_entropy if max_entropy > 0 else 0.0
```

If `diversity_score(population, "format_marker") < 0.20`, force-mutate 2 individuals' format markers.

---

## 7. Pitfalls When Evolving Prompts for 1B-Class Models

Small models (1-2B parameters) are qualitatively different from large models (70B+) for prompt optimization. These pitfalls are documented from empirical observation across the ablation rounds.

### Pitfall 1: Wording Sensitivity

1B models are extremely sensitive to single-word changes. In the ablation data:
- `"Fix:"` → 80% accuracy on code_debug (4/5)
- `"Fix the bug. Output the fixed function..."` → 40% accuracy (2/5)

**Lesson:** Verbose prompts often confuse small models. Ultra-minimal prompts (`"Fact:"`, `"Fix:"`, `"Answer:"`) consistently outperform elaborated instructions. The GA should **strongly prefer shorter prompts** and penalize verbose genomes.

### Pitfall 2: Overfitting to the 19-Question Set

With only 19 factual questions, a prompt that happens to work on 15 of them might be memorizing patterns rather than generalizing. Symptoms from ablation:
- `"Answer in 1-3 words if possible..."` scored 0/5 (all wrong), but `"Answer the question directly..."` scored 2/5 — same model, similar verbosity, wildly different results.

**Mitigation:**
- Run **3 separate GA trials** with different random seeds per model
- Select the final prompt by majority vote across trials, NOT the single best round-3 score
- After GA completes, validate the winner on a held-out set of 5 factual questions (separate from the 19)

### Pitfall 3: Constraint Contradiction

Small models struggle with contradictory or nested constraints. Example:
- `"Keep under 15 words. No preamble. Answer the question directly."` — this worked OK (40%).
- `"Answer in 1-3 words if possible. Exact fact only. No explanation."` — scored 0%.

**Hypothesis:** "If possible" introduces ambiguity that 1B models can't resolve. The GA's constraint-pool design should avoid hedging language ("if possible", "try to", "you may").

### Pitfall 4: Format Marker Fragility

Small models are highly sensitive to format markers:
- `"Answer:"` prefix without a following space after colon caused some models to concatenate the prefix with the answer.
- Trailing `":"` vs `": "` vs `""` can shift accuracy by 1-2 questions (5-10 pp).

**Fix:** Normalize format during rendering: always add exactly one space after a colon prefix.

### Pitfall 5: Temperature Sensitivity

Ablation used temperature=0.0 throughout. For the GA, temperature=0.0 is correct: we want deterministic evaluation of prompt quality. Sampling noise from temperature>0 would mask the actual prompt fitness.

**Exception:** During convergence, if top-3 accuracy delta < 5%, run a **tiebreak evaluation** at temperature=0.1 with 3 samples, using majority vote. This can disambiguate equally-good prompts.

### Pitfall 6: Model-Specific Prompt Affinities

From ablation data, each model has a "sweet spot":

| Model | Best Prompt Style | Accuracy (factual) |
|-------|-------------------|-------------------|
| qwen2.5-1.5b | Moderate constraints, medium length (15-30 words) | 40% (v1-basic) |
| qwen2.5-math-1.5b | Not yet tested on factuals; excels at structured tasks | Unknown |
| qwen2.5-coder-1.5b | Not yet tested on factuals; trained on code, may prefer structured formats | Unknown |
| smollm2-1.7b | Not yet tested on factuals; small model, may prefer ultra-short prompts | Unknown |

**Action:** Run the GA independently per model (not jointly). Each model may converge to a completely different prompt structure. The final deliverable is the best model+prompt combo.

### Pitfall 7: GGUF Quantization Noise

Q4_K_M quantization adds non-deterministic noise. Even at temperature=0.0, GGUF models can produce slightly different outputs due to GPU kernel nondeterminism. This means fitness evaluations have inherent variance.

**Mitigation:**
- Run each prompt through **all 19 questions twice** and take the average score (if nondeterminism is observed).
- Use a fixed random seed for the LLM inference backend.

---

## 8. Complete Algorithm Pseudocode

```python
def run_genetic_evolution(
    model_name: str,
    questions: list[dict],          # training-v3.json filtered to factual
    population_size: int = 12,
    generations: int = 3,
    mutation_rate_adj: float = 1.0,  # adjust mutation rates
    convergence_threshold: float = 5.0,
    seed_prompts: list[str] = None,
) -> tuple[dict, list]:
    """
    Evolve a system prompt for a given model on factual QA.
    
    Returns:
        best_genome: the genome with highest fitness across all generations
        history: per-round metrics
    """
    # 1. Initialize population
    population = initialize_population(population_size, seed_prompts)
    
    history = []
    
    for gen in range(generations):
        print(f"\n=== Generation {gen+1} ===")
        
        # 2. Evaluate fitness for each individual
        fitness_scores = []
        for genome in population:
            prompt = render_genome(genome)
            accuracy = evaluate_on_factuals(model_name, prompt, questions)
            fitness_scores.append(accuracy)
            # Penalize length (>200 chars)
            if len(prompt) > 200:
                fitness_scores[-1] *= 0.8
        
        # 3. Track metrics
        sorted_idx = sorted(range(len(fitness_scores)), 
                          key=lambda i: fitness_scores[i], reverse=True)
        top3_mean = sum(fitness_scores[i] for i in sorted_idx[:3]) / 3.0
        top1_idx = sorted_idx[0]
        
        history.append({
            "generation": gen + 1,
            "best_fitness": fitness_scores[top1_idx],
            "best_prompt": render_genome(population[top1_idx]),
            "top3_mean": top3_mean,
            "population_fitness": list(fitness_scores),
            "diversity": {slot: diversity_score(population, slot) 
                         for slot in ["prefix", "format_marker"]},
        })
        
        # 4. Check convergence
        if check_convergence(history, gen, convergence_threshold):
            print(f"  Converged at generation {gen+1}")
            break
        
        # 5. Generate next population
        next_population = []
        
        # Elitism: carry over best individual
        next_population.append(population[top1_idx])
        
        # Crossover (60% of remaining slots)
        while len(next_population) < population_size:
            if len(next_population) < population_size * 0.6:
                parent_a = tournament_select(population, fitness_scores, k=3)
                parent_b = tournament_select(population, fitness_scores, k=3)
                child = crossover(parent_a, parent_b)
                child = mutate(child, CONSTRAINT_POOL, THESAURUS)
            elif len(next_population) < population_size * 0.85:
                # Mutation of top performers
                parent = tournament_select(population, fitness_scores, k=3)
                child = mutate(parent, CONSTRAINT_POOL, THESAURUS)
            else:
                # Random fresh blood
                child = generate_random_genome()
            next_population.append(child)
        
        population = next_population
    
    # Return best genome across all generations
    all_genomes = []
    all_fitness = []
    for g_idx, f_scores in enumerate(history):
        for p_idx, genome in enumerate(population if g_idx == len(history)-1 else []):
            pass  # need to store per-generation populations
    # Simplified: return best from final generation
    best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
    return population[best_idx], history
```

---

## 9. Prior Art & Connections

### Automated Prompt Optimization (APE)

**Zhou et al. (2023)** "Automatic Prompt Engineer" — APE uses LLMs to generate and score prompt candidates. Key insight: for factual QA, "direct" prompts (like "Answer:") consistently outperform "instruction" prompts (like "Provide the correct answer to this question"). Our ablation data confirms this: the minimal prompts (`"Fact:"`, `"Answer:"`) dramatically outperform elaborated ones.

**Connection to GA:** APE's random candidate generation is essentially mutation without crossover. Our GA adds crossover to recombine features from winning prompts, which should converge faster.

### OPRO

**Yang et al. (2024)** "Large Language Models as Optimizers" — OPRO uses an LLM to iteratively refine prompts by describing past attempts and their scores. This is similar to our GA's mutation step, but OPRO uses natural language description of the optimization trajectory rather than formal genetic operators.

**Key difference:** OPRO requires an LLM to generate the next candidates (expensive). Our GA uses pre-defined operators (cheap, deterministic). For 1B models where we can't afford to run an LLM-based optimizer, the GA approach is more practical.

### DSPy

**Khattab et al. (2023/2024)** — DSPy treats prompts as optimizable parameters using Bayesian optimization or random search. Its `dspy.teleprompt` module supports:
- `BootstrapFewShot`: Iteratively improves examples
- `BootstrapFewShotWithRandomSearch`: Random prompt variations
- `MIPRO`: Bayesian optimization over prompt structure

**Connection to GA:** DSPy's random search is a simpler form of our GA (mutation only, no crossover). Our approach adds structured crossover and tournament selection. DSPy's `MIPRO` uses Bayesian optimization which is more sample-efficient but requires a surrogate model — for 19 questions, the overhead isn't worth it.

### Prompt Engineering Competition Learnings

From the AMD Hackathon context (this project is an AMD hackathon entry), winning prompt strategies for small models on factual QA follow consistent patterns:

1. **Ultra-minimal prompts dominate.** The ablation data is clear: `"Fix:"` (2 chars!) gets 80% accuracy, while the multi-sentence instruction gets 40%. This contradicts conventional wisdom that "more detailed prompts are better."
2. **Prefix matters.** `"Fact:"` vs `"Answer:"` vs no prefix can shift accuracy by 10-20 points.
3. **Single constraint beats multiple constraints.** "Use exact names, dates, and numbers. Keep under 15 words. No preamble." is probably overwrought. One strong constraint may be better than three.
4. **Format markers are double-edged.** Adding `" ->"` or `"Answer:"` as a marker can help extract the answer token, but can also confuse small models into repeating the marker.

### Key References

| Paper/Method | Year | Key Technique | Relevance |
|-------------|------|---------------|-----------|
| APE (Zhou et al.) | 2023 | LLM-generated prompt candidates | Prompt as genome concept |
| OPRO (Yang et al.) | 2024 | LLM-in-the-loop optimization | Iterative refinement loop |
| DSPy (Khattab et al.) | 2024 | Programmatic prompt optimization | Random search + bootstrapping |
| TextGrad (Yuksekgonul et al.) | 2024 | Gradient-based prompt optimization | Mutation operators inspiration |
| EvoPrompt (Bai et al.) | 2024 | Genetic algorithm for discrete prompts | Direct prior art for our approach |

---

## 10. Evaluation Protocol

### Per-Round Evaluation

```python
def evaluate_on_factuals(model_name: str, system_prompt: str, 
                         questions: list[dict], num_runs: int = 1) -> float:
    """
    Evaluate a prompt on all factual questions.
    
    Returns accuracy as a float 0.0-1.0.
    """
    from llama_cpp import Llama
    
    model = load_model(model_name)  # cached
    correct = 0
    
    for q in questions:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q["prompt"]},
        ]
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=64,
            temperature=0.0,
        )
        answer = response["choices"][0]["message"]["content"].strip()
        
        if fuzzy_match(answer, q["expected_answer"]):
            correct += 1
    
    return correct / len(questions)
```

### Grading Cascade (reused from existing ablation script)

```python
def fuzzy_match(answer: str, expected: str) -> bool:
    """4-cascade fuzzy match from existing ablation infrastructure."""
    a, e = answer.strip().lower(), expected.strip().lower()
    if not a or not e:
        return False
    # 1. Exact match
    if a == e:
        return True
    # 2. Substring (short strings)
    if len(e) <= 20 and e in a:
        return True
    if len(a) <= 20 and a in e:
        return True
    # 3. Numeric tolerance (1%)
    import re
    nums_a = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", a)]
    nums_e = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", e)]
    if nums_a and nums_e:
        an, en = nums_a[-1], nums_e[-1]
        if en != 0 and abs((an - en) / en) <= 0.01:
            return True
        if an == en:
            return True
    # 4. Token overlap >= 70%
    ta = set(tok for tok in re.split(r"[^a-zA-Z0-9.]+", a) if tok)
    te = set(tok for tok in re.split(r"[^a-zA-Z0-9.]+", e) if tok)
    if len(te) > 0 and len(ta & te) / len(te) >= 0.70:
        return True
    return False
```

### Held-Out Validation

After 3 GA rounds, reserve 5 of the 19 factual questions for validation (randomly selected, same across all trials). Train on the remaining 14. This reduces overfitting risk. If validation accuracy is significantly lower than training accuracy (>15 pp drop), discard the prompt and re-run with higher mutation rate.

---

## 11. Expected Output Artifacts

After running the GA for all 4 models:

```
results/
  genetic_evolution/
    qwen2.5-1.5b/
      trial_0_best_prompt.json
      trial_1_best_prompt.json  
      trial_2_best_prompt.json
      history.json          # per-round metrics
    qwen2.5-math-1.5b/
      ...
    qwen2.5-coder-1.5b/
      ...
    smollm2-1.7b/
      ...
    summary.csv             # model, best_prompt, accuracy, convergence_round
    winning_combo.json      # overall best model + prompt
```

Each `best_prompt.json`:

```json
{
  "model": "qwen2.5-1.5b",
  "trial": 0,
  "genome": {
    "prefix": "Fact:",
    "format_instruction": "Answer the question directly.",
    "constraint_clause": "Use exact names, dates, and numbers.",
    "verbosity_level": "ultra-terse",
    "format_marker": ""
  },
  "rendered_prompt": "Fact: Answer the question directly. Use exact names, dates, and numbers.",
  "training_accuracy": 0.78,
  "validation_accuracy": 0.60,
  "convergence_round": 2,
  "top3_delta": 2.3
}
```

---

## 12. Implementation Checklist (for the next stage)

- [ ] `genome.py` — Genome dataclass, `render_genome()`, `generate_random_genome()`
- [ ] `mutation.py` — `mutate()`, constraint pool, rephrase thesaurus
- [ ] `crossover.py` — `crossover()`, uniform slot-level
- [ ] `selection.py` — `tournament_select()`, `rank_select()`
- [ ] `evaluate.py` — `evaluate_on_factuals()`, wraps llama-cpp-python inference
- [ ] `ga_loop.py` — `run_genetic_evolution()`, convergence check, diversity tracking
- [ ] `run_experiment.py` — Orchestrates 4 models × 3 trials, saves results
- [ ] `config.py` — All constants: population size, mutation rates, thresholds, model paths, question paths
- [ ] `test_genetic.py` — Unit tests for genome, mutation, crossover, selection
