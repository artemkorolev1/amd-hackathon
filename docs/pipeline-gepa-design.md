# Pipeline-Integrated GEPA Design

## Problem
Previous GEPA runs tested the bare model (prompt + question → answer → grade). This misses
pipeline-level failures: classifier misroutes, deterministic solvers not reached, post-processing
bugs. Improving the prompt in isolation doesn't translate to pipeline accuracy.

## Solution
Wrap Pipeline.process() as the evaluation function inside GEPA.

## Architecture

### Genotype
A single candidate is a pair: (category, system_prompt). When evaluated:
1. Inject the candidate system_prompt into the pipeline for that category
2. Run Pipeline.process() on all questions of that category
3. Grade using the official fuzzy_match cascade
4. Return (accuracy, avg_latency, avg_tokens) as objectives

### Modification to Pipeline
Pipeline() already accepts a `routing_table` dict in __init__:
```
pipe = Pipeline(routing_table={"math": {"system_prompt": "...", "decoding": {...}}})
```
When a routing_table entry exists for a category, its system_prompt overrides
the dynamic_prompts.py default. This is the clean injection point — no source
changes needed to the Pipeline class.

### Evaluation Loop
```
for candidate in population:
    pipe = Pipeline(routing_table={category: {"system_prompt": candidate.prompt}})
    pipe._load_local_model()  # loads model once
    results = pipe.process_batch(questions)  # all questions for this category
    candidate.fitness = grade_all(results, expected_answers)
    pipe.close()
```

### Performance Consideration
Pipeline.process() is ~1-3s per question (model loading + inference). For 19 questions × 8
population × 3 generations = ~456 calls → ~10-20 minutes per category. This is acceptable
for GPU runs.

### Seed Prompts
Use the existing _CATEGORY_PROMPTS from dynamic_prompts.py as seed prompts.
Each seed is the `low`, `medium`, `high` prompts for one category.

### Mutation Operators
Same as existing GEPA: prompt shortening/lengthening, constraint injection,
prefix/suffix changes, temperature variation. But the fitness is pipeline accuracy,
not bare-model accuracy.

## Implementation Plan

1. Create agent/pipeline_gepa.py that:
   - Takes a category and a set of candidate system prompts
   - For each candidate: builds a Pipeline with that prompt, runs process_batch, grades
   - Runs the NSGA-II evolutionary loop (same as gepa_category_runner.py)
   - Reports Pareto-optimal prompts + their pipeline accuracy

2. Start with factual (81% baseline, deterministic solver works for most)
3. Then move to math (63% — most room), logic (68%), summarization (75%)

## Key Difference from Bare-Model GEPA
- Each eval loads the full Pipeline (model + classifier + solvers) ≈ 1-3s per question
- Scoring uses the official grader, not the GEPA runner's simple fuzzy_match
- Failures are REAL pipeline failures — misroutes count as wrong answers
- The prompt that scores highest actually works in the deployed system
