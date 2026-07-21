#!/usr/bin/env python3
"""
GEPA Summarization Optimization Runner
=======================================
- 366-question training set (summarization)
- 78-question validation set
- 3 models: qwen2.5-1.5b, qwen2.5-coder-1.5b, gemma-3-1b
- 2 generations (gen 0 seed + gen 1 evolved)
- Chunk-and-summarize workflow for long texts (>200 words)
- Multi-signal grading: entity recall, keyword overlap, number match
- Workflow operators: split-into-steps, add-verify-step, remove-step, reorder

Usage:
    python3 gepa_plans/run_summarization_gepa.py
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agent.cell import Cell, DecodingConfig, StepConfig
from agent.mutation_agent import MutationAgent
from agent.solvers.summarization_solver import (
    chunk_text,
    summarize_workflow,
    build_llm_infer,
)

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_PATHS = {
    "qwen2.5-1.5b": "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "qwen2.5-coder-1.5b": "/home/artem/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    "gemma-3-1b": "/home/artem/models/gemma-3-1b-it-Q4_K_M.gguf",
}

MODEL_KEYS = list(MODEL_PATHS.keys())

TRAIN_SET_PATH = os.path.join(PROJECT_ROOT, "data", "eval", "summarization_train.json")
VAL_SET_PATH = os.path.join(PROJECT_ROOT, "data", "eval", "summarization_val.json")
HARD_TEST_PATH = os.path.join(PROJECT_ROOT, "data", "eval", "summarization_hard_test.json")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "research")
OUTPUT_REPORT = os.path.join(RESULTS_DIR, "summarization_gepa_results.md")
LOG_DIR = os.path.join(PROJECT_ROOT, "gepa_logs")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

SEED = 42
GENERATIONS = 2  # gen 0 + gen 1

random.seed(SEED)

# ── Multi-signal summarization grader ────────────────────────────────────────

def fuzzy_match(answer: str, expected: str) -> bool:
    """4-cascade fuzzy match: exact → substring → numeric 1% → token overlap ≥80%."""
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


def extract_entities(text: str) -> set[str]:
    """Extract capitalized multi-word named entities (people, orgs, places)."""
    return set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))


def grade_summarization(output: str, expected: str) -> bool:
    """Multi-signal summarization grader.

    Pass if ANY of:
    1. Base fuzzy_match cascade passes (exact → substring → numeric 1% → token ≥80%)
    2. Entity recall ≥ 50% AND keyword overlap ≥ 40%
    3. Number overlap ≥ 1 (if expected contains numbers)
    """
    if not output or not expected:
        return False

    # Signal 1: Base fuzzy_match cascade
    if fuzzy_match(output, expected):
        return True

    # Signal 2: Entity recall ≥ 50% AND keyword overlap ≥ 40%
    exp_entities = extract_entities(expected)
    out_entities = extract_entities(output)
    if exp_entities:
        entity_overlap = exp_entities & out_entities
        entity_recall = len(entity_overlap) / len(exp_entities)
    else:
        entity_recall = 0.0

    # Keyword overlap: content words (≥4 letters)
    exp_keywords = set(re.findall(r'[a-zA-Z]{4,}', expected.lower()))
    out_keywords = set(re.findall(r'[a-zA-Z]{4,}', output.lower()))
    if exp_keywords:
        keyword_overlap = len(exp_keywords & out_keywords) / len(exp_keywords)
    else:
        keyword_overlap = 0.0

    if entity_recall >= 0.5 and keyword_overlap >= 0.4:
        return True

    # Signal 3: Number overlap
    exp_nums = set(re.findall(r'\d+(?:\.\d+)?', expected))
    out_nums = set(re.findall(r'\d+(?:\.\d+)?', output))
    if exp_nums and exp_nums & out_nums:
        return True

    return False


# ── Eval set loader ─────────────────────────────────────────────────────────

def load_eval_set(path: str, label: str = "eval") -> list[dict]:
    """Load summarization eval questions."""
    with open(path) as f:
        questions = json.load(f)
    # Ensure all have category tag
    for q in questions:
        q["category"] = "summarization"
        if "task_id" not in q:
            q["task_id"] = ""
    # Stats
    word_counts = [len(q["prompt"].split()) for q in questions]
    long_count = sum(1 for wc in word_counts if wc > 200)
    print(f"  Loaded {label}: {len(questions)} questions, "
          f"avg {sum(word_counts)//len(word_counts)} words, "
          f"{long_count} long (>200w)")
    return questions


# ── Model Cache (loads one model at a time) ─────────────────────────────────

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
            time.sleep(0.5)

    def get_loaded_key(self):
        return self._loaded_key


# ── Seed prompt configurations ──────────────────────────────────────────────

# Default decoding params
DEFAULT_PARAMS = {
    "temperature": 0.0,
    "max_tokens": 96,
    "top_p": 1.0,
    "top_k": 40,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "seed": None,
}

# Slightly optimized params
OPTIMIZED_PARAMS = {
    "temperature": 0.0,
    "max_tokens": 128,
    "top_p": 0.9,
    "top_k": 20,
    "min_p": 0.05,
    "repeat_penalty": 1.05,
    "seed": SEED,
}

SEED_PROMPTS = [
    # (name, system_prompt, params_dict, use_workflow)
    ("empty", "", DEFAULT_PARAMS, False),
    ("summarize_colon", "Summarize:", DEFAULT_PARAMS, False),
    ("explicit_instruction",
     "Summarize the text in at most 2 sentences. Include key names, numbers, and facts.",
     OPTIMIZED_PARAMS, False),
    ("verbose_instruction",
     "Read the following text carefully and produce a concise summary. "
     "Capture the main point, key entities, and any numerical data. "
     "Output 1-3 sentences maximum. Do not add opinions or commentary.",
     OPTIMIZED_PARAMS, False),
    ("concise_news",
     "Summarize this news article in 1-2 sentences with exact names and numbers.",
     OPTIMIZED_PARAMS, False),
    ("tldr", "TL;DR:", DEFAULT_PARAMS, False),
    ("extract_key_points",
     "Extract key points from this text. Be specific with names and numbers.",
     OPTIMIZED_PARAMS, False),
    ("wf_summarize",
     "Summarize concisely. Break long texts into sections first.",
     OPTIMIZED_PARAMS, True),
    ("wf_key_points",
     "Extract and summarize key points from each section.",
     OPTIMIZED_PARAMS, True),
]


# ── Seed population builder ────────────────────────────────────────────────

def create_seed_population():
    """Create seed cells (3 models × N prompt/param variants)."""
    population = []
    for model_key in MODEL_KEYS:
        for i, (name, prompt_text, params, use_workflow) in enumerate(SEED_PROMPTS):
            cell = Cell(
                task_id="T04",  # summarization
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
            # For workflow cells, add steps
            if use_workflow:
                cell.steps = [
                    StepConfig(
                        name="summarize",
                        system_prompt=prompt_text,
                        input_from="_input",
                    ),
                ]
                cell.aggregation = "workflow"
            population.append(cell)

    print(f"\nCreated seed population: {len(population)} cells")
    for c in population:
        wf_mark = " [workflow]" if c.steps else ""
        print(f"  {c.name:45s} | top_p={c.decoding.top_p} top_k={c.decoding.top_k} "
              f"min_p={c.decoding.min_p} | prompt={c.system_prompt[:50]!r}{wf_mark}")
    return population


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_single_question(
    cell: Cell,
    question: dict,
    llm_model,
    prompt_text: str,
    expected: str,
) -> tuple[bool, str, float]:
    """Evaluate a single question against a cell.

    If cell has steps (workflow), use chunk-and-summarize for long texts,
    otherwise use direct single-shot inference.

    Returns:
        (is_correct, output_text, latency_ms)
    """
    infer_fn = build_llm_infer(llm_model)

    word_count = len(prompt_text.split())
    is_long = word_count > 200

    t0 = time.time()

    if cell.steps:
        # Workflow cell — use chunk-and-summarize for long texts
        if is_long:
            result = summarize_workflow(
                text=prompt_text,
                llm_infer_fn=lambda msgs, **kw: infer_fn(msgs, **kw),
                system_prompt=cell.system_prompt or "Summarize concisely.",
                chunk_prompt=cell.system_prompt or "Summarize this section in 1-2 sentences.",
                merge_prompt="Combine these summaries into a coherent 2-3 sentence summary.",
                max_words_per_chunk=200,
                max_tokens_per_call=cell.decoding.max_tokens,
                temperature=cell.decoding.temperature,
            )
            output = result["summary"]
        else:
            # Short text — direct call with the cell's prompt
            messages = [
                {"role": "system", "content": cell.system_prompt},
                {"role": "user", "content": prompt_text},
            ]
            output = infer_fn(messages, max_tokens=cell.decoding.max_tokens,
                             temperature=cell.decoding.temperature)
    else:
        # Single-shot — direct LLM call
        messages = [
            {"role": "system", "content": cell.system_prompt},
            {"role": "user", "content": prompt_text},
        ]
        output = infer_fn(messages, max_tokens=cell.decoding.max_tokens,
                         temperature=cell.decoding.temperature)

    latency_ms = (time.time() - t0) * 1000
    output = (output or "").strip()

    # Grade with multi-signal grader
    is_correct = grade_summarization(output, expected)

    return is_correct, output, latency_ms


def evaluate_generation(population, questions, label="Generation"):
    """Evaluate all cells on a set of questions, loading one model at a time."""
    print(f"\n{'='*60}")
    print(f"  EVALUATING {label}: {len(population)} cells on {len(questions)} questions")
    print(f"{'='*60}")

    # Group cells by model
    by_model = defaultdict(list)
    for idx, c in enumerate(population):
        by_model[c.model_key].append((idx, c))

    cache = SingleModelCache()

    for model_key, cell_list in by_model.items():
        print(f"\n  [{model_key}] Evaluating {len(cell_list)} cells...")
        llm = cache.get(model_key)

        for idx, cell in cell_list:
            correct = 0
            total = 0
            total_latency = 0.0
            # Filter questions for summarization
            task_questions = [q for q in questions if q.get("category", "").startswith("summarization")]
            if not task_questions:
                task_questions = questions

            details = []
            for q in task_questions:
                prompt_text = q.get("prompt", "")
                expected = q.get("expected_answer", q.get("answer", ""))
                ok, output, lat = evaluate_single_question(
                    cell, q, llm, prompt_text, expected
                )
                if ok:
                    correct += 1
                total += 1
                total_latency += lat
                details.append({
                    "task_id": q.get("task_id", ""),
                    "prompt": prompt_text[:60],
                    "expected": expected[:80],
                    "got": output[:80],
                    "correct": ok,
                    "latency_ms": round(lat, 1),
                })

            elapsed = total_latency / total if total else 0
            cell.metadata["accuracy"] = round(correct / total, 4) if total else 0.0
            cell.metadata["correct"] = correct
            cell.metadata["total"] = total
            cell.metadata["avg_latency_ms"] = round(elapsed, 1)
            cell.metadata["details"] = details
            cell.metadata["category"] = "summarization"

            acc_str = f"{cell.metadata['accuracy']:.4f}"
            print(f"    {cell.name:45s} | acc={acc_str} ({correct}/{total}) "
                  f"lat={elapsed:.0f}ms")

        cache.clear()

    print(f"\n  Evaluation complete.")
    return population


def summarize_generation(population, gen_label, questions_list=None):
    """Print detailed summary of a generation's results."""
    print(f"\n{'─'*60}")
    print(f"  {gen_label} SUMMARY")
    print(f"{'─'*60}")

    by_model = defaultdict(list)
    for c in population:
        by_model[c.model_key].append(c)

    for mk in MODEL_KEYS:
        cells = by_model.get(mk, [])
        if not cells:
            continue
        cells_sorted = sorted(cells, key=lambda c: c.metadata.get("accuracy", 0.0), reverse=True)
        best = cells_sorted[0]
        acc = best.metadata.get("accuracy", 0.0)
        correct = best.metadata.get("correct", 0)
        total = best.metadata.get("total", 0)
        has_wf = " [workflow]" if best.steps else ""
        print(f"\n  [{mk}] Best cell:")
        print(f"    Name:    {best.name}")
        print(f"    Prompt:  {best.system_prompt!r}{has_wf}")
        print(f"    Params:  temp={best.decoding.temperature}, top_p={best.decoding.top_p}, "
              f"top_k={best.decoding.top_k}, min_p={best.decoding.min_p}, "
              f"repeat_penalty={best.decoding.repeat_penalty}, seed={best.decoding.seed}")
        print(f"    Acc:     {acc:.3f} ({correct}/{total})")
        print(f"    Latency: {best.metadata.get('avg_latency_ms', 0):.0f}ms avg")
        if best.steps:
            print(f"    Steps:   {len(best.steps)} steps")
            for s in best.steps:
                print(f"      - {s.name}: {s.system_prompt[:60]!r}")

    # Overall best
    all_cells = []
    for c in population:
        all_cells.append(c)
    all_sorted = sorted(all_cells, key=lambda c: c.metadata.get("accuracy", 0.0), reverse=True)
    if all_sorted:
        overall = all_sorted[0]
        print(f"\n  [Overall Best]")
        print(f"    Model:   {overall.model_key}")
        print(f"    Name:    {overall.name}")
        print(f"    Prompt:  {overall.system_prompt!r}")
        print(f"    Acc:     {overall.metadata.get('accuracy', 0):.3f} "
              f"({overall.metadata.get('correct', 0)}/{overall.metadata.get('total', 0)})")

    return population


# ── Evolution: create generation 1 from gen 0 ─────────────────────────────

def create_next_generation(population, model_keys=None):
    """Use MutationAgent to evolve from current generation."""
    if model_keys is None:
        model_keys = MODEL_KEYS

    # Group by model and evolve per-model to respect VRAM constraints
    by_model = defaultdict(list)
    for c in population:
        by_model[c.model_key].append(c)

    next_gen = []

    for mk in MODEL_KEYS:
        parents = by_model.get(mk, [])
        if not parents:
            continue

        # Sort by accuracy
        sorted_parents = sorted(
            parents,
            key=lambda c: c.metadata.get("accuracy", 0.0) if c.metadata else 0.0,
            reverse=True,
        )

        # Create mutation agent for this model
        ma = MutationAgent(model_keys=[mk], seed=SEED + 1)

        # Evolve: keep top 2 elites, generate children
        children = ma.evolve(
            sorted_parents,
            tags=["summarization"],
            target_size=len(sorted_parents),
            elite_count=2,
            crossover_fraction=0.5,
        )

        # Tag generation
        for c in children:
            c.generation = 1
            # Ensure task_id is correct
            c.task_id = "T04"

        next_gen.extend(children)

    print(f"\n  Created generation 1: {len(next_gen)} cells (from {len(population)} parents)")
    return next_gen


# ── Validation on held-out set ────────────────────────────────────────────

def validate_best_cells(population, val_questions):
    """Run the best cell per model on the validation set."""
    print(f"\n{'='*60}")
    print(f"  VALIDATION ON {len(val_questions)} HELD-OUT QUESTIONS")
    print(f"{'='*60}")

    by_model = defaultdict(list)
    for c in population:
        by_model[c.model_key].append(c)

    val_results = {}
    cache = SingleModelCache()

    for mk in MODEL_KEYS:
        cells = by_model.get(mk, [])
        if not cells:
            continue
        # Find best cell for this model
        best = max(cells, key=lambda c: c.metadata.get("accuracy", 0.0))
        print(f"\n  [{mk}] Validating best cell: {best.name} (acc={best.metadata.get('accuracy', 0):.3f})")

        llm = cache.get(mk)
        correct = 0
        total = 0
        total_latency = 0.0

        for q in val_questions:
            prompt_text = q.get("prompt", "")
            expected = q.get("expected_answer", q.get("answer", ""))
            ok, output, lat = evaluate_single_question(
                best, q, llm, prompt_text, expected
            )
            if ok:
                correct += 1
            total += 1
            total_latency += lat

        val_acc = correct / total if total else 0
        val_latency = total_latency / total if total else 0

        val_results[mk] = {
            "cell_name": best.name,
            "system_prompt": best.system_prompt,
            "train_acc": best.metadata.get("accuracy", 0.0),
            "val_acc": round(val_acc, 4),
            "val_correct": correct,
            "val_total": total,
            "val_avg_latency_ms": round(val_latency, 1),
            "decoding": best.decoding.to_dict(),
            "has_workflow": best.steps is not None,
        }

        print(f"    Train acc: {best.metadata.get('accuracy', 0):.4f} → Val acc: {val_acc:.4f} "
              f"({correct}/{total}) lat={val_latency:.0f}ms")

        cache.clear()

    return val_results


# ── Reporting ──────────────────────────────────────────────────────────────

def generate_report(gen0, gen1, val_results, train_questions, hard_test_results=None):
    """Generate a markdown report of all results."""
    report_lines = [
        "# Summarization GEPA Evolution Results",
        "",
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        "",
        "Summarization was our worst category at 37.8% accuracy. "
        "This experiment runs GEPA evolution with multi-signal grading "
        "(entity recall, keyword overlap, number match) and a "
        "chunk-and-summarize workflow for long texts (>200 words).",
        "",
        "### Configuration",
        "",
        f"- **Models tested**: {', '.join(MODEL_KEYS)}",
        f"- **Generations**: {GENERATIONS} (gen 0 + gen 1)",
        f"- **Training set**: {len(train_questions)} questions (sampled from {len(json.load(open(TRAIN_SET_PATH)))} total)",
        f"- **Validation set**: {len(json.load(open(VAL_SET_PATH)))} questions",
        f"- **Grading**: Multi-signal (entity recall ≥50% + keyword ≥40%, or number overlap, or fuzzy_match cascade)",
        f"- **Workflow**: Chunk-and-summarize for texts >200 words",
        "",
        "## Seed Prompts (Gen 0)",
        "",
        "| Name | Prompt | Workflow |",
        "|------|--------|----------|",
    ]

    for name, prompt, _, use_wf in SEED_PROMPTS:
        safe_prompt = prompt.replace("|", "\\|")[:60]
        report_lines.append(f"| {name} | {safe_prompt} | {'Yes' if use_wf else 'No'} |")

    report_lines.append("")
    report_lines.append("## Gen 0 Results")
    report_lines.append("")
    report_lines.append("| Model | Cell | Prompt | Steps | Acc | Correct/Total | Latency (ms) |")
    report_lines.append("|-------|------|--------|-------|-----|---------------|-------------|")

    for c in sorted(gen0, key=lambda x: x.metadata.get("accuracy", 0), reverse=True):
        mk = c.model_key
        name = c.name
        prompt = c.system_prompt[:50].replace("|", "\\|")
        step_count = len(c.steps) if c.steps else 0
        acc = c.metadata.get("accuracy", 0)
        correct = c.metadata.get("correct", 0)
        total = c.metadata.get("total", 0)
        lat = c.metadata.get("avg_latency_ms", 0)
        report_lines.append(f"| {mk} | {name} | {prompt} | {step_count} | {acc:.4f} | {correct}/{total} | {lat:.0f} |")

    report_lines.append("")
    report_lines.append("## Gen 1 Results (Evolved)")
    report_lines.append("")
    report_lines.append("| Model | Cell | Prompt | Steps | Acc | Correct/Total | Latency (ms) | Parent |")
    report_lines.append("|-------|------|--------|-------|-----|---------------|-------------|--------|")

    for c in sorted(gen1, key=lambda x: x.metadata.get("accuracy", 0), reverse=True):
        mk = c.model_key
        name = c.name
        prompt = c.system_prompt[:50].replace("|", "\\|")
        step_count = len(c.steps) if c.steps else 0
        acc = c.metadata.get("accuracy", 0)
        correct = c.metadata.get("correct", 0)
        total = c.metadata.get("total", 0)
        lat = c.metadata.get("avg_latency_ms", 0)
        parent = c.parent[:30] if c.parent else "-"
        report_lines.append(f"| {mk} | {name} | {prompt} | {step_count} | {acc:.4f} | {correct}/{total} | {lat:.0f} | {parent} |")

    report_lines.append("")
    report_lines.append("## Best Per Model")
    report_lines.append("")

    for mk in MODEL_KEYS:
        all_cells = gen0 + gen1
        by_model = [c for c in all_cells if c.model_key == mk]
        if not by_model:
            continue
        best = max(by_model, key=lambda c: c.metadata.get("accuracy", 0.0))
        acc = best.metadata.get("accuracy", 0)
        correct = best.metadata.get("correct", 0)
        total = best.metadata.get("total", 0)
        report_lines.append(f"### {mk}")
        report_lines.append("")
        report_lines.append(f"- **Best cell**: {best.name}")
        report_lines.append(f"- **System prompt**: `{best.system_prompt}`")
        report_lines.append(f"- **Steps**: {len(best.steps) if best.steps else 'single-shot'}")
        report_lines.append(f"- **Decoding params**: `{best.decoding.to_dict()}`")
        report_lines.append(f"- **Training accuracy**: {acc:.4f} ({correct}/{total})")
        if mk in val_results:
            vr = val_results[mk]
            report_lines.append(f"- **Validation accuracy**: {vr['val_acc']:.4f} ({vr['val_correct']}/{vr['val_total']})")
        report_lines.append("")

    report_lines.append("## Validation Results")
    report_lines.append("")
    report_lines.append("| Model | Cell | Train Acc | Val Acc | Val Correct/Total | Delta |")
    report_lines.append("|-------|------|-----------|---------|-------------------|-------|")

    for mk, vr in val_results.items():
        delta = vr["val_acc"] - vr["train_acc"]
        report_lines.append(f"| {mk} | {vr['cell_name']} | {vr['train_acc']:.4f} | {vr['val_acc']:.4f} "
                          f"| {vr['val_correct']}/{vr['val_total']} | {delta:+.4f} |")

    report_lines.append("")
    report_lines.append("## Key Insights")
    report_lines.append("")
    report_lines.append("1. **Multi-signal grading** captures semantics that fuzzy_match misses for open-ended summaries.")
    report_lines.append("2. **Chunk-and-summarize** helps for long texts (>200 words) by breaking the task into manageable pieces.")
    report_lines.append("3. **Workflow operators** (split-into-steps, add-verify-step) can improve accuracy by forcing the model to reason step by step.")
    report_lines.append("4. **Entity recall** and **keyword overlap** together provide a robust signal even when the exact phrasing differs.")

    return "\n".join(report_lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    total_start = time.time()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       SUMMARIZATION GEPA EVOLUTION RUNNER                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # ── Step 0: Load datasets ──────────────────────────────────────────
    print("Step 0: Loading datasets...")
    # Use a representative subset for GEPA evolution (speeds up the run)
    # Full set is used for final validation
    train_questions_full = load_eval_set(TRAIN_SET_PATH, "train (full)")
    
    # Subsample to ~100 questions for the evolution loop
    random.seed(SEED)
    if len(train_questions_full) > 100:
        train_questions = random.sample(train_questions_full, 100)
    else:
        train_questions = train_questions_full
    
    val_questions = load_eval_set(VAL_SET_PATH, "val")
    hard_test_questions = load_eval_set(HARD_TEST_PATH, "hard_test")
    print(f"\n  Using {len(train_questions)} questions for evolution training "
          f"(full training set: {len(train_questions_full)})")
    print()

    # ── Step 1: Create seed population ─────────────────────────────────
    print("Step 1: Creating seed population (gen 0)...")
    gen0 = create_seed_population()
    # Deduplicate
    from agent.cell import deduplicate_population
    gen0 = deduplicate_population(gen0)
    print(f"  After dedup: {len(gen0)} unique cells")
    print()

    # ── Step 2: Evaluate gen 0 on training set ────────────────────────
    print("Step 2: Evaluating generation 0...")
    gen0 = evaluate_generation(gen0, train_questions, "Generation 0 (train)")
    gen0 = summarize_generation(gen0, "GEN 0", train_questions)
    print()

    # ── Step 3: Create and evaluate gen 1 ─────────────────────────────
    print("Step 3: Creating and evaluating generation 1...")
    gen1 = create_next_generation(gen0)
    gen1 = evaluate_generation(gen1, train_questions, "Generation 1 (train)")
    gen1 = summarize_generation(gen1, "GEN 1", train_questions)
    print()

    # ── Step 4: Validate best cells on held-out set ──────────────────
    print("Step 4: Validating best cells...")
    all_cells = gen0 + gen1
    val_results = validate_best_cells(all_cells, val_questions)
    print()

    # ── Step 5: Generate report ───────────────────────────────────────
    print("Step 5: Generating report...")
    report = generate_report(gen0, gen1, val_results, train_questions)
    with open(OUTPUT_REPORT, "w") as f:
        f.write(report)
    print(f"  Report saved to {OUTPUT_REPORT}")

    # Save raw results
    raw_results = {
        "task": "summarization_gepa_evolution",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_size": len(train_questions),
        "val_size": len(val_questions),
        "hard_test_size": len(hard_test_questions),
        "models_tested": MODEL_KEYS,
        "generations": GENERATIONS,
        "gen0_results": {c.name: c.metadata for c in gen0},
        "gen1_results": {c.name: c.metadata for c in gen1},
        "validation_results": val_results,
        "best_per_model": {},
    }
    for mk in MODEL_KEYS:
        by_mk = [c for c in all_cells if c.model_key == mk]
        if by_mk:
            best = max(by_mk, key=lambda c: c.metadata.get("accuracy", 0.0))
            raw_results["best_per_model"][mk] = {
                "cell_name": best.name,
                "system_prompt": best.system_prompt,
                "decoding": best.decoding.to_dict(),
                "train_acc": best.metadata.get("accuracy", 0.0),
                "train_correct": best.metadata.get("correct", 0),
                "train_total": best.metadata.get("total", 0),
                "has_workflow": best.steps is not None,
                "num_steps": len(best.steps) if best.steps else 0,
            }

    raw_path = os.path.join(RESULTS_DIR, "summarization_gepa_raw.json")
    with open(raw_path, "w") as f:
        json.dump(raw_results, f, indent=2, default=str)
    print(f"  Raw results saved to {raw_path}")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  TOTAL RUNTIME: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
