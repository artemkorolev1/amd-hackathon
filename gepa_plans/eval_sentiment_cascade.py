#!/usr/bin/env python3
"""
gepa_plans/eval_sentiment_cascade.py — Evaluate the 2-level cascading sentiment classifier.

Runs the full cascade (Level 1 coarse → Level 2 fine-grained) on the 100
validation questions and reports:

  - Coarse accuracy (we have ground truth via expected_answer)
  - Fine coverage (% of questions producing a valid fine emotion)
  - Judge-validation rate (% of fine emotions deemed reasonable)
  - Per-emotion distribution
  - Most common emotions

Usage:
  python3 gepa_plans/eval_sentiment_cascade.py

Output:
  - eval_results/sentiment_cascade_TIMESTAMP.json
"""

import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.expanduser("/home/artem/dev/amd-hackathon")
DATA_DIR = f"{BASE}/data/eval"
RESULTS_DIR = f"{BASE}/eval_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, BASE)

# Model config
MODEL_PATH = os.path.join(BASE, "models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
MODEL_KEY = "qwen2.5-1.5b"

# Default generation kwargs for coarse level
COARSE_KWARGS = {
    "temperature": 0.0,
    "top_p": 0.9,
    "top_k": 20,
    "min_p": 0.05,
    "seed": 42,
    "max_tokens": 32,
}

# Fine level kwargs (slightly higher temperature for diversity)
FINE_KWARGS = {
    "temperature": 0.3,
    "top_p": 0.9,
    "top_k": 20,
    "min_p": 0.05,
    "seed": 42,
    "max_tokens": 48,
}

# Judge kwargs (deterministic)
JUDGE_KWARGS = {
    "temperature": 0.0,
    "top_p": 0.9,
    "top_k": 20,
    "min_p": 0.05,
    "seed": 42,
    "max_tokens": 8,
}


# ── Imports ──────────────────────────────────────────────────────────────────

def _import_modules():
    """Lazy-import cascade and hybrid modules."""
    from agent.solvers.sentiment_cascade import (
        classify_sentiment_cascade,
        validate_emotion_is_reasonable,
        EMOTION_TAXONOMY,
        ALL_EMOTIONS,
    )
    from agent.solvers.sentiment_hybrid import classify_sentiment_hybrid
    return classify_sentiment_cascade, validate_emotion_is_reasonable, EMOTION_TAXONOMY, ALL_EMOTIONS, classify_sentiment_hybrid


# ── Data loading ─────────────────────────────────────────────────────────────

def load_val_split():
    """Load validation split."""
    path = f"{DATA_DIR}/sentiment_val.json"
    if not os.path.exists(path):
        print(f"  ✗ Validation file not found: {path}")
        return []
    with open(path) as f:
        data = json.load(f)
    return data


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(model_path: str):
    """Load a GGUF model with llama.cpp."""
    try:
        from llama_cpp import Llama
    except ImportError:
        print("  ✗ llama-cpp-python not installed.")
        print("    Install with: pip install llama-cpp-python")
        return None

    print(f"  Loading model: {model_path}")
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=4,
        verbose=False,
    )
    print(f"  Model loaded: {model_path}")
    return llm


def make_llm_infer(llm, gen_kwargs: dict):
    """
    Create an llm_infer_fn that wraps a llama_cpp.Llama instance.
    """
    temperature = gen_kwargs.get("temperature", 0.0)
    top_p = gen_kwargs.get("top_p", 0.9)
    top_k = gen_kwargs.get("top_k", 20)
    min_p = gen_kwargs.get("min_p", 0.05)
    repeat_penalty = gen_kwargs.get("repeat_penalty", 1.1)
    seed = gen_kwargs.get("seed", 42)
    max_tokens = gen_kwargs.get("max_tokens", 32)

    def llm_infer(system: str, user: str) -> str:
        full_prompt = f"{system}\n\n{user}"
        output = llm(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repeat_penalty=repeat_penalty,
            seed=seed,
            echo=False,
            stop=["\n\n", "---"],
        )
        model_text = output["choices"][0]["text"].strip() if output.get("choices") else ""
        return model_text

    return llm_infer


# ── Main evaluation ──────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("SENTIMENT CASCADE EVALUATION")
    print("=" * 70)
    print(f"  Model:     {MODEL_KEY}")
    print(f"  Model file: {MODEL_PATH}")
    print(f"  Coarse gen: {COARSE_KWARGS}")
    print(f"  Fine gen:   {FINE_KWARGS}")

    # ── Step 1: Import modules ──────────────────────────────────────────────
    print("\n--- Step 1: Importing modules ---")
    try:
        cascade_fn, validate_fn, emotion_taxonomy, all_emotions, hybrid_fn = _import_modules()
        print("  ✓ Modules imported")
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        sys.exit(1)

    # ── Step 2: Load validation data ────────────────────────────────────────
    print("\n--- Step 2: Loading validation data ---")
    val_data = load_val_split()
    if not val_data:
        print("  ✗ No validation data loaded.")
        sys.exit(1)
    print(f"  Loaded {len(val_data)} validation questions")

    # ── Step 3: Load model ──────────────────────────────────────────────────
    print("\n--- Step 3: Loading model ---")
    if not os.path.exists(MODEL_PATH):
        print(f"  ✗ Model file not found: {MODEL_PATH}")
        sys.exit(1)

    llm = load_model(MODEL_PATH)
    if llm is None:
        sys.exit(1)

    # Create llm_infer_fn for each level
    coarse_llm = make_llm_infer(llm, COARSE_KWARGS)
    fine_llm = make_llm_infer(llm, FINE_KWARGS)
    judge_llm = make_llm_infer(llm, JUDGE_KWARGS)

    # ── Step 4: Run cascade on each question ───────────────────────────────
    print(f"\n--- Step 4: Running cascade on {len(val_data)} questions ---")

    results = []
    coarse_correct = 0
    fine_valid = 0
    judge_approved = 0
    fine_emotion_counts: Counter = Counter()
    coarse_counts: Counter = Counter()
    total_judge_attempts = 0

    start_time = time.time()

    for i, item in enumerate(val_data):
        task_id = item.get("task_id", f"q{i}")
        prompt = item["prompt"]
        expected_coarse = item["expected_answer"].strip().lower()

        # --- Level 1 + 2 cascade ---
        cascade_result = cascade_fn(
            text=prompt,
            llm_infer_fn=coarse_llm,
            fine_llm_infer_fn=fine_llm,
            coarse_classifier_fn=hybrid_fn,
        )

        coarse_label = cascade_result["coarse_label"]
        fine_emotion = cascade_result["fine_emotion"]

        # Track coarse accuracy
        is_coarse_correct = coarse_label == expected_coarse
        if is_coarse_correct:
            coarse_correct += 1

        # Track fine coverage
        if fine_emotion != "unknown":
            fine_valid += 1

        coarse_counts[coarse_label] += 1
        fine_emotion_counts[fine_emotion] += 1

        # --- Validation via LLM-as-judge ---
        if fine_emotion != "unknown":
            total_judge_attempts += 1
            judge_result = validate_fn(
                text=prompt,
                coarse=coarse_label,
                fine=fine_emotion,
                judge_llm_fn=judge_llm,  # pass None to skip LLM judge for speed; using rule-based only
            )
            if judge_result["reasonable"]:
                judge_approved += 1
        else:
            judge_result = {"reasonable": False, "method": "none", "raw_judge_output": ""}

        results.append({
            "task_id": task_id,
            "prompt_preview": prompt[:120] + ("..." if len(prompt) > 120 else ""),
            "expected_coarse": expected_coarse,
            "coarse_label": coarse_label,
            "coarse_source": cascade_result["coarse_source"],
            "coarse_confidence": cascade_result["coarse_confidence"],
            "coarse_correct": is_coarse_correct,
            "fine_emotion": fine_emotion,
            "fine_source": cascade_result["fine_source"],
            "fine_confidence": cascade_result["fine_confidence"],
            "judge_reasonable": judge_result.get("reasonable", False),
            "judge_method": judge_result.get("method", "none"),
            "judge_raw": judge_result.get("raw_judge_output", ""),
            "full_path": cascade_result["full_path"],
        })

        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"    {i+1}/{len(val_data)} ({rate:.1f} q/s) "
                  f"coarse_acc={coarse_correct/(i+1)*100:.1f}% "
                  f"fine_cov={fine_valid/(i+1)*100:.1f}%")

    total_time = time.time() - start_time

    # ── Step 5: Compute metrics ────────────────────────────────────────────
    print("\n--- Step 5: Metrics ---")

    total = len(val_data)
    coarse_accuracy = coarse_correct / total * 100 if total > 0 else 0.0
    fine_coverage = fine_valid / total * 100 if total > 0 else 0.0
    judge_rate = judge_approved / total_judge_attempts * 100 if total_judge_attempts > 0 else 0.0

    # Per-class coarse accuracy
    coarse_by_class = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        exp = r["expected_coarse"]
        coarse_by_class[exp]["total"] += 1
        if r["coarse_correct"]:
            coarse_by_class[exp]["correct"] += 1

    # Per-source breakdown
    source_counts = Counter(r["coarse_source"] for r in results)
    source_accuracy = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        src = r["coarse_source"]
        source_accuracy[src]["total"] += 1
        if r["coarse_correct"]:
            source_accuracy[src]["correct"] += 1

    # Per-emotion counts
    emotion_counts_sorted = sorted(fine_emotion_counts.items(), key=lambda x: -x[1])
    top_emotions = emotion_counts_sorted[:10]

    # ── Print report ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CASCADE EVALUATION REPORT")
    print("=" * 70)
    print(f"  Total questions:         {total}")
    print(f"  Total time:              {total_time:.1f}s")
    print(f"  Throughput:              {total/total_time:.1f} q/s")
    print()
    print(f"  Coarse accuracy:         {coarse_accuracy:.1f}% ({coarse_correct}/{total})")
    print(f"  Fine coverage:           {fine_coverage:.1f}% ({fine_valid}/{total})")
    print(f"  Judge-validation rate:   {judge_rate:.1f}% "
          f"({judge_approved}/{total_judge_attempts}) "
          f"(judge attempted on {total_judge_attempts} non-unknown emotions)")

    print("\n--- Per-Class Coarse Accuracy ---")
    for label in ["positive", "negative", "neutral", "mixed"]:
        if label in coarse_by_class:
            info = coarse_by_class[label]
            acc = info["correct"] / info["total"] * 100 if info["total"] > 0 else 0
            print(f"  {label:12s}: {acc:.1f}% ({info['correct']}/{info['total']})")

    print("\n--- Decision Source Breakdown (Coarse) ---")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        info = source_accuracy[src]
        acc = info["correct"] / info["total"] * 100 if info["total"] > 0 else 0
        print(f"  {src:20s}: {count:3d} questions  accuracy={acc:.1f}% ({info['correct']}/{info['total']})")

    print("\n--- Top 10 Fine Emotions ---")
    for emotion, count in top_emotions:
        pct = count / total * 100 if total > 0 else 0
        print(f"  {emotion:20s}: {count:3d} ({pct:.1f}%)")

    print("\n--- Per-Emotion Distribution ---")
    # Show by coarse category
    for coarse_cat, emotions in emotion_taxonomy.items():
        cat_total = sum(fine_emotion_counts[e] for e in emotions)
        if cat_total == 0:
            cat_pct = 0
        else:
            cat_pct = cat_total / total * 100 if total > 0 else 0
        print(f"\n  {coarse_cat} ({cat_total} total, {cat_pct:.1f}%):")
        for emotion in emotions:
            count = fine_emotion_counts.get(emotion, 0)
            if count > 0:
                pct = count / total * 100 if total > 0 else 0
                print(f"    {emotion:20s}: {count:3d} ({pct:.1f}%)")

    # Fine source breakdown
    fine_source_counts = Counter(r["fine_source"] for r in results)
    print("\n--- Fine Emotion Source ---")
    for src, count in sorted(fine_source_counts.items(), key=lambda x: -x[1]):
        print(f"  {src:20s}: {count:3d}")

    # ── Step 6: Save results ───────────────────────────────────────────────
    print("\n--- Step 6: Saving results ---")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{RESULTS_DIR}/sentiment_cascade_{timestamp}.json"

    eval_results = {
        "model": MODEL_KEY,
        "model_path": MODEL_PATH,
        "coarse_gen_kwargs": COARSE_KWARGS,
        "fine_gen_kwargs": FINE_KWARGS,
        "judge_gen_kwargs": JUDGE_KWARGS,
        "timestamp": timestamp,
        "total_questions": total,
        "total_time_seconds": round(total_time, 2),
        "throughput_qps": round(total / total_time, 2) if total_time > 0 else 0,
        "metrics": {
            "coarse_accuracy_pct": round(coarse_accuracy, 2),
            "coarse_correct": coarse_correct,
            "coarse_total": total,
            "fine_coverage_pct": round(fine_coverage, 2),
            "fine_valid": fine_valid,
            "judge_validation_rate_pct": round(judge_rate, 2),
            "judge_approved": judge_approved,
            "judge_attempted": total_judge_attempts,
            "per_class_coarse_accuracy": {
                label: {
                    "accuracy_pct": round(
                        info["correct"] / info["total"] * 100, 2
                    ) if info["total"] > 0 else 0,
                    "correct": info["correct"],
                    "total": info["total"],
                }
                for label, info in sorted(coarse_by_class.items())
            },
            "source_breakdown": {
                src: {
                    "count": count,
                    "accuracy_pct": round(
                        source_accuracy[src]["correct"] / source_accuracy[src]["total"] * 100, 2
                    ) if source_accuracy[src]["total"] > 0 else 0,
                }
                for src, count in sorted(source_counts.items())
            },
            "fine_emotion_distribution": dict(
                sorted(fine_emotion_counts.items(), key=lambda x: -x[1])
            ),
            "top_emotions": top_emotions,
            "fine_source_breakdown": dict(
                sorted(fine_source_counts.items(), key=lambda x: -x[1])
            ),
        },
        "per_question": results,
    }

    with open(output_path, "w") as f:
        json.dump(eval_results, f, indent=2, default=str)

    print(f"  ✓ Results saved to {output_path}")
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
