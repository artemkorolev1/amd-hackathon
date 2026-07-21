#!/usr/bin/env python3
"""
gepa_plans/eval_sentiment.py — Reusable sentiment evaluation pipeline.

Evaluates a given model on train/val/hard-test splits and reports:
  - Per-set accuracy
  - Per-difficulty accuracy
  - Per-source accuracy (hybrid mode)
  - Confusion matrix
  - Individual question results

Supports two modes:
  1. Pure LLM (--no-hybrid):  direct LLM inference on every question
  2. Hybrid VADER + LLM (default): routes simple cases to VADER, hard cases to LLM

Usage:
  python3 gepa_plans/eval_sentiment.py --model gemma-3-1b \\
    --prompt "Analyze the tone as positive, negative, neutral, or mixed." \\
    --top_p 0.9 --top_k 20 --min_p 0.05

  # Disable hybrid mode
  python3 gepa_plans/eval_sentiment.py --model gemma-3-1b --no-hybrid

Requires: llama-cpp-python (for GGUF model inference), vaderSentiment
"""
import argparse
import json
import os
import re
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

# Model paths relative to BASE
MODEL_PATHS = {
    "qwen2.5-1.5b": "models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "qwen2.5-coder-1.5b": "models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    "gemma-3-1b": "models/gemma-3-1b-it-Q4_K_M.gguf",
    "phi-4-mini": "models/phi-4-mini-instruct-q4_k_m.gguf",
}


# ── Format normalization ─────────────────────────────────────────────────────

# Try to import the high-quality format_normalizer; fall back to simple regex
try:
    sys.path.insert(0, BASE)
    from agent.solvers.format_normalizer import normalize_sentiment_output
    HAS_FORMAT_NORMALIZER = True
except ImportError:
    HAS_FORMAT_NORMALIZER = False
    
    def normalize_sentiment_output(text):
        """Fallback simple normalizer when format_normalizer is unavailable."""
        return normalize_answer_simple(text), "low"


def normalize_answer_simple(text):
    """Extract sentiment label from model output using regex."""
    if not text or not text.strip():
        return "unknown"
    
    t = text.strip().lower()
    
    # Try to find "positive", "negative", "neutral", "mixed" in output
    # Look for explicit sentiment statement first
    patterns = [
        r'(?:^|[\s,;.:!?])positive(?:[\s,;.:!?]|$)',
        r'(?:^|[\s,;.:!?])negative(?:[\s,;.:!?]|$)',
        r'(?:^|[\s,;.:!?])neutral(?:[\s,;.:!?]|$)',
        r'(?:^|[\s,;.:!?])mixed(?:[\s,;.:!?]|$)',
    ]
    
    # Check for "sentiment: X" pattern
    sentiment_match = re.search(r'sentiment[:\s]+(positive|negative|neutral|mixed)', t)
    if sentiment_match:
        return sentiment_match.group(1)
    
    # Check for answer: X pattern
    answer_match = re.search(r'(?:answer|label)[:\s]+(positive|negative|neutral|mixed)', t)
    if answer_match:
        return answer_match.group(1)
    
    # Check for standalone sentiment words with word boundaries
    labels = ["positive", "negative", "neutral", "mixed"]
    for label in labels:
        pattern = r'(?:^|[\s,;.:!?"\'({])' + re.escape(label) + r'(?:[\s,;.:!?"\')}]|$)'
        if re.search(pattern, t):
            return label
    
    # Fallback: substring match
    if "positive" in t:
        return "positive"
    if "negative" in t:
        return "negative"
    if "neutral" in t:
        return "neutral"
    if "mixed" in t:
        return "mixed"
    
    return "unknown"


# ── Hybrid VADER + LLM classifier ────────────────────────────────────────────

# Try to import the hybrid classifier
try:
    sys.path.insert(0, BASE)
    from agent.solvers.sentiment_hybrid import classify_sentiment_hybrid
    HAS_HYBRID = True
except ImportError as e:
    HAS_HYBRID = False
    print(f"  Note: hybrid classifier not available ({e}). Will use pure LLM.")


# ── Data loading ─────────────────────────────────────────────────────────────

def load_split(name):
    """Load a sentiment split (train, val, hard_test)."""
    path = f"{DATA_DIR}/sentiment_{name}.json"
    if not os.path.exists(path):
        print(f"  ✗ Split not found: {path}")
        return []
    with open(path) as f:
        data = json.load(f)
    return data


# ── Model inference ──────────────────────────────────────────────────────────

def run_inference(model_key, prompts, system_prompt, generation_kwargs):
    """
    Run inference on a list of prompts using llama.cpp.
    Returns list of (prompt, expected_answer, model_output, predicted_label).
    """
    model_path = MODEL_PATHS.get(model_key)
    if not model_path:
        print(f"  ✗ Unknown model key: {model_key}")
        print(f"    Available: {list(MODEL_PATHS.keys())}")
        return None
    
    full_path = f"{BASE}/{model_path}"
    if not os.path.exists(full_path):
        print(f"  ✗ Model file not found: {full_path}")
        print("    Cannot run evaluation. Use --dry-run to validate setup.")
        return None
    
    try:
        from llama_cpp import Llama
    except ImportError:
        print("  ✗ llama-cpp-python not installed.")
        print("    Install with: pip install llama-cpp-python")
        return None
    
    print(f"  Loading model: {model_key} ({full_path})")
    llm = Llama(
        model_path=full_path,
        n_ctx=2048,
        n_threads=4,
        verbose=False,
    )
    
    # Extract generation parameters
    temperature = generation_kwargs.get("temperature", 0.7)
    top_p = generation_kwargs.get("top_p", 0.9)
    top_k = generation_kwargs.get("top_k", 40)
    min_p = generation_kwargs.get("min_p", 0.05)
    repeat_penalty = generation_kwargs.get("repeat_penalty", 1.1)
    seed = generation_kwargs.get("seed", 42)
    max_tokens = generation_kwargs.get("max_tokens", 128)
    
    results = []
    print(f"  Running inference on {len(prompts)} prompts...")
    
    for i, (prompt_text, expected_answer) in enumerate(prompts):
        full_prompt = f"{system_prompt}\n\n{prompt_text}"
        
        start = time.time()
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
        elapsed = time.time() - start
        
        model_text = output["choices"][0]["text"].strip() if output.get("choices") else ""
        predicted, confidence = normalize_sentiment_output(model_text)
        
        results.append({
            "prompt": prompt_text,
            "expected_answer": expected_answer,
            "model_output": model_text,
            "predicted": predicted,
            "correct": predicted == expected_answer,
            "latency": round(elapsed, 2),
        })
        
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(prompts)} done ({(i+1)/len(prompts)*100:.0f}%)")
    
    return results


# ── Hybrid inference ─────────────────────────────────────────────────────────


def run_inference_hybrid(model_key, prompts, system_prompt, generation_kwargs):
    """
    Run inference using hybrid VADER + LLM classifier.

    Returns list of results dicts with additional 'source' field.
    """
    model_path = MODEL_PATHS.get(model_key)
    if not model_path:
        print(f"  ✗ Unknown model key: {model_key}")
        print(f"    Available: {list(MODEL_PATHS.keys())}")
        return None

    full_path = f"{BASE}/{model_path}"
    if not os.path.exists(full_path):
        print(f"  ✗ Model file not found: {full_path}")
        print("    Cannot run evaluation. Use --dry-run to validate setup.")
        return None

    try:
        from llama_cpp import Llama
    except ImportError:
        print("  ✗ llama-cpp-python not installed.")
        print("    Install with: pip install llama-cpp-python")
        return None

    print(f"  Loading model: {model_key} ({full_path})")
    llm = Llama(
        model_path=full_path,
        n_ctx=2048,
        n_threads=4,
        verbose=False,
    )

    # Extract generation parameters
    temperature = generation_kwargs.get("temperature", 0.7)
    top_p = generation_kwargs.get("top_p", 0.9)
    top_k = generation_kwargs.get("top_k", 40)
    min_p = generation_kwargs.get("min_p", 0.05)
    repeat_penalty = generation_kwargs.get("repeat_penalty", 1.1)
    seed = generation_kwargs.get("seed", 42)
    max_tokens = generation_kwargs.get("max_tokens", 128)

    # Create llm_infer_fn for the hybrid classifier
    def llm_infer(system, user):
        full_prompt = f"{system}\n\n{user}"
        start = time.time()
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
            stop=["\\n\\n", "---"],
        )
        elapsed = time.time() - start
        model_text = output["choices"][0]["text"].strip() if output.get("choices") else ""
        return model_text

    results = []
    print(f"  Running hybrid inference on {len(prompts)} prompts...")

    for i, (prompt_text, expected_answer) in enumerate(prompts):
        start = time.time()
        hybrid_result = classify_sentiment_hybrid(
            text=prompt_text,
            llm_infer_fn=llm_infer,
            system_prompt=system_prompt,
        )
        elapsed = time.time() - start

        # Normalize the expected label (strip whitespace, lowercase)
        expected = expected_answer.strip().lower() if expected_answer else "unknown"
        predicted = hybrid_result["label"]
        source = hybrid_result["source"]

        results.append({
            "prompt": prompt_text,
            "expected_answer": expected_answer,
            "model_output": "",  # Hybrid doesn't produce a single model output string
            "hybrid_result": hybrid_result,
            "predicted": predicted,
            "source": source,
            "correct": predicted == expected,
            "latency": round(elapsed, 2),
        })

        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(prompts)} done ({(i+1)/len(prompts)*100:.0f}%)")

    return results


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(results, split_name, difficulty_filter=None):
    """Compute accuracy and confusion matrix from results."""
    if not results:
        return {"accuracy": 0.0, "total": 0}
    
    if difficulty_filter:
        filtered = [r for r in results if r.get("difficulty") == difficulty_filter]
    else:
        filtered = list(results)
    
    if not filtered:
        return {"accuracy": 0.0, "total": 0}
    
    correct = sum(1 for r in filtered if r["correct"])
    total = len(filtered)
    accuracy = correct / total * 100 if total > 0 else 0.0
    
    # Confusion matrix
    labels = ["positive", "negative", "neutral", "mixed"]
    confusion = defaultdict(lambda: defaultdict(int))
    for r in filtered:
        confusion[r["expected_answer"]][r["predicted"]] += 1
    
    return {
        "split": split_name,
        "difficulty": difficulty_filter or "all",
        "accuracy": round(accuracy, 2),
        "correct": correct,
        "total": total,
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


def compute_metrics_by_source(results):
    """Compute accuracy broken down by decision source (hybrid mode)."""
    sources = defaultdict(list)
    for r in results:
        src = r.get("source", "unknown")
        sources[src].append(r)

    source_metrics = {}
    for src, items in sorted(sources.items()):
        correct = sum(1 for it in items if it["correct"])
        total = len(items)
        acc = correct / total * 100 if total > 0 else 0.0
        source_metrics[src] = {
            "accuracy": round(acc, 2),
            "correct": correct,
            "total": total,
        }
    return source_metrics


def print_metrics_report(metrics_list, all_results, hybrid=False):
    """Print formatted accuracy report."""
    print("\n" + "=" * 70)
    print("SENTIMENT EVALUATION REPORT")
    print("=" * 70)
    
    for m in metrics_list:
        if m["total"] == 0:
            continue
        acc_str = f"{m['accuracy']:.1f}%"
        label = f"{m['split']}/{m['difficulty']}" if m['difficulty'] != 'all' else m['split']
        print(f"  {label:30s} {acc_str:>8s}  ({m['correct']}/{m['total']})")
    
    # Print overfit gap if we have both train and val
    train_all = next((m for m in metrics_list if m['split'] == 'train' and m['difficulty'] == 'all'), None)
    val_all = next((m for m in metrics_list if m['split'] == 'val' and m['difficulty'] == 'all'), None)
    if train_all and val_all and train_all['total'] > 0 and val_all['total'] > 0:
        gap = round(train_all['accuracy'] - val_all['accuracy'], 2)
        print(f"\n  {'Overfit gap (train - val):':30s} {gap:>8.1f}%")
    
    # Confusion matrices
    print("\n--- Confusion Matrices ---")
    for m in metrics_list:
        if m["total"] == 0 or not m.get("confusion"):
            continue
        print(f"\n  {m['split']} ({m['difficulty']}):")
        labels = ["positive", "negative", "neutral", "mixed"]
        print(f"  {'':12s}", end="")
        for l in labels:
            print(f"{l:>12s}", end="")
        print()
        for actual in labels:
            print(f"  {actual:12s}", end="")
            for predicted in labels:
                val = m["confusion"].get(actual, {}).get(predicted, 0)
                print(f"{val:>12d}", end="")
            print()
    
    # Per-source accuracy (hybrid mode only)
    if hybrid:
        print("\n--- Per-Source Accuracy ---")
        for split_name, results in all_results.items():
            if not results:
                continue
            src_metrics = compute_metrics_by_source(results)
            if not src_metrics:
                continue
            print(f"\n  {split_name.replace('hard_test', 'test')}:")
            for src, sm in src_metrics.items():
                acc_str = f"{sm['accuracy']:.1f}%"
                print(f"    {src:20s} {acc_str:>8s}  ({sm['correct']}/{sm['total']})")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sentiment Evaluation Pipeline")
    parser.add_argument("--model", default="gemma-3-1b",
                        choices=list(MODEL_PATHS.keys()),
                        help="Model key to evaluate")
    parser.add_argument("--prompt", default="Analyze the tone as positive, negative, neutral, or mixed.",
                        help="System prompt for sentiment")
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repeat_penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "hard_test"],
                        help="Splits to evaluate (default: all)")
    parser.add_argument("--output", default=None,
                        help="Output path (default: eval_results/sentiment_hybrid_TIMESTAMP.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate setup without running inference")
    parser.add_argument("--no-hybrid", action="store_false", dest="hybrid", default=True,
                        help="Disable hybrid VADER+LLM routing (use pure LLM)")
    parser.add_argument("--hybrid", action="store_true", dest="hybrid",
                        help="Use hybrid VADER+LLM routing (default)")
    
    args = parser.parse_args()
    use_hybrid = args.hybrid
    
    generation_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "repeat_penalty": args.repeat_penalty,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
    }
    
    print(f"Sentiment Evaluation Pipeline")
    print(f"  Model:     {args.model}")
    print(f"  Mode:      {'Hybrid VADER+LLM' if use_hybrid else 'Pure LLM'}")
    print(f"  Prompt:    {args.prompt[:60]}...")
    print(f"  Params:    top_p={args.top_p}, top_k={args.top_k}, min_p={args.min_p}, "
          f"temp={args.temperature}, seed={args.seed}")
    print(f"  Splits:    {', '.join(args.splits)}")
    
    # Load all splits
    split_data = {}
    for split_name in args.splits:
        items = load_split(split_name)
        if items:
            split_data[split_name] = items
            print(f"  Loaded {split_name}: {len(items)} questions")
    
    if not split_data:
        print("  ✗ No split data loaded. Run gepa_plans/build_splits.py first.")
        sys.exit(1)
    
    if args.dry_run:
        print("\n✓ Dry-run complete. All files and model paths validated.")
        for split_name, items in split_data.items():
            diff_c = Counter(it.get("difficulty", "unknown") for it in items)
            ans_c = Counter(it.get("expected_answer", "unknown") for it in items)
            print(f"  {split_name}: {len(items)} items "
                  f"(easy={diff_c.get('easy',0)}, medium={diff_c.get('medium',0)}, hard={diff_c.get('hard',0)}) "
                  f"(pos={ans_c.get('positive',0)}, neg={ans_c.get('negative',0)}, "
                  f"neu={ans_c.get('neutral',0)}, mix={ans_c.get('mixed',0)})")
        
        model_path = f"{BASE}/{MODEL_PATHS.get(args.model, '')}"
        if os.path.exists(model_path):
            print(f"\n  Model file exists: {model_path}")
        else:
            print(f"\n  ! Model file not found: {model_path}")
            print("  (This is OK for dry-run if model hasn't been downloaded yet)")
        return
    
    # Run inference on each split
    all_results = {}
    for split_name, items in split_data.items():
        print(f"\n--- Evaluating {split_name} ({len(items)} questions) ---")
        prompts = [(it["prompt"], it["expected_answer"]) for it in items]
        
        if use_hybrid:
            if not HAS_HYBRID:
                print("  ✗ Hybrid classifier not available. Run without --no-hybrid or install dependencies.")
                sys.exit(1)
            results = run_inference_hybrid(args.model, prompts, args.prompt, generation_kwargs)
        else:
            results = run_inference(args.model, prompts, args.prompt, generation_kwargs)
        if results is None:
            sys.exit(1)
        
        # Attach difficulty to results
        for i, r in enumerate(results):
            r["difficulty"] = items[i].get("difficulty", "unknown") if i < len(items) else "unknown"
        
        all_results[split_name] = results
    
    # Compute metrics
    metrics = []
    for split_name, results in all_results.items():
        split_short = split_name.replace("hard_test", "test")
        metrics.append(compute_metrics(results, split_short))
        for diff in ["easy", "medium", "hard"]:
            diff_metrics = compute_metrics(results, split_short, diff)
            if diff_metrics["total"] > 0:
                metrics.append(diff_metrics)
    
    print_metrics_report(metrics, all_results, hybrid=use_hybrid)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "hybrid" if use_hybrid else "eval"
    output_path = args.output or f"{RESULTS_DIR}/sentiment_{suffix}_{timestamp}.json"
    
    eval_results = {
        "model": args.model,
        "system_prompt": args.prompt,
        "generation_kwargs": generation_kwargs,
        "timestamp": timestamp,
        "hybrid_mode": use_hybrid,
        "metrics": metrics,
        "per_question": {split: results for split, results in all_results.items()},
    }
    
    # Add per-source metrics if hybrid mode
    if use_hybrid:
        eval_results["per_source_accuracy"] = {
            split: compute_metrics_by_source(results)
            for split, results in all_results.items()
        }
    
    with open(output_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
