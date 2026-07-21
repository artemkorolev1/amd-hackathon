#!/usr/bin/env python3
"""GEPA: Prompt ablation for factual knowledge category.

Tests 3 prompt variants on factual questions using qwen2.5-1.5b-instruct:
  1. Empty system prompt (just user message)
  2. Minimal "Answer:" system prompt
  3. Verbose prompt with full instructions

Outputs accuracy per variant to stdout and a JSON file.
"""

import json
import os
import sys
import re
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gepa_factual")

MODEL_PATH = "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"

# Prompt variants
PROMPT_VARIANTS = {
    "empty": {
        "system": "",
        "description": "No system prompt — just user message"
    },
    "minimal": {
        "system": "Answer:",
        "description": "Minimal 'Answer:' prefix"
    },
    "verbose": {
        "system": (
            "Answer the question directly. "
            "Address every part of multi-part questions. "
            "Keep the answer under 50 words. "
            "Use exact names, dates, and numbers. "
            "No preamble, no closing."
        ),
        "description": "Full verbose prompt from dynamic_prompts.py factual/low tier"
    },
}

# Load factual questions from training-v3 (19 entries) + some from factual_combined_80
def load_factual_questions():
    questions = []
    
    # From training-v3.json (19 factual)
    with open("/home/artem/dev/amd-hackathon/data/eval/training-v3.json") as f:
        data = json.load(f)
    for d in data:
        if d.get("category") == "factual":
            questions.append({
                "prompt": d["prompt"],
                "expected": d["expected_answer"],
                "source": d.get("source", "training-v3"),
            })
    
    # From factual_combined_80.json (58 factual) — sample some
    with open("/home/artem/dev/amd-hackathon/data/eval/factual_combined_80.json") as f:
        data = json.load(f)
    for d in data:
        questions.append({
            "prompt": d["prompt"],
            "expected": d["expected_answer"],
            "source": d.get("source", "factual_combined_80"),
        })
    
    return questions

def normalize(text):
    """Normalize for comparison — lowercase, strip punctuation, collapse spaces."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def evaluate_variant(llm, questions, variant_name, variant_config):
    """Evaluate a prompt variant on all questions."""
    system_prompt = variant_config["system"]
    results = []
    correct = 0
    total = 0
    
    for i, q in enumerate(questions):
        prompt = q["prompt"]
        expected = q["expected"]
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            resp = llm.create_chat_completion(
                messages=messages,
                max_tokens=64,
                temperature=0.0,
                top_p=1.0,
                top_k=40,
                min_p=0.0,
                repeat_penalty=1.0,
                seed=None,
                stop=["\n\n", "Question:", "Context:"],
            )
            answer = resp["choices"][0]["message"]["content"] or ""
        except Exception as e:
            logger.warning(f"  [{i}] Inference error: {e}")
            answer = ""
        
        # Compare normalized
        answer_norm = normalize(answer)
        expected_norm = normalize(expected)
        is_correct = expected_norm == answer_norm or expected_norm in answer_norm or answer_norm in expected_norm
        
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            "prompt": prompt[:80],
            "expected": expected,
            "answer": answer[:120],
            "correct": is_correct,
        })
        
        if i < 3 or not is_correct:
            status = "✓" if is_correct else "✗"
            logger.info(f"  [{status}] Q{i}: {prompt[:60]}... -> '{answer[:80]}' (expected: '{expected}')")
    
    accuracy = correct / total * 100 if total > 0 else 0
    return {
        "variant": variant_name,
        "description": variant_config["description"],
        "accuracy": round(accuracy, 1),
        "correct": correct,
        "total": total,
        "results": results,
    }

def main():
    from llama_cpp import Llama
    
    questions = load_factual_questions()
    logger.info(f"Loaded {len(questions)} factual questions")
    
    # Load model
    logger.info(f"Loading model from {MODEL_PATH}...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_gpu_layers=-1,
        n_threads=4,
        flash_attn=True,
        verbose=False,
    )
    logger.info("Model loaded")
    
    all_results = {}
    
    for variant_name, variant_config in PROMPT_VARIANTS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing variant: {variant_name} — {variant_config['description']}")
        logger.info(f"{'='*60}")
        
        start = time.time()
        result = evaluate_variant(llm, questions, variant_name, variant_config)
        elapsed = time.time() - start
        
        logger.info(f"\n{variant_name}: {result['correct']}/{result['total']} = {result['accuracy']}% ({elapsed:.1f}s)")
        all_results[variant_name] = result
    
    # Summary table
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    for vn, vr in sorted(all_results.items(), key=lambda x: x[1].get("accuracy", 0), reverse=True):
        logger.info(f"  {vr['variant']:12s}  {vr['accuracy']:5.1f}%  ({vr['correct']}/{vr['total']})  — {vr['description']}")
    
    # Save results
    output_path = "/home/artem/dev/amd-hackathon/gepa_factual_ablation_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
