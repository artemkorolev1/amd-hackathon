#!/usr/bin/env python3
"""
Evaluate qwen_instruct_gen0.json prompt variants on qwen2.5-1.5b-instruct
against 19 factual questions. Saves results to qwen_instruct_gen0_results.json.
"""
import json
import sys
import time
from pathlib import Path

# Add the project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.gepa_runner import _model_cache, fuzzy_match

MODEL_KEY = "qwen2.5-1.5b"
VARIANT_FILE = Path("/home/artem/dev/amd-hackathon/gepa_plans/qwen_instruct_gen0.json")
DATASET_FILE = Path("/home/artem/dev/amd-hackathon/data/eval/training-v3.json")
OUTPUT_FILE = Path("/home/artem/dev/amd-hackathon/gepa_plans/qwen_instruct_gen0_results.json")

def main():
    print("=" * 60)
    print("GEPA Runner: Evaluating gen0 on qwen2.5-1.5b-instruct (GPU)")
    print("=" * 60)

    # 1. Load variants
    print("\n[1] Loading variants...")
    with open(VARIANT_FILE, "r") as f:
        plan = json.load(f)
    variants = plan["variants"]
    previous_best_accuracy = plan.get("previous_best_accuracy", 0.368)
    print(f"    Loaded {len(variants)} variants from gen0 plan")

    # 2. Load dataset & filter to factual
    print("\n[2] Loading dataset & filtering factual questions...")
    with open(DATASET_FILE, "r") as f:
        all_data = json.load(f)
    factual_questions = [q for q in all_data if q.get("category") == "factual"]
    print(f"    {len(factual_questions)} factual questions found")

    # 3. Load model (GPU)
    print("\n[3] Loading model on GPU...")
    llm = _model_cache.get(MODEL_KEY)
    print(f"    Model loaded successfully")

    # 4. Evaluate each variant
    results = []
    print("\n[4] Evaluating variants...")
    for i, variant in enumerate(variants):
        name = variant["name"]
        system_prompt = variant["system_prompt"]
        temperature = variant.get("temperature", 0.0)
        max_tokens = variant.get("max_tokens", 64)

        print(f"\n  --- Variant {i+1}/8: '{name}' ---")
        print(f"      system_prompt: {repr(system_prompt)}")
        print(f"      temperature={temperature}, max_tokens={max_tokens}")

        correct = 0
        total_latency = 0.0
        details = []

        for j, q in enumerate(factual_questions):
            prompt_text = q["prompt"]
            expected = q["expected_answer"]

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text},
            ]

            start = time.time()
            try:
                response = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                elapsed_ms = (time.time() - start) * 1000
                answer = response["choices"][0]["message"]["content"].strip()
                usage = response.get("usage", {})
                tok_count = usage.get("completion_tokens", len(answer.split()))

                passed = fuzzy_match(answer, expected)
                if passed:
                    correct += 1

                total_latency += elapsed_ms

                details.append({
                    "question_id": q.get("task_id", f"q{j}"),
                    "prompt": prompt_text[:80],
                    "expected": expected,
                    "answer": answer,
                    "passed": passed,
                    "latency_ms": round(elapsed_ms, 1),
                })

                if passed:
                    status = "PASS"
                else:
                    status = "FAIL"
                print(f"      [{j+1:2d}/19] {status} | ans={repr(answer[:40]):44s} | exp={repr(expected[:40]):40s} | {elapsed_ms:6.1f}ms")

            except Exception as e:
                elapsed_ms = (time.time() - start) * 1000
                total_latency += 100  # penalty
                print(f"      [{j+1:2d}/19] ERROR | {e}")
                details.append({
                    "question_id": q.get("task_id", f"q{j}"),
                    "prompt": prompt_text[:80],
                    "expected": expected,
                    "answer": f"[ERROR: {e}]",
                    "passed": False,
                    "latency_ms": 100,
                })

        accuracy = correct / len(factual_questions)
        avg_latency = total_latency / len(factual_questions)
        print(f"      => Accuracy: {correct}/{len(factual_questions)} = {accuracy:.4f}")
        print(f"      => Avg latency: {avg_latency:.1f}ms")

        results.append({
            "name": name,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "accuracy": round(accuracy, 4),
            "correct": correct,
            "total": len(factual_questions),
            "avg_latency_ms": round(avg_latency, 1),
            "details": details,
        })

    # 5. Find best variant
    best = max(results, key=lambda r: r["accuracy"])
    improvement = best["accuracy"] > previous_best_accuracy

    print("\n" + "=" * 60)
    print(f"RESULTS SUMMARY")
    print("=" * 60)
    for r in results:
        marker = " ★ BEST" if r["name"] == best["name"] else ""
        print(f"  {r['name']:25s}  acc={r['accuracy']:.4f}  ({r['correct']}/{r['total']})  avg_lat={r['avg_latency_ms']:.1f}ms{marker}")
    print(f"\n  Previous best accuracy: {previous_best_accuracy}")
    print(f"  Best variant: {best['name']} ({best['accuracy']:.4f})")
    print(f"  Improvement: {improvement}")

    # 6. Save results
    output = {
        "generation": 0,
        "model": "qwen2.5-1.5b-instruct",
        "total_questions": len(factual_questions),
        "previous_best_accuracy": previous_best_accuracy,
        "improvement": improvement,
        "best_variant": {
            "name": best["name"],
            "accuracy": best["accuracy"],
            "correct": best["correct"],
            "total": best["total"],
            "avg_latency_ms": best["avg_latency_ms"],
        },
        "results": [
            {
                "name": r["name"],
                "system_prompt": r["system_prompt"],
                "accuracy": r["accuracy"],
                "correct": r["correct"],
                "total": r["total"],
                "avg_latency_ms": r["avg_latency_ms"],
            }
            for r in results
        ],
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {OUTPUT_FILE}")

    # Also print detailed per-question results for debugging
    print("\n" + "=" * 60)
    print("DETAILED PER-QUESTION BREAKDOWN")
    print("=" * 60)
    for q_idx, q in enumerate(factual_questions):
        print(f"\n  Q{q_idx+1}: {q['prompt'][:60]}")
        print(f"      Expected: {q['expected_answer']}")
        for r in results:
            detail = r["details"][q_idx]
            status = "PASS" if detail["passed"] else "FAIL"
            print(f"      {r['name']:25s} {status} | {detail['answer'][:40]}")

    print("\nDone!")

if __name__ == "__main__":
    main()
