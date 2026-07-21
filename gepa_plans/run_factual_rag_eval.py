#!/usr/bin/env python3
"""RAG-style eval: FactDB retrieval + LLM answer.
Retrieves top-3 facts from FactDB, injects them as context into the LLM, measures accuracy vs baseline (no retrieval)."""
import sys, os, json, time, re

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["PYTHONUNBUFFERED"] = "1"

from eval_common import fuzzy_match
from agent.solvers.fact_db import FactDB

MODEL_PATH = "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
FACT_DB_PATH = "/home/artem/dev/amd-hackathon/data/facts/facts.db"
EVAL_PATH = "/home/artem/dev/amd-hackathon/data/eval/factual_combined_80.json"
OUT_PATH = "/home/artem/dev/amd-hackathon/gepa_plans/factual_rag_eval_results.json"

PROMPTS = [
    {"name": "baseline_no_rag", "system_prompt": "Answer the question directly. Use exact names, dates, and numbers. Keep under 15 words."},
    {"name": "rag_context",      "system_prompt": "Use the provided facts to answer the question. Answer directly with exact names, dates, and numbers."},
]

with open(EVAL_PATH) as f:
    questions = json.load(f)
print(f"Eval set: {len(questions)} factual questions")

from llama_cpp import Llama

results = {}
for prompt_cfg in PROMPTS:
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt_cfg['name']}")
    print(f"{'='*60}")

    llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_gpu_layers=-1, verbose=False, seed=42)
    db = FactDB(FACT_DB_PATH) if prompt_cfg["name"] == "rag_context" else None

    correct = 0
    total = 0
    total_latency = 0.0
    per_q = []

    for q in questions:
        prompt_txt = q.get("prompt", "")
        expected = q.get("expected_answer", "")

        message = {"role": "user", "content": prompt_txt}

        if prompt_cfg["name"] == "rag_context":
            retrieved = db.query(prompt_txt, k=3)
            context_parts = []
            for score, rq, ra, src in retrieved:
                context_parts.append(f"Q: {rq}\nA: {ra}")
            context_str = "\n\n".join(context_parts) if context_parts else ""
            sys_msg = "Answer the question using the following facts:\n\n" + context_str + \
                      "\n\nAnswer directly with exact names, dates, and numbers."
            messages = [{"role": "system", "content": sys_msg}, message]
        else:
            messages = [{"role": "system", "content": prompt_cfg["system_prompt"]}, message]

        t0 = time.time()
        try:
            resp = llm.create_chat_completion(messages=messages, max_tokens=64, temperature=0.0)
            output = resp["choices"][0]["message"]["content"] or ""
        except Exception as e:
            output = ""
        elapsed = (time.time() - t0) * 1000

        ok = fuzzy_match(output, expected)
        if ok:
            correct += 1
        total += 1
        total_latency += elapsed

        if total <= 3 or not ok:
            per_q.append({"prompt": prompt_txt[:60], "expected": expected[:60], "got": output[:60], "correct": ok, "latency_ms": round(elapsed, 1)})

    results[prompt_cfg["name"]] = {
        "accuracy": correct / total if total else 0,
        "correct": correct,
        "total": total,
        "avg_latency_ms": round(total_latency / total, 1) if total else 0,
        "model": "qwen2.5-1.5b-instruct",
    }
    print(f"  Accuracy: {correct}/{total} = {correct/total:.4f}  Latency: {results[prompt_cfg['name']]['avg_latency_ms']:.1f}ms")

    del llm
    if db:
        db.close()

improvement = results.get("rag_context", {}).get("accuracy", 0) - results.get("baseline_no_rag", {}).get("accuracy", 0)
print(f"\n{'='*60}")
print(f"Improvement with RAG: {improvement:+.4f} ({improvement*100:+.1f}pp)")

output = {
    "task": "factual_rag_evaluation",
    "date": "2026-07-13",
    "dataset": "factual_combined_80",
    "num_questions": len(questions),
    "model": "qwen2.5-1.5b-instruct",
    "fact_db": "facts.db (17,661 facts)",
    "results": results,
    "improvement": round(improvement, 4),
    "per_question": per_q,
}

with open(OUT_PATH, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {OUT_PATH}")
