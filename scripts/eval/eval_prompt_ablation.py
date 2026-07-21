#!/usr/bin/env python3
"""
Prompt ablation study — for each category, test the best specialist model
with 3-4 prompt variants and measure which prompt is most successful.
"""
import json
import re
import sys
import time
import gc
from typing import Dict, List

MODEL_PATHS = {
    "qwen2.5-1.5b": "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "qwen2.5-math-1.5b": "/home/artem/models/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf",
    "gemma-3-1b": "/home/artem/models/gemma-3-1b-it-Q4_K_M.gguf",
}

# ── PROMPT VARIANTS PER CATEGORY ───────────────────────────────────────────
# Each variant has: model, system_prompt, temperature, max_tokens
PROMPT_VARIANTS = {
    "code_debug": [  # Best model: Math-1.5B
        {
            "name": "v1-basic",
            "model": "qwen2.5-math-1.5b",
            "system_prompt": "Fix the bug. Output ONLY the corrected function inside ```python ... ```. No explanation.",
            "temperature": 0.0,
            "max_tokens": 512,
        },
        {
            "name": "v2-terse",
            "model": "qwen2.5-math-1.5b",
            "system_prompt": "Fix the bug. Output the fixed function in a fenced code block. No explanation.",
            "temperature": 0.0,
            "max_tokens": 512,
        },
        {
            "name": "v3-preserve-signature",
            "model": "qwen2.5-math-1.5b",
            "system_prompt": "Fix the bug. Output ONLY the corrected function inside ```python ... ```. Preserve original function name and signature exactly. Handle edge cases. No explanation.",
            "temperature": 0.0,
            "max_tokens": 512,
        },
        {
            "name": "v4-minimal",
            "model": "qwen2.5-math-1.5b",
            "system_prompt": "Fix:",
            "temperature": 0.0,
            "max_tokens": 512,
        },
    ],
    "code_gen": [  # Best model: Gemma-1B
        {
            "name": "v1-basic",
            "model": "gemma-3-1b",
            "system_prompt": "Write the requested Python function. Output ONLY the function inside ```python ... ```. Keep the exact function name. No explanation.",
            "temperature": 0.0,
            "max_tokens": 512,
        },
        {
            "name": "v2-detailed",
            "model": "gemma-3-1b",
            "system_prompt": "Write the requested Python function. Output ONLY the function inside ```python ... ```. Preserve exact name and signature. Handle edge cases like empty inputs, None, type errors. No explanation.",
            "temperature": 0.0,
            "max_tokens": 512,
        },
        {
            "name": "v3-minimal",
            "model": "gemma-3-1b",
            "system_prompt": "Write the function. Output ONLY the code in a fenced block.",
            "temperature": 0.0,
            "max_tokens": 512,
        },
        {
            "name": "v4-terse",
            "model": "gemma-3-1b",
            "system_prompt": "Code:",
            "temperature": 0.0,
            "max_tokens": 512,
        },
    ],
    "factual": [  # Best model: Qwen-1.5B
        {
            "name": "v1-basic",
            "model": "qwen2.5-1.5b",
            "system_prompt": "Answer the question directly. Use exact names, dates, and numbers. Keep under 15 words. No preamble.",
            "temperature": 0.0,
            "max_tokens": 64,
        },
        {
            "name": "v2-short",
            "model": "qwen2.5-1.5b",
            "system_prompt": "Answer in 1-3 words if possible. Exact fact only. No explanation.",
            "temperature": 0.0,
            "max_tokens": 32,
        },
        {
            "name": "v3-force-guess",
            "model": "qwen2.5-1.5b",
            "system_prompt": "Answer the question. If you don't know, give your best guess. Use exact names. Keep extremely short.",
            "temperature": 0.0,
            "max_tokens": 64,
        },
        {
            "name": "v4-direct",
            "model": "qwen2.5-1.5b",
            "system_prompt": "Answer:",
            "temperature": 0.0,
            "max_tokens": 64,
        },
    ],
    "logic": [  # Best model: Math-1.5B
        {
            "name": "v1-basic",
            "model": "qwen2.5-math-1.5b",
            "system_prompt": "Solve the logic puzzle. End with 'Answer: <conclusion>' on its own line. Keep conclusion short.",
            "temperature": 0.0,
            "max_tokens": 256,
        },
        {
            "name": "v2-step-by-step",
            "model": "qwen2.5-math-1.5b",
            "system_prompt": "Solve the logic puzzle step by step. Deduce from premises. End with 'Answer: <conclusion>'. Keep conclusion short.",
            "temperature": 0.0,
            "max_tokens": 256,
        },
        {
            "name": "v3-terse",
            "model": "qwen2.5-math-1.5b",
            "system_prompt": "Output ONLY the answer. One word or short phrase. No explanation.",
            "temperature": 0.0,
            "max_tokens": 32,
        },
        {
            "name": "v4-table",
            "model": "qwen2.5-math-1.5b",
            "system_prompt": "Use a truth table or deductive chain. End with 'Answer: <conclusion>'.",
            "temperature": 0.0,
            "max_tokens": 256,
        },
    ],
    "math": [  # Best model: Math-1.5B
        {
            "name": "v1-basic",
            "model": "qwen2.5-math-1.5b",
            "system_prompt": "Solve the math problem. End with 'Answer: <value>' on its own line. Use standard decimal format.",
            "temperature": 0.0,
            "max_tokens": 512,
        },
        {
            "name": "v2-terse",
            "model": "qwen2.5-math-1.5b",
            "system_prompt": "Output ONLY the final numeric answer. No explanation. No working. Just the number.",
            "temperature": 0.0,
            "max_tokens": 32,
        },
        {
            "name": "v3-step-by-step",
            "model": "qwen2.5-math-1.5b",
            "system_prompt": "Solve step by step. Show brief working (2-3 steps). Double-check. End with 'Answer: <value>'.",
            "temperature": 0.0,
            "max_tokens": 512,
        },
        {
            "name": "v4-verify",
            "model": "qwen2.5-math-1.5b",
            "system_prompt": "Solve step by step. Verify your calculation. End with 'Answer: <value>'.",
            "temperature": 0.0,
            "max_tokens": 512,
        },
    ],
    "ner": [  # Best model: Gemma-1B
        {
            "name": "v1-structured",
            "model": "gemma-3-1b",
            "system_prompt": "Extract entities. Output: CATEGORY: value1, value2; CATEGORY: value3. Use PERSON, ORG, LOC, DATE, DISEASE. No explanation.",
            "temperature": 0.0,
            "max_tokens": 256,
        },
        {
            "name": "v2-list",
            "model": "gemma-3-1b",
            "system_prompt": "List all named entities. Format: * entity_name (type). No explanation.",
            "temperature": 0.0,
            "max_tokens": 256,
        },
        {
            "name": "v3-examples",
            "model": "gemma-3-1b",
            "system_prompt": "Extract all named entities. Output as:\nPERSON: name1, name2;\nORGANIZATION: org1;\nLOCATION: loc1, loc2;\nDATE: date1;\n\nNo explanation, no sentences.",
            "temperature": 0.0,
            "max_tokens": 256,
        },
        {
            "name": "v4-minimal",
            "model": "gemma-3-1b",
            "system_prompt": "Entities:",
            "temperature": 0.0,
            "max_tokens": 256,
        },
    ],
    "sentiment": [  # Best model: Qwen-1.5B
        {
            "name": "v1-basic",
            "model": "qwen2.5-1.5b",
            "system_prompt": "Classify the sentiment. Output EXACTLY one word: positive, negative, neutral, or mixed. No explanation.",
            "temperature": 0.0,
            "max_tokens": 32,
        },
        {
            "name": "v2-sarcasm",
            "model": "qwen2.5-1.5b",
            "system_prompt": "Classify the sentiment. Output EXACTLY one word: positive, negative, neutral, or mixed. Watch for sarcasm — sarcasm is NEGATIVE. Default to negative when uncertain. No explanation.",
            "temperature": 0.0,
            "max_tokens": 32,
        },
        {
            "name": "v3-label-only",
            "model": "qwen2.5-1.5b",
            "system_prompt": "Sentiment:",
            "temperature": 0.0,
            "max_tokens": 16,
        },
        {
            "name": "v4-detailed",
            "model": "qwen2.5-1.5b",
            "system_prompt": "Classify the sentiment. First output the label: positive, negative, neutral, or mixed. Then a one-sentence reason. Sarcasm = negative. Hedging = neutral.",
            "temperature": 0.0,
            "max_tokens": 64,
        },
    ],
    "summarization": [  # Best model: Gemma-1B
        {
            "name": "v1-basic",
            "model": "gemma-3-1b",
            "system_prompt": "Summarize in at most 2 sentences. Include key names, numbers, and facts. No preamble.",
            "temperature": 0.0,
            "max_tokens": 256,
        },
        {
            "name": "v2-bullet",
            "model": "gemma-3-1b",
            "system_prompt": "Summarize as bullet points. Each bullet one key fact. Include names, numbers, dates. No preamble.",
            "temperature": 0.0,
            "max_tokens": 256,
        },
        {
            "name": "v3-terse",
            "model": "gemma-3-1b",
            "system_prompt": "One sentence summary:",
            "temperature": 0.0,
            "max_tokens": 128,
        },
        {
            "name": "v4-direct",
            "model": "gemma-3-1b",
            "system_prompt": "Summarize:",
            "temperature": 0.0,
            "max_tokens": 256,
        },
    ],
}

# ── Fuzzy match ─────────────────────────────────────────────────────────────
def fuzzy_match(answer: str, expected: str) -> bool:
    a, e = answer.strip().lower(), expected.strip().lower()
    if not a or not e: return False
    if a == e: return True
    if len(e) <= 20 and e in a: return True
    if len(a) <= 20 and a in e: return True
    nums_a = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", a)]
    nums_e = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", e)]
    if nums_a and nums_e:
        an, en = nums_a[-1], nums_e[-1]
        if en != 0 and abs((an - en) / en) <= 0.01: return True
        if an == en: return True
    ta = set(tok for tok in re.split(r"[^a-zA-Z0-9.]+", a) if tok)
    te = set(tok for tok in re.split(r"[^a-zA-Z0-9.]+", e) if tok)
    if len(te) > 0 and len(ta & te) / len(te) >= 0.8: return True
    return False

# ── Model cache ─────────────────────────────────────────────────────────────
class ModelCache:
    def __init__(self):
        self._models = {}
    def get(self, name: str):
        if name not in self._models:
            from llama_cpp import Llama
            print(f"  [Load] {name}...", file=sys.stderr)
            t0 = time.time()
            self._models[name] = Llama(model_path=MODEL_PATHS[name], n_gpu_layers=-1, n_ctx=2048, verbose=False)
            print(f"  [Load] done in {time.time()-t0:.1f}s", file=sys.stderr)
        return self._models[name]
    def run(self, name: str, system_prompt: str, user_prompt: str, temp=0.0, max_tok=512):
        llm = self.get(name)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        t0 = time.time()
        r = llm.create_chat_completion(messages=messages, max_tokens=max_tok, temperature=temp)
        elapsed = time.time() - t0
        return r["choices"][0]["message"]["content"].strip(), elapsed
    def unload_all(self):
        for k in list(self._models.keys()): del self._models[k]
        self._models = {}; gc.collect(); import torch; torch.cuda.empty_cache()

# ── Run ablation ────────────────────────────────────────────────────────────
def main():
    dataset_path = "/home/artem/dev/amd-hackathon/data/eval/training-v3.json"
    with open(dataset_path) as f:
        all_q = json.load(f)

    # Take 5 questions per category
    import random
    random.seed(42)
    by_cat = {}
    for q in all_q:
        by_cat.setdefault(q["category"], []).append(q)
    # Use same 5 per category for fair comparison
    test_questions = {}
    for cat in sorted(PROMPT_VARIANTS.keys()):
        test_questions[cat] = random.sample(by_cat[cat], min(5, len(by_cat[cat])))

    cache = ModelCache()
    all_results = {}  # category -> variant_name -> {"correct": N, "total": N, "detail": [...]}

    for cat, variants in sorted(PROMPT_VARIANTS.items()):
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"CATEGORY: {cat}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        qs = test_questions[cat]
        cat_results = {}

        for variant in variants:
            vname = variant["name"]
            model_name = variant["model"]
            sys_prompt = variant["system_prompt"]
            temp = variant["temperature"]
            max_tok = variant["max_tokens"]

            correct = 0
            details = []
            for i, q in enumerate(qs):
                answer, elapsed = cache.run(model_name, sys_prompt, q["prompt"], temp, max_tok)
                ok = fuzzy_match(answer, q["expected_answer"])
                if ok: correct += 1
                details.append({
                    "task_id": q.get("task_id", f"q_{i}"),
                    "expected": q["expected_answer"],
                    "got": answer,
                    "correct": ok,
                    "time_s": round(elapsed, 2),
                })
                marker = "✓" if ok else "✗"
                print(f"  {cat:15s} {vname:20s} {marker} {answer[:50]}", file=sys.stderr)

            acc = correct / len(qs) * 100
            cat_results[vname] = {
                "model": model_name,
                "prompt": sys_prompt,
                "temperature": temp,
                "max_tokens": max_tok,
                "correct": correct,
                "total": len(qs),
                "accuracy": round(acc, 1),
                "details": details,
            }
            print(f"  → {vname}: {correct}/{len(qs)} = {acc:.1f}%", file=sys.stderr)

        all_results[cat] = cat_results

    cache.unload_all()

    # ── SUMMARY TABLE ────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}", file=sys.stderr)
    print("PROMPT ABLATION SUMMARY", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    print(f"{'Category':<18} {'Variant':<22} {'Model':<20} {'Acc':<8} {'Best?':<6}", file=sys.stderr)
    print(f"{'-'*74}", file=sys.stderr)

    best_variants = {}
    for cat, variants in sorted(all_results.items()):
        best_acc = -1
        best_vname = ""
        for vname, vdata in sorted(variants.items()):
            acc = vdata["accuracy"]
            is_best = "★" if acc >= max(v["accuracy"] for v in variants.values()) else ""
            # Also mark if tied
            print(f"{cat:<18} {vname:<22} {vdata['model']:<20} {acc:<8} {is_best:<6}", file=sys.stderr)
            if acc > best_acc:
                best_acc = acc
                best_vname = vname
        best_variants[cat] = best_vname
        # Show best variant highlighted
        best_data = variants[best_vname]
        print(f"{'':>18} {'─'*70}", file=sys.stderr)

    print(f"\n\nBEST VARIANT PER CATEGORY:", file=sys.stderr)
    print(f"{'Category':<18} {'Variant':<22} {'Prompt (first 80 chars)':<60}", file=sys.stderr)
    print(f"{'-'*100}", file=sys.stderr)
    for cat in sorted(PROMPT_VARIANTS.keys()):
        vname = best_variants[cat]
        prompt = all_results[cat][vname]["prompt"]
        prompt_short = prompt[:77] + "..." if len(prompt) > 80 else prompt
        acc = all_results[cat][vname]["accuracy"]
        print(f"{cat:<18} {vname:<22} {prompt_short:<60} {acc}%", file=sys.stderr)

    # ── Save ─────────────────────────────────────────────────────────────────
    out = {"variants_per_category": all_results, "best_per_category": best_variants}
    out_path = "/home/artem/dev/amd-hackathon/data/eval/prompt_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
