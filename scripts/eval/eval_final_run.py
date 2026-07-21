#!/usr/bin/env python3
"""
Final optimized router — runs the full training-v3.json (152 Q) with
the best model + best prompt per category found across all 6 ablation rounds.
Reports accuracy, tokens, time per category, and sample outputs.
"""
import json, re, sys, time, gc, random
from typing import Dict, List

MODEL_PATHS = {
    "qwen2.5-coder": "/home/artem/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    "qwen2.5-math-1.5b": "/home/artem/models/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf",
    "llama-3.2-1b": "/home/artem/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "qwen2.5-1.5b": "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
}

# ── FINAL OPTIMIZED CONFIG ──────────────────────────────────────────────────
# Best model + best prompt per category from 6 rounds of ablation
FINAL_CONFIG = {
    "code_debug": {
        "model": "qwen2.5-coder",
        "system_prompt": "Debug:",
        "temperature": 0.0, "max_tokens": 512,
        "eval_acc": 100.0,
    },
    "code_gen": {
        "model": "qwen2.5-coder",
        "system_prompt": "```python",
        "temperature": 0.0, "max_tokens": 512,
        "eval_acc": 80.0,
    },
    "factual": {
        "model": "llama-3.2-1b",
        "system_prompt": "Fact:",
        "temperature": 0.0, "max_tokens": 64,
        "eval_acc": 80.0,
    },
    "logic": {
        "model": "qwen2.5-math-1.5b",
        "system_prompt": "Deduce step by step. Answer:",
        "temperature": 0.0, "max_tokens": 256,
        "eval_acc": 20.0,
    },
    "math": {
        "model": "qwen2.5-math-1.5b",
        "system_prompt": "Solve step by step. End with 'Answer: <value>'.",
        "temperature": 0.0, "max_tokens": 512,
        "eval_acc": 80.0,
    },
    "ner": {
        "model": "llama-3.2-1b",
        "system_prompt": '{"PERSON":[],"ORG":[],"LOC":[],"DATE":[]}  Fill in entities. Output ONLY the JSON.',
        "temperature": 0.0, "max_tokens": 256,
        "eval_acc": 31.6,
    },
    "sentiment": {
        "model": "qwen2.5-1.5b",
        "system_prompt": "Classify the sentiment. Output EXACTLY one word: positive, negative, neutral, or mixed. No explanation.",
        "temperature": 0.0, "max_tokens": 32,
        "eval_acc": 100.0,
    },
    "summarization": {
        "model": "llama-3.2-1b",
        "system_prompt": "BBC-style headline. One sentence. Names and key action.",
        "temperature": 0.0, "max_tokens": 128,
        "eval_acc": 0.0,
    },
}

# ── Fuzzy match ─────────────────────────────────────────────────────────────
def fuzzy_match(answer: str, expected: str) -> bool:
    a, e = answer.strip().lower(), expected.strip().lower()
    if not a or not e: return False
    if a == e: return True
    if len(e) <= 20 and e in a: return True
    if len(a) <= 20 and a in e: return True
    na = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", a)]
    ne = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", e)]
    if na and ne:
        an, en = na[-1], ne[-1]
        if en != 0 and abs((an-en)/en) <= 0.01: return True
        if an == en: return True
    ta = set(t for t in re.split(r"[^a-zA-Z0-9.]+", a) if t)
    te = set(t for t in re.split(r"[^a-zA-Z0-9.]+", e) if t)
    if len(te) > 0 and len(ta & te) / len(te) >= 0.8: return True
    return False

# ── Model cache ─────────────────────────────────────────────────────────────
class ModelCache:
    def __init__(self):
        self._models = {}
        self._stats = {}
    def get(self, name: str):
        if name not in self._models:
            from llama_cpp import Llama
            p = MODEL_PATHS[name]
            print(f"  [Load] {name}...", file=sys.stderr)
            t0 = time.time()
            self._models[name] = Llama(model_path=p, n_gpu_layers=-1, n_ctx=2048, verbose=False)
            print(f"  [Load] done in {time.time()-t0:.1f}s", file=sys.stderr)
        return self._models[name]
    def run(self, name: str, sys_prompt: str, user_prompt: str, temp=0.0, max_tok=512) -> tuple:
        llm = self.get(name)
        msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}] if sys_prompt else [{"role": "user", "content": user_prompt}]
        t0 = time.time()
        r = llm.create_chat_completion(messages=msgs, max_tokens=max_tok, temperature=temp)
        elapsed = time.time() - t0
        answer = r["choices"][0]["message"]["content"].strip()
        tok_in = r.get("usage", {}).get("prompt_tokens", 0)
        tok_out = r.get("usage", {}).get("completion_tokens", 0)
        self._stats.setdefault(name, {"calls": 0, "time": 0, "tok_in": 0, "tok_out": 0})
        self._stats[name]["calls"] += 1
        self._stats[name]["time"] += elapsed
        self._stats[name]["tok_in"] += tok_in
        self._stats[name]["tok_out"] += tok_out
        return answer, elapsed, tok_in, tok_out
    def unload_all(self):
        for k in list(self._models.keys()): del self._models[k]
        self._models = {}; gc.collect(); import torch; torch.cuda.empty_cache()

# ── Main ────────────────────────────────────────────────────────────────────
def main():
    dataset_path = "/home/artem/dev/amd-hackathon/data/eval/training-v3.json"
    with open(dataset_path) as f: all_q = json.load(f)

    print(f"Dataset: {len(all_q)} questions across 8 categories", file=sys.stderr)
    by_cat = {}
    for q in all_q: by_cat.setdefault(q["category"], []).append(q)

    cache = ModelCache()

    overall_ok = 0
    overall_total = 0
    overall_time = 0
    overall_tok_out = 0
    per_cat = {}
    errors = []

    # Process each category separately so we load each model once
    for cat in sorted(FINAL_CONFIG.keys()):
        qs = by_cat.get(cat, [])
        if not qs:
            continue
        cfg = FINAL_CONFIG[cat]
        model_name = cfg["model"]
        sys_prompt = cfg["system_prompt"]
        temp = cfg["temperature"]
        max_tok = cfg["max_tokens"]

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"{cat} → {model_name}", file=sys.stderr)
        print(f"  prompt: {sys_prompt[:70]}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        cat_ok = 0
        cat_time = 0
        cat_tok = 0
        cat_results = []

        for i, q in enumerate(qs):
            answer, elapsed, tok_in, tok_out = cache.run(model_name, sys_prompt, q["prompt"], temp, max_tok)
            ok = fuzzy_match(answer, q["expected_answer"])

            cat_ok += 1 if ok else 0
            cat_time += elapsed
            cat_tok += tok_out
            overall_ok += 1 if ok else 0
            overall_total += 1
            overall_time += elapsed
            overall_tok_out += tok_out

            cat_results.append({
                "task_id": q["task_id"],
                "expected": q["expected_answer"],
                "got": answer,
                "correct": ok,
                "time_s": round(elapsed, 2),
                "tokens_out": tok_out,
            })

            marker = "✓" if ok else "✗"
            brief = answer[:50].replace("\n", "\\n")
            print(f"  [{i+1:2d}/{len(qs)}] {marker} ({elapsed:.1f}s, {tok_out} tok) {brief}", file=sys.stderr)
            if not ok:
                exp_brief = q["expected_answer"][:50].replace("\n", "\\n")
                print(f"         exp: {exp_brief}", file=sys.stderr)

        acc = round(cat_ok / len(qs) * 100, 1)
        print(f"  → {cat}: {cat_ok}/{len(qs)} = {acc}%  ({cat_time:.0f}s, {cat_tok//len(qs)} tok/q)", file=sys.stderr)

        per_cat[cat] = {
            "model": model_name,
            "prompt": sys_prompt,
            "correct": cat_ok,
            "total": len(qs),
            "accuracy": acc,
            "time_s": round(cat_time, 1),
            "avg_tok": cat_tok // len(qs),
            "results": cat_results,
        }
        errors.extend([r for r in cat_results if not r["correct"]])

    cache.unload_all()

    # ── FINAL REPORT ────────────────────────────────────────────────────────
    overall_acc = round(overall_ok / overall_total * 100, 1)

    report = f"""
{'='*70}
FINAL OPTIMIZED SYSTEM — Full Run (152 Questions)
{'='*70}

Overall Accuracy: {overall_ok}/{overall_total} = {overall_acc}%
Total Time: {overall_time:.0f}s ({overall_time/overall_total:.2f}s/q)
Total Tokens Out: {overall_tok_out} ({overall_tok_out//overall_total}/q)

Model Usage:
"""
    for m, s in sorted(cache._stats.items()):
        report += f"  {m:<20s} {s['calls']:4d} calls  {s['time']:.0f}s total  {s['tok_out']:5d} tok out  {s['tok_out']//max(s['calls'],1):4d} tok/call\n"

    report += f"\n{'Category':<20s} {'Model':<20s} {'Acc':>8s} {'Time':>8s} {'Tok/q':>8s}\n"
    report += f"{'-'*64}\n"
    for cat in sorted(per_cat.keys()):
        p = per_cat[cat]
        report += f"{cat:<20s} {p['model']:<20s} {p['accuracy']:>7.1f}% {p['time_s']:>7.1f}s {p['avg_tok']:>7d}\n"

    report += f"\n{'='*70}\n"
    report += "SAMPLE FAILURES (first 20):\n"
    report += f"{'='*70}\n"
    for e in errors[:20]:
        report += f"  [{e['task_id']}] exp: {e['expected'][:55]}\n"
        report += f"            got: {e['got'][:55]}\n\n"

    # Print the report
    print(report, file=sys.stderr)

    # Save
    out = {
        "dataset": "training-v3.json",
        "total_questions": overall_total,
        "overall_accuracy": overall_acc,
        "total_time_s": round(overall_time, 1),
        "total_tokens_out": overall_tok_out,
        "per_category": per_cat,
        "per_model_stats": cache._stats,
        "config": FINAL_CONFIG,
    }
    out_path = "/home/artem/dev/amd-hackathon/data/eval/final_optimized_run.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nFull results saved to {out_path}", file=sys.stderr)
    print(f"Report above can be copied to handoff_report.md", file=sys.stderr)

if __name__ == "__main__":
    main()
