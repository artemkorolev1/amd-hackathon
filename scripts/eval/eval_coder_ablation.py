#!/usr/bin/env python3
"""Test Qwen2.5-Coder on code_gen, code_debug, math, and ner."""
import json, re, sys, time, gc

MODEL_PATHS = {
    "qwen2.5-coder": "/home/artem/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    "qwen2.5-math-1.5b": "/home/artem/models/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf",
    "gemma-3-1b": "/home/artem/models/gemma-3-1b-it-Q4_K_M.gguf",
    "qwen2.5-1.5b": "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
}

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

VARIANTS = {
    "code_debug": [
        {"name": "coder-fix",     "model": "qwen2.5-coder", "sys": "Fix:", "temp": 0.0, "tok": 512},
        {"name": "coder-debug",   "model": "qwen2.5-coder", "sys": "Debug:", "temp": 0.0, "tok": 512},
        {"name": "coder-empty",   "model": "qwen2.5-coder", "sys": "", "temp": 0.0, "tok": 512},
        {"name": "coder-fix-bug", "model": "qwen2.5-coder", "sys": "Fix the bug. Output the corrected function in backticks. No explanation.", "temp": 0.0, "tok": 512},
        # Comparison: Math-1.5B with best prompt
        {"name": "math-fix",      "model": "qwen2.5-math-1.5b", "sys": "Fix:", "temp": 0.0, "tok": 512},
    ],
    "code_gen": [
        {"name": "coder-code",        "model": "qwen2.5-coder", "sys": "Code:", "temp": 0.0, "tok": 512},
        {"name": "coder-backticks",   "model": "qwen2.5-coder", "sys": "```python", "temp": 0.0, "tok": 512},
        {"name": "coder-empty",       "model": "qwen2.5-coder", "sys": "", "temp": 0.0, "tok": 512},
        {"name": "coder-full",        "model": "qwen2.5-coder", "sys": "Write the Python function. Output ONLY the function inside ```python ... ```. Keep exact name. No explanation.", "temp": 0.0, "tok": 512},
        {"name": "coder-def",         "model": "qwen2.5-coder", "sys": "Write the function. def", "temp": 0.0, "tok": 512},
        {"name": "coder-no-markdown", "model": "qwen2.5-coder", "sys": "Write the Python function. No markdown. Just the code starting with 'def'.", "temp": 0.0, "tok": 512},
        # Comparison: Gemma-1B with best prompt
        {"name": "gemma-backticks",   "model": "gemma-3-1b", "sys": "```python", "temp": 0.0, "tok": 512},
    ],
    "math": [
        {"name": "coder-step",        "model": "qwen2.5-coder", "sys": "Solve step by step. End with 'Answer: <value>'.", "temp": 0.0, "tok": 512},
        {"name": "coder-calc",        "model": "qwen2.5-coder", "sys": "Calc:", "temp": 0.0, "tok": 512},
        {"name": "coder-empty",       "model": "qwen2.5-coder", "sys": "", "temp": 0.0, "tok": 512},
        {"name": "coder-answer",      "model": "qwen2.5-coder", "sys": "Solve. Answer:", "temp": 0.0, "tok": 512},
        {"name": "coder-python",      "model": "qwen2.5-coder", "sys": "Write a Python function to solve this, then call it. Output the answer.", "temp": 0.0, "tok": 512},
        # Comparison: Math-1.5B with best prompt
        {"name": "math-step",         "model": "qwen2.5-math-1.5b", "sys": "Solve step by step. End with 'Answer: <value>'.", "temp": 0.0, "tok": 512},
    ],
    "ner": [
        {"name": "coder-entities",    "model": "qwen2.5-coder", "sys": "Entities:", "temp": 0.0, "tok": 256},
        {"name": "coder-empty",       "model": "qwen2.5-coder", "sys": "", "temp": 0.0, "tok": 256},
        {"name": "coder-structured",  "model": "qwen2.5-coder", "sys": "Extract all named entities. Output as: CATEGORY: value1, value2; Use PERSON, ORG, LOC, DATE. No explanation.", "temp": 0.0, "tok": 256},
        {"name": "coder-list",        "model": "qwen2.5-coder", "sys": "Extract named entities. Format: * name (type). List format only.", "temp": 0.0, "tok": 256},
        {"name": "coder-extract",     "model": "qwen2.5-coder", "sys": "Extract all named entities. Output each on its own line: Name (Type)", "temp": 0.0, "tok": 256},
        # Comparison: Gemma-1B with best prompt
        {"name": "gemma-entities",    "model": "gemma-3-1b", "sys": "Entities:", "temp": 0.0, "tok": 256},
        {"name": "qwen-entities",     "model": "qwen2.5-1.5b", "sys": "Extract named entities. Format: * name (type). List format only.", "temp": 0.0, "tok": 256},
    ],
}

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
    def run(self, name, sys_prompt, user_prompt, temp=0.0, max_tok=512):
        llm = self.get(name)
        msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}] if sys_prompt else [{"role": "user", "content": user_prompt}]
        t0 = time.time()
        r = llm.create_chat_completion(messages=msgs, max_tokens=max_tok, temperature=temp)
        return r["choices"][0]["message"]["content"].strip(), time.time()-t0
    def unload_all(self):
        for k in list(self._models.keys()): del self._models[k]
        self._models = {}; gc.collect(); import torch; torch.cuda.empty_cache()

def main():
    dataset_path = "/home/artem/dev/amd-hackathon/data/eval/training-v3.json"
    with open(dataset_path) as f: all_q = json.load(f)
    import random
    random.seed(42)
    by_cat = {}
    for q in all_q: by_cat.setdefault(q["category"], []).append(q)
    test_qs = {cat: random.sample(by_cat[cat], min(5, len(by_cat[cat]))) for cat in VARIANTS}

    cache = ModelCache()
    all_results = {}

    for cat, variants in sorted(VARIANTS.items()):
        print(f"\n{'='*60}", file=sys.stderr); print(f"CODER TEST: {cat}", file=sys.stderr); print(f"{'='*60}", file=sys.stderr)
        qs = test_qs[cat]
        cat_results = {}
        for v in variants:
            correct = 0; details = []
            for i, q in enumerate(qs):
                answer, elapsed = cache.run(v["model"], v["sys"], q["prompt"], v["temp"], v["tok"])
                ok = fuzzy_match(answer, q["expected_answer"])
                if ok: correct += 1
                details.append({"expected": q["expected_answer"], "got": answer, "correct": ok})
                print(f"  {v['name']:25s} {'✓' if ok else '✗'} {answer[:55]}", file=sys.stderr)
            acc = round(correct/len(qs)*100, 1)
            cat_results[v["name"]] = {"model": v["model"], "prompt": v["sys"], "correct": correct, "total": len(qs), "accuracy": acc, "details": details}
            print(f"  → {v['name']}: {correct}/{len(qs)} = {acc}%", file=sys.stderr)
        all_results[cat] = cat_results

    cache.unload_all()

    # Summary
    print(f"\n\n{'='*70}", file=sys.stderr)
    print("QWEN2.5-CODER ABLATION vs Math/Gemma/Qwen baselines", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    for cat, variants in sorted(all_results.items()):
        print(f"\n{cat}:", file=sys.stderr)
        best_acc = max(v["accuracy"] for v in variants.values())
        for vname, vdata in sorted(variants.items()):
            acc = vdata["accuracy"]
            star = " ★" if acc >= best_acc else ""
            print(f"  {vname:25s} {vdata['model']:20s} {acc:5.1f}%{star}", file=sys.stderr)

    out_path = "/home/artem/dev/amd-hackathon/data/eval/coder_ablation.json"
    with open(out_path, "w") as f: json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
