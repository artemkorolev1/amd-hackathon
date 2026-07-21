#!/usr/bin/env python3
"""Test SmolLM2-1.7B and Llama 3.2-1B on weak categories + code_gen."""
import json, re, sys, time, gc

MODEL_PATHS = {
    "smollm2-1.7b": "/home/artem/models/smollm2-1.7b-instruct-q4_k_m.gguf",
    "llama-3.2-1b": "/home/artem/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
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
    "code_gen": [
        {"name": "smollm-code",      "model": "smollm2-1.7b", "sys": "```python", "temp": 0.0, "tok": 512},
        {"name": "smollm-full",      "model": "smollm2-1.7b", "sys": "Write the Python function. Output ONLY the function inside ```python ... ```. Keep exact name. No explanation.", "temp": 0.0, "tok": 512},
        {"name": "smollm-empty",     "model": "smollm2-1.7b", "sys": "", "temp": 0.0, "tok": 512},
        {"name": "llama-code",       "model": "llama-3.2-1b", "sys": "```python", "temp": 0.0, "tok": 512},
        {"name": "llama-full",       "model": "llama-3.2-1b", "sys": "Write the Python function. Output ONLY the function inside ```python ... ```. No explanation.", "temp": 0.0, "tok": 512},
        {"name": "llama-empty",      "model": "llama-3.2-1b", "sys": "", "temp": 0.0, "tok": 512},
    ],
    "factual": [
        {"name": "smollm-direct",    "model": "smollm2-1.7b", "sys": "Answer directly. Exact names and numbers. Under 15 words.", "temp": 0.0, "tok": 64},
        {"name": "smollm-empty",     "model": "smollm2-1.7b", "sys": "", "temp": 0.0, "tok": 64},
        {"name": "smollm-answer",    "model": "smollm2-1.7b", "sys": "Answer:", "temp": 0.0, "tok": 64},
        {"name": "llama-direct",     "model": "llama-3.2-1b", "sys": "Answer directly. Exact names and numbers. Under 15 words.", "temp": 0.0, "tok": 64},
        {"name": "llama-empty",      "model": "llama-3.2-1b", "sys": "", "temp": 0.0, "tok": 64},
        {"name": "llama-answer",     "model": "llama-3.2-1b", "sys": "Answer:", "temp": 0.0, "tok": 64},
    ],
    "logic": [
        {"name": "smollm-deduce",    "model": "smollm2-1.7b", "sys": "Deduce step by step. Answer:", "temp": 0.0, "tok": 256},
        {"name": "smollm-empty",     "model": "smollm2-1.7b", "sys": "", "temp": 0.0, "tok": 256},
        {"name": "smollm-table",     "model": "smollm2-1.7b", "sys": "Use elimination. Answer:", "temp": 0.0, "tok": 256},
        {"name": "llama-deduce",     "model": "llama-3.2-1b", "sys": "Deduce step by step. Answer:", "temp": 0.0, "tok": 256},
        {"name": "llama-empty",      "model": "llama-3.2-1b", "sys": "", "temp": 0.0, "tok": 256},
        {"name": "llama-answer",     "model": "llama-3.2-1b", "sys": "What follows? Answer:", "temp": 0.0, "tok": 256},
    ],
    "ner": [
        {"name": "smollm-entities",  "model": "smollm2-1.7b", "sys": "Entities:", "temp": 0.0, "tok": 256},
        {"name": "smollm-list",      "model": "smollm2-1.7b", "sys": "List entities. One per line: Name (Type).", "temp": 0.0, "tok": 256},
        {"name": "smollm-empty",     "model": "smollm2-1.7b", "sys": "", "temp": 0.0, "tok": 256},
        {"name": "llama-entities",   "model": "llama-3.2-1b", "sys": "Entities:", "temp": 0.0, "tok": 256},
        {"name": "llama-list",       "model": "llama-3.2-1b", "sys": "List entities. One per line: Name (Type).", "temp": 0.0, "tok": 256},
        {"name": "llama-empty",      "model": "llama-3.2-1b", "sys": "", "temp": 0.0, "tok": 256},
    ],
    "summarization": [
        {"name": "smollm-2sent",     "model": "smollm2-1.7b", "sys": "Summarize in at most 2 sentences. Key names, numbers, facts.", "temp": 0.0, "tok": 256},
        {"name": "smollm-facts",     "model": "smollm2-1.7b", "sys": "Key facts only. Who, what, where?", "temp": 0.0, "tok": 256},
        {"name": "smollm-empty",     "model": "smollm2-1.7b", "sys": "", "temp": 0.0, "tok": 256},
        {"name": "llama-2sent",      "model": "llama-3.2-1b", "sys": "Summarize in at most 2 sentences. Key names, numbers, facts.", "temp": 0.0, "tok": 256},
        {"name": "llama-facts",      "model": "llama-3.2-1b", "sys": "Key facts only. Who, what, where?", "temp": 0.0, "tok": 256},
        {"name": "llama-empty",      "model": "llama-3.2-1b", "sys": "", "temp": 0.0, "tok": 256},
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
        print(f"\n{'='*60}", file=sys.stderr); print(f"{cat} — SmolLM2 + Llama", file=sys.stderr); print(f"{'='*60}", file=sys.stderr)
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
    print("SMALLM2 + LLAMA — potential assessment", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    for cat, variants in sorted(all_results.items()):
        print(f"\n{cat}:", file=sys.stderr)
        best_acc = max(v["accuracy"] for v in variants.values())
        for vname, vdata in sorted(variants.items()):
            acc = vdata["accuracy"]
            star = " ★ BEST" if acc >= best_acc else ""
            print(f"  {vname:25s} {vdata['model']:20s} {acc:5.1f}%{star}", file=sys.stderr)

    out_path = "/home/artem/dev/amd-hackathon/data/eval/smollm_llama_test.json"
    with open(out_path, "w") as f: json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
