#!/usr/bin/env python3
"""
Round 5 — targeted prompt fixes for categories still improving.
code_gen: fix function name drift. factual: push past 80%. ner: push past 40%.
"""
import json, re, sys, time, gc

MODEL_PATHS = {
    "qwen2.5-coder": "/home/artem/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
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
    "code_gen": [  # Coder: fix function name drift
        {"name": "exact-name",  "model": "qwen2.5-coder", "sys": "Use the EXACT function name from the request. No renaming. Output in ```python.", "tok": 512},
        {"name": "name-first",  "model": "qwen2.5-coder", "sys": "Write the function. The name must match the specification exactly. ```python", "tok": 512},
        {"name": "def-strict",  "model": "qwen2.5-coder", "sys": "```python\ndef", "tok": 512},
        {"name": "preserve-sig","model": "qwen2.5-coder", "sys": "Implement the function preserving EXACT function name and signature. Code only.", "tok": 512},
        {"name": "match-exact", "model": "qwen2.5-coder", "sys": "Read the task carefully. Copy the exact function name. Implement it. ```python", "tok": 512},
        {"name": "backticks-def","model": "qwen2.5-coder", "sys": "```python\ndef ", "tok": 512},  # trailing space
    ],
    "factual": [  # Llama: push past 80%
        {"name": "exact-answer","model": "llama-3.2-1b", "sys": "Answer with the exact name, number, or date. One short phrase.", "tok": 64},
        {"name": "fact-prefix", "model": "llama-3.2-1b", "sys": "Fact:", "tok": 64},
        {"name": "be-specific", "model": "llama-3.2-1b", "sys": "Provide the precise fact requested. Be very specific.", "tok": 64},
        {"name": "short-fact",  "model": "llama-3.2-1b", "sys": "Short factual answer:", "tok": 64},
        {"name": "answer-only", "model": "llama-3.2-1b", "sys": "", "tok": 64},  # empty — pure gen
        {"name": "name-number", "model": "llama-3.2-1b", "sys": "Name, number, or date. Exact. No extra words.", "tok": 64},
    ],
    "ner": [  # Llama: push past 40%
        {"name": "json-only",   "model": "llama-3.2-1b", "sys": '{"PERSON":[],"ORG":[],"LOC":[],"DATE":[]}  Fill in entities. Output ONLY the JSON.', "tok": 256},
        {"name": "list-types",  "model": "llama-3.2-1b", "sys": "PERSON: ...\nORG: ...\nLOC: ...\nDATE: ...\nFill in each category. No other text.", "tok": 256},
        {"name": "json-precise","model": "llama-3.2-1b", "sys": "Output entities as JSON with keys PERSON, ORG, LOC, DATE. Only entities in the text.", "tok": 256},
        {"name": "extract-all", "model": "llama-3.2-1b", "sys": "Find every named entity. Output JSON: {PERSON:[], ORG:[], LOC:[], DATE:[]}", "tok": 256},
        {"name": "no-preamble", "model": "llama-3.2-1b", "sys": "JSON only:", "tok": 256},
        {"name": "compact",     "model": "llama-3.2-1b", "sys": "Extract: persons, orgs, locations, dates. JSON format, no other text.", "tok": 256},
    ],
    "summarization": [  # One more try with Llama + different framing
        {"name": "extract-event","model": "llama-3.2-1b", "sys": "What event happened? Who was involved? Where? One sentence.", "tok": 128},
        {"name": "title-gen",   "model": "llama-3.2-1b", "sys": "Generate a news headline for this text. Title only.", "tok": 128},
        {"name": "bbc-style",   "model": "llama-3.2-1b", "sys": "BBC-style headline. One sentence. Names and key action.", "tok": 128},
        {"name": "lead-sentence","model": "llama-3.2-1b", "sys": "Write the lead sentence of a news article about this. One sentence.", "tok": 128},
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

    cache = ModelCache()
    all_results = {}

    for cat, variants in sorted(VARIANTS.items()):
        print(f"\n{'='*60}", file=sys.stderr); print(f"{cat}", file=sys.stderr); print(f"{'='*60}", file=sys.stderr)
        qs = random.sample(by_cat[cat], min(5, len(by_cat[cat])))
        cat_results = {}
        for v in variants:
            correct = 0; details = []
            for i, q in enumerate(qs):
                answer, elapsed = cache.run(v["model"], v["sys"], q["prompt"], 0.0, v["tok"])
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
    print("ROUND 5 — Targeted fixes for code_gen, factual, ner, summarization", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    prev_best = {"code_gen": 80, "factual": 80, "ner": 40, "summarization": 0}
    for cat, variants in sorted(all_results.items()):
        best_acc = max(v["accuracy"] for v in variants.values())
        prev = prev_best.get(cat, 0)
        delta = best_acc - prev
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "—"
        print(f"\n{cat}: best={best_acc}% (prev={prev}% {arrow})", file=sys.stderr)
        for vname, vdata in sorted(variants.items()):
            acc = vdata["accuracy"]
            star = " ★" if acc >= best_acc else ""
            print(f"  {vname:25s} {vdata['model']:20s} {acc:5.1f}%{star}", file=sys.stderr)
            if star:
                print(f"  {'':25} prompt: {vdata['prompt'][:80]}", file=sys.stderr)

    out_path = "/home/artem/dev/amd-hackathon/data/eval/round5_fixes.json"
    with open(out_path, "w") as f: json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
