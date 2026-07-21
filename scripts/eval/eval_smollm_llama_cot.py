#!/usr/bin/env python3
"""
SmolLM2 + Llama on weak categories (factual, logic, ner, summarization)
with 4 smarter prompts each: CoT, few-shot, rephrased task.
"""
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

TEMPLATES = {
    "factual": [
        {"name": "cot-reason",   "sys": "Think step by step. What is the exact answer? Use precise names and numbers.", "tok": 128},
        {"name": "short-answer", "sys": "Answer in 1-5 words. Exact fact only.", "tok": 32},
        {"name": "qa-format",    "sys": "Q: What is the answer? A:", "tok": 64},
        {"name": "direct-fact",  "sys": "Give the precise fact. Name, number, or date.", "tok": 64},
    ],
    "logic": [
        {"name": "cot-premises", "sys": "Let's reason step by step. Consider each premise carefully. What necessarily follows? Answer:", "tok": 256},
        {"name": "eliminate",    "sys": "Eliminate impossible cases one by one. What must be true? Answer:", "tok": 256},
        {"name": "table-reason", "sys": "Use a table to track possibilities. Rule out contradictions. Answer:", "tok": 256},
        {"name": "terse-conclusion", "sys": "What conclusion follows? One word or short phrase. Answer:", "tok": 64},
    ],
    "ner": [
        {"name": "cot-extract",  "sys": "Read the text carefully. Find every named entity. List them one per line with their type.", "tok": 256},
        {"name": "json-format",  "sys": "Extract named entities as JSON: {\"PERSON\":[],\"ORG\":[],\"LOC\":[],\"DATE\":[]}", "tok": 256},
        {"name": "category-list","sys": "PERSON: ...; ORG: ...; LOC: ...; DATE: ...", "tok": 256},
        {"name": "list-all",     "sys": "List all named entities. One per line: Name (Type). No other text.", "tok": 256},
    ],
    "summarization": [
        {"name": "headline",     "sys": "What is the headline for this news story? One sentence capturing the main event with names.", "tok": 128},
        {"name": "who-what",     "sys": "Extract: who did what, when, and where? Write one sentence.", "tok": 128},
        {"name": "title-only",   "sys": "Write a title for this text. Just the title, no prefix.", "tok": 128},
        {"name": "cot-headline", "sys": "Find the most important information. Who is involved? What happened? Write a one-sentence headline.", "tok": 128},
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

    for cat, templates in sorted(TEMPLATES.items()):
        print(f"\n{'='*60}", file=sys.stderr); print(f"{cat}", file=sys.stderr); print(f"{'='*60}", file=sys.stderr)
        qs = random.sample(by_cat[cat], min(5, len(by_cat[cat])))
        cat_results = {}

        for model_key, model_name in [("smollm", "smollm2-1.7b"), ("llama", "llama-3.2-1b")]:
            for tpl in templates:
                vname = f"{model_key}-{tpl['name']}"
                correct = 0; details = []
                for i, q in enumerate(qs):
                    answer, elapsed = cache.run(model_name, tpl["sys"], q["prompt"], 0.0, tpl["tok"])
                    ok = fuzzy_match(answer, q["expected_answer"])
                    if ok: correct += 1
                    details.append({"expected": q["expected_answer"], "got": answer, "correct": ok})
                    print(f"  {vname:25s} {'✓' if ok else '✗'} {answer[:55]}", file=sys.stderr)
                acc = round(correct/len(qs)*100, 1)
                cat_results[vname] = {"model": model_name, "prompt": tpl["sys"], "correct": correct, "total": len(qs), "accuracy": acc, "details": details}
                print(f"  → {vname}: {correct}/{len(qs)} = {acc}%", file=sys.stderr)

        all_results[cat] = cat_results

    cache.unload_all()

    # Summary
    models_tested = ["smollm2-1.7b", "llama-3.2-1b"]
    print(f"\n\n{'='*70}", file=sys.stderr)
    print("SMOLIM2 + LLAMA — CoT/few-shot prompts", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    for cat, variants in sorted(all_results.items()):
        print(f"\n{cat}:", file=sys.stderr)
        best_acc = max(v["accuracy"] for v in variants.values())
        for vname, vdata in sorted(variants.items()):
            acc = vdata["accuracy"]
            star = " ★" if acc >= best_acc else ""
            print(f"  {vname:30s} {vdata['model']:20s} {acc:5.1f}%{star}", file=sys.stderr)

    # Compare with previous best
    prev_best = {"factual": 40, "logic": 20, "ner": 20, "summarization": 0}  # from earlier rounds
    print(f"\n{'='*70}", file=sys.stderr)
    print("IMPROVEMENT OVER PREVIOUS BEST:", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    for cat in sorted(prev_best.keys()):
        variants = all_results.get(cat, {})
        if variants:
            best = max(v["accuracy"] for v in variants.values())
            prev = prev_best[cat]
            delta = best - prev
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "—"
            print(f"  {cat:20s} prev={prev}% → highest={best}% {arrow} (delta {delta:+.0f})", file=sys.stderr)
            if best > prev:
                # show the winning prompt
                for vn, vd in variants.items():
                    if vd["accuracy"] == best:
                        print(f"    Winner: {vn:30s} {vd['prompt'][:70]}", file=sys.stderr)
                        break

    out_path = "/home/artem/dev/amd-hackathon/data/eval/smollm_llama_cot.json"
    with open(out_path, "w") as f: json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
