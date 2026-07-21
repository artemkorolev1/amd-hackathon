#!/usr/bin/env python3
"""
Round 4: Pick best model per category, test 4 refined prompts on each.
Models: Qwen2.5-Coder (code_debug, code_gen, ner), Math-1.5B (math, logic), Qwen-1.5B (factual, sentiment)
"""
import json, re, sys, time, gc

MODEL_PATHS = {
    "qwen2.5-coder": "/home/artem/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    "qwen2.5-math-1.5b": "/home/artem/models/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf",
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

# 4 new prompts per category + include the best from Round 3 as baseline
VARIANTS = {
    "code_debug": [  # Best model: Qwen2.5-Coder
        {"name": "prev-best-debug", "model": "qwen2.5-coder", "sys": "Debug:", "temp": 0.0, "tok": 512},
        {"name": "refined-1",       "model": "qwen2.5-coder", "sys": "Fix the bug:", "temp": 0.0, "tok": 512},
        {"name": "refined-2",       "model": "qwen2.5-coder", "sys": "Correct this function:", "temp": 0.0, "tok": 512},
        {"name": "refined-3",       "model": "qwen2.5-coder", "sys": "Bug: return the fixed code in backticks.", "temp": 0.0, "tok": 512},
        {"name": "refined-4",       "model": "qwen2.5-coder", "sys": "The function has a bug. Fix it:", "temp": 0.0, "tok": 512},
    ],
    "code_gen": [  # Best model: Qwen2.5-Coder with "```python"
        {"name": "prev-best-backticks", "model": "qwen2.5-coder", "sys": "```python", "temp": 0.0, "tok": 512},
        {"name": "refined-1",           "model": "qwen2.5-coder", "sys": "```python\ndef", "temp": 0.0, "tok": 512},
        {"name": "refined-2",           "model": "qwen2.5-coder", "sys": "Write the function. ```python", "temp": 0.0, "tok": 512},
        {"name": "refined-3",           "model": "qwen2.5-coder", "sys": "Implement the Python function:", "temp": 0.0, "tok": 512},
        {"name": "refined-4",           "model": "qwen2.5-coder", "sys": "Code. No explanation. def", "temp": 0.0, "tok": 512},
    ],
    "math": [  # Best model: Math-1.5B
        {"name": "prev-best-step",      "model": "qwen2.5-math-1.5b", "sys": "Solve step by step. End with 'Answer: <value>'.", "temp": 0.0, "tok": 512},
        {"name": "refined-1",           "model": "qwen2.5-math-1.5b", "sys": "Work through this math problem. Final Answer:", "temp": 0.0, "tok": 512},
        {"name": "refined-2",           "model": "qwen2.5-math-1.5b", "sys": "Calculate. Show steps. End: 'Answer: <value>'.", "temp": 0.0, "tok": 512},
        {"name": "refined-3",           "model": "qwen2.5-math-1.5b", "sys": "Math problem. Solve it. Then write 'Answer: <value>'.", "temp": 0.0, "tok": 512},
        {"name": "refined-4",           "model": "qwen2.5-math-1.5b", "sys": "Solve: output ONLY the final number on the last line.", "temp": 0.0, "tok": 512},
    ],
    "factual": [  # Best model: Qwen-1.5B (stuck at 40%)
        {"name": "prev-best-direct",    "model": "qwen2.5-1.5b", "sys": "Answer the question directly. Use exact names, dates, numbers. Under 15 words.", "temp": 0.0, "tok": 64},
        {"name": "refined-1",           "model": "qwen2.5-1.5b", "sys": "Fact: give the precise answer. Names and numbers exactly. 1-2 words if possible.", "temp": 0.0, "tok": 64},
        {"name": "refined-2",           "model": "qwen2.5-1.5b", "sys": "What is the exact answer? Be specific with names, dates, figures.", "temp": 0.0, "tok": 64},
        {"name": "refined-3",           "model": "qwen2.5-1.5b", "sys": "Short factual answer. Exact wording. No extra sentences.", "temp": 0.0, "tok": 64},
        {"name": "refined-4",           "model": "qwen2.5-1.5b", "sys": "", "temp": 0.0, "tok": 64},
    ],
    "logic": [  # Best model: Math-1.5B (stuck at 20%)
        {"name": "prev-best-deduce",    "model": "qwen2.5-math-1.5b", "sys": "Deduce step by step. Answer:", "temp": 0.0, "tok": 256},
        {"name": "refined-1",           "model": "qwen2.5-math-1.5b", "sys": "Logic puzzle. Reason from premises. What must be true? Answer:", "temp": 0.0, "tok": 256},
        {"name": "refined-2",           "model": "qwen2.5-math-1.5b", "sys": "Use elimination: rule out impossible cases. Answer:", "temp": 0.0, "tok": 256},
        {"name": "refined-3",           "model": "qwen2.5-math-1.5b", "sys": "Think step by step. The answer is a single word or short phrase. Answer:", "temp": 0.0, "tok": 256},
        {"name": "refined-4",           "model": "qwen2.5-math-1.5b", "sys": "Solve. What conclusion follows? Answer:", "temp": 0.0, "tok": 256},
    ],
    "ner": [  # Best model: Qwen2.5-Coder (40%)
        {"name": "prev-best-list",      "model": "qwen2.5-coder", "sys": "Extract named entities. Format: * name (type). List only.", "temp": 0.0, "tok": 256},
        {"name": "refined-1",           "model": "qwen2.5-coder", "sys": "List every named entity. One per line: Name (CATEGORY). No other text.", "temp": 0.0, "tok": 256},
        {"name": "refined-2",           "model": "qwen2.5-coder", "sys": "Named entities: PERSON, ORG, LOC, DATE, DISEASE. List each on its own line.", "temp": 0.0, "tok": 256},
        {"name": "refined-3",           "model": "qwen2.5-coder", "sys": "Extract: all people, organizations, locations, dates from the text.", "temp": 0.0, "tok": 256},
        {"name": "refined-4",           "model": "qwen2.5-coder", "sys": "Entities:", "temp": 0.0, "tok": 256},
    ],
    "sentiment": [  # Best model: Qwen-1.5B (100%)
        {"name": "prev-best-exact",     "model": "qwen2.5-1.5b", "sys": "Classify the sentiment. Output EXACTLY one word: positive, negative, neutral, or mixed. No explanation.", "temp": 0.0, "tok": 32},
        {"name": "refined-1",           "model": "qwen2.5-1.5b", "sys": "Sentiment (one word):", "temp": 0.0, "tok": 16},
        {"name": "refined-2",           "model": "qwen2.5-1.5b", "sys": "positive, negative, neutral, or mixed? One word only.", "temp": 0.0, "tok": 16},
        {"name": "refined-3",           "model": "qwen2.5-1.5b", "sys": "Classify: choose one — positive, negative, neutral, mixed.", "temp": 0.0, "tok": 32},
        {"name": "refined-4",           "model": "qwen2.5-1.5b", "sys": "Sentiment: positive/negative/neutral/mixed.", "temp": 0.0, "tok": 16},
    ],
    "summarization": [  # All 0%. One final try with Coder + Qwen
        {"name": "coder-summary",       "model": "qwen2.5-coder", "sys": "Summarize the text in 1-2 sentences. Key facts only.", "temp": 0.0, "tok": 256},
        {"name": "coder-key-points",    "model": "qwen2.5-coder", "sys": "Key points from this text:", "temp": 0.0, "tok": 256},
        {"name": "qwen-summary",        "model": "qwen2.5-1.5b", "sys": "Short summary. Names and numbers. 1 sentence.", "temp": 0.0, "tok": 128},
        {"name": "qwen-who-what",       "model": "qwen2.5-1.5b", "sys": "Who, what, when, where? Extract key facts.", "temp": 0.0, "tok": 128},
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
    # Use 5 per category
    test_qs = {cat: random.sample(by_cat[cat], min(5, len(by_cat[cat]))) for cat in VARIANTS}

    cache = ModelCache()
    all_results = {}

    for cat, variants in sorted(VARIANTS.items()):
        print(f"\n{'='*60}", file=sys.stderr); print(f"{cat}", file=sys.stderr); print(f"{'='*60}", file=sys.stderr)
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
    print("ROUND 4 — BEST MODEL + 4 REFINED PROMPTS", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    final_config = {}
    for cat, variants in sorted(all_results.items()):
        print(f"\n{cat}:", file=sys.stderr)
        best_acc = max(v["accuracy"] for v in variants.values())
        best_v = None
        for vname, vdata in sorted(variants.items()):
            acc = vdata["accuracy"]
            star = " ★" if acc >= best_acc else ""
            print(f"  {vname:25s} {vdata['model']:20s} {acc:5.1f}%{star}", file=sys.stderr)
            if acc >= best_acc and best_v is None or (best_v and acc > best_v["accuracy"]):
                best_v = vdata
                best_v["name"] = vname
        if best_v:
            final_config[cat] = {
                "model": best_v["model"],
                "prompt": best_v["prompt"],
                "accuracy": best_acc,
            }

    print(f"\n{'='*70}", file=sys.stderr)
    print("FINAL CONFIG — best model + prompt per category:", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    for cat, cfg in sorted(final_config.items()):
        print(f"  {cat:20s} → {cfg['model']:20s} ({cfg['accuracy']}%)  {cfg['prompt'][:60]}", file=sys.stderr)

    out_path = "/home/artem/dev/amd-hackathon/data/eval/final_config_ablation.json"
    with open(out_path, "w") as f: json.dump({"final_config": final_config, "full_results": all_results}, f, indent=2)
    print(f"\nSaved to {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
