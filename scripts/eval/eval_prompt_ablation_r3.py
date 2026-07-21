#!/usr/bin/env python3
"""
Round 3 prompt ablation — develop the best prompts further.
For each category, take the winning prompt style and create refinements.
"""
import json, re, sys, time, gc

MODEL_PATHS = {
    "qwen2.5-1.5b": "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "qwen2.5-math-1.5b": "/home/artem/models/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf",
    "gemma-3-1b": "/home/artem/models/gemma-3-1b-it-Q4_K_M.gguf",
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

# ── PROMPT VARIANTS (developed from best performers) ────────────────────────
VARIANTS = {
    "code_debug": [  # Best was "Fix:" (80%). Variations:
        {"name": "v-best-fix",      "model": "qwen2.5-math-1.5b", "sys": "Fix:", "temp": 0.0, "tok": 512},
        {"name": "v-empty",          "model": "qwen2.5-math-1.5b", "sys": "", "temp": 0.0, "tok": 512},
        {"name": "v-fix-bug",       "model": "qwen2.5-math-1.5b", "sys": "Fix the bug. Output the fixed function in backticks. No explanation.", "temp": 0.0, "tok": 512},
        {"name": "v-corrected-code","model": "qwen2.5-math-1.5b", "sys": "Corrected code:", "temp": 0.0, "tok": 512},
        {"name": "v-debug",          "model": "qwen2.5-math-1.5b", "sys": "Debug:", "temp": 0.0, "tok": 512},
        {"name": "v-return-only",    "model": "qwen2.5-math-1.5b", "sys": "Return the fixed function. Code only.", "temp": 0.0, "tok": 512},
    ],
    "code_gen": [  # Best was several at 60%. Refine:
        {"name": "v-best-basic",     "model": "gemma-3-1b", "sys": "Write the requested Python function. Output ONLY the function inside ```python ... ```. Keep the exact function name. No explanation.", "temp": 0.0, "tok": 512},
        {"name": "v-empty",          "model": "gemma-3-1b", "sys": "", "temp": 0.0, "tok": 512},
        {"name": "v-no-backticks",   "model": "gemma-3-1b", "sys": "Write the Python function. Output pure code only — no ```python, no markdown fences, no explanation. Just the function definition starting with 'def'.", "temp": 0.0, "tok": 512},
        {"name": "v-function-only",  "model": "gemma-3-1b", "sys": "Write the function. def", "temp": 0.0, "tok": 512},
        {"name": "v-terse",          "model": "gemma-3-1b", "sys": "Code:", "temp": 0.0, "tok": 512},
        {"name": "v-backticks-only", "model": "gemma-3-1b", "sys": "```python", "temp": 0.0, "tok": 512},
        {"name": "v-qwen-instead",  "model": "qwen2.5-1.5b", "sys": "Write the Python function. Output ONLY the function inside ```python ... ```. No explanation.", "temp": 0.0, "tok": 512},
    ],
    "factual": [  # Best was "Answer directly..." (40%). Refine:
        {"name": "v-best-direct",    "model": "qwen2.5-1.5b", "sys": "Answer the question directly. Use exact names, dates, and numbers. Keep under 15 words. No preamble.", "temp": 0.0, "tok": 64},
        {"name": "v-empty",          "model": "qwen2.5-1.5b", "sys": "", "temp": 0.0, "tok": 64},
        {"name": "v-answer",         "model": "qwen2.5-1.5b", "sys": "Answer:", "temp": 0.0, "tok": 64},
        {"name": "v-fact-only",      "model": "qwen2.5-1.5b", "sys": "State the exact fact. One sentence. Use precise names and numbers.", "temp": 0.0, "tok": 64},
        {"name": "v-who-what",       "model": "qwen2.5-1.5b", "sys": "Answer with the exact name, number, or date requested. Short and precise.", "temp": 0.0, "tok": 64},
        {"name": "v-guess-if-needed","model": "qwen2.5-1.5b", "sys": "Give the most likely answer. Use exact names and numbers. Keep it short.", "temp": 0.0, "tok": 64},
    ],
    "logic": [  # Best was "Step by step, Answer:" (20%). Refine:
        {"name": "v-best-step",      "model": "qwen2.5-math-1.5b", "sys": "Solve the logic puzzle step by step. Deduce from premises. End with 'Answer: <conclusion>' on its own line.", "temp": 0.0, "tok": 256},
        {"name": "v-empty",          "model": "qwen2.5-math-1.5b", "sys": "", "temp": 0.0, "tok": 256},
        {"name": "v-eliminate",      "model": "qwen2.5-math-1.5b", "sys": "Use process of elimination. Consider each possibility. Rule out contradictions. End with 'Answer: <conclusion>'.", "temp": 0.0, "tok": 256},
        {"name": "v-truth-table",    "model": "qwen2.5-math-1.5b", "sys": "Use a truth table or grid to track possibilities. End with 'Answer: <conclusion>'.", "temp": 0.0, "tok": 256},
        {"name": "v-terse",          "model": "qwen2.5-math-1.5b", "sys": "Deduce step by step. Answer:", "temp": 0.0, "tok": 256},
        {"name": "v-if-then",        "model": "qwen2.5-math-1.5b", "sys": "Apply logical reasoning. If-then deductions. Eliminate impossibilities. Answer:", "temp": 0.0, "tok": 256},
    ],
    "math": [  # Best was step-by-step + Answer: (80%). Refine:
        {"name": "v-best-step",      "model": "qwen2.5-math-1.5b", "sys": "Solve the math problem step by step. End with 'Answer: <value>' on its own line. Use standard decimal format.", "temp": 0.0, "tok": 512},
        {"name": "v-empty",          "model": "qwen2.5-math-1.5b", "sys": "", "temp": 0.0, "tok": 512},
        {"name": "v-verify",         "model": "qwen2.5-math-1.5b", "sys": "Solve step by step. Double-check your calculation. End with 'Answer: <value>'.", "temp": 0.0, "tok": 512},
        {"name": "v-brief",          "model": "qwen2.5-math-1.5b", "sys": "Solve. Show brief working. Answer:", "temp": 0.0, "tok": 512},
        {"name": "v-final-answer",   "model": "qwen2.5-math-1.5b", "sys": "Work through it. Final Answer:", "temp": 0.0, "tok": 512},
        {"name": "v-calc",           "model": "qwen2.5-math-1.5b", "sys": "Calc:", "temp": 0.0, "tok": 512},
    ],
    "ner": [  # Best was structured category format with Qwen (40%). Refine:
        {"name": "q-best-structured","model": "qwen2.5-1.5b", "sys": "Extract all named entities. Output as: CATEGORY: value1, value2; CATEGORY: value3. Use PERSON, ORG, LOC, DATE, DISEASE. No explanation.", "temp": 0.0, "tok": 256},
        {"name": "q-empty",          "model": "qwen2.5-1.5b", "sys": "", "temp": 0.0, "tok": 256},
        {"name": "q-entities",       "model": "qwen2.5-1.5b", "sys": "Entities:", "temp": 0.0, "tok": 256},
        {"name": "q-names-only",     "model": "qwen2.5-1.5b", "sys": "Find all names, organizations, locations, dates. List them.", "temp": 0.0, "tok": 256},
        {"name": "q-extract-list",   "model": "qwen2.5-1.5b", "sys": "Extract named entities. Format: * name (type). List format only.", "temp": 0.0, "tok": 256},
        {"name": "q-category-pairs", "model": "qwen2.5-1.5b", "sys": "Output entities as: PERSON=..., ORG=..., LOC=..., DATE=..., DISEASE=...", "temp": 0.0, "tok": 256},
        # Also test Gemma variants since they tied
        {"name": "g-best-entities",  "model": "gemma-3-1b", "sys": "Entities:", "temp": 0.0, "tok": 256},
        {"name": "g-structured",     "model": "gemma-3-1b", "sys": "Extract entities. PERSON=..., ORG=..., LOC=...", "temp": 0.0, "tok": 256},
        {"name": "g-empty",          "model": "gemma-3-1b", "sys": "", "temp": 0.0, "tok": 256},
    ],
    "sentiment": [  # Best was 100%! Test if we can keep it perfect with variations:
        {"name": "v-best",           "model": "qwen2.5-1.5b", "sys": "Classify the sentiment. Output EXACTLY one word: positive, negative, neutral, or mixed. No explanation.", "temp": 0.0, "tok": 32},
        {"name": "v-empty",          "model": "qwen2.5-1.5b", "sys": "", "temp": 0.0, "tok": 32},
        {"name": "v-sentiment",      "model": "qwen2.5-1.5b", "sys": "Sentiment:", "temp": 0.0, "tok": 16},
        {"name": "v-sarcasm-guard",  "model": "qwen2.5-1.5b", "sys": "Classify: positive, negative, neutral, or mixed. Sarcasm = negative. Hedging = neutral.", "temp": 0.0, "tok": 32},
        {"name": "v-label-only",     "model": "qwen2.5-1.5b", "sys": "Label:", "temp": 0.0, "tok": 16},
    ],
    "summarization": [  # All 0%. Try radically different approaches:
        {"name": "q-best-2sent",     "model": "qwen2.5-1.5b", "sys": "Summarize in at most 2 sentences. Include key names and numbers. No preamble.", "temp": 0.0, "tok": 256},
        {"name": "q-empty",          "model": "qwen2.5-1.5b", "sys": "", "temp": 0.0, "tok": 256},
        {"name": "q-key-facts",      "model": "qwen2.5-1.5b", "sys": "Key facts only. Who, what, when, where? 1-2 sentences.", "temp": 0.0, "tok": 256},
        {"name": "q-bullet",         "model": "qwen2.5-1.5b", "sys": "Bullet points of important facts. Each bullet one fact.", "temp": 0.0, "tok": 256},
        {"name": "q-one-line",       "model": "qwen2.5-1.5b", "sys": "One sentence summary.", "temp": 0.0, "tok": 128},
        {"name": "q-extract",        "model": "qwen2.5-1.5b", "sys": "Extract the most important information from this text: names, numbers, events.", "temp": 0.0, "tok": 256},
        {"name": "g-best-2sent",     "model": "gemma-3-1b", "sys": "Summarize in at most 2 sentences. Include key names and numbers.", "temp": 0.0, "tok": 256},
        {"name": "g-empty",          "model": "gemma-3-1b", "sys": "", "temp": 0.0, "tok": 256},
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
        print(f"\n{'='*60}", file=sys.stderr); print(f"CATEGORY: {cat}", file=sys.stderr); print(f"{'='*60}", file=sys.stderr)
        qs = test_qs[cat]
        cat_results = {}
        for v in variants:
            correct = 0; details = []
            for i, q in enumerate(qs):
                answer, elapsed = cache.run(v["model"], v["sys"], q["prompt"], v["temp"], v["tok"])
                ok = fuzzy_match(answer, q["expected_answer"])
                if ok: correct += 1
                details.append({"expected": q["expected_answer"], "got": answer, "correct": ok, "time_s": round(elapsed, 2)})
                print(f"  {cat:15s} {v['name']:20s} {'✓' if ok else '✗'} {answer[:50]}", file=sys.stderr)
            acc = round(correct/len(qs)*100, 1)
            cat_results[v["name"]] = {"model": v["model"], "prompt": v["sys"], "correct": correct, "total": len(qs), "accuracy": acc, "details": details}
            print(f"  → {v['name']}: {correct}/{len(qs)} = {acc}%", file=sys.stderr)
        all_results[cat] = cat_results

    cache.unload_all()

    # Summary
    print(f"\n\n{'='*70}", file=sys.stderr)
    print("ROUND 3 PROMPT ABLATION — Best-variant refinements", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    for cat, variants in sorted(all_results.items()):
        print(f"\n{cat}:", file=sys.stderr)
        best_acc = max(v["accuracy"] for v in variants.values())
        for vname, vdata in sorted(variants.items()):
            acc = vdata["accuracy"]
            star = " ★ BEST" if acc >= best_acc else ""
            print(f"  {vname:<25} {vdata['model']:<20} {acc:5.1f}% {star}", file=sys.stderr)
            if star:
                print(f"  {'':25} prompt: {vdata['prompt'][:80]}", file=sys.stderr)

    out_path = "/home/artem/dev/amd-hackathon/data/eval/prompt_ablation_round3.json"
    with open(out_path, "w") as f: json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
