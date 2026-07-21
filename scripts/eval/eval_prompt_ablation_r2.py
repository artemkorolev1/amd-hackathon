#!/usr/bin/env python3
"""
Round 2 prompt ablation
 — Gemma-1B on code_gen: more creative prompt variants
 — Qwen-1.5B on NER: test if generalist beats Gemma
 — Qwen-1.5B on summarization: test if generalist is any better
"""
import json, re, sys, time, gc

MODEL_PATHS = {
    "qwen2.5-1.5b": "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
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

# ── PROMPT VARIANTS ─────────────────────────────────────────────────────────
VARIANTS = {
    "code_gen": [  # Gemma-1B — test more creative prompts
        {
            "name": "v1-basic",
            "model": "gemma-3-1b",
            "system_prompt": "Write the requested Python function. Output ONLY the function inside ```python ... ```. Keep the exact function name. No explanation.",
            "temperature": 0.0,
            "max_tokens": 512,
        },
        {
            "name": "v5-code-only",
            "model": "gemma-3-1b",
            "system_prompt": "Code:",
            "temperature": 0.0,
            "max_tokens": 512,
        },
        {
            "name": "v6-just-function",
            "model": "gemma-3-1b",
            "system_prompt": "Write the Python function. Output the function body only. No backticks. No markdown. No explanation.",
            "temperature": 0.0,
            "max_tokens": 512,
        },
        {
            "name": "v7-def-starter",
            "model": "gemma-3-1b",
            "system_prompt": "",
            "temperature": 0.0,
            "max_tokens": 512,
        },
        {
            "name": "v8-example",
            "model": "gemma-3-1b",
            "system_prompt": "Example:\nUser: Write a function that returns the sum of a list.\nAssistant: def sum_list(lst):\n    return sum(lst)\n\nNow write the function requested below. Output ONLY the function definition. No backticks.",
            "temperature": 0.0,
            "max_tokens": 512,
        },
        {
            "name": "v9-terse-fix",
            "model": "gemma-3-1b",
            "system_prompt": "Write the function. def",
            "temperature": 0.0,
            "max_tokens": 512,
        },
        {
            "name": "v10-no-backticks",
            "model": "gemma-3-1b",
            "system_prompt": "Write the Python function. Output pure code only — no ```python, no markdown fences, no explanation. Just the function definition starting with 'def'.",
            "temperature": 0.0,
            "max_tokens": 512,
        },
        {
            "name": "v11-qwen-prompt",
            "model": "qwen2.5-1.5b",
            "system_prompt": "Write the requested Python function. Output ONLY the function inside ```python ... ```. Keep the exact function name and signature. Handle edge cases. No explanation.",
            "temperature": 0.0,
            "max_tokens": 512,
        },
    ],
    "ner": [  # Qwen-1.5B — can the generalist do better than Gemma?
        {
            "name": "q-v1-entities",
            "model": "qwen2.5-1.5b",
            "system_prompt": "Entities:",
            "temperature": 0.0,
            "max_tokens": 256,
        },
        {
            "name": "q-v2-structured",
            "model": "qwen2.5-1.5b",
            "system_prompt": "Extract all named entities. Output as: CATEGORY: value1, value2; CATEGORY: value3. Use PERSON, ORG, LOC, DATE, DISEASE. No explanation.",
            "temperature": 0.0,
            "max_tokens": 256,
        },
        {
            "name": "q-v3-list",
            "model": "qwen2.5-1.5b",
            "system_prompt": "List every named entity found in the text. Format: * entity_name (type). No explanation.",
            "temperature": 0.0,
            "max_tokens": 256,
        },
        {
            "name": "q-v4-short",
            "model": "qwen2.5-1.5b",
            "system_prompt": "Extract entities. Output: PERSON names, ORG names, LOC names. Short list format. No explanation.",
            "temperature": 0.0,
            "max_tokens": 128,
        },
    ],
    "summarization": [  # Qwen-1.5B — can the generalist do better than Gemma?
        {
            "name": "q-v1-2-sentences",
            "model": "qwen2.5-1.5b",
            "system_prompt": "Summarize in at most 2 sentences. Include key names and numbers. No preamble.",
            "temperature": 0.0,
            "max_tokens": 256,
        },
        {
            "name": "q-v2-facts",
            "model": "qwen2.5-1.5b",
            "system_prompt": "Extract the key facts only. 1-2 sentences. Include names, numbers, dates. No preamble.",
            "temperature": 0.0,
            "max_tokens": 256,
        },
        {
            "name": "q-v3-bullet",
            "model": "qwen2.5-1.5b",
            "system_prompt": "Key facts as bullet points. Each bullet one fact. Include names, numbers. No preamble.",
            "temperature": 0.0,
            "max_tokens": 256,
        },
        {
            "name": "q-v4-terse",
            "model": "qwen2.5-1.5b",
            "system_prompt": "Summarize:",
            "temperature": 0.0,
            "max_tokens": 256,
        },
    ],
}

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
    def run(self, name, sys_prompt, user_prompt, temp=0.0, max_tok=512):
        llm = self.get(name)
        msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}] if sys_prompt else [{"role": "user", "content": user_prompt}]
        t0 = time.time()
        r = llm.create_chat_completion(messages=msgs, max_tokens=max_tok, temperature=temp)
        return r["choices"][0]["message"]["content"].strip(), time.time()-t0
    def unload_all(self):
        for k in list(self._models.keys()): del self._models[k]
        self._models = {}; gc.collect(); import torch; torch.cuda.empty_cache()

# ── Main ────────────────────────────────────────────────────────────────────
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
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"CATEGORY: {cat}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        qs = random.sample(by_cat[cat], min(5, len(by_cat[cat])))
        cat_results = {}

        for variant in variants:
            vname = variant["name"]
            model_name = variant["model"]
            correct = 0
            details = []
            for i, q in enumerate(qs):
                answer, elapsed = cache.run(model_name, variant["system_prompt"], q["prompt"],
                                            variant["temperature"], variant["max_tokens"])
                ok = fuzzy_match(answer, q["expected_answer"])
                if ok: correct += 1
                details.append({"task_id": q.get("task_id", f"q_{i}"), "expected": q["expected_answer"],
                                "got": answer, "correct": ok, "time_s": round(elapsed, 2)})
                print(f"  {cat:15s} {vname:20s} {'✓' if ok else '✗'} {answer[:50]}", file=sys.stderr)

            acc = correct / len(qs) * 100
            cat_results[vname] = {"model": model_name, "prompt": variant["system_prompt"],
                                  "correct": correct, "total": len(qs), "accuracy": round(acc, 1), "details": details}
            print(f"  → {vname}: {correct}/{len(qs)} = {acc:.1f}%", file=sys.stderr)

        all_results[cat] = cat_results

    cache.unload_all()

    # Summary
    print(f"\n\n{'='*70}", file=sys.stderr)
    print("ROUND 2 PROMPT ABLATION SUMMARY", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    print(f"{'Category':<18} {'Variant':<22} {'Model':<20} {'Acc':<8} {'Best?':<6}", file=sys.stderr)
    print(f"{'-'*74}", file=sys.stderr)

    for cat, variants in sorted(all_results.items()):
        best_acc = max(v["accuracy"] for v in variants.values())
        for vname, vdata in sorted(variants.items()):
            acc = vdata["accuracy"]
            is_best = "★" if acc >= best_acc else ""
            print(f"{cat:<18} {vname:<22} {vdata['model']:<20} {acc:<8} {is_best:<6}", file=sys.stderr)
        print(f"{'':>18} {'─'*70}", file=sys.stderr)

    # Comparison with Round 1
    print(f"\n\nBEST PER CATEGORY (Round 2):", file=sys.stderr)
    for cat in sorted(all_results.keys()):
        variants = all_results[cat]
        best_v = max(variants.values(), key=lambda v: v["accuracy"])
        print(f"  {cat:20s} → {best_v['model']:20s} {best_v['accuracy']}%  '{best_v['prompt'][:60]}'", file=sys.stderr)

    out_path = "/home/artem/dev/amd-hackathon/data/eval/prompt_ablation_round2.json"
    with open(out_path, "w") as f: json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
