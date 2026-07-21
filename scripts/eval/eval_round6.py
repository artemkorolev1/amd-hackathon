#!/usr/bin/env python3
"""Round 6 — push factual past 80% with ultra-minimal Llama prompts."""
import json, re, sys, time, gc, random

MODEL_PATHS = {"llama-3.2-1b": "/home/artem/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"}

def fuzzy_match(answer, expected):
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

# Genuinely different prompt framings for factual QA
VARIANTS = [
    {"name": "fact-colon",  "sys": "Fact:", "tok": 64},
    {"name": "empty",       "sys": "", "tok": 64},
    {"name": "answer-colon","sys": "Answer:", "tok": 64},
    {"name": "q-prefix",    "sys": "Q: {prompt}\nA:", "tok": 64, "inject_prompt": True},  # format Q/A
    {"name": "single-word", "sys": "One word: the exact name, number, or date.", "tok": 32},
    {"name": "colon-only",  "sys": ":", "tok": 32},
    {"name": "be-precise",  "sys": "Be precise.", "tok": 64},
    {"name": "give-fact",   "sys": "Give the exact fact.", "tok": 64},
]

class ModelCache:
    def __init__(self):
        self._models = {}
    def get(self, name):
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
    with open("/home/artem/dev/amd-hackathon/data/eval/training-v3.json") as f:
        all_q = json.load(f)
    by_cat = {}
    for q in all_q: by_cat.setdefault(q["category"], []).append(q)
    
    cache = ModelCache()
    
    for cat in ["factual", "code_gen"]:
        print(f"\n{'='*60}", file=sys.stderr); print(f"{cat} — Llama push", file=sys.stderr); print(f"{'='*60}", file=sys.stderr)
        qs = random.sample(by_cat[cat], min(5, len(by_cat[cat])))
        
        for v in VARIANTS:
            correct = 0
            for i, q in enumerate(qs):
                user_prompt = v.get("inject_prompt", False) and v["sys"].format(prompt=q["prompt"]) or q["prompt"]
                actual_sys = "" if v.get("inject_prompt", False) else v["sys"]
                answer, elapsed = cache.run("llama-3.2-1b", actual_sys, user_prompt, 0.0, v["tok"])
                ok = fuzzy_match(answer, q["expected_answer"])
                if ok: correct += 1
                print(f"  {cat:15s} {v['name']:20s} {'✓' if ok else '✗'} {answer[:55]}", file=sys.stderr)
            acc = round(correct/len(qs)*100, 1)
            print(f"  → {v['name']}: {correct}/{len(qs)} = {acc}%", file=sys.stderr)
    
    cache.unload_all()

if __name__ == "__main__":
    main()
