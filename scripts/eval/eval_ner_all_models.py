#!/usr/bin/env python3
"""Run all 4 remaining models on NER (19 questions) with best prompts."""
import json, re, sys, time, gc

MODEL_PATHS = {
    "qwen2.5-1.5b": "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "qwen2.5-coder": "/home/artem/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    "qwen2.5-math-1.5b": "/home/artem/models/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf",
    "llama-3.2-1b": "/home/artem/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
}

def fuzzy_match(answer, expected):
    a, e = answer.strip().lower(), expected.strip().lower()
    if not a or not e: return False
    if a == e: return True
    if len(e) <= 20 and e in a: return True
    if len(a) <= 20 and a in e: return True
    na = re.findall(r"-?\d+(?:\.\d+)?", a)
    ne = re.findall(r"-?\d+(?:\.\d+)?", e)
    if na and ne:
        an, en = float(na[-1]), float(ne[-1])
        if en != 0 and abs((an-en)/en) <= 0.01: return True
        if an == en: return True
    ta = set(t for t in re.split(r"[^a-zA-Z0-9.]+", a) if t)
    te = set(t for t in re.split(r"[^a-zA-Z0-9.]+", e) if t)
    if len(te) > 0 and len(ta & te) / len(te) >= 0.8: return True
    return False

# Best NER prompt per model from ablation
VARIANTS = [
    ("qwen2.5-1.5b", "Extract named entities. Format: * name (type). List format only."),
    ("qwen2.5-coder", "Extract named entities. Format: * name (type). List only."),
    ("qwen2.5-math-1.5b", "Entities:"),
    ("llama-3.2-1b", '{"PERSON":[],"ORG":[],"LOC":[],"DATE":[]}  Fill in entities. Output ONLY the JSON.'),
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
    def run(self, name, sys_prompt, user_prompt, temp=0.0, max_tok=256):
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
    ner_qs = [q for q in all_q if q["category"] == "ner"]
    print(f"NER questions: {len(ner_qs)}", file=sys.stderr)
    
    cache = ModelCache()
    all_results = {}
    
    for model_name, system_prompt in VARIANTS:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Model: {model_name}", file=sys.stderr)
        print(f"Prompt: {system_prompt[:60]}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        
        total_ok = 0
        total_time = 0
        total_tok = 0
        results = []
        
        for i, q in enumerate(ner_qs):
            answer, elapsed = cache.run(model_name, system_prompt, q["prompt"], 0.0, 256)
            ok = fuzzy_match(answer, q["expected_answer"])
            tok = len(answer.split())
            total_ok += 1 if ok else 0
            total_time += elapsed
            total_tok += tok
            
            results.append({
                "task_id": q["task_id"],
                "expected": q["expected_answer"],
                "got": answer,
                "correct": ok,
                "time_s": round(elapsed, 2),
                "tokens": tok,
            })
            
            marker = "✓" if ok else "✗"
            print(f"  [{i+1:2d}/{len(ner_qs)}] {marker} ({elapsed:.1f}s) {answer[:55]}", file=sys.stderr)
            if not ok:
                print(f"        exp: {q['expected_answer'][:55]}", file=sys.stderr)
        
        acc = round(total_ok / len(ner_qs) * 100, 1)
        print(f"  → {model_name}: {total_ok}/{len(ner_qs)} = {acc}%  ({total_time:.0f}s, {total_tok//len(ner_qs)} tok/q)", file=sys.stderr)
        all_results[model_name] = {
            "accuracy": acc,
            "correct": total_ok,
            "total": len(ner_qs),
            "total_time_s": round(total_time, 1),
            "avg_tok": total_tok // len(ner_qs),
            "results": results,
        }
    
    cache.unload_all()
    
    # Summary table
    print(f"\n\n{'='*60}", file=sys.stderr)
    print("NER — ALL MODELS COMPARISON", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"{'Model':<25} {'Acc':<8} {'Time':<8} {'Tok/q':<8}", file=sys.stderr)
    print(f"{'-'*50}", file=sys.stderr)
    for model_name in [v[0] for v in VARIANTS]:
        r = all_results[model_name]
        print(f"{model_name:<25} {r['accuracy']}%  {r['total_time_s']:.0f}s  {r['avg_tok']:<8}", file=sys.stderr)
    
    # Save
    out_path = "/home/artem/dev/amd-hackathon/data/eval/ner_all_models.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
