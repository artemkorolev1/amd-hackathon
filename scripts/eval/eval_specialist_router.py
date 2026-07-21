#!/usr/bin/env python3
"""
Per-Category Specialist Router v2
 — One model per category with category-specific prompts, temps, and params.
 — Based on empirical eval from training-v3.json (152 Q).
"""
import json
import re
import sys
import time
import gc
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
MODEL_PATHS = {
    "qwen2.5-1.5b": "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "qwen2.5-math-1.5b": "/home/artem/models/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf",
    "gemma-3-1b": "/home/artem/models/gemma-3-1b-it-Q4_K_M.gguf",
}

# ---------------------------------------------------------------------------
# PER-CATEGORY CONFIG — one specialist per category with tailored prompts
# ---------------------------------------------------------------------------
# From eval (152 Q):
#   code_debug  → Math-1.5B   (94.7%)
#   code_gen    → Gemma-1B    (42.1%)
#   factual     → Qwen-1.5B   (21.1%)
#   logic       → Math-1.5B   (15.8%)
#   math        → Math-1.5B   (89.5%)
#   ner         → Gemma-1B    (26.3%)
#   sentiment   → Qwen-1.5B   (94.7%)
#   summarization → Gemma-1B (0.0% — placeholder)

PER_CATEGORY_CONFIG = {
    "code_debug": {
        "model": "qwen2.5-math-1.5b",
        "system_prompt": (
            "Fix the bug. Output ONLY the corrected function inside "
            "```python ... ```. Preserve the original function name and "
            "signature. No explanation."
        ),
        "temperature": 0.0,
        "max_tokens": 512,
    },
    "code_gen": {
        "model": "gemma-3-1b",
        "system_prompt": (
            "Write the requested Python function. Output ONLY the function "
            "inside ```python ... ```. Preserve the exact function name "
            "and signature. Handle edge cases. No explanation."
        ),
        "temperature": 0.0,
        "max_tokens": 512,
    },
    "factual": {
        "model": "qwen2.5-1.5b",
        "system_prompt": (
            "Answer the question directly with the exact fact. "
            "Use exact names, dates, and numbers. Keep under 15 words. "
            "No preamble, no explanation."
        ),
        "temperature": 0.0,
        "max_tokens": 64,
    },
    "logic": {
        "model": "qwen2.5-math-1.5b",
        "system_prompt": (
            "Solve the logic puzzle. Deduce the answer step by step. "
            "End with 'Answer: <conclusion>' on its own line. "
            "Keep the conclusion to one word or short phrase."
        ),
        "temperature": 0.0,
        "max_tokens": 256,
    },
    "math": {
        "model": "qwen2.5-math-1.5b",
        "system_prompt": (
            "Solve the math problem step by step. "
            "End with 'Answer: <value>' on its own line. "
            "Use standard decimal format. Round to nearest tenth if needed."
        ),
        "temperature": 0.0,
        "max_tokens": 512,
    },
    "ner": {
        "model": "gemma-3-1b",
        "system_prompt": (
            "Extract all named entities from the text. "
            "Output as: CATEGORY: value1, value2; CATEGORY: value3. "
            "Use categories like PERSON, ORGANIZATION, LOCATION, DATE, DISEASE, GENE. "
            "Each entity must explicitly appear in the text. "
            "No explanation, no sentences, no preamble."
        ),
        "temperature": 0.0,
        "max_tokens": 256,
    },
    "sentiment": {
        "model": "qwen2.5-1.5b",
        "system_prompt": (
            "Classify the sentiment. "
            "Output EXACTLY one word: positive, negative, neutral, or mixed. "
            "Sarcasm and dismissiveness are NEGATIVE. Default to negative when uncertain. "
            "No explanation. No preamble."
        ),
        "temperature": 0.0,
        "max_tokens": 64,
    },
    "summarization": {
        "model": "gemma-3-1b",
        "system_prompt": (
            "Summarize the text in at most 2 sentences. "
            "Include key names, numbers, and facts. "
            "No preamble, no 'Here is a summary'. "
            "Output ONLY the summary text."
        ),
        "temperature": 0.0,
        "max_tokens": 256,
    },
}

# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------
def fuzzy_match(answer: str, expected: str) -> bool:
    a, e = answer.strip().lower(), expected.strip().lower()
    if not a or not e:
        return False
    if a == e:
        return True
    if len(e) <= 20 and e in a:
        return True
    if len(a) <= 20 and a in e:
        return True
    nums_a = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", a)]
    nums_e = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", e)]
    if nums_a and nums_e:
        a_num, e_num = nums_a[-1], nums_e[-1]
        if e_num != 0 and abs((a_num - e_num) / e_num) <= 0.01:
            return True
        if a_num == e_num:
            return True
    ta = set(tok for tok in re.split(r"[^a-zA-Z0-9.]+", a) if tok)
    te = set(tok for tok in re.split(r"[^a-zA-Z0-9.]+", e) if tok)
    if len(te) > 0 and len(ta & te) / len(te) >= 0.8:
        return True
    return False

# ---------------------------------------------------------------------------
# Task classifier (Qwen-1.5B chat)
# ---------------------------------------------------------------------------
class TaskClassifier:
    def __init__(self):
        self.llm = None

    def _load(self):
        if self.llm is None:
            from llama_cpp import Llama
            print("  [Router] Loading classifier...", file=sys.stderr)
            self.llm = Llama(
                model_path=MODEL_PATHS["qwen2.5-1.5b"],
                n_gpu_layers=-1, n_ctx=1024, verbose=False,
            )

    def classify(self, prompt: str) -> str:
        self._load()
        messages = [
            {"role": "system", "content": (
                "You classify questions into one of: code_debug, code_gen, factual, "
                "logic, math, ner, sentiment, summarization. Return ONLY the single word."
            )},
            {"role": "user", "content": f"Classify: {prompt[:800]}"},
        ]
        r = self.llm.create_chat_completion(
            messages=messages, max_tokens=15, temperature=0.0
        )
        label = r["choices"][0]["message"]["content"].strip().lower()
        valid = {"code_debug", "code_gen", "factual", "logic", "math", "ner", "sentiment", "summarization"}
        if label in valid:
            return label
        # Keyword fallback
        pl = prompt.lower()
        if any(w in pl for w in ["fix","debug","bug","error","broken","incorrect","wrong"]): return "code_debug"
        if any(w in pl for w in ["def ","function","return","write a","implement","code"]): return "code_gen"
        if any(w in pl for w in ["entity","entities","extract","named entity","person","location","organization","disease","gene"]): return "ner"
        if any(w in pl for w in ["sum","number","average","total","calculate","how many","percent","ratio","compute","what is"]): return "math"
        if any(w in pl for w in ["if","then","therefore","conclusion","argument","must be","syllogism","deduce","infer"]): return "logic"
        if any(w in pl for w in ["feeling","opinion","sentiment","positive","negative","neutral","tone"]): return "sentiment"
        if any(w in pl for w in ["summarize","summary","brief","overview","concise"]): return "summarization"
        return "factual"

    def unload(self):
        if self.llm:
            del self.llm; self.llm = None
            gc.collect()
            import torch; torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------
class ModelCache:
    def __init__(self):
        self._models = {}

    def get(self, name: str):
        if name not in self._models:
            from llama_cpp import Llama
            print(f"  [Load] {name}...", file=sys.stderr)
            t0 = time.time()
            self._models[name] = Llama(
                model_path=MODEL_PATHS[name],
                n_gpu_layers=-1, n_ctx=2048, verbose=False,
            )
            print(f"  [Load] {name} ready in {time.time()-t0:.1f}s", file=sys.stderr)
        return self._models[name]

    def run(self, name: str, system_prompt: str, user_prompt: str,
            temperature: float = 0.0, max_tokens: int = 512) -> tuple:
        llm = self.get(name)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        t0 = time.time()
        r = llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens,
            temperature=temperature,
        )
        elapsed = time.time() - t0
        answer = r["choices"][0]["message"]["content"].strip()
        tok_in = r.get("usage", {}).get("prompt_tokens", 0)
        tok_out = r.get("usage", {}).get("completion_tokens", 0)
        return answer, elapsed, tok_in, tok_out

    def unload_all(self):
        for name in list(self._models.keys()):
            del self._models[name]
        self._models = {}
        gc.collect()
        import torch; torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Main eval
# ---------------------------------------------------------------------------
def evaluate(questions: List[dict], label: str = ""):
    classifier = TaskClassifier()
    cache = ModelCache()

    results = []
    by_cat = {}
    overall_ok = 0
    total_time = 0
    total_tok_out = 0
    n = len(questions)

    for i, q in enumerate(questions):
        prompt = q["prompt"]
        expected = q["expected_answer"]
        true_cat = q["category"]

        # 1. Classify
        pred_cat = classifier.classify(prompt)

        # 2. Get config for predicted category
        cfg = PER_CATEGORY_CONFIG.get(pred_cat, PER_CATEGORY_CONFIG["factual"])
        model_name = cfg["model"]

        # 3. Run the specialist
        answer, elapsed, tok_in, tok_out = cache.run(
            model_name, cfg["system_prompt"], prompt,
            cfg["temperature"], cfg["max_tokens"],
        )

        # 4. Judge
        ok = fuzzy_match(answer, expected)
        overall_ok += 1 if ok else 0
        total_time += elapsed
        total_tok_out += tok_out

        by_cat.setdefault(true_cat, {"correct": 0, "total": 0, "cls_ok": 0})
        by_cat[true_cat]["total"] += 1
        by_cat[true_cat]["correct"] += 1 if ok else 0
        if pred_cat == true_cat:
            by_cat[true_cat]["cls_ok"] += 1

        marker = "✓" if ok else "✗"
        cls_mark = "✓" if pred_cat == true_cat else "✗"
        print(
            f"[{i+1:3d}/{n}] {true_cat:15s}→{model_name:20s} {cls_mark}{marker} "
            f"({elapsed:.2f}s) {answer[:50]}",
            file=sys.stderr,
        )

        results.append({
            "task_id": q.get("task_id", f"q_{i}"),
            "true_category": true_cat,
            "predicted_category": pred_cat,
            "model": model_name,
            "temperature": cfg["temperature"],
            "time_s": round(elapsed, 2),
            "tokens_out": tok_out,
            "answer": answer,
            "expected": expected,
            "correct": ok,
        })

    classifier.unload()
    cache.unload_all()

    # Summary
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"SPECIALIST ROUTER — {label}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Overall: {overall_ok}/{n} = {overall_ok/n*100:.1f}%", file=sys.stderr)
    print(f"Total time: {total_time:.0f}s ({total_time/n:.2f}s/q)", file=sys.stderr)
    print(f"Total tokens out: {total_tok_out} ({total_tok_out//n}/q)", file=sys.stderr)

    print(f"\n{'Category':<20} {'Accuracy':<10} {'Model':<20} {'Cls%':<8}", file=sys.stderr)
    print(f"{'-'*60}", file=sys.stderr)
    for cat in sorted(by_cat.keys()):
        v = by_cat[cat]
        acc = v["correct"]/v["total"]*100
        cls_acc = v["cls_ok"]/v["total"]*100
        mdl = PER_CATEGORY_CONFIG.get(cat, {}).get("model", "?")
        print(f"{cat:<20} {v['correct']:2d}/{v['total']:2d}={acc:5.1f}% {mdl:<20} {cls_acc:5.0f}%", file=sys.stderr)

    print(f"\n{'='*60}", file=sys.stderr)
    print("FAILURES:", file=sys.stderr)
    for r in results:
        if not r["correct"]:
            print(f"  [{r['true_category']:15s}]→{r['model']:20s} exp:{r['expected'][:50]}", file=sys.stderr)
            print(f"  {'':19s}  got:{r['answer'][:60]}", file=sys.stderr)
            print(file=sys.stderr)

    return {"total": n, "correct": overall_ok, "accuracy": round(overall_ok/n*100, 1), "results": results}


def main():
    dataset_path = "/home/artem/dev/amd-hackathon/data/eval/training-v3.json"
    with open(dataset_path) as f:
        all_questions = json.load(f)

    # Sample 5 per category = 40
    import random
    random.seed(42)
    by_cat = {}
    for q in all_questions:
        by_cat.setdefault(q["category"], []).append(q)
    subset = []
    for cat in sorted(by_cat.keys()):
        subset.extend(random.sample(by_cat[cat], min(5, len(by_cat[cat]))))
    random.shuffle(subset)

    print(f"Running {len(subset)} questions (5 per category)", file=sys.stderr)
    result = evaluate(subset, "40Q — per-category specialist")

    # Save
    out_path = "/home/artem/dev/amd-hackathon/data/eval/specialist_router_40q.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
