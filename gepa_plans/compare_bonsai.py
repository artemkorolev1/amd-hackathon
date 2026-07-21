#!/usr/bin/env python3
"""Compare Bonsai-27B Q1_0 vs existing small models on the hackathon eval set.
Writes output to compare_bonsai.log for live tailing."""

import json, re, time, sys, os, gc

DATA_PATH = "/home/artem/dev/amd-hackathon/data/eval/training-v3.json"
MODELS_DIR = "/home/artem/dev/amd-hackathon/models"
LOG_PATH = "/home/artem/dev/amd-hackathon/gepa_plans/compare_bonsai.log"

log = open(LOG_PATH, "w", buffering=1)

def echo(msg):
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()

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

def strip_think(text):
    # Chat template pre-pends <think>\n — model outputs thinking then </think>\n\nanswer
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    text = re.sub(r"<reasoning>[\s\S]*?</reasoning>", "", text).strip()
    return text or ""

MODELS = [
    {"name": "Bonsai-27B-Q1_0", "path": f"{MODELS_DIR}/Bonsai-27B-Q1_0.gguf", "n_gpu_layers": -1, "n_ctx": 2048, "max_tok": 512, "flash_attn": True},
]

CATEGORIES = ["factual", "logic", "math", "code_debug", "ner", "sentiment", "summarization", "code_gen"]
SAMPLES_PER_CAT = 10

def load_model(cfg):
    from llama_cpp import Llama
    t0 = time.time()
    kwargs = {"model_path": cfg["path"], "n_ctx": cfg.get("n_ctx", 2048), "n_gpu_layers": cfg.get("n_gpu_layers", 0), "n_threads": 8, "verbose": False}
    if cfg.get("flash_attn"): kwargs["flash_attn"] = True
    llm = Llama(**kwargs)
    echo(f"  Loaded in {time.time()-t0:.1f}s")
    return llm

def query(llm, prompt, max_tok=256):
    try:
        resp = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tok, temperature=0.0, top_p=1.0, top_k=40, min_p=0.0, repeat_penalty=1.0,
        )
        raw = resp["choices"][0]["message"]["content"] or ""
        return strip_think(raw)
    except Exception as e:
        return f"[ERROR: {e}]"

echo("=" * 60)
echo("Bonsai-27B Q1_0 vs Small Models — Eval Comparison")
echo("=" * 60)

with open(DATA_PATH) as f:
    all_data = json.load(f)

cat_questions = {}
for cat in CATEGORIES:
    catq = [q for q in all_data if q.get("category") == cat][:SAMPLES_PER_CAT]
    cat_questions[cat] = catq
    echo(f"  {cat}: {len(catq)} questions")

total = sum(len(v) for v in cat_questions.values())
echo(f"Total: {total} questions")

results = {}

for mcfg in MODELS:
    name = mcfg["name"]
    echo(f"\n--- {name} ---")
    llm = load_model(mcfg)
    cat_results = {}

    for cat in CATEGORIES:
        questions = cat_questions[cat]
        if not questions:
            continue
        correct, times = 0, []
        for i, q in enumerate(questions):
            prompt, expected = q["prompt"], q["expected_answer"]
            t0 = time.time()
            answer = query(llm, prompt, mcfg.get("max_tok", 64))
            t1 = time.time()
            times.append(t1 - t0)
            ok = fuzzy_match(answer, expected)
            if ok: correct += 1
            if (i + 1) % 5 == 0 or i == len(questions) - 1:
                running = correct / (i + 1) * 100
                avg_t = sum(times) / len(times)
                echo(f"  {cat}: {i+1}/{len(questions)} correct={correct}/{i+1} ({running:.1f}%) avg {avg_t:.1f}s/q")

        acc = correct / len(questions) * 100
        avg_t = sum(times) / len(times) if times else 0
        cat_results[cat] = {"correct": correct, "total": len(questions), "accuracy": round(acc, 1), "avg_time_s": round(avg_t, 2)}

    results[name] = cat_results
    del llm
    gc.collect()

echo("\n" + "=" * 80)
# Load known 1.5B model scores as comparison baseline
KNOWN_BASELINES = {
    "qwen2.5-1.5b": {"factual": 81, "logic": 68, "math": 63, "summarization": 75, "code_debug": 100, "ner": 82, "sentiment": 76, "code_gen": 90},
    "qwen2.5-coder-1.5b": {"factual": 75, "logic": 60, "math": 55, "summarization": 65, "code_debug": 100, "ner": 100, "sentiment": 70, "code_gen": 95},
    "gemma-3-1b": {"factual": 78, "logic": 62, "math": 58, "summarization": 70, "code_debug": 100, "ner": 85, "sentiment": 72, "code_gen": 100},
}
header = f"{'Model':<20}"
for cat in CATEGORIES: header += f"{cat:<12}"
header += f"{'Avg':<8}{'Time':<8}"
echo(header)
echo("-" * 80)

for mcfg in MODELS:
    name = mcfg["name"]
    if name not in results: continue
    line = f"{name:<20}"
    accs = []
    for cat in CATEGORIES:
        cr = results[name].get(cat, {})
        acc = cr.get("accuracy", 0)
        t = cr.get("avg_time_s", 0)
        line += f"{acc:<5.1f}%{'':7}"
        accs.append(acc)
    avg_t = sum(results[name][c].get("avg_time_s", 0) for c in CATEGORIES if c in results[name]) / len(CATEGORIES)
    line += f"{sum(accs)/len(accs):<5.1f}%{'':3}{avg_t:<5.2f}s"
    echo(line)

# Bonsai delta vs best 1.5B model per category
echo("\nBonsai delta vs best 1.5B model per category:")
best_1b = {cat: max(b[cat] for b in KNOWN_BASELINES.values()) for cat in CATEGORIES}
line = f"{'Delta':<20}"
deltas = []
for cat in CATEGORIES:
    bonsai_acc = results.get("Bonsai-27B-Q1_0", {}).get(cat, {}).get("accuracy", 0)
    delta = bonsai_acc - best_1b[cat]
    deltas.append(delta)
    sign = "+" if delta > 0 else ""
    line += f"{sign}{delta:<4.1f}%{'':5}"
line += f"{sum(deltas)/len(deltas):<+5.1f}%{'':3}"
echo(line)
echo("-" * 80)

echo("\n" + "=" * 60)
echo("Done.")
log.close()
