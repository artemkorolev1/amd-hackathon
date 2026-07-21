#!/usr/bin/env python3
"""
Fast NER Eval — focused: 6 models × 2 prompts on simple NER, then best 2 models × 4 prompts on hard.
"""
import json, logging, os, re, sys, time
from pathlib import Path
logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))
# Add user site-packages for llama_cpp
sys.path.insert(0, os.path.expanduser("~/.local/lib/python3.12/site-packages"))
from agent.solvers.deterministic import solve_ner as old_solve_ner
from agent.solvers.prototype_ner_v3 import solve_ner as proto_solve_ner

MODELS = {
    "Qwen2.5-1.5B": os.path.expanduser("~/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    "Qwen2.5-Coder-1.5B": os.path.expanduser("~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
    "Qwen2.5-Math-1.5B": os.path.expanduser("~/models/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf"),
    "Gemma-3-1B": os.path.expanduser("~/models/gemma-3-1b-it-Q4_K_M.gguf"),
    "SmolLM2-1.7B": os.path.expanduser("~/models/smollm2-1.7b-instruct-q4_k_m.gguf"),
    "Llama-3.2-1B": os.path.expanduser("~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
}

# 2 primary prompts for round 1
PRIMARY_PROMPTS = {
    "v2_structured": "Extract all named entities from the following text. Use these entity types: person, group, corporation, location, event, product, creative_work. Output each entity on its own line exactly as: type: value. If the entity has {@...@} markers in the text, preserve them. Cover ALL named entities. No preamble, no commentary. Output ONLY the entity lines.",
    "v3_fewshot": "Extract all named entities from the text below. Output each on its own line as: type: value\n\nExample:\nText: Sitting out here watching {@Philadelphia Police@} shake hands with Trump supporters at 12th and Arch.\nOutput:\ncorporation: {@Philadelphia Police@}\nperson: Trump\nlocation: 12th and Arch\n\nNow extract from this text:",
}

# 4 prompts for top models
ALL_PROMPTS = {
    "v1_terse": "Extract all named entities from the text. Output each on its own line as: type: value. No extra text.",
    "v2_structured": "Extract all named entities from the following text. Use these entity types: person, group, corporation, location, event, product, creative_work. Output each entity on its own line exactly as: type: value. If the entity has {@...@} markers in the text, preserve them. Cover ALL named entities. No preamble, no commentary. Output ONLY the entity lines.",
    "v3_fewshot": "Extract all named entities from the text below. Output each on its own line as: type: value\n\nExample:\nText: Sitting out here watching {@Philadelphia Police@} shake hands with Trump supporters at 12th and Arch.\nOutput:\ncorporation: {@Philadelphia Police@}\nperson: Trump\nlocation: 12th and Arch\n\nNow extract from this text:",
    "v4_cot": "Extract named entities from the text step by step:\n1. Identify entities in {@...@} markers.\n2. Find hashtags (#word) and @mentions.\n3. Find capitalized proper names.\n4. Find locations and organizations.\n5. Classify each as: person, group, corporation, location, event, product, or creative_work.\n6. Output each on its own line: type: value\n\nText:",
}

def load_llama(path, ngl=-1):
    from llama_cpp import Llama
    return Llama(model_path=path, n_gpu_layers=ngl, n_ctx=4096, verbose=False)

def infer(llm, sys_p, user_t, max_tokens=512, temp=0.1):
    try:
        r = llm.create_chat_completion(
            [{"role":"system","content":sys_p},{"role":"user","content":user_t}],
            max_tokens=max_tokens, temperature=temp,
            stop=["</s>","<|end|>","<|im_end|>","<end_of_turn>"])
        return r["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"ERR:{e}"

def norm_lines(t):
    out = []
    for line in t.split("\n"):
        l = line.strip()
        if not l or l.startswith(("Here","The","I ","Note","Output","Sure","Step")):
            continue
        l = re.sub(r"^\d+[\.\)]\s*","",l)
        if ":" in l:
            out.append(l)
    return out

def score(pred_lines, exp_text):
    ex = set()
    for line in exp_text.strip().split("\n"):
        l = line.strip()
        if ":" in l: ex.add(l)
    pr = set(pred_lines)
    inter = ex & pr
    p = len(inter)/len(pr) if pr else 0
    r = len(inter)/len(ex) if ex else 0
    f1 = 2*p*r/(p+r) if p+r>0 else 0
    exact = 1.0 if pr==ex and len(ex)>0 else 0
    return f1, exact, p, r

def get_text(prompt):
    t = prompt.strip()
    t = re.sub(r"^(?:Extract\s+entities:?\s*)","",t)
    m = re.search(r"(?:from:)\s*['\"`]?(.+?)['\"`]?\s*$",t,re.DOTALL)
    if m: return m.group(1).strip()
    return t

def main():
    qs = []
    for fname, src in [("input/dev_40.json","dev_40"), ("input/complexity_40.json","complexity_40")]:
        with open(str(_HERE/fname)) as f:
            data = json.load(f)
        for q in data:
            if q.get("category") == "ner":
                qs.append({"source":src, "prompt":q["prompt"],
                    "expected":q.get("expected_answer",q.get("answer","")),
                    "id":q.get("task_id",f"{src}-{len(qs)}")})

    # Load training-v3
    with open(str(_HERE/"data"/"eval"/"training-v3.json")) as f:
        data = json.load(f)
    for q in data:
        if q.get("category") == "ner":
            qs.append({"source":"training-v3","prompt":q["prompt"],
                "expected":q["expected_answer"],"id":q.get("task_id",f"train-{len(qs)}")})

    print(f"Total NER questions: {len(qs)}")
    results = []

    # Deterministic baselines (fast)
    for q in qs:
        old = old_solve_ner(q["prompt"],"ner")
        fl = norm_lines(old) if old else []
        f1, ex, _, _ = score(fl, q["expected"])
        results.append({"id":q["id"],"src":q["source"],"solver":"old_deterministic","prompt":"none","f1":f1,"exact":ex})
        proto = proto_solve_ner(q["prompt"],"ner")
        pl = norm_lines(proto) if proto else []
        f1, ex, _, _ = score(pl, q["expected"])
        results.append({"id":q["id"],"src":q["source"],"solver":"prototype_v3","prompt":"none","f1":f1,"exact":ex})

    # Round 1: all models on simple questions with 2 prompts
    simple_qs = [q for q in qs if q["source"] != "training-v3"]
    hard_qs = [q for q in qs if q["source"] == "training-v3"]
    
    for mname, mpath in MODELS.items():
        if not os.path.exists(mpath):
            print(f"SKIP {mname}: not found", file=sys.stderr)
            continue
        print(f"\n--- {mname} (round 1: {len(simple_qs)} simple × 2 prompts) ---", file=sys.stderr)
        llm = load_llama(mpath)
        
        for pname, sp in PRIMARY_PROMPTS.items():
            for q in simple_qs:
                st = time.time()
                txt = get_text(q["prompt"])
                ans = infer(llm, sp, txt)
                el = time.time()-st
                lines = norm_lines(ans)
                f1, ex, _, _ = score(lines, q["expected"])
                results.append({"id":q["id"],"src":q["source"],"solver":mname,"prompt":pname,"f1":f1,"exact":ex,"latency":round(el,1)})
                print(f"  {q['id'][:25]:25s} {pname:15s} F1={f1:.2f} ({el:.1f}s)", file=sys.stderr)
        del llm

    # Round 2: top 2 models on ALL questions with 4 prompts
    # Determine top 2 from round 1
    from collections import defaultdict
    perf = defaultdict(list)
    for r in results:
        if r["solver"] in MODELS and r["src"] != "training-v3":
            perf[r["solver"]].append(r["f1"])
    top2 = sorted(perf.items(), key=lambda x: sum(x[1])/len(x[1]) if x[1] else 0, reverse=True)[:2]
    top2_names = [t[0] for t in top2]
    print(f"\n--- Top 2 models from round 1: {top2_names} ---", file=sys.stderr)

    for mname in top2_names:
        mpath = MODELS.get(mname)
        if not mpath or not os.path.exists(mpath):
            continue
        print(f"\n--- {mname} (round 2: ALL {len(qs)} questions × 4 prompts) ---", file=sys.stderr)
        llm = load_llama(mpath)
        for pname, sp in ALL_PROMPTS.items():
            # Skip prompts already done for simple_qs
            todo_qs = simple_qs + hard_qs  # do all questions fresh for all prompts
            for q in todo_qs:
                st = time.time()
                txt = get_text(q["prompt"])
                ans = infer(llm, sp, txt)
                el = time.time()-st
                lines = norm_lines(ans)
                f1, ex, _, _ = score(lines, q["expected"])
                results.append({"id":q["id"],"src":q["source"],"solver":mname,"prompt":pname,"f1":f1,"exact":ex,"latency":round(el,1)})
                print(f"  {q['id'][:25]:25s} {pname:15s} F1={f1:.2f} ({el:.1f}s)", file=sys.stderr)
        del llm

    out = _HERE / "eval_results" / "ner_comparison.json"
    os.makedirs(out.parent, exist_ok=True)
    with open(out,"w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}", file=sys.stderr)

    # Summary
    print("\n"+"="*80)
    print("NER AGENT COMPARISON")
    print("="*80)
    agg = defaultdict(list)
    for r in results:
        key = (r.get("solver"), r.get("prompt","none"))
        agg[key].append(r["f1"])
    
    for (solver, prompt), f1s in sorted(agg.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True):
        avg = sum(f1s)/len(f1s)
        exacts = sum(1 for r in results if r.get("solver")==solver and r.get("prompt","none")==prompt and r["exact"]==1)
        total = sum(1 for r in results if r.get("solver")==solver and r.get("prompt","none")==prompt)
        print(f"  {solver:25s} / {prompt:20s} F1={avg:.3f} exact={exacts}/{total}")
        if solver not in MODELS and solver not in ("old_deterministic","prototype_v3"):
            break  # Only print top LLM combos
        break  # Only print top 2
    
    print(f"\nTotal results: {len(results)}")

if __name__ == "__main__":
    main()
