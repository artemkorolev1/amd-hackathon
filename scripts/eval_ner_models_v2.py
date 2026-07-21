#!/usr/bin/env python3
"""NER Model Eval v2 — runs all 6 GGUF models on 19 training-v3 NER questions with 4 prompt variants."""
import json, logging, os, re, sys, time
from pathlib import Path
logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, os.path.expanduser("~/.local/lib/python3.12/site-packages"))

from agent.solvers.deterministic import solve_ner as old_solve
from agent.solvers.prototype_ner_v3 import solve_ner as proto_solve

MODELS = {
    "Qwen2.5-1.5B": os.path.expanduser("~/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    "Qwen2.5-Coder-1.5B": os.path.expanduser("~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
    "Qwen2.5-Math-1.5B": os.path.expanduser("~/models/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf"),
    "Gemma-3-1B": os.path.expanduser("~/models/gemma-3-1b-it-Q4_K_M.gguf"),
    "SmolLM2-1.7B": os.path.expanduser("~/models/smollm2-1.7b-instruct-q4_k_m.gguf"),
    "Llama-3.2-1B": os.path.expanduser("~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
}

PROMPTS = {
    "v1_terse": "Extract all named entities from the text. Output each on its own line as: type: value. No extra text.",
    "v2_structured": "Extract all named entities from the following text. Use these entity types: person, group, corporation, location, event, product, creative_work. Output each entity on its own line exactly as: type: value. Cover ALL named entities.",
    "v3_fewshot": "Extract all named entities from the text. Output each on its own line as: type: value.\n\nExample:\nText: Watching {@Philadelphia Police@} shake hands with Trump supporters at 12th and Arch.\nOutput:\ncorporation: {@Philadelphia Police@}\nperson: Trump\nlocation: 12th and Arch\n\nNow extract from this text:",
    "v4_cot": "Extract named entities step by step: 1. Find {@...@} markers. 2. Find hashtags (#word) and @mentions. 3. Find capitalized proper names. 4. Classify each as: person, group, corp, location, event, product, creative_work. 5. Output: type: value\n\nText:",
}

def load_llm(path):
    from llama_cpp import Llama
    return Llama(model_path=path, n_gpu_layers=-1, n_ctx=4096, verbose=False)

def infer(llm, sys_p, user_t, max_tok=512, temp=0.1):
    try:
        r = llm.create_chat_completion(
            [{"role":"system","content":sys_p},{"role":"user","content":user_t}],
            max_tokens=max_tok, temperature=temp,
            stop=["</s>","<|end|>","<|im_end|>","<end_of_turn>"])
        return r["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"ERROR: {e}"

def norm_lines(t):
    out = []
    for line in t.split("\n"):
        l = line.strip()
        if not l or l.lower().startswith(("here","the entit","output","sure","i ","note:","step","based on")):
            continue
        if ":" in l:
            l = re.sub(r"^\d+[\.\)]\s*","",l)
            out.append(l)
    return out

def score(pl, et):
    ex = set(l.strip() for l in et.strip().split("\n") if ":" in l.strip())
    pr = set(pl)
    inter = ex & pr
    p = len(inter)/len(pr) if pr else 0
    r = len(inter)/len(ex) if ex else 0
    f1 = 2*p*r/(p+r) if p+r>0 else 0
    return f1, int(pr==ex and len(ex)>0)

def main():
    with open(str(_HERE/"data"/"eval"/"training-v3.json")) as f:
        data = json.load(f)
    ner_qs = [q for q in data if q.get("category")=="ner"]
    print(f"NER questions: {len(ner_qs)}")

    all_results = []

    # Deterministic baselines
    for q in ner_qs:
        for sn, fn in [("old_det",old_solve),("proto_v3",proto_solve)]:
            ans = fn(q["prompt"],"ner")
            lines = norm_lines(ans) if ans else []
            f1, ex = score(lines, q["expected_answer"])
            all_results.append({"id":q.get("task_id","?"),"solver":sn,"prompt":"none","f1":f1,"exact":ex,"answer":ans or "","latency_s":0})

    det = [r for r in all_results if r["solver"]=="proto_v3"]
    print(f"proto_v3: avg F1={sum(r['f1'] for r in det)/len(det):.3f} exact={sum(r['exact'] for r in det)}/{len(det)}")

    # Models
    for mname, mpath in MODELS.items():
        if not os.path.exists(mpath):
            print(f"SKIP {mname}")
            continue
        print(f"\n{mname}: loading...")
        llm = load_llm(mpath)
        for pname, sp in PROMPTS.items():
            for q in ner_qs:
                ut = re.sub(r"^Extract\s+entities:\s*","",q["prompt"]).strip()
                st = time.time()
                ans = infer(llm, sp, ut)
                el = time.time()-st
                lines = norm_lines(ans)
                f1, ex = score(lines, q["expected_answer"])
                all_results.append({"id":q.get("task_id","?"),"solver":mname,"prompt":pname,"f1":f1,"exact":ex,"answer":ans,"latency_s":round(el,1)})
                sys.stderr.write(f"  {mname:20s} {pname:15s} {q.get('task_id','?'):30s} F1={f1:.2f} ({el:.1f}s)\n")
        del llm

    # Save
    out = _HERE/"eval_results"/"ner_comparison_v2.json"
    out.parent.mkdir(exist_ok=True)
    with open(out,"w") as f:
        json.dump(all_results,f,indent=2)
    print(f"\nSaved to {out}")

    # Summary
    print("\n"+"="*80)
    print("NER COMPARISON (training-v3)")
    print("="*80)
    from collections import defaultdict
    agg = defaultdict(list)
    for r in all_results:
        agg[(r["solver"],r["prompt"])].append(r["f1"])
    for (s,p), f1s in sorted(agg.items(),key=lambda x:sum(x[1])/len(x[1]),reverse=True):
        ex = sum(1 for r in all_results if r["solver"]==s and r["prompt"]==p and r["exact"]==1)
        print(f"  {s:25s} / {p:20s} F1={sum(f1s)/len(f1s):.3f} exact={ex}/{len(f1s)}")

if __name__=="__main__":
    main()
