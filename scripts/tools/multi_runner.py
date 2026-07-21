#!/usr/bin/env python3
"""Multi-model runner — runs one eval set through multiple GGUF models sequentially."""
import argparse, gc, json, os, re, sys, time, logging, concurrent.futures, subprocess
logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger("multi-runner")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from agent.pre_filter import stage0
from agent.category_filter import classify_with_detail as _stage2_detail
from agent.complexity import score as mlm_complexity
from agent.solvers.deterministic import solve_arithmetic, solve_logic, solve_sentiment, solve_ner, solve_factual_qa, solve_code_debugging
from agent.dynamic_prompts import build_system_prompt, get_max_tokens as dp_max_tokens, get_stop_sequences as dp_stop_sequences
from agent.run_logger import RunLogger

_DOC_HEADER_RE = re.compile(r'^(HEADLINE:|LEGAL BRIEF|STATEMENT BY|ARTICLE:|TRANSCRIPT:|MEMO:|PRESS RELEASE:)', re.I|re.M)
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.I)
def strip_think(t): s=_THINK_RE.sub("",t).strip(); return s if s!=t.strip() else t.split("</think>",1)[1].strip() if "</think>" in t else t.strip()
def secondary_category(rs,p):
    for c,_ in sorted(rs.items(),key=lambda x:-x[1]): 
        if c!=p: return c
    return ""

def find_gguf(d):
    from pathlib import Path
    r=[]; p=Path(d).expanduser().resolve()
    if p.is_dir():
        for f in sorted(p.glob("*.gguf")): r.append({"path":str(f),"name":f.name,"size_gb":round(f.stat().st_size/1e9,1)})
    return r

def run_model(cfg, questions, ngl, nctx, nth, rl):
    from llama_cpp import Llama
    mp, mn = cfg["path"], cfg["name"]
    logger.warning("Loading %s (%.1fGB, ngl=%d)", mn, cfg["size_gb"], ngl)
    llm = Llama(model_path=mp, n_ctx=nctx, n_gpu_layers=ngl, n_threads=nth, flash_attn=True, verbose=False)
    logger.warning("Model %s ready", mn)
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    def infer(msgs, mt, ss, to=60):
        def _c(): return llm.create_chat_completion(messages=msgs, max_tokens=mt, temperature=0.0, stop=ss)
        try: r=ex.submit(_c).result(timeout=to); return strip_think(r["choices"][0]["message"]["content"] or ""), r.get("usage",{})
        except: return "", {}
    for i,q in enumerate(questions):
        tid = q.get("task_id",f"idx_{i}")
        prompt = q.get("prompt",q.get("question",""))
        rl.start_question(tid, prompt, model_name=mn,
                          difficulty=q.get("difficulty", ""))
        t0=time.monotonic(); s0=stage0(prompt); s0ms=(time.monotonic()-t0)*1000
        rl.log_pre_filter(action=s0.action, answer=s0.direct_answer or "", category_hint=s0.category or "", flags=str(s0.flags), elapsed_ms=s0ms)
        if s0.action=="bypass" and s0.direct_answer: rl.finish_question(s0.direct_answer,0); print(s0.direct_answer.replace("\n","\\n")); continue
        t0=time.monotonic(); detail=_stage2_detail(prompt); s2ms=(time.monotonic()-t0)*1000
        cat, sd, rs = detail["category"], detail["score_delta"], detail["raw_scores"]
        lower, ovr = prompt.lower(), []
        if cat not in ("summarization",) and _DOC_HEADER_RE.search(prompt): cat="summarization"; sd=1.0; ovr.append("doc_header→summarization")
        if cat!="summarization" and re.search(r"\bsummariz[ei]",lower): cat="summarization"; sd=1.0; ovr.append("summarize_keyword→summarization")
        if cat=="math" and len(prompt)>600 and not re.search(r'[=×÷]|\\frac|\\int|\\sum|\bsolve\b|\bcalculate\b|\bcompute\b|\bfind\b.*\b(?:value|sum|product|ratio)\b',prompt,re.I): cat="summarization"; sd=1.0; ovr.append("long_math→summarization")
        if cat not in ("ner",) and re.search(r"\b(extract|identify|list)\b.{0,40}\b(named entity|entities|people mentioned|organizations mentioned)\b",lower): cat="ner"; sd=1.0; ovr.append("extract_keyword→ner")
        if cat=="sentiment" and re.search(r"\bfor (someone|a person)\b.{0,50}\b(with your|of your)\b",prompt,re.I): rl.log_post_processing("backhanded_compliment→negative"); rl.finish_question("negative",0); print("negative"); continue
        rl.log_category_filter(category=cat, category_4way=detail.get("category_4way",""), confidence=detail.get("confidence",0.0), score_delta=sd, raw_scores=rs, overrides="; ".join(ovr), elapsed_ms=s2ms)
        if ovr and rl._current: rl._current.keyword_overrides_applied="; ".join(ovr)
        t0=time.monotonic(); cx=mlm_complexity(prompt); cxms=(time.monotonic()-t0)*1000
        rl.log_complexity(cx,"MiniLM-L6-v2+LogReg",cxms)
        # Decision timing
        t_dec = time.monotonic()
        mc=len(set(o.strip().lower() for o in re.findall(r"(?<!\w)[a-dA-D]\)\s",prompt)))>=3
        mt=int(dp_max_tokens(cat,cx)); sp=dp_stop_sequences(cat)
        sysp=build_system_prompt(cat,cx)
        rl.log_decision(solver_name="local_llm", model=mn, max_tokens=mt, temperature=0.0, system_prompt=sysp[:300], prompt_version=f"standard/{cat}/cx={cx:.2f}", elapsed_ms=(time.monotonic()-t_dec)*1000)
        dcm={"math":"math_arithmetic","sentiment":"sentiment","factual":"other_complex","code_debug":"code_debugging"}
        da=None
        if cat in dcm:
            for sf in [solve_arithmetic,solve_logic,solve_sentiment,solve_ner,solve_factual_qa,solve_code_debugging]:
                try: a=sf(prompt,dcm[cat]); da=a; break
                except: pass
        if da: rl.log_deterministic([("det",da)],da,0,0); rl.finish_question(da,0); print(da.replace("\n","\\n")); continue
        msgs=[{"role":"system","content":sysp},{"role":"user","content":prompt}]
        t0=time.monotonic(); ans,usage=infer(msgs,mt,sp); llmms=(time.monotonic()-t0)*1000
        pt=usage.get("prompt_tokens",0); ct=usage.get("completion_tokens",0); tt=usage.get("total_tokens",0)
        retry=False
        if not ans:
            t0=time.monotonic(); a2,u2=infer([{"role":"user","content":prompt}],mt,sp); llmms+=(time.monotonic()-t0)*1000
            if a2: ans=a2; pt+=u2.get("prompt_tokens",0); ct+=u2.get("completion_tokens",0); tt+=u2.get("total_tokens",0)
            retry=True
        rl.log_local_llm(elapsed_ms=llmms, retry=retry, prompt_tokens=pt, completion_tokens=ct, total_tokens=tt)
        pp=""
        if cat=="math" and ans:
            m=re.search(r"\bAnswer:\s*(.+)",ans,re.I|re.DOTALL)
            if m: ans=m.group(1).strip().split("\n")[0].strip(); pp="math_answer_extract"
        rl.log_post_processing(pp)
        rl.finish_question(ans or "", llmms)
        print(ans.replace("\n","\\n"))
    del llm; gc.collect()
    logger.warning("Unloaded %s", mn)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", required=True)
    parser.add_argument("--model-dir", default="~/models/")
    parser.add_argument("--models")
    parser.add_argument("--gpu", action="store_true", default=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--n-threads", type=int, default=2)
    args = parser.parse_args()
    ngl=0 if args.cpu else -1
    if args.models:
        models=[]
        for p in args.models.split(","):
            p=p.strip()
            if os.path.exists(p): models.append({"path":os.path.abspath(p),"name":os.path.basename(p),"size_gb":round(os.path.getsize(p)/1e9,1)})
    else: models=find_gguf(args.model_dir)
    if not models: logger.error("No models"); sys.exit(1)
    logger.warning("Found %d models:", len(models))
    for m in models: logger.warning("  [%.1fGB] %s", m["size_gb"], m["name"])
    with open(args.eval) as f: d=json.load(f)
    questions=d.get("questions",d) if isinstance(d,dict) else d
    logger.warning("Loaded %d questions", len(questions))
    try: pv=subprocess.run(["git","describe","--tags","--always"],capture_output=True,text=True,cwd=_HERE,timeout=5).stdout.strip() or "unknown"
    except: pv="unknown"
    rl=RunLogger(run_number=None, pipeline_version=pv, model_path="multi: "+", ".join(m["name"] for m in models), fireworks_model="(disabled)", fireworks_key_set=False, n_gpu_layers=ngl, n_ctx=args.n_ctx, n_threads=args.n_threads, num_questions=len(questions)*len(models), eval_source=os.path.basename(args.eval))
    for mc in models: run_model(mc, questions, ngl, args.n_ctx, args.n_threads, rl)
    xd=os.path.join(_HERE,"eval_results")
    try: fp=rl.write_xlsx(xd); logger.warning("Run log written to %s", fp)
    except Exception as e: logger.warning("Failed to write run log: %s", e)
    logger.warning("Done — %d q × %d models = %d rows", len(questions), len(models), len(questions)*len(models))

if __name__=="__main__": main()
