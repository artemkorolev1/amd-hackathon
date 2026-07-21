#!/usr/bin/env python3
"""Quick smoke test of the renamed V12E pipeline — runs one query step by step."""
import os, sys, time, json

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
os.environ["COMPLEXITY_MODEL_DIR"] = os.path.join(os.getcwd(), "shared", "classifiers", "best_complexity_model")

from agent.complexity_filter import score as c_score  # heuristic per-category scorer
from agent.category_filter import classify
from agent.solvers import local_model
from agent.dynamic_prompts import build_system_prompt, get_max_tokens
from agent.solvers.deterministic import solve_arithmetic, solve_logic, solve_sentiment, solve_ner, solve_factual_qa, solve_code_debugging
from agent.pre_filter import stage0

# Test prompt — math arithmetic
prompt = "What is 2+2?"
print("=" * 60)
print(f"  PROMPT: {prompt}")
print("=" * 60)

# Step 1: Pre-filter (was Stage 0)
t0 = time.time()
s0 = stage0(prompt)
t1 = time.time()
print(f"\n[1] PRE-FILTER ({t1-t0:.4f}s)")
print(f"    action={s0.action}, category={s0.category}, direct={s0.direct_answer}")

if s0.action == "bypass" and s0.direct_answer:
    print(f"    → BYPASS answer: {s0.direct_answer}")
else:
    # Step 2: Category filter (was Stage 2)
    cat, conf, scores = classify(prompt)
    t2 = time.time()
    print(f"\n[2] CATEGORY FILTER ({t2-t1:.4f}s)")
    print(f"    category={cat}, confidence={conf:.4f}")
    top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
    print(f"    top3: {dict(top3)}")

    # Step 3: Complexity field (was Stage 3)
    cx = c_score(prompt, cat)
    t3 = time.time()
    print(f"\n[3] COMPLEXITY FIELD ({t3-t2:.4f}s)")
    print(f"    score={cx:.4f} → {'low' if cx<0.3 else 'medium' if cx<0.7 else 'high'}")

    # Step 4: Deterministic solvers
    det_map = {"math":"math_arithmetic","logic":"logical_reasoning","sentiment":"sentiment",
               "ner":"named_entity_recognition","factual":"factual_knowledge",
               "code_debug":"code_debugging","summarization":"summarization"}
    det_ans = None
    for sfn in [solve_arithmetic, solve_logic, solve_sentiment, solve_ner, solve_factual_qa, solve_code_debugging]:
        try:
            a = sfn(prompt, det_map.get(cat, "general"))
            if a:
                det_ans = a
                break
        except Exception as e:
            pass
    t4 = time.time()
    print(f"\n[4] DETERMINISTIC ({t4-t3:.4f}s)")
    print(f"    answer={det_ans or '—'}")
    print(f"    resolved_by={'deterministic' if det_ans else 'local_model'}")

    if not det_ans:
        # Step 5: Local model (was LoRA)
        sys_p = build_system_prompt(cat, cx)
        msgs = [{"role":"system","content":sys_p},{"role":"user","content":prompt}]
        ans = local_model.chat_completion(msgs, category=cat,
                                          max_tokens=get_max_tokens(cat, cx))
        t5 = time.time()
        print(f"\n[5] LOCAL MODEL ({t5-t4:.4f}s)")
        print(f"    answer={ans[:120] if ans else '(empty)'}")
    else:
        ans = det_ans

print(f"\n{'='*60}")
print(f"  FINAL: {ans}")
print(f"{'='*60}")
