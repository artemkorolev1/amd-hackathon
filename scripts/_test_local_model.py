#!/usr/bin/env python3
"""Quick smoke test — query that hits the local model."""
import os, sys, time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
os.environ["COMPLEXITY_MODEL_DIR"] = os.path.join(os.getcwd(), "shared", "classifiers", "best_complexity_model")

from agent.complexity_filter import score as c_score  # heuristic per-category scorer
from agent.category_filter import classify
from agent.solvers import local_model
from agent.dynamic_prompts import build_system_prompt, get_max_tokens
from agent.pre_filter import stage0

# A factual query the deterministic solver won't answer
prompt = "What is the capital of New Zealand?"
print("=" * 60)
print(f"  PROMPT: {prompt}")
print("=" * 60)

s0 = stage0(prompt)
print(f"\n[1] PRE-FILTER: action={s0.action}")
cat, conf, _ = classify(prompt)
print(f"[2] CATEGORY: {cat} (conf={conf:.2f})")
cx = c_score(prompt, cat)
print(f"[3] COMPLEXITY: {cx:.4f}")

# Check deterministic (factual_knowledge — the expected mapping)
from agent.solvers.deterministic import solve_factual_qa
ans = solve_factual_qa(prompt, "factual_knowledge")
print(f"[4] DET (factual): {ans or '—'}")

# Fall through to local model
sys_p = build_system_prompt(cat, cx)
msgs = [{"role":"system","content":sys_p},{"role":"user","content":prompt}]
print(f"[5] Running local model (may take ~15s)...")
t0 = time.time()
ans = local_model.chat_completion(msgs, category=cat, max_tokens=get_max_tokens(cat, cx))
elapsed = time.time() - t0
print(f"    → answer: {ans[:200] if ans else '(empty)'}")
print(f"    → latency: {elapsed:.1f}s")
print(f"\n{'='*60}")
print(f"  FINAL: {ans}")
print(f"{'='*60}")
