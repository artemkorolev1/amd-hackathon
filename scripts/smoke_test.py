#!/usr/bin/env python3
"""Smoke test for v12e pipeline — Qwen + LoRA + new complexity."""
import os, sys, time, json

# Set working dir to project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

os.environ["COMPLEXITY_MODEL_DIR"] = "/home/artem/dev/amd-hackathon-shared/classifiers/best_complexity_model"
os.environ["FIREWORKS_API_KEY"] = ""

# Load the pipeline components
from agent.complexity import score as c_score  # MiniLM ML scorer
from agent.category_filter import classify
from agent.solvers import local_model
from agent.dynamic_prompts import build_system_prompt, get_max_tokens
from agent.solvers.deterministic import solve_arithmetic

prompts = [
    ("q_math", "What is 2+2?"),
    ("q_logic", "If it is raining, the ground is wet. It is raining. What can you conclude?"),
    ("q_ner", "Identify all named entities in: Elon Musk founded Tesla and SpaceX in California."),
]

for tid, prompt in prompts:
    print(f"\n{'='*60}")
    print(f"  [{tid}] {prompt}")
    print(f"{'='*60}")

    # Try deterministic solver first
    det_ans = solve_arithmetic(prompt, "math_arithmetic")
    print(f"  Deterministic: {det_ans or '—'}")

    # Stage 2
    cat, conf, scores = classify(prompt)
    print(f"  S2: {cat} (conf={conf:.2f})")

    # Complexity
    cx = c_score(prompt)
    print(f"  Complexity: {cx:.4f}")

    # LoRA inference
    sys_prompt = build_system_prompt(cat, cx)
    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
    t0 = time.time()
    ans = local_model.chat_completion(messages, category=cat, max_tokens=get_max_tokens(cat, cx))
    elapsed = time.time() - t0
    print(f"  LoRA ({cat}): {ans[:120] if ans else '(empty)'} [{elapsed:.1f}s]")
