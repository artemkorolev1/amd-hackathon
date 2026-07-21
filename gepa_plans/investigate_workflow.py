#!/usr/bin/env python3
"""
Investigate why qwen2.5-math-1.5b ignores multi-step workflow instructions.

Tests:
  1. Config A: qwen2.5-1.5b-instruct for ALL steps (3 questions)
  2. Config B: qwen2.5-math-1.5b for ALL steps (3 questions)
  3. Single-prompt (no workflow) on both models (1 question)
  4. Config C: Llama-3.2-1B-Instruct for ALL steps (1 question)
  5. Raw message dump analysis

Saves exact input/output dumps to workflow_investigation.md
"""
import gc
import json
import os
import sys
import time
import textwrap

sys.path.insert(0, '/home/artem/dev/amd-hackathon')
from agent.cell import Cell, StepConfig, DecodingConfig
from agent.workflow import WorkflowEngine, build_step_messages

DATA_PATH = "/home/artem/dev/amd-hackathon/data/eval/math_combined_80.json"
OUT_PATH = "/home/artem/dev/amd-hackathon/gepa_plans/workflow_investigation.md"

MODEL_INSTRUCT = "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODEL_MATH = "/home/artem/models/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf"
MODEL_LLAMA = "/home/artem/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

# ── Data ─────────────────────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    all_qs = json.load(f)
test_qs = all_qs[:3]

# ── Helper to dump messages ──────────────────────────────────────────────────
def dump_messages(messages):
    """Return a string representation of messages dict."""
    lines = []
    for i, msg in enumerate(messages):
        lines.append(f"  --- Message {i} (role={msg['role']}) ---")
        content = msg['content']
        lines.append(textwrap.indent(content, "  "))
    return "\n".join(lines)

def short(s, maxlen=150):
    return s[:maxlen] + "..." if len(s) > maxlen else s


# ── Investigation runner ─────────────────────────────────────────────────────
results = []

def load_model(path, label):
    print(f"\n{'='*60}", flush=True)
    print(f"Loading {label}...", flush=True)
    from llama_cpp import Llama
    llm = Llama(model_path=path, n_ctx=2048, n_gpu_layers=-1, verbose=False)
    print(f"  Loaded. GPU layers: -1", flush=True)
    return llm

def unload_model(llm):
    del llm
    gc.collect()
    if 'torch' in sys.modules:
        import torch
        torch.cuda.empty_cache()
    time.sleep(1)

def run_config(label, model_path, model_key, steps, questions, do_single_prompt=False):
    """Run a config and return results dict."""
    print(f"\n{'#'*70}", flush=True)
    print(f"## CONFIG: {label}", flush=True)
    print(f"{'#'*70}", flush=True)
    
    llm = load_model(model_path, model_key)
    
    results_for_label = []
    
    for qi, q in enumerate(questions):
        prompt = q["prompt"]
        expected = q.get("expected_answer", "?")
        print(f"\n{'─'*50}", flush=True)
        print(f"Q{qi}: {short(prompt, 80)}", flush=True)
        print(f"Expected: {expected}", flush=True)
        
        if do_single_prompt:
            # Single-prompt multi-step test
            single_prompt = "Work through this problem: first plan, then solve step by step, then put the answer in \\boxed{}."
            messages = [
                {"role": "system", "content": "You are a helpful math assistant."},
                {"role": "user", "content": f"{prompt}\n\n{single_prompt}"},
            ]
            print(f"\n  [single prompt] Messages sent:", flush=True)
            print(dump_messages(messages), flush=True)
            
            t0 = time.time()
            resp = llm.create_chat_completion(messages=messages, max_tokens=512, temperature=0.0)
            result = resp["choices"][0]["message"]["content"]
            elapsed = time.time() - t0
            
            print(f"\n  Raw output ({elapsed*1000:.0f}ms):", flush=True)
            print(textwrap.indent(result, "    "), flush=True)
            
            results_for_label.append({
                "q_idx": qi,
                "prompt": prompt,
                "expected": expected,
                "type": "single_prompt",
                "messages": messages,
                "raw_output": result,
                "latency_ms": round(elapsed * 1000, 1),
            })
        else:
            # 3-step workflow
            cell = Cell(
                task_id="math",
                model_key=model_key,
                steps=[
                    StepConfig(name="plan", system_prompt="Analyze this math problem. List the steps to solve it. Output ONLY the numbered plan, nothing else.",
                               model_key=model_key),
                    StepConfig(name="solve", system_prompt="Execute the plan above step by step. Show each calculation. Put the final answer in \\boxed{}.",
                               model_key=model_key),
                    StepConfig(name="compose", system_prompt="Format the answer clearly. Start with 'The answer is \\boxed{number}'.",
                               model_key=model_key),
                ],
            )
            
            artifacts = {"_input": prompt}
            
            for step_idx, step in enumerate(cell.steps):
                step_model_key = step.model_key or cell.model_key
                sys_prompt = step.system_prompt
                
                # Build messages the same way as the workflow engine
                prior_keys = [k for k in artifacts if k != "_input"]
                if prior_keys:
                    last_key = prior_keys[-1]
                    last_out = artifacts[last_key]
                    if len(last_out) > 400:
                        last_out = last_out[:400] + "\n...[truncated]"
                    user_msg = f"Original problem:\n{prompt}\n\n=== [{last_key}] output ===\n{last_out}\n\nNow: {step.system_prompt}"
                    if len(prior_keys) > 1:
                        sys_prompt += f"\n(Previous steps: {', '.join(prior_keys[:-1])})"
                else:
                    user_msg = prompt
                
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg},
                ]
                
                print(f"\n  [{step.name}] (model={step_model_key})", flush=True)
                print(f"  Messages sent:", flush=True)
                print(dump_messages(messages), flush=True)
                
                t0 = time.time()
                resp = llm.create_chat_completion(messages=messages, max_tokens=256, temperature=0.0)
                result = resp["choices"][0]["message"]["content"]
                elapsed = time.time() - t0
                
                artifacts[step.name] = result
                
                print(f"\n  Raw output ({elapsed*1000:.0f}ms):", flush=True)
                print(textwrap.indent(result, "    "), flush=True)
                
                # Check if it followed the plan (for solve step)
                if step.name == "solve" and prior_keys:
                    plan_text = artifacts.get("plan", "")
                    follows = "plan" in result.lower()[:50] or any(word in result.lower()[:100] for word in ["step 1", "step1", "1.", "first"])
                    print(f"\n  >>> Follows plan instruction? {'YES' if follows else 'NO (may be regenerating)'}", flush=True)
                
                results_for_label.append({
                    "q_idx": qi,
                    "step": step.name,
                    "expected": expected,
                    "messages": messages,
                    "raw_output": result,
                    "latency_ms": round(elapsed * 1000, 1),
                })
            
            final = artifacts.get(cell.steps[-1].name, "N/A")
            print(f"\n  FINAL: {short(final, 120)}", flush=True)
    
    unload_model(llm)
    return results_for_label


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1: Config A — instruct for ALL steps (3 questions)
# ═════════════════════════════════════════════════════════════════════════════
results_a = run_config("A: qwen2.5-1.5b-instruct (ALL steps)", MODEL_INSTRUCT, "qwen2.5-1.5b",
                       [
                           StepConfig(name="plan"),
                           StepConfig(name="solve"),
                           StepConfig(name="compose"),
                       ], test_qs)

# ═════════════════════════════════════════════════════════════════════════════
# TEST 2: Config B — math for ALL steps (3 questions)
# ═════════════════════════════════════════════════════════════════════════════
results_b = run_config("B: qwen2.5-math-1.5b (ALL steps)", MODEL_MATH, "qwen2.5-math-1.5b",
                       [
                           StepConfig(name="plan"),
                           StepConfig(name="solve"),
                           StepConfig(name="compose"),
                       ], test_qs)

# ═════════════════════════════════════════════════════════════════════════════
# TEST 3: Single-prompt multi-step (both models, 1 question)
# ═════════════════════════════════════════════════════════════════════════════
results_sp_instruct = run_config("Single-prompt: instruct", MODEL_INSTRUCT, "qwen2.5-1.5b",
                                 [], test_qs[:1], do_single_prompt=True)
results_sp_math = run_config("Single-prompt: math", MODEL_MATH, "qwen2.5-math-1.5b",
                             [], test_qs[:1], do_single_prompt=True)

# ═════════════════════════════════════════════════════════════════════════════
# TEST 4: Config C — Llama-3.2 for ALL steps (1 question)
# ═════════════════════════════════════════════════════════════════════════════
results_c = run_config("C: Llama-3.2-1B-Instruct (ALL steps)", MODEL_LLAMA, "llama-3.2-1b",
                       [
                           StepConfig(name="plan"),
                           StepConfig(name="solve"),
                           StepConfig(name="compose"),
                       ], test_qs[:1])

# ═════════════════════════════════════════════════════════════════════════════
# TEST 5: Raw dump investigation — math specialist on Q0
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'#'*70}", flush=True)
print(f"## RAW DUMP: Math specialist Q0 — plan→solve (simulating workflow)", flush=True)
print(f"{'#'*70}", flush=True)

llm_math2 = load_model(MODEL_MATH, "qwen2.5-math-1.5b (raw dump)")

q0 = test_qs[0]
prompt0 = q0["prompt"]

# Step 1: plan (manually with math model)
plan_messages = [
    {"role": "system", "content": "Analyze this math problem. List the steps to solve it. Output ONLY the numbered plan, nothing else."},
    {"role": "user", "content": prompt0},
]
print(f"\n  [plan] EXACT messages:", flush=True)
print(dump_messages(plan_messages), flush=True)

resp = llm_math2.create_chat_completion(messages=plan_messages, max_tokens=256, temperature=0.0)
plan_output = resp["choices"][0]["message"]["content"]
print(f"\n  [plan] EXACT response:", flush=True)
print(textwrap.indent(plan_output, "    "), flush=True)

# Step 2: solve (feed plan)
solve_user = f"Original problem:\n{prompt0}\n\n=== [plan] output ===\n{plan_output[:400]}\n\nNow: Execute the plan above step by step. Show each calculation. Put the final answer in \\boxed{{}}."
solve_messages = [
    {"role": "system", "content": "Execute the plan above step by step. Show each calculation. Put the final answer in \\boxed{}."},
    {"role": "user", "content": solve_user},
]
print(f"\n  [solve] EXACT messages:", flush=True)
print(dump_messages(solve_messages), flush=True)

resp = llm_math2.create_chat_completion(messages=solve_messages, max_tokens=256, temperature=0.0)
solve_output = resp["choices"][0]["message"]["content"]
print(f"\n  [solve] EXACT response:", flush=True)
print(textwrap.indent(solve_output, "    "), flush=True)

# Check if response uses the plan or regenerates
print(f"\n  >>> Plan was: {short(plan_output, 80)}", flush=True)
print(f"  >>> Solve output uses plan numbers? {'YES' if any(n in solve_output for n in ['30', 'lollipop', 'bag']) else 'CHECKING...'}", flush=True)

unload_model(llm_math2)

print(f"\n{'#'*70}", flush=True)
print("ALL TESTS COMPLETE", flush=True)
print(f"{'#'*70}", flush=True)
