#!/usr/bin/env python3
"""Test math workflow with DIFFERENT models per step.
Plan/compose on instruct model, solve on math specialist."""
import json, os, sys, time
os.environ['PYTHONUNBUFFERED'] = '1'
sys.path.insert(0, '/home/artem/dev/amd-hackathon')
from agent.cell import Cell, StepConfig
from agent.workflow import WorkflowEngine

DATA_PATH = "/home/artem/dev/amd-hackathon/data/eval/math_combined_80.json"
MATH_PATH = "/home/artem/models/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf"
INSTRUCT_PATH = "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"

with open(DATA_PATH) as f:
    all_qs = json.load(f)
test_qs = all_qs[:3]

from llama_cpp import Llama

# Load both models
print("Loading instruct model (plan/compose)...", flush=True)
llm_instruct = Llama(model_path=INSTRUCT_PATH, n_ctx=2048, n_gpu_layers=-1, verbose=False)
print("Loading math specialist (solve)...", flush=True)
llm_math = Llama(model_path=MATH_PATH, n_ctx=2048, n_gpu_layers=-1, verbose=False)

model_cache = {
    "qwen2.5-1.5b": llm_instruct,
    "qwen2.5-math-1.5b": llm_math,
}

def infer_fn(messages, max_tokens, temperature, model_key="qwen2.5-1.5b"):
    llm = model_cache.get(model_key, llm_instruct)
    resp = llm.create_chat_completion(
        messages=messages, max_tokens=max_tokens or 256, temperature=temperature or 0.0,
    )
    return resp["choices"][0]["message"]["content"]

# Wrap to capture model_key — WorkflowEngine needs a way to pass model per step
# Currently it uses cell.model_key or step.model_key, but infer_fn doesn't get model_key
# Quick fix: make a closure per step
def make_infer(model_key):
    def fn(messages, max_tokens, temperature):
        return infer_fn(messages, max_tokens, temperature, model_key)
    return fn

# 3-step workflow: instruct → math → instruct
cell = Cell(
    task_id="math",
    model_key="qwen2.5-math-1.5b",  # default
    steps=[
        StepConfig(name="plan", system_prompt="Analyze this math problem. List the steps to solve it. Output ONLY the numbered plan, nothing else.",
                   model_key="qwen2.5-1.5b"),
        StepConfig(name="solve", system_prompt="Execute the plan above step by step. Show each calculation. Put the final answer in \\boxed{}.",
                   model_key="qwen2.5-math-1.5b"),
        StepConfig(name="compose", system_prompt="Format the answer clearly. Start with 'The answer is \\\\boxed{number}'.",
                   model_key="qwen2.5-1.5b"),
    ],
)

for qi, q in enumerate(test_qs):
    prompt = q["prompt"]
    expected = q.get("expected_answer", "?")
    print(f"\n{'='*70}", flush=True)
    print(f"Q{qi}: {prompt[:80]}...", flush=True)
    print(f"Expected: {expected}", flush=True)

    # Run each step manually with the right model
    artifacts = {"_input": prompt}
    
    for step in cell.steps:
        step_model = step.model_key or cell.model_key
        step_max_tokens = (step.decoding or cell.decoding).max_tokens
        step_temp = (step.decoding or cell.decoding).temperature
        
        sys_prompt = step.system_prompt
        user_msg = artifacts.get("_input", prompt)
        
        # For continuation steps, include prior output
        prior_keys = [k for k in artifacts if k != "_input"]
        if prior_keys:
            last_key = prior_keys[-1]
            last_out = artifacts[last_key]
            if len(last_out) > 400:
                last_out = last_out[:400] + "\n...truncated"
            user_msg = f"Original problem:\n{prompt}\n\n=== [{last_key}] output ===\n{last_out}\n\nNow: {step.system_prompt}"
            if len(prior_keys) > 1:
                sys_prompt += f"\n(Previous steps: {', '.join(prior_keys[:-1])})"
        
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_msg}]
        
        print(f"  [{step.name}] model={step_model}...", flush=True)
        t0 = time.time()
        llm = model_cache[step_model]
        resp = llm.create_chat_completion(messages=messages, max_tokens=256, temperature=0.0)
        result = resp["choices"][0]["message"]["content"]
        elapsed = time.time() - t0
        artifacts[step.name] = result
        
        out = result[:100] + "..." if len(result) > 100 else result
        print(f"    {elapsed*1000:.0f}ms | {out}", flush=True)
    
    final = artifacts[cell.steps[-1].name]
    print(f"  FINAL: {final[:120]}", flush=True)

print(f"\n{'='*70}", flush=True)
print("DONE", flush=True)
