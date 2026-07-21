#!/usr/bin/env python3
"""Quick test: Run math 5-step workflow on 3 questions via GPU."""
import json, os, sys, time
os.environ['PYTHONUNBUFFERED'] = '1'

sys.path.insert(0, '/home/artem/dev/amd-hackathon')
from agent.cell import Cell, StepConfig, DecodingConfig
from agent.workflow import WorkflowEngine, MATH_3STEP_WORKFLOW

DATA_PATH = "/home/artem/dev/amd-hackathon/data/eval/math_combined_80.json"
MODEL_PATH = "/home/artem/models/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf"

with open(DATA_PATH) as f:
    all_qs = json.load(f)
test_qs = all_qs[:3]

print(f"Testing {len(test_qs)} math questions with 5-step workflow", flush=True)
print(f"Model: {MODEL_PATH}", flush=True)

from llama_cpp import Llama
t0 = time.time()
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_gpu_layers=-1, verbose=False)
print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)

def infer_fn(messages, max_tokens, temperature):
    resp = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens or 256,
        temperature=temperature or 0.0,
    )
    return resp["choices"][0]["message"]["content"]

engine = WorkflowEngine(infer_fn)

cell = Cell(
    task_id='math',
    model_key='qwen2.5-math-1.5b',
    steps=MATH_3STEP_WORKFLOW,
)

for qi, q in enumerate(test_qs):
    prompt = q["prompt"]
    expected = q.get("expected_answer", "?")
    print(f"\n{'='*70}", flush=True)
    print(f"Q{qi}: {prompt[:80]}...", flush=True)
    print(f"Expected: {expected}", flush=True)
    
    t_start = time.time()
    result = engine.run(cell, prompt)
    elapsed = time.time() - t_start
    
    final = result["final_answer"]
    print(f"Final answer: {final[:100]}", flush=True)
    print(f"Time: {elapsed:.1f}s total", flush=True)
    print(f"Steps:", flush=True)
    for sr in result["step_results"]:
        out = sr["output"][:80] + "..." if len(sr["output"]) > 80 else sr["output"]
        print(f"  [{sr['step']}] {sr['latency_ms']:.0f}ms | {out}", flush=True)

print(f"\n{'='*70}", flush=True)
print("DONE", flush=True)
