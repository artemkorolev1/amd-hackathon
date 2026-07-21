#!/usr/bin/env python3
"""Run 3-step workflow (all steps) with Llama-3.2-1B on all 94 math questions."""
import json, os, sys, time, gc
os.environ['PYTHONUNBUFFERED'] = '1'
sys.path.insert(0, '/home/artem/dev/amd-hackathon')

from agent.cell import Cell, StepConfig
from agent.workflow import WorkflowEngine

DATA_PATH = "/home/artem/dev/amd-hackathon/data/eval/math_combined_80.json"
RESULTS_PATH = "/home/artem/dev/amd-hackathon/gepa_plans/llama_math_results.json"

MODEL_PATH = "/home/artem/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

with open(DATA_PATH) as f:
    questions = json.load(f)
print(f"Loaded {len(questions)} math questions", flush=True)

from llama_cpp import Llama
print("Loading Llama-3.2-1B on GPU...", flush=True)
t0 = time.time()
llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_gpu_layers=-1, verbose=False)
print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

def infer_fn(messages, max_tokens, temperature):
    resp = llm.create_chat_completion(messages=messages, max_tokens=max_tokens or 256, temperature=temperature or 0.0)
    return resp["choices"][0]["message"]["content"]

engine = WorkflowEngine(infer_fn)

# 3-step workflow, all on Llama
cell = Cell(
    task_id="math",
    model_key="llama-3.2-1b",
    steps=[
        StepConfig(name="plan", system_prompt="Create a numbered plan to solve this math problem. List the calculations needed. Output ONLY the plan."),
        StepConfig(name="solve", system_prompt="Execute the plan step by step. Show each calculation. Put the final answer in \\boxed{}."),
        StepConfig(name="compose", system_prompt="Format: The answer is \\boxed{number}."),
    ],
)

results = {}
correct = 0
total_start = time.time()

for qi, q in enumerate(questions):
    prompt = q["prompt"]
    expected = str(q.get("expected_answer", ""))
    
    t_q = time.time()
    result = engine.run(cell, prompt)
    elapsed = time.time() - t_q
    
    final_answer = result["final_answer"].strip()
    is_correct = final_answer.lower() == expected.lower() or expected in final_answer
    
    if is_correct:
        correct += 1
    
    results[q.get("task_id", f"q{qi}")] = {
        "expected": expected,
        "got": final_answer[:100],
        "correct": is_correct,
        "latency_ms": result["total_latency"],
        "steps": result["step_results"],
    }
    
    mark = "✓" if is_correct else "✗"
    print(f"  {mark} Q{qi+1}: expected={expected}, got={final_answer[:60]}, lat={elapsed*1000:.0f}ms", flush=True)
    
    # Partial save every 10 questions
    if (qi + 1) % 10 == 0:
        acc = correct / (qi + 1)
        out = {"questions_processed": qi + 1, "accuracy": round(acc, 3), "correct": correct, "results": results}
        with open(RESULTS_PATH, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  [save] {correct}/{qi+1} = {acc:.3f}  ({elapsed*1000:.0f}ms/q avg)", flush=True)

# Final
total_elapsed = time.time() - total_start
accuracy = correct / len(questions) if questions else 0
out = {
    "model": "Llama-3.2-1B-Instruct",
    "workflow": "3-step (plan-solve-compose)",
    "questions": len(questions),
    "correct": correct,
    "accuracy": round(accuracy, 3),
    "total_time_s": round(total_elapsed, 1),
    "avg_latency_ms": round(total_elapsed / len(questions) * 1000, 1) if questions else 0,
    "results": results,
}
with open(RESULTS_PATH, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n{'='*60}", flush=True)
print(f"LLAMA MATH EVAL DONE: {correct}/{len(questions)} = {accuracy:.3f}", flush=True)
print(f"Total: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)", flush=True)
print(f"Results: {RESULTS_PATH}", flush=True)
