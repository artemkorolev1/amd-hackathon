#!/usr/bin/env python3
"""CPU benchmark of the full pipeline (simulating harness.py behavior)."""
import time, json, os, sys

os.environ['N_GPU_LAYERS'] = '0'
os.environ['N_THREADS'] = '2'
os.environ['N_CTX'] = '2048'

sys.path.insert(0, '/home/artem/dev/amd-hackathon')
sys.path.insert(0, '/home/artem/dev/amd-hackathon/scripts')

from agent.pre_filter import stage0
from agent.category_filter import classify_with_detail as _stage2_detail
from agent.solvers.deterministic import (
    solve_arithmetic, solve_logic, solve_sentiment,
    solve_ner, solve_factual_qa, solve_code_debugging,
)
from llama_cpp import Llama

# Load model (with timing)
t0 = time.time()
print("Loading model...", file=sys.stderr)
llm = Llama(
    model_path='/home/artem/dev/amd-hackathon/models/qwen2.5-1.5b-instruct-q4_k_m.gguf',
    n_ctx=2048, n_gpu_layers=0, n_threads=2, flash_attn=True, verbose=False
)
model_load_time = time.time() - t0
print(f"Model loaded in {model_load_time:.2f}s", file=sys.stderr)

# Load eval set
with open('/home/artem/dev/amd-hackathon/data/eval/primary/eval_mini_10.json') as f:
    questions = json.load(f)

# Warmup
print("Warming up model...", file=sys.stderr)
llm.create_chat_completion(
    messages=[{'role': 'user', 'content': 'Hello'}],
    max_tokens=5, temperature=0.0
)
print("Warmup done.\n", file=sys.stderr)

# Deterministic solver map
DET_SOLVER = {
    'math_arithmetic': solve_arithmetic,
    'sentiment': solve_sentiment,
    'other_complex': solve_factual_qa,
    'code_debugging': solve_code_debugging,
}

total_time = 0.0
det_count = 0
llm_count = 0
bypass_count = 0
per_q_times = []

for i, q in enumerate(questions):
    prompt = q['prompt']
    cat_label = q.get('category', 'unknown')
    t_q = time.monotonic()
    
    # Stage 0 - pre-filter (bypass for greetings, pure arithmetic)
    s0 = stage0(prompt)
    if s0.action == 'bypass' and s0.direct_answer:
        elapsed = time.monotonic() - t_q
        total_time += elapsed
        bypass_count += 1
        per_q_times.append(elapsed)
        print(f'Q{i+1} [{cat_label:12s}] BYPASS {elapsed:.3f}s ans={s0.direct_answer}')
        continue
    
    # Stage 2 - classifier (regex, no model)
    t_s2 = time.monotonic()
    detail = _stage2_detail(prompt)
    category = detail['category']
    score_delta = detail['score_delta']
    t_s2 = time.monotonic() - t_s2
    
    # Try deterministic solver if high-confidence
    if category in DET_SOLVER and score_delta >= 0.5:
        solver = DET_SOLVER[category]
        t_det = time.monotonic()
        try:
            det_answer = solver(prompt)
        except Exception:
            det_answer = None
        t_det = time.monotonic() - t_det
        if det_answer and str(det_answer).strip():
            elapsed = time.monotonic() - t_q
            total_time += elapsed
            det_count += 1
            per_q_times.append(elapsed)
            print(f'Q{i+1} [{cat_label:12s}] DET({category}) {elapsed:.3f}s (cls={t_s2:.3f}s, solver={t_det:.3f}s) ans={str(det_answer)[:60]}')
            continue
    
    # LLM inference
    t_inf = time.monotonic()
    resp = llm.create_chat_completion(
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=150, temperature=0.0, stop=['\n\n']
    )
    inf_time = time.monotonic() - t_inf
    
    elapsed = time.monotonic() - t_q
    total_time += elapsed
    llm_count += 1
    per_q_times.append(elapsed)
    usage = resp.get('usage', {})
    ans = resp['choices'][0]['message']['content'][:60]
    print(f'Q{i+1} [{cat_label:12s}] LLM({category}) {elapsed:.3f}s (cls={t_s2:.3f}s, inf={inf_time:.3f}s, pt={usage.get("prompt_tokens","?")}, ct={usage.get("completion_tokens","?")}) ans={ans}')

print(f'\n--- SUMMARY ---')
print(f'Model load:  {model_load_time:.3f}s')
print(f'Total:       {total_time:.3f}s for {len(questions)} questions')
print(f'Avg:         {total_time/len(questions):.3f}s/q')
print(f'Bypass:      {bypass_count}, Deterministic: {det_count}, LLM: {llm_count}')
print(f'Per-question times: {[f"{t:.3f}" for t in per_q_times]}')
print(f'')
print(f'EXTRAPOLATION:')
avg_t = total_time / len(questions)
print(f'  {len(questions)}q → {total_time:.1f}s total ({avg_t:.3f}s/q)')
print(f'  300q → {avg_t * 300:.1f}s (without model load)')
print(f'  300q → {avg_t * 300 + model_load_time:.1f}s (with model load)')
print(f'  Deadline: 600s')
if avg_t * 300 + model_load_time <= 600:
    print(f'  ✅ 300 questions CAN complete within 600s on CPU')
else:
    print(f'  ❌ 300 questions CANNOT complete within 600s on CPU')
    
    # What if 75% bypass/deterministic (caveman router)
    print(f'')
    print(f'  WITH CAVEMAN ROUTER (75% tokens saved):')
    # If 75% of questions are handled by deterministic solvers at ~0.1s each
    det_300 = int(300 * 0.75)
    llm_300 = 300 - det_300
    est_det_time = det_300 * 0.1  # 0.1s per deterministic question
    est_llm_time = 0
    llm_qs = [t for i, t in enumerate(per_q_times) if i >= bypass_count]
    if llm_qs:
        avg_llm = sum(llm_qs) / len(llm_qs)
        est_llm_time = llm_300 * avg_llm
    total_est = est_det_time + est_llm_time + model_load_time
    print(f'    {det_300} det questions × ~0.1s = {est_det_time:.0f}s')
    print(f'    {llm_300} LLM questions × ~{avg_llm:.2f}s = {est_llm_time:.0f}s')
    print(f'    Total est: {total_est:.0f}s')
    if total_est <= 600:
        print(f'    ✅ Likely achievable with router')
    else:
        print(f'    ❌ Still not achievable')
