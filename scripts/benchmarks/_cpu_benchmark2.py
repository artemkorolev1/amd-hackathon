#!/usr/bin/env python3
"""CPU benchmark using actual harness.py with per-question timing."""
import subprocess
import time
import json
import sys
import os

HERE = '/home/artem/dev/amd-hackathon'

def run_benchmark(eval_path, label, n_threads=2):
    """Run the harness with per-question timing by patching the _process function."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {label}")
    print(f"Eval: {eval_path}")
    print(f"N_THREADS={n_threads}")
    print(f"{'='*60}")
    
    # Read the eval set first
    with open(eval_path) as f:
        data = json.load(f)
    questions = data.get("questions", data) if isinstance(data, dict) else data
    n_q = len(questions)
    print(f"Questions: {n_q}")
    
    # Run with time
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{HERE}:{HERE}/scripts"
    env['MODEL_PATH'] = f"{HERE}/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    env['N_GPU_LAYERS'] = '0'
    env['N_THREADS'] = str(n_threads)
    env['FIREWORKS_API_KEY'] = '***'  # Disable Fireworks
    
    t0 = time.time()
    result = subprocess.run(
        [f'{HERE}/.venv/bin/python', f'{HERE}/scripts/harness.py', '--cpu', eval_path],
        capture_output=True, text=True, timeout=300, env=env
    )
    wall_time = time.time() - t0
    
    # Parse stderr for warnings
    stderr_lines = result.stderr.strip().split('\n')
    warnings = [l for l in stderr_lines if l.startswith('WARNING:')]
    
    # Parse stdout for answers (one per question)
    stdout_lines = [l for l in result.stdout.strip().split('\n') if l.strip()]
    answers = stdout_lines
    
    print(f"\nWall time: {wall_time:.3f}s")
    print(f"Mode: {'CPU' if 'N_GPU_LAYERS=0' in str(env) else 'GPU'}")
    print(f"Model load warnings: {[w for w in warnings if 'Loading' in w]}")
    print(f"Answers: {len(answers)}/{n_q}")
    
    return {
        'label': label,
        'eval_path': eval_path,
        'n_questions': n_q,
        'wall_time': wall_time,
        'answers': answers,
        'n_threads': n_threads,
        'returncode': result.returncode,
    }


# Benchmark 1: eval_mini_10 (10 questions, mixed categories)
r1 = run_benchmark(
    f'{HERE}/data/eval/primary/eval_mini_10.json',
    'Mini 10 (mixed categories)',
    n_threads=2
)

# Benchmark 2: factual_combined_80 (80 factual questions — mostly deterministic)
r2 = run_benchmark(
    f'{HERE}/data/eval/factual_combined_80.json',
    'Factual 80 (mostly deterministic)',
    n_threads=2
)

# Benchmark 3: validation-v3.json (48 questions, 8 categories)
r3 = run_benchmark(
    f'{HERE}/data/eval/validation-v3.json',
    'Validation 48 (8 categories)',
    n_threads=2
)

# Summary
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
for r in [r1, r2, r3]:
    avg = r['wall_time'] / r['n_questions']
    extrap_300 = avg * 300
    status = "✅" if extrap_300 <= 600 else "❌"
    print(f"\n{r['label']}:")
    print(f"  {r['n_questions']}q → {r['wall_time']:.1f}s total ({avg:.3f}s/q)")
    print(f"  Extrapolated 300q: {extrap_300:.0f}s")
    print(f"  Deadline 600s: {status}")
    print(f"  Return code: {r['returncode']}")
