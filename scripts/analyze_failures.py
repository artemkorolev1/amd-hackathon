#!/usr/bin/env python3
"""Analyze each GSM8K failure: which solver path caused it and why."""
import sys, os, gc, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['MODEL_PATH'] = '/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf'
os.environ['LOG_LEVEL'] = 'ERROR'
import logging
logging.getLogger('pipeline').setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

from agent.classifier import classify
from agent.solvers.deterministic import *
from scripts.grade_answer import fuzzy_match
import pandas as pd
from agent.pipeline import Pipeline, PipelineConfig

cfg = PipelineConfig(
    model_path='/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf',
    n_gpu_layers=-1, n_ctx=2048, n_threads=4,
    consensus_samples=1, category_model_map={},
)
pipe = Pipeline(config=cfg)

df = pd.read_parquet('/tmp/gsm8k_test.parquet')
N = 50

results = []
for i in range(N):
    row = df.iloc[i]
    q = row['question']
    expected = row['answer'].split('####')[-1].strip() if '####' in row['answer'] else row['answer'].strip()

    cat, method, conf = classify(q)
    t_start = time.time()
    try:
        ans = pipe.process(q)
    except Exception as e:
        ans = ''
    elapsed = time.time() - t_start

    if i > 0 and i % 10 == 0:
        gc.collect()

    correct = fuzzy_match(ans, expected)
    results.append({
        'num': i+1,
        'correct': correct,
        'classified': cat,
        'expected': expected,
        'got': ans[:60] if ans else '(empty)',
        'time': round(elapsed, 1),
        'question': q[:100],
    })
    if not correct:
        status = f"FAIL (cat={cat})"
        print(f'[{i+1}] {status}: exp={expected:>8} got={str(ans)[:40]:>40} time={elapsed:.1f}s')

pipe.close()

# Summary
correct_count = sum(1 for r in results if r['correct'])
wrong = [r for r in results if not r['correct']]
print(f'\n=== Summary: {correct_count}/{N} correct ({correct_count/N*100:.1f}%) ===')
print(f'Failures: {len(wrong)}')

# Categorize failures
categories = {'misroute': [], 'tora_code': [], 'tora_exec': [], 'det_solved_wrong': [], 'llm_fallback': []}
for r in wrong:
    got = r['got']
    if r['classified'] != 'math':
        categories['misroute'].append(r)
    elif 'Traceback' in got or 'Error' in got or '(empty)' in got:
        categories['tora_exec'].append(r)
    else:
        categories['tora_code'].append(r)

for cat, items in categories.items():
    print(f'\n  {cat}: {len(items)}')
    for r in items[:5]:
        print(f'    [{r["num"]}] exp={r["expected"]:>8} got={r["got"][:40]}')

# Save to JSON for analysis
with open('/tmp/gsm8k_failure_analysis.json', 'w') as f:
    json.dumps(results, indent=2)
    print(f'\nFull results saved to /tmp/gsm8k_failure_analysis.json')
