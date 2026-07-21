#!/usr/bin/env python3
"""Classify each GSM8K failure: misroute vs ToRA error vs deterministic/LLM fail."""
import sys, os, gc, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['MODEL_PATH'] = '/dev/null'
os.environ['LOG_LEVEL'] = 'ERROR'

import logging
logging.getLogger('pipeline').setLevel(logging.ERROR)

from agent.classifier import classify
from scripts.grade_answer import fuzzy_match
import pandas as pd
from agent.pipeline import Pipeline, PipelineConfig

df = pd.read_parquet('/tmp/gsm8k_test.parquet')
N = 50

cfg = PipelineConfig(
    model_path='/dev/null',
    n_gpu_layers=0,
    n_ctx=512,
    n_threads=2,
    consensus_samples=1,
    category_model_map={},
)
pipe = Pipeline(config=cfg)

misroutes = []
tora_fails = []
det_fails = []
other_fails = []

for i in range(N):
    row = df.iloc[i]
    question = row['question']
    expected = row['answer'].split('####')[-1].strip() if '####' in row['answer'] else row['answer'].strip()

    cat, method, conf = classify(question)

    t_start = time.time()
    try:
        answer = pipe.process(question)
    except Exception as e:
        answer = ''
    elapsed = time.time() - t_start

    if i > 0 and i % 10 == 0:
        gc.collect()

    correct = fuzzy_match(answer, expected)

    if not correct:
        entry = {
            'num': i+1,
            'question': question[:80],
            'expected': expected,
            'got': answer[:50] if answer else '(empty)',
            'classified': cat,
            'elapsed': round(elapsed, 1),
        }
        if cat != 'math':
            misroutes.append(entry)
        else:
            tora_fails.append(entry)

pipe.close()

print(f'\n=== GSM8K Failure Analysis (50 problems) ===')
print(f'Total failures: {len(misroutes) + len(tora_fails)}')

print(f'\n--- MISROUTED (classified as other instead of math) ---')
print(f'Count: {len(misroutes)}')
for e in misroutes:
    print(f'  [{e["num"]}] cat={e["classified"]} | expected={e["expected"]:>8} | got={e["got"][:40]}')
    print(f'         Q: {e["question"][:70]}')

print(f'\n--- CORRECTLY ROUTED as math but ToRA/pipeline wrong ---')
print(f'Count: {len(tora_fails)}')
for e in tora_fails:
    print(f'  [{e["num"]}] expected={e["expected"]:>8} | got={e["got"][:40]}')
    print(f'         Q: {e["question"][:70]}')

print(f'\nSummary: {len(misroutes)} misrouted, {len(tora_fails)} correctly-classified math failures')
