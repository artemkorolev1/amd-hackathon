#!/usr/bin/env python3
"""Test secondary_mathfact resolver."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from agent.secondary_mathfact import resolve_mathfact
import pandas as pd

df = pd.read_parquet('/tmp/gsm8k_test.parquet')
indices = [1, 4, 16, 39, 40]

print('=== All 5 misrouted GSM8K problems ===')
for idx in indices:
    row = df.iloc[idx]
    q = row['question']
    result = resolve_mathfact('factual', q)
    status = 'FIXED' if result == 'math' else 'STILL FACTUAL'
    print(f'  [{idx+1}] {status}')

real_factual = [
    'What is the capital of France?',
    'Who discovered penicillin?',
    'When was the Declaration of Independence signed?',
    'What is the population of Japan?',
    'Define photosynthesis in simple terms.',
    'Explain the theory of relativity.',
    'Who was the 16th president of the United States?',
    'How many continents are there?',
    'What is the meaning of life?',
    'What is the distance from Earth to the Moon?',
    'What is the total population of Europe?',
]
print('\n=== Negative test: actual factual problems ===')
ok = 0
for q in real_factual:
    result = resolve_mathfact('factual', q)
    status = 'OK' if result == 'factual' else 'WRONG -> math'
    if result == 'factual':
        ok += 1
    print(f'  {status}: {q[:70]}')
print(f'\nNegative test: {ok}/{len(real_factual)} correct')
