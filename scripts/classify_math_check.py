#!/usr/bin/env python3
"""Classify all GSM8K problems: how many are correctly identified as math?"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from agent.classifier import classify
import pandas as pd

df = pd.read_parquet('/tmp/gsm8k_test.parquet')
N = 50

correct_math = 0
misrouted = []

for i in range(N):
    row = df.iloc[i]
    question = row['question']
    expected = row['answer'].split('####')[-1].strip() if '####' in row['answer'] else row['answer'].strip()
    
    cat, method, conf = classify(question)
    
    if cat != 'math':
        misrouted.append((i+1, cat, question[:90], expected))
    else:
        correct_math += 1

print(f'=== GSM8K Classification Analysis (50 problems) ===')
print(f'Classified as math: {correct_math}/50 ({correct_math/50*100:.1f}%)')
print(f'Misrouted (classified as other): {len(misrouted)}')

print(f'\nMisrouted breakdown:')
from collections import Counter
cats = Counter(e[1] for e in misrouted)
for cat, count in cats.most_common():
    print(f'  {cat}: {count}')

print(f'\nDetails:')
for num, cat, q, exp in misrouted:
    print(f'  [{num}] cat={cat} | expected={exp:>8}')
    print(f'         Q: {q}')
