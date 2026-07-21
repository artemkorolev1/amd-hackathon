#!/usr/bin/env python3
"""Debug: compare math vs factual scores for misrouted GSM8K problems."""
import sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the scoring functions directly
from agent.category_filter import (
    _score_math, _score_factual, _score_code_debug, _score_code_gen,
    _score_logic, _score_ner, _score_sentiment, _score_summarization,
)

import pandas as pd

df = pd.read_parquet('/tmp/gsm8k_test.parquet')
problem_indices = [1, 4, 16, 39, 40]  # 0-indexed for the 5 misrouted

scorers = {
    'math': _score_math,
    'factual': _score_factual,
    'code_gen': _score_code_gen,
    'code_debug': _score_code_debug,
    'logic': _score_logic,
    'ner': _score_ner,
    'sentiment': _score_sentiment,
    'summarization': _score_summarization,
}

for idx in problem_indices:
    row = df.iloc[idx]
    question = row['question']
    
    print(f'Q{idx+1}: {question[:80]}...')
    scores = {name: scorer(question) for name, scorer in scorers.items()}
    winner = max(scores, key=scores.get)
    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])[:5]
    
    print(f'  Winner: {winner} (score={scores[winner]:.1f})')
    for name, score in sorted_scores[:5]:
        print(f'    {name}: {score:.1f}')
    
    # Check math score details
    lower = question.lower()
    nums = re.findall(r'\d+(?:\.\d+)?', question)
    print(f'  Digits found: {nums}')
    print()
