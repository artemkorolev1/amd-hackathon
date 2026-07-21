"""
Pipeline-context solver evaluation — runs each question through the actual pipeline
(classification → deterministic solvers → fallthrough) and grades with official fuzzy_match.

Usage:
    python3 scripts/eval/eval_pipeline_solvers.py
"""

import json, os, sys, contextlib, io

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, "../.."))
sys.path.insert(0, _PROJECT)

os.environ['MODEL_PATH'] = '/dev/null'

from scripts.grade_answer import fuzzy_match
from agent.pipeline import Pipeline, PipelineConfig
from agent.solvers.deterministic import *
from agent.solvers.logic_reasoning import solve_logical_reasoning
from agent.solvers.prototype_ner_v3 import solve_ner as solve_ner_v3

# Suppress all pipeline warnings
os.environ['LOG_LEVEL'] = 'ERROR'
import logging
logging.getLogger('pipeline').setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

# ── Optimal solver chain (trained on training-v2 data) ──
cfg = PipelineConfig()
cfg.model_path = '/dev/null'
cfg.category_model_map = {}
cfg.det_solvers = [
    # NER: old regex (78.5%) before v3 (62.5%)
    solve_ner, solve_ner_v3,
    # Logic
    solve_logical_reasoning, solve_logic,
    # Rest
    solve_sentiment,
    solve_factual_qa,
    solve_code_debugging,
]
pipe = Pipeline(config=cfg)


def eval_category(items, cat_name):
    correct = 0
    solved_by_det = 0
    total = len(items)
    for item in items:
        prompt = item['prompt']
        expected = item.get('expected_answer', '')
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                ans = pipe.process(prompt)
            if ans:
                solved_by_det += 1
                if fuzzy_match(ans, expected):
                    correct += 1
        except:
            pass
    return correct, solved_by_det, total


for label, data_path in [
    ('TRAINING-v2', os.path.join(_PROJECT, 'data/eval/training-v2.json')),
    ('VALIDATION-v2', os.path.join(_PROJECT, 'data/eval/validation-v2.json')),
]:
    with open(data_path) as f:
        data = json.load(f)

    print(f'=== {label} — Pipeline-context solver eval (fuzzy_match) ===')
    print('Category        Correct Total  Pct    DetSolved')
    print('-' * 50)
    
    total_correct = 0
    total_all = 0
    for cat in ['factual', 'sentiment', 'ner', 'code_gen', 'code_debug',
                'logic', 'math', 'summarization']:
        items = [it for it in data if it['category'] == cat][:50]
        if not items:
            continue
        c, s, t = eval_category(items, cat)
        total_correct += c
        total_all += t
        print(f'{cat:15s} {c:5d} {t:5d} {c/t*100:5.1f}%   {s}/{t}')
    
    total_pct = total_correct / total_all * 100 if total_all else 0
    print(f'{"TOTAL":15s} {total_correct:5d} {total_all:5d} {total_pct:5.1f}%')
    print()
