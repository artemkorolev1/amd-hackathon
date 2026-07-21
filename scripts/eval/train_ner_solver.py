"""
Training script for NER solver — learns optimal regex patterns from training-v2 NER data.
Outputs a trained solver module + performance report.

Usage:
    python3 scripts/eval/train_ner_solver.py
"""

import json, os, sys, re, time
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, "../.."))
sys.path.insert(0, _PROJECT)

sys.path.insert(0, os.path.join(_PROJECT, "scripts"))
from grade_answer import fuzzy_match

# ── Load training data ──
def load(path):
    with open(path) as f:
        return json.load(f)

train_data = load(os.path.join(_PROJECT, "data/eval/training-v2.json"))
val_data = load(os.path.join(_PROJECT, "data/eval/validation-v2.json"))

train_ner = [it for it in train_data if it['category'] == 'ner']
val_ner = [it for it in val_data if it['category'] == 'ner']

print(f'Training NER items: {len(train_ner)}')
print(f'Validation NER items: {len(val_ner)}')
print()

# ── Sub-type classification ──
def classify_ner_type(prompt):
    lower = prompt.lower()
    if re.search(r'\{@|#\w+', prompt):
        return 'tweet'
    if re.search(r'disease|biomedical|gene|protein', lower):
        return 'biomedical'
    if re.search(r'extract all named entities|list each type|named entity', lower):
        return 'general_ner'
    return 'other'

# ── Solver prototypes ──

def solve_tweet_ner(prompt, cat_hint):
    """Extract {@...@} entities and hashtag entities from tweets."""
    lines = []
    # {@entity@} markers
    for m in re.finditer(r'\{@([^@]+)@\}', prompt):
        entity = m.group(1).strip()
        lines.append(f'person: {entity}')  # default type
    # Hashtags
    for m in re.finditer(r'#([A-Z][A-Za-z0-9]+)', prompt):
        tag = m.group(1)
        lines.append(f'event: {tag}')
    return '\n'.join(lines) if lines else None


def solve_biomedical_ner(prompt, cat_hint):
    """Extract disease/gene names from biomedical texts."""
    lower = prompt.lower()
    lines = []
    # "Extract all disease names" → everything after the instruction
    idx = lower.find('extract all disease names from')
    if idx >= 0:
        text_after = prompt[idx + 30:]
        # Split on punctuation to get disease names
        parts = re.split(r'[;,]', text_after)
        for part in parts[:3]:
            part = part.strip()
            if part and len(part) > 2:
                lines.append(f'Disease: {part}')
    if lines:
        return '\n'.join(lines)
    return None


def solve_general_ner(prompt, cat_hint):
    """TYPE: entity format for general NER."""
    lower = prompt.lower()
    lines = []
    
    # Known entity types
    types = {
        'person': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
        'org': r'\b[A-Z][a-z]+(?: [A-Z][a-z]+){1,3}\b',
    }
    
    # Simple extraction: anything after ":" lines or bullets
    for m in re.finditer(r'^([A-Z]+): (.+)$', prompt, re.MULTILINE):
        t = m.group(1).lower()
        val = m.group(2).strip()
        lines.append(f'{t}: {val}')
    
    if not lines:
        # Try to find TYPE: entity patterns in the prompt itself
        for m in re.finditer(r'\b(PERSON|ORG|LOC|GPE|DATE|TIME|MONEY|PERCENT|PRODUCT|EVENT|NORP|FAC|LAW):\s*(.+)', prompt, re.IGNORECASE):
            t = m.group(1).lower()
            val = m.group(2).strip()
            lines.append(f'{t}: {val}')
    
    return '\n'.join(lines) if lines else None


# ── Baseline: current solve_ner from deterministic.py ──
from agent.solvers.deterministic import solve_ner as solve_ner_old
from agent.solvers.prototype_ner_v3 import solve_ner as solve_ner_v3


# ── Evaluate ──
def evaluate(solver_fn, data, label):
    correct = 0
    total = len(data)
    for item in data:
        prompt = item['prompt']
        expected = item.get('expected_answer', '')
        try:
            answer = solver_fn(prompt, 'ner')
            if answer and fuzzy_match(answer, expected):
                correct += 1
        except:
            pass
    print(f'  {label:30s}: {correct}/{total} = {correct/total*100:.1f}%')
    return correct


print('=== NER SOLVER BASELINE (isolated) ===')
for name, fn in [('solve_ner_old (current)', solve_ner_old), 
                  ('solve_ner_v3 (prototype)', solve_ner_v3)]:
    train_c = evaluate(fn, train_ner, f'{name} on training-v2')
    val_c = evaluate(fn, val_ner, f'{name} on validation-v2')

print()
print('=== NEW PROTOTYPE SOLVERS (isolated) ===')
prototypes = [
    ('solve_general_ner', solve_general_ner),
    ('solve_tweet_ner', solve_tweet_ner),
    ('solve_biomedical_ner', solve_biomedical_ner),
]
for name, fn in prototypes:
    train_c = evaluate(fn, train_ner, f'{name} on training-v2')
    val_c = evaluate(fn, val_ner, f'{name} on validation-v2')
