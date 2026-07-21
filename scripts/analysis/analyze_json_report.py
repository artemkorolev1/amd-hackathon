#!/usr/bin/env python3
"""Produce final JSON summary report."""
import json, os
from collections import Counter, defaultdict
from statistics import mean, median

DATA_PATH = os.path.expanduser('/home/artem/dev/amd-hackathon/data/eval/training-v1.json')
with open(DATA_PATH) as f:
    data = json.load(f)

report = {}

# Basic stats
report['total_entries'] = len(data)
report['categories'] = sorted(set(e['category'] for e in data))
report['sources'] = sorted(set(e['source'] for e in data))
report['num_sources'] = len(report['sources'])

# 1. Quality
cat_counts = Counter(e['category'] for e in data)
diff_counts = Counter(e['difficulty'] for e in data)
source_counts = Counter(e['source'] for e in data)

# Category imbalance: coefficient of variation
cat_vals = list(cat_counts.values())
mean_cat = mean(cat_vals)
std_cat = (sum((v - mean_cat)**2 for v in cat_vals) / len(cat_vals))**0.5
report['category_balance'] = {
    'counts': dict(cat_counts),
    'cv': round(std_cat / mean_cat, 3),
    'note': f"code_debug is short by {int(200-114)} entries vs the 200-target (7.5% vs 13.2%)"
}

# Difficulty per category
diff_by_cat = {}
for cat in sorted(cat_counts):
    entries = [e for e in data if e['category'] == cat]
    c = Counter(e['difficulty'] for e in entries)
    diff_by_cat[cat] = dict(c)
report['difficulty_by_category'] = diff_by_cat

# Sources per category
src_by_cat = defaultdict(set)
for e in data:
    src_by_cat[e['category']].add(e['source'])
report['sources_per_category'] = {cat: sorted(list(srcs)) for cat, srcs in src_by_cat.items()}

# Single source categories
report['single_source_categories'] = [cat for cat, srcs in src_by_cat.items() if len(srcs) == 1]

# Answer format analysis
import re
def classify_answer(a):
    a = a.strip()
    if not a: return "empty"
    if '\n' in a and ('def ' in a or 'class ' in a or 'return ' in a or '    ' in a):
        return "code_block"
    if '\n' in a: return "multi_sentence"
    if a.startswith('{') and a.endswith('}'): return "dict_json"
    if a.startswith('[') and a.endswith(']'): return "list"
    if a.lower() in ('true','false','yes','no'): return "boolean"
    try:
        int(a); return "integer"
    except: pass
    try:
        float(a.replace(',','')); return "number"
    except: pass
    if re.match(r'^[0-4]\.\s', a): return "multi_choice_index"
    if a in ('POSITIVE','NEGATIVE'): return "binary_polarity"
    if len(a.split()) <= 3: return "short_phrase"
    return "sentence"

fmt_by_cat = defaultdict(Counter)
for e in data:
    fmt_by_cat[e['category']][classify_answer(e['expected_answer'])] += 1
report['answer_formats_per_category'] = {cat: dict(c) for cat, c in sorted(fmt_by_cat.items())}

# Prompt length stats
prompt_lens_by_cat = defaultdict(list)
for e in data:
    prompt_lens_by_cat[e['category']].append(len(e['prompt']))
report['prompt_lengths'] = {}
for cat in sorted(prompt_lens_by_cat):
    l = prompt_lens_by_cat[cat]
    report['prompt_lengths'][cat] = {
        'avg': round(mean(l), 1), 'median': int(median(l)),
        'min': min(l), 'max': max(l)
    }

# Answer length stats
ans_lens_by_cat = defaultdict(list)
for e in data:
    ans_lens_by_cat[e['category']].append(len(e['expected_answer'].strip()))
report['answer_lengths'] = {}
for cat in sorted(ans_lens_by_cat):
    l = ans_lens_by_cat[cat]
    report['answer_lengths'][cat] = {
        'avg': round(mean(l), 1), 'median': int(median(l)),
        'min': min(l), 'max': max(l)
    }

# Trivial answer ratio
trivial_count = sum(1 for e in data if classify_answer(e['expected_answer']) in ('binary_polarity', 'integer', 'multi_choice_index'))
report['trivial_answer_ratio'] = round(trivial_count / len(data) * 100, 1)

# Coverage gaps
richness_checks = {
    "Evol-Instruct mutations": ['rewrite', 'make it more difficult', 'harder version', 'evolved', 'add constraint'],
    "formatting constraints (JSON/XML/YAML)": ['json', 'xml', 'yaml', 'csv format', 'output as'],
    '"for a 10-year-old" / audience targeting': ['10-year-old', 'explain like', 'beginner', 'simplify', 'for a child'],
    "chain-of-thought / step-by-step": ['step by step', 'explain your reasoning', "let's think"],
}
all_text = ' '.join(e['prompt'] + ' ' + e['expected_answer'] for e in data).lower()
report['richness_features'] = {}
for feature, kws in richness_checks.items():
    found = sum(1 for kw in kws if kw in all_text)
    report['richness_features'][feature] = {'keyword_matches': found, 'present': found > 0}

# Overlap between code_gen and code_debug
code_gen_funcs = set()
code_debug_funcs = set()
for e in data:
    m = re.search(r'function `(\w+)', e['prompt'])
    if m:
        if e['category'] == 'code_gen': code_gen_funcs.add(m.group(1))
        if e['category'] == 'code_debug': code_debug_funcs.add(m.group(1))
    m2 = re.search(r'def\s+(\w+)', e['prompt'])
    if m2:
        if e['category'] == 'code_gen': code_gen_funcs.add(m2.group(1))
        if e['category'] == 'code_debug': code_debug_funcs.add(m2.group(1))
report['code_gen_debug_overlap'] = len(code_gen_funcs & code_debug_funcs)

# Exact duplicate check
from collections import defaultdict as dd
prompt_map = dd(list)
for i, e in enumerate(data):
    prompt_map[e['prompt'].strip()].append(e['task_id'])
dupes = {k: v for k, v in prompt_map.items() if len(v) > 1}
report['exact_duplicate_prompts'] = len(dupes)

print(json.dumps(report, indent=2))
