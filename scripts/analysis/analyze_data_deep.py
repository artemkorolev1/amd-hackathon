#!/usr/bin/env python3
"""Deeper analysis on specific findings."""
import json, os
from collections import Counter

DATA_PATH = os.path.expanduser('/home/artem/dev/amd-hackathon/data/eval/training-v1.json')
with open(DATA_PATH) as f:
    data = json.load(f)

# ── Sentiment answers are all len 8? ──
sent_entries = [e for e in data if e['category'] == 'sentiment']
answer_vals = Counter(e['expected_answer'].strip() for e in sent_entries)
print("=== SENTIMENT ANSWER VALUES ===")
for v, cnt in answer_vals.most_common(10):
    print(f"  {repr(v)}: {cnt}")

# ── Math answer distribution ──
math_entries = [e for e in data if e['category'] == 'math']
math_answers = Counter(e['expected_answer'].strip() for e in math_entries)
print("\n=== MATH ANSWER SAMPLE (first 30) ===")
for e in math_entries[:30]:
    print(f"  {e['task_id']}: '{e['expected_answer'].strip()}' (len={len(e['expected_answer'].strip())})")
print(f"  ... ({len(math_entries)} total)")

# ── Factual answer distribution ──
fact_entries = [e for e in data if e['category'] == 'factual']
fact_lens = [len(e['expected_answer'].strip()) for e in fact_entries]
print("\n=== FACTUAL ANSWER STATS ===")
print(f"  Avg length: {sum(fact_lens)/len(fact_lens):.0f}")
print(f"  Answer length <= 5: {sum(1 for l in fact_lens if l <= 5)}")
print(f"  Answer length <= 10: {sum(1 for l in fact_lens if l <= 10)}")
fact_multi_choice = [e for e in fact_entries if '\nChoices:' in e['prompt'] or e['expected_answer'].strip().startswith(('A)', 'B)', 'C)', 'D)'))]
print(f"  Multi-choice style: {len(fact_multi_choice)}")

# Check mmlu answers specifically
mmlu_entries = [e for e in fact_entries if e['source'] == 'mmlu']
mmlu_answers = [e['expected_answer'].strip() for e in mmlu_entries]
print(f"\n  MMLU answers are just single words/phrases:")
for a in mmlu_answers[:10]:
    print(f"    {repr(a)}")

# ── Check logic dict format ──
logic_entries = [e for e in data if e['category'] == 'logic']
logic_dict_entries = [e for e in logic_entries if 'dict' in e['expected_answer'][:10] or e['expected_answer'].strip().startswith('{')]
print(f"\n=== LOGIC: dict-format entries = {len(logic_dict_entries)} out of {len(logic_entries)}")
# Show one
if logic_dict_entries:
    print(f"  Sample dict answer: {logic_dict_entries[0]['expected_answer'][:200]}")

# Check logiqa entries (logic non-dict)
logiqa_entries = [e for e in logic_entries if e['source'] == 'logiqa']
print(f"  LogiQA entries: {len(logiqa_entries)}")
print(f"  LogiQA sample answer: {logiqa_entries[0]['expected_answer'][:200] if logiqa_entries else 'N/A'}")

# Zebra logic bench
zebra_entries = [e for e in logic_entries if e['source'] == 'zebra_logic_bench']
print(f"  Zebra Logic Bench entries: {len(zebra_entries)}")
if zebra_entries:
    print(f"  Zebra sample answer: {zebra_entries[0]['expected_answer'][:200]}")

# ── Redundancy via fuzzy matching ──
print("\n=== FUZZY REDUNDANCY CHECK ===")
# Group by source, check if prompts share same first 50 chars
from collections import defaultdict
for src in set(e['source'] for e in data):
    entries = [e for e in data if e['source'] == src]
    prefix_map = defaultdict(list)
    for i, e in enumerate(entries):
        prefix = e['prompt'].strip()[:80]
        prefix_map[prefix].append((i, e['task_id']))
    multi = {k: v for k, v in prefix_map.items() if len(v) > 1}
    if multi:
        print(f"  {src}: {len(multi)} prefix collisions (same first 80 chars)")
        for pref, hits in list(multi.items())[:2]:
            print(f"    '{pref}...' -> {len(hits)} entries")

# ── Summarization answer lengths ──
summ_entries = [e for e in data if e['category'] == 'summarization']
summ_word_counts = [len(e['expected_answer'].split()) for e in summ_entries]
avg_wc = sum(summ_word_counts) / len(summ_word_counts)
print(f"\n=== SUMMARIZATION ===")
print(f"  Avg answer word count: {avg_wc:.1f}")
print(f"  Min: {min(summ_word_counts)}, Max: {max(summ_word_counts)}")

# ── code_gen vs code_debug ──
print(f"\n=== CODE CATEGORIES ===")
code_gen = [e for e in data if e['category'] == 'code_gen']
code_debug = [e for e in data if e['category'] == 'code_debug']
print(f"  code_gen: {len(code_gen)} entries from {set(e['source'] for e in code_gen)}")
print(f"  code_debug: {len(code_debug)} entries from {set(e['source'] for e in code_debug)}")
# code_debug all "hard"?
debug_diffs = Counter(e['difficulty'] for e in code_debug)
print(f"  code_debug difficulty distribution: {dict(debug_diffs)}")

# Check if code_debug prompts are unique or if they mirror code_gen
code_gen_first_lines = set()
for e in code_gen:
    # Get just the function signature or first line of task description
    lines = e['prompt'].strip().split('\n')
    for line in lines:
        if 'def ' in line:
            code_gen_first_lines.add(line.strip()[:50])
            break

debug_first_lines = set()
for e in code_debug:
    lines = e['prompt'].strip().split('\n')
    for line in lines:
        if 'def ' in line:
            debug_first_lines.add(line.strip()[:50])
            break

overlap = code_gen_first_lines & debug_first_lines
print(f"  Overlapping function definitions (code_gen x code_debug): {len(overlap)} out of {len(debug_first_lines)} debug funcs")

# ── Difficulty balance assessment ──
print("\n=== DIFFICULTY BALANCE ===")
# Some categories are 100% one difficulty
for cat in ['code_debug', 'summarization', 'logic']:
    entries = [e for e in data if e['category'] == cat]
    diffs = Counter(e['difficulty'] for e in entries)
    print(f"  {cat}: 100% {list(diffs.keys())[0] if len(diffs)==1 else 'mixed'} -> {dict(diffs)}")
