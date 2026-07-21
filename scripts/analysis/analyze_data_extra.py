#!/usr/bin/env python3
"""Extra checks on content overlap and structure."""
import json, os, re
from collections import Counter

DATA_PATH = os.path.expanduser('/home/artem/dev/amd-hackathon/data/eval/training-v1.json')
with open(DATA_PATH) as f:
    data = json.load(f)

# ── Check if code_debug prompts are derived from code_gen (HumanEval) ──
# code_debug all from humanevalpack (114 entries)
# code_gen from humaneval (48) + mbpp (152)
code_gen_humaneval = [e for e in data if e['source'] == 'humaneval']
code_debug = [e for e in data if e['category'] == 'code_debug']

# Extract function names from code_gen
gen_funcs = set()
for e in code_gen_humaneval:
    m = re.search(r'def\s+(\w+)', e['prompt'])
    if m:
        gen_funcs.add(m.group(1))

# Extract function names from code_debug prompts
debug_funcs = set()
for e in code_debug:
    # The prompt has format: "Fix the bug in this Python function:\n\n<buggy_code>\n\nTask: Write a Python function `func_name`..."
    m = re.search(r'function `(\w+)', e['prompt'])
    if m:
        debug_funcs.add(m.group(1))
    else:
        m2 = re.search(r'def\s+(\w+)', e['prompt'])
        if m2:
            debug_funcs.add(m2.group(1))

print("=== CODE CATEGORY OVERLAP ===")
print(f"  code_gen (humaneval): {len(code_gen_humaneval)} entries, {len(gen_funcs)} unique functions")
print(f"  code_debug (humanevalpack): {len(code_debug)} entries, {len(debug_funcs)} unique functions")
overlap = gen_funcs & debug_funcs
print(f"  Overlapping function names: {len(overlap)}")
if overlap:
    print(f"  Functions in both: {sorted(overlap)[:10]}")

# Check if code_debug prompts contain the SAME task descriptions as code_gen
for fn in list(overlap)[:3]:
    gen_entry = next(e for e in code_gen_humaneval if fn in e['prompt'])
    debug_entry = next(e for e in code_debug if fn in e['prompt'])
    print(f"\n  Function '{fn}':")
    print(f"    code_gen prompt first 100 chars: {gen_entry['prompt'][:100]}")
    print(f"    code_debug prompt first 100 chars: {debug_entry['prompt'][:100]}")
    print(f"    code_gen answer first 80 chars: {gen_entry['expected_answer'][:80]}")
    print(f"    code_debug answer first 80 chars: {debug_entry['expected_answer'][:80]}")

# ── Check zebra logic bench format variety ──
zebra = [e for e in data if e['source'] == 'zebra_logic_bench']
print(f"\n=== ZEBRA LOGIC BENCH ===")
puzzle_sizes = Counter()
for e in zebra:
    m = re.search(r'(\d+) houses?', e['prompt'])
    if m:
        puzzle_sizes[f"{m.group(1)} houses"] += 1
print(f"  Puzzle size distribution: {dict(puzzle_sizes)}")

answer_previews = Counter()
for e in zebra[:20]:
    a = e['expected_answer'][:50]
    answer_previews[a] += 1
print(f"  Unique answer previews (first 50 chars) in first 20: {len(answer_previews)}")
print(f"  Sample answer: {zebra[0]['expected_answer'][:200]}")

# ── How much of the dataset is just "Positve/Negative" or just "number"? ──
print("\n=== TRIVIAL ANSWER RATIO ===")
all_answers = [e['expected_answer'].strip() for e in data]
# Single word classification (POSITIVE/NEGATIVE) - 200 entries
polarity_count = sum(1 for a in all_answers if a in ('POSITIVE', 'NEGATIVE'))
print(f"  Binary polarity (POSITIVE/NEGATIVE): {polarity_count} ({polarity_count/len(data)*100:.1f}%)")
# Pure number
number_count = sum(1 for a in all_answers if re.match(r'^-?\d+$', a))
print(f"  Pure integer answers: {number_count} ({number_count/len(data)*100:.1f}%)")
# Multi-choice single letter/number
mc_count = sum(1 for a in all_answers if re.match(r'^[0-4]\.\s', a))
print(f"  Multi-choice index (0./1./etc): {mc_count} ({mc_count/len(data)*100:.1f}%)")
# Combined trivial
trivial = polarity_count + number_count + mc_count
print(f"  Total 'trivial' format answers: {trivial} ({trivial/len(data)*100:.1f}%)")

# ── How many entries are NON-trivial (multi-sentence, code) ──
non_trivial = sum(1 for a in all_answers if '\n' in a or len(a) > 80)
print(f"  Multi-line or long answers (>80 chars): {non_trivial} ({non_trivial/len(data)*100:.1f}%)")
