#!/usr/bin/env python3
"""Comprehensive analysis of training-v1.json dataset."""
import json, sys, os
from collections import Counter, defaultdict
from statistics import mean, median

DATA_PATH = os.path.expanduser('/home/artem/dev/amd-hackathon/data/eval/training-v1.json')
with open(DATA_PATH) as f:
    data = json.load(f)

print(f"Total entries: {len(data)}")
print(f"Fields: {list(data[0].keys())}")
print("=" * 72)

# ── 1. Quality Check ──
print("\n## 1. QUALITY CHECK")
empty_fields = []
for i, entry in enumerate(data):
    for key in ['prompt', 'expected_answer', 'category', 'source', 'difficulty', 'task_id']:
        val = entry.get(key)
        if val is None or (isinstance(val, str) and val.strip() == ''):
            empty_fields.append((i, key, val))
        elif key == 'category' and val not in ['factual','math','sentiment','summarization','ner','code_gen','code_debug','logic']:
            empty_fields.append((i, f"invalid_category={val}", val))
        elif key == 'difficulty' and val not in ['easy','medium','hard']:
            empty_fields.append((i, f"invalid_difficulty={val}", val))

print(f"Entries with empty/missing/invalid fields: {len(empty_fields)}")
for e in empty_fields[:20]:
    print(f"  Index {e[0]}: {e[1]} = {repr(e[2])}")

# Check for non-string types
bad_types = []
for i, entry in enumerate(data):
    for key in entry:
        if not isinstance(entry[key], str):
            bad_types.append((i, key, type(entry[key]).__name__, entry[key]))
print(f"Entries with non-string field values: {len(bad_types)}")
for b in bad_types[:10]:
    print(f"  Index {b[0]}: field '{b[1]}' is {b[2]} = {repr(b[3])}")

# ── 2. Category Balance ──
print("\n## 2. CATEGORY BALANCE")
cat_counts = Counter(e['category'] for e in data)
for cat, cnt in sorted(cat_counts.items()):
    print(f"  {cat}: {cnt} ({cnt/len(data)*100:.1f}%)")

# ── 3. Difficulty Distribution ──
print("\n## 3. DIFFICULTY DISTRIBUTION")
diff_counts = Counter(e['difficulty'] for e in data)
for d, cnt in sorted(diff_counts.items()):
    print(f"  {d}: {cnt} ({cnt/len(data)*100:.1f}%)")

print("\n  Per category:")
diff_by_cat = defaultdict(Counter)
for e in data:
    diff_by_cat[e['category']][e['difficulty']] += 1
for cat in sorted(diff_by_cat.keys()):
    c = diff_by_cat[cat]
    total = sum(c.values())
    parts = ", ".join(f"{d}={c[d]}" for d in ['easy','medium','hard'] if c[d])
    print(f"    {cat}: {parts} (total={total})")

# ── 4. Source Variety ──
print("\n## 4. SOURCE VARIETY")
source_counts = Counter(e['source'] for e in data)
print(f"Number of distinct source datasets: {len(source_counts)}")
for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
    print(f"  {src}: {cnt} ({cnt/len(data)*100:.1f}%)")

print("\n  Per-category source breakdown:")
source_by_cat = defaultdict(set)
for e in data:
    source_by_cat[e['category']].add(e['source'])
for cat in sorted(source_by_cat.keys()):
    srcs = source_by_cat[cat]
    print(f"    {cat}: {len(srcs)} source(s): {', '.join(sorted(srcs))}")
    if len(srcs) == 1:
        print(f"      *** Single-source only! ***")

# ── 5. Format Variety ──
print("\n## 5. FORMAT VARIETY (answer format categories)")
def classify_answer(answer):
    a = answer.strip()
    if not a:
        return "empty"
    if '\n' in a and ('def ' in a or 'class ' in a or 'return ' in a):
        return "code_block"
    if '\n' in a:
        return "multi_sentence"
    if a.startswith('{') and a.endswith('}'):
        return "dict"
    if a.startswith('[') and a.endswith(']'):
        return "list"
    if a.startswith('"') and a.endswith('"') or a.startswith("'") and a.endswith("'"):
        return "quoted_string"
    # multi-choice patterns (A, B, C, D)
    if re.match(r'^[A-D](\)|\.|\s)', a):
        return "multi_choice"
    # boolean
    if a.lower() in ('true', 'false', 'yes', 'no'):
        return "boolean"
    # number
    try:
        float(a.replace(',', ''))
        return "number"
    except ValueError:
        pass
    if len(a.split()) <= 3:
        return "short_phrase"
    return "sentence"

import re
fmt_counts = Counter()
fmt_by_cat = defaultdict(Counter)
for e in data:
    fmt = classify_answer(e['expected_answer'])
    fmt_counts[fmt] += 1
    fmt_by_cat[e['category']][fmt] += 1

for fmt, cnt in fmt_counts.most_common():
    print(f"  {fmt}: {cnt}")
print("\n  Per-category format variety:")
for cat in sorted(fmt_by_cat.keys()):
    fmts = fmt_by_cat[cat]
    print(f"    {cat}: {len(fmts)} format(s): {dict(fmts)}")
    if len(fmts) <= 2:
        print(f"      *** Low format variety! ***")

# ── 6. Prompt Length Distribution ──
print("\n## 6. PROMPT LENGTH DISTRIBUTION (characters)")
prompt_lens = [len(e['prompt']) for e in data]
print(f"  Overall: avg={mean(prompt_lens):.0f}, median={median(prompt_lens):.0f}, min={min(prompt_lens)}, max={max(prompt_lens)}")

print("\n  Per category:")
for cat in sorted(cat_counts.keys()):
    lens = [len(e['prompt']) for e in data if e['category'] == cat]
    print(f"    {cat}: avg={mean(lens):.0f}, median={median(lens):.0f}, min={min(lens)}, max={max(lens)}")

# ── 7. Answer Length Analysis ──
print("\n## 7. ANSWER LENGTH ANALYSIS (characters)")
ans_lens = [len(e['expected_answer']) for e in data]
print(f"  Overall: avg={mean(ans_lens):.0f}, median={median(ans_lens):.0f}, min={min(ans_lens)}, max={max(ans_lens)}")

# Very short answers
short_answers = [(i, e) for i, e in enumerate(data) if len(e['expected_answer'].strip()) <= 2]
print(f"  Answers <= 2 chars: {len(short_answers)}")
for idx, e in short_answers[:10]:
    print(f"    Index {idx}: prompt='{e['prompt'][:80]}...' answer='{e['expected_answer']}'")

# Empty-ish
empty_answers = [(i, e) for i, e in enumerate(data) if len(e['expected_answer'].strip()) == 0]
print(f"  Completely empty answers: {len(empty_answers)}")

print("\n  Per category:")
for cat in sorted(cat_counts.keys()):
    lens = [len(e['expected_answer']) for e in data if e['category'] == cat]
    print(f"    {cat}: avg={mean(lens):.0f}, median={median(lens):.0f}, min={min(lens)}, max={max(lens)}")

# ── 8. Redundancy ──
print("\n## 8. REDUNDANCY (near-duplicate prompts)")
from collections import defaultdict
prompt_texts = defaultdict(list)
for i, e in enumerate(data):
    prompt_texts[e['prompt'].strip()].append((i, e['task_id'], e['category']))
exact_dupes = {k: v for k, v in prompt_texts.items() if len(v) > 1}
print(f"Exact duplicate prompts (same text, diff task_id): {len(exact_dupes)}")
for prompt, entries in list(exact_dupes.items())[:10]:
    ids = [e[1] for e in entries]
    cats = [e[2] for e in entries]
    print(f"  Prompt len={len(prompt)}, {entries[0][0]}-{entries[-1][0]}: task_ids={ids}, cats={cats}")

# Check for source+prompt repeats (same source and same prompt)
src_prompt_map = defaultdict(list)
for i, e in enumerate(data):
    key = (e['source'], e['prompt'].strip())
    src_prompt_map[key].append((i, e['task_id']))
cross_source_dupes = {k: v for k, v in src_prompt_map.items() if len(v) > 1}
print(f"\nExact same prompt+source: {len(cross_source_dupes)}")
# If zero exact dupes, also check for high text similarity
if len(cross_source_dupes) == 0:
    print("  (All prompts unique within their source)")

# ── 9. Coverage Gaps ──
print("\n## 9. COVERAGE ANALYSIS")
# Check for richness features mentioned in plan
all_prompts = [e['prompt'] for e in data]
all_answers = [e['expected_answer'] for e in data]
all_text = ' '.join(all_prompts + all_answers).lower()

richness_checks = {
    "mixed signals / conflicting info": ['although', 'however', 'despite', 'nevertheless', 'on the other hand', 'but the'],
    "multi-hop reasoning": ['if...then', 'because', 'since', 'therefore', 'in order to', 'as a result'],
    "Evol-Instruct mutations": ['rewrite', 'make it more difficult', 'harder version', 'complex version', 'evolved', 'add constraint'],
    "formatting constraints": ['json', 'xml', 'yaml', 'csv format', 'output as', 'in the format of', 'following format'],
    "\"for a 10-year-old\" audience changes": ['10-year-old', 'explain like', 'simple terms', 'beginner', 'layman', 'simplify'],
    "audience targeting": ['for a child', 'for beginners', 'explain to', 'audience'],
    "chain-of-thought prompting": ['step by step', 'think step', 'reason step', 'let\'s think', 'explain your reasoning'],
    "negative constraints": ['without using', 'do not use', 'not allowed', 'cannot use', 'don\'t mention'],
}

print("  Checking for planned richness features:")
for feature, keywords in richness_checks.items():
    found = sum(1 for kw in keywords if kw in all_text)
    sample_matches = []
    for kw in keywords:
        for p in all_prompts:
            if kw in p.lower():
                sample_matches.append(kw)
                break
        if len(sample_matches) > 5:
            break
    status = "✓ " if found >= 2 else "✗ "
    print(f"    {status}{feature}: {found}/{len(keywords)} keyword groups found")
    if sample_matches:
        print(f"       examples: {sample_matches[:4]}")

# ── 10. Sample entries per category ──
print("\n## 10. SAMPLE ENTRIES PER CATEGORY (manual review)")
for cat in sorted(cat_counts.keys()):
    entries = [e for e in data if e['category'] == cat]
    print(f"\n  --- {cat} ({len(entries)} entries) ---")
    for e in entries[:3]:
        prompt_preview = e['prompt'][:120].replace('\n', ' | ')
        answer_preview = e['expected_answer'][:80].replace('\n', ' | ')
        print(f"    Task: {e['task_id']} | Diff: {e['difficulty']} | Src: {e['source']}")
        print(f"    Prompt: {prompt_preview}...")
        print(f"    Answer: {answer_preview}...")
        print()

# ── Summary Statistics ──
print("\n" + "=" * 72)
print("## SUMMARY STATISTICS")
print(f"Total entries: {len(data)}")
print(f"Categories: {len(cat_counts)}")
print(f"Sources: {len(source_counts)}")
print(f"Empty/invalid fields: {len(empty_fields)}")
print(f"Exact duplicate prompts: {len(exact_dupes)}")
print(f"Min answer len: {min(ans_lens)} (empty answers: {len(empty_answers)})")
print(f"Min prompt len: {min(prompt_lens)}")
print(f"Easy: {diff_counts.get('easy',0)}, Medium: {diff_counts.get('medium',0)}, Hard: {diff_counts.get('hard',0)}")
print("=" * 72)
