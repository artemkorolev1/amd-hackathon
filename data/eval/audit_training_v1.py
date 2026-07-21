#!/usr/bin/env python3
"""Deep quality audit of training-v1.json dataset."""
import json
import re
from collections import Counter, defaultdict

with open('/home/artem/dev/amd-hackathon/data/eval/training-v1.json') as f:
    data = json.load(f)

print(f"Total entries: {len(data)}")

# ──────────────────────────────────────────────
# 1. Struct checks
# ──────────────────────────────────────────────
missing_fields = []
for i, e in enumerate(data):
    for k in ('category', 'prompt', 'expected_answer', 'source', 'difficulty', 'task_id'):
        if k not in e:
            missing_fields.append((i, k))
print(f"\nEntries with missing fields: {len(missing_fields)}")
for i, k in missing_fields[:20]:
    print(f"  Entry {i}: missing '{k}'")

# ──────────────────────────────────────────────
# 2. Category-level sampling and inspection
# ──────────────────────────────────────────────
problems = []  # list of dicts with entry index, category, problem_type, evidence

def add_problem(idx, cat, ptype, evidence):
    problems.append({
        'idx': idx,
        'category': cat,
        'problem_type': ptype,
        'evidence': evidence[:200],  # truncate long evidence
        'prompt_snippet': data[idx]['prompt'][:150],
        'expected_answer_snippet': data[idx]['expected_answer'][:150],
        'source': data[idx].get('source',''),
        'difficulty': data[idx].get('difficulty',''),
        'task_id': data[idx].get('task_id',''),
    })

# --- 2a. Check for truncated entries (summarization: 1200 char, imdb: 800 char)
for i, e in enumerate(data):
    prompt = e['prompt']
    ea = e['expected_answer']
    cat = e['category']
    src = e.get('source','')
    
    # Prompt truncation markers
    if len(prompt) >= 1190 and cat == 'summarization':
        # Check if it ends mid-sentence (no period, no newline)
        last_char = prompt.rstrip()[-1] if prompt.rstrip() else ''
        if last_char not in '.!?)\n"':
            add_problem(i, cat, 'truncated_prompt',
                        f"Prompt is {len(prompt)} chars, ends abruptly with '{last_char}' (likely 1200-char truncation)")
    
    if len(prompt) >= 790 and src == 'imdb':
        last_char = prompt.rstrip()[-1] if prompt.rstrip() else ''
        if last_char not in '.!?)\n"':
            add_problem(i, cat, 'truncated_prompt',
                        f"Prompt is {len(prompt)} chars (IMDB 800-char truncation marker)")
    
    # Expected answer truncation
    if len(ea) >= 1190 and cat == 'summarization':
        last_char = ea.rstrip()[-1] if ea.rstrip() else ''
        if last_char not in '.!?)\n"':
            add_problem(i, cat, 'truncated_expected_answer',
                        f"Expected answer is {len(ea)} chars, ends abruptly with '{last_char}'")
    
    if len(ea) >= 790 and src == 'imdb':
        last_char = ea.rstrip()[-1] if ea.rstrip() else ''
        if last_char not in '.!?)\n"':
            add_problem(i, cat, 'truncated_expected_answer',
                        f"Expected answer is {len(ea)} chars (IMDB 800-char truncation marker)")

# --- 2b. Template artifacts
for i, e in enumerate(data):
    prompt = e['prompt']
    ea = e['expected_answer']
    cat = e['category']
    src = e.get('source','')
    
    # GSM8K "####" artifact in expected_answer
    if '####' in ea and src == 'gsm8k':
        add_problem(i, cat, 'template_artifact_gsm8k',
                    f"Expected answer contains '####' (GSM8K template artifact): '{ea[:100]}'")
    
    # BIO tag indices in NER output
    if cat == 'ner':
        # Check if expected_answer looks like BIO tags with indices
        if re.search(r'B-[A-Z]+|I-[A-Z]+|O[^a-z]', ea):
            pass  # BIO tags are correct for NER
        # But check for index numbers mixed in
        if re.search(r'\b\d+\s*[:\]]', ea[:200]):
            add_problem(i, cat, 'template_artifact_ner_indices',
                        f"NER expected answer seems to contain indices: '{ea[:100]}'")
    
    # JSON-formatted puzzle solutions in logic
    if cat == 'logic' and '{' in ea and '"' in ea:
        add_problem(i, cat, 'template_artifact_json_in_logic',
                    f"Logic expected answer contains JSON: '{ea[:100]}'")
    
    # Code: check for leftover docstring/formatter artifacts
    if cat in ('code_gen', 'code_debug'):
        if '>>> ' not in prompt and '```' not in prompt:
            pass  # Not all code prompts have these
        # Check if expected_answer has markdown fences
        if '```' in ea:
            add_problem(i, cat, 'template_artifact_markdown_code',
                        f"Expected answer contains markdown code fences: '{ea[:100]}'")

# --- 2c. Meaningless expected answers (single token answers that don't answer)
meaningless_patterns = [
    r'^\s*0\s*$', r'^\s*1\s*$', r'^\s*2\s*$', r'^\s*3\s*$', r'^\s*4\s*$', r'^\s*5\s*$',
    r'^\s*true\s*$', r'^\s*false\s*$',
    r'^\s*yes\s*$', r'^\s*no\s*$',
    r'^\s*positive\s*$', r'^\s*negative\s*$',
    r'^\s*neutral\s*$',
]
for i, e in enumerate(data):
    ea = e['expected_answer'].strip()
    cat = e['category']
    
    # Single token numeric/simple answers
    for pat in meaningless_patterns:
        if re.match(pat, ea, re.IGNORECASE) and cat not in ('sentiment',):
            # For sentiment, "positive"/"negative" is appropriate
            if cat == 'sentiment' and re.match(r'^\s*(positive|negative|neutral)\s*$', ea, re.IGNORECASE):
                continue
            add_problem(i, cat, 'meaningless_expected_answer',
                        f"Expected answer is just '{ea}' which doesn't answer the question")
            break
    
    # Very short answers for categories that need full answers
    if len(ea) < 5 and cat in ('math', 'logic', 'factual', 'summarization', 'code_gen'):
        if re.match(r'^\d+$', ea):  # numeric answers are valid for math
            continue
        if re.match(r'^[A-Z]\)', ea):  # multiple choice letter
            continue
        if cat != 'ner' and len(ea) < 3:
            add_problem(i, cat, 'suspiciously_short_expected_answer',
                        f"Expected answer is only {len(ea)} chars: '{ea}'")

# --- 2d. Prompt doesn't match category
# We'll check by looking for keywords that indicate a mismatch
category_keywords = {
    'sentiment': ['sentiment', 'positive', 'negative', 'feeling', 'opinion', 'review', 'emotion', 'atta'],
    'ner': ['entity', 'name', 'person', 'organization', 'location', 'ner', 'named entity'],
    'math': ['solve', 'calculate', 'compute', 'how many', 'what is', 'equation', 'number', 'sum', 'difference', 'product'],
    'code_gen': ['write a function', 'implement', 'python function', 'code', 'program', 'def '],
    'code_debug': ['fix', 'bug', 'debug', 'error', 'incorrect'],
    'summarization': ['summarize', 'summary', 'article', 'news', 'TL;DR', 'condense'],
    'factual': ['question', 'answer', 'who', 'what', 'when', 'where', 'why', 'how'],
    'logic': ['logic', 'reason', 'deduce', 'infer', 'puzzle', 'if.*then'],
}

for i, e in enumerate(data):
    prompt_lower = e['prompt'].lower()
    cat = e['category']
    
    # Check if math prompt doesn't contain any math-like keywords
    if cat == 'math':
        if not any(kw in prompt_lower for kw in category_keywords['math']):
            # Check if it's just a plain number question
            if not re.search(r'\d+', e['prompt']):
                add_problem(i, cat, 'category_mismatch',
                            f"Math entry has no math keywords or numbers in prompt")
    
    # Check summarization that doesn't ask to summarize
    if cat == 'summarization':
        if not any(kw in prompt_lower for kw in category_keywords['summarization']):
            add_problem(i, cat, 'category_mismatch',
                        f"Summarization prompt lacks summary keywords")

# --- 2e. Bad expected answers for math
for i, e in enumerate(data):
    if e['category'] == 'math':
        ea = e['expected_answer'].strip()
        prompt = e['prompt']
        # Check if expected answer is obviously not a number when math problem asks for one
        if re.search(r'\b(?:how many|what is|calculate|find|compute|determine|sum of|difference)\b', prompt.lower()):
            if not re.search(r'\d', ea) and len(ea) > 0:
                # Could be a word answer - flag for review
                if not re.match(r'^[A-Z]\)', ea) and len(ea) < 100:
                    add_problem(i, 'math', 'non_numeric_math_answer',
                                f"Math problem asks for computation but expected answer is non-numeric: '{ea[:80]}'")

# --- 2f. Code: check if solution is compilable
# (basic syntax check)
for i, e in enumerate(data):
    if e['category'] in ('code_gen', 'code_debug'):
        ea = e['expected_answer']
        if 'def ' not in ea and 'function ' not in ea and 'class ' not in ea and 'lambda' not in ea:
            if len(ea) > 10 and not ea.strip().startswith('```'):
                add_problem(i, e['category'], 'code_not_function',
                            f"Expected answer doesn't look like code (no def/function): '{ea[:80]}'")

# --- 2g. Duplicate/near-duplicate prompts
prompt_lookup = defaultdict(list)
for i, e in enumerate(data):
    # Normalize whitespace for comparison
    norm = re.sub(r'\s+', ' ', e['prompt']).strip()
    prompt_lookup[norm].append(i)

dup_count = 0
for norm, indices in prompt_lookup.items():
    if len(indices) > 1:
        dup_count += 1
        cats = [data[j]['category'] for j in indices]
        add_problem(indices[0], cats[0], 'duplicate_prompt',
                    f"Exact same prompt appears {len(indices)}x at indices {indices}, categories: {cats}")
        # Only flag the first occurrence

print(f"\n=== DUPLICATE PROMPTS: {dup_count} sets ===")
for norm, indices in prompt_lookup.items():
    if len(indices) > 1:
        print(f"  Indices {indices}: categories={[data[j]['category'] for j in indices]}")
        print(f"    Prompt: {norm[:100]}...")

# --- 2h. NER label quality check
for i, e in enumerate(data):
    if e['category'] == 'ner':
        ea = e['expected_answer']
        # Check for invalid BIO tags
        if not re.search(r'B-[A-Z]+|I-[A-Z]+', ea) and not re.search(r'\bO\b', ea):
            add_problem(i, 'ner', 'ner_no_bio_tags',
                        f"NER expected answer has no BIO tags: '{ea[:100]}'")

# ──────────────────────────────────────────────
# 3. Summary statistics
# ──────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"AUDIT SUMMARY")
print(f"{'='*80}")
print(f"Total entries: {len(data)}")
print(f"Total problems flagged: {len(problems)}")

by_cat = Counter(p['category'] for p in problems)
by_type = Counter(p['problem_type'] for p in problems)

print(f"\nProblems by category:")
for cat, cnt in by_cat.most_common():
    print(f"  {cat}: {cnt}")

print(f"\nProblems by type:")
for ptype, cnt in by_type.most_common():
    print(f"  {ptype}: {cnt}")

# Worst category
if by_cat:
    worst_cat = by_cat.most_common(1)[0]
    print(f"\nWorst-performing category: {worst_cat[0]} ({worst_cat[1]} problems)")

# Top 5 worst individual entries
print(f"\nTop 5 worst individual entries (most issues):")
entry_problem_count = Counter(p['idx'] for p in problems)
for idx, cnt in entry_problem_count.most_common(5):
    e = data[idx]
    print(f"\n{'─'*80}")
    print(f"  INDEX: {idx} | CATEGORY: {e['category']} | SOURCE: {e.get('source','')} | DIFFICULTY: {e.get('difficulty','')} | TASK_ID: {e.get('task_id','')}")
    print(f"  PROBLEM COUNT: {cnt}")
    print(f"  PROMPT: {e['prompt'][:300]}")
    print(f"  EXPECTED ANSWER: {e['expected_answer'][:300]}")
    # List specific problems for this entry
    for p in problems:
        if p['idx'] == idx:
            print(f"    ⚠ [{p['problem_type']}] {p['evidence'][:150]}")

print(f"\n{'='*80}")
print(f"COMPLETE PROBLEM DUMP (all {len(problems)} problems)")
print(f"{'='*80}")
for p in problems:
    print(f"[{p['idx']:4d}] [{p['category']:14s}] [{p['problem_type']:35s}] {p['evidence'][:120]}")

# ──────────────────────────────────────────────
# 4. Recommendations
# ──────────────────────────────────────────────
print(f"\n{'='*80}")
print("RECOMMENDATIONS")
print(f"{'='*80}")

fix_types = {'truncated_prompt', 'truncated_expected_answer'}
remove_types = {'meaningless_expected_answer', 'suspiciously_short_expected_answer', 'category_mismatch'}
regenerate_types = {'template_artifact_gsm8k', 'template_artifact_json_in_logic', 'code_not_function', 'ner_no_bio_tags', 'non_numeric_math_answer'}

fix_count = sum(1 for p in problems if p['problem_type'] in fix_types)
remove_count = sum(1 for p in problems if p['problem_type'] in remove_types)
regenerate_count = sum(1 for p in problems if p['problem_type'] in regenerate_types)

print(f"  Entries to FIX (truncation issues): {fix_count}")
print(f"  Entries to REMOVE (meaningless/mismatched): {remove_count}")
print(f"  Entries to REGENERATE (template artifacts, bad labels): {regenerate_count}")
print(f"  Entries to REVIEW (other): {len(problems) - fix_count - remove_count - regenerate_count}")
