#!/usr/bin/env python3
"""Inspect training-v3.json categories, prompt formats, and answer types."""
import json

with open('/home/artem/dev/amd-hackathon/data/eval/training-v3.json') as f:
    data = json.load(f)

print(f"Total questions: {len(data)}")

# Category distribution
cats = {}
for q in data:
    c = q['category']
    cats[c] = cats.get(c, 0) + 1
print(f"\nCategories: {json.dumps(cats, indent=2)}")

# Difficulty distribution
diff = {}
for q in data:
    d = q.get('difficulty', 'unknown')
    diff[d] = diff.get(d, 0) + 1
print(f"\nDifficulty: {json.dumps(diff, indent=2)}")

# Show first question of each category
seen = set()
print("\n=== Sample questions by category ===")
for q in data:
    c = q['category']
    if c not in seen:
        seen.add(c)
        answer = q['expected_answer']
        print(f"\n--- {c} (difficulty={q['difficulty']}) ---")
        print(f"PROMPT ({len(q['prompt'])} chars): {q['prompt'][:250]}")
        print(f"ANSWER ({len(answer)} chars): {answer[:150]}")
        print(f"  starts_with_question: {q['prompt'].strip().startswith(('What','Who','How','Why','Is','Are','Do','Does','Write','Fix','Return','Sum','Find'))}")

# Check prompt patterns
code_prompts = [q['prompt'] for q in data if q['category'] in ('code_gen', 'code_debug')]
for i, p in enumerate(code_prompts[:3]):
    print(f"\n--- Code prompt {i+1} ---")
    print(p[:300])
