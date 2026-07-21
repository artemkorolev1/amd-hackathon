#!/usr/bin/env python3
"""Patch duplicate questions in build-B-40.json."""
import json

with open('/home/artem/dev/amd-hackathon/data/eval/generated/build-B-40.json') as f:
    B = json.load(f)
with open('/home/artem/dev/amd-hackathon/data/eval/generated/build-A-40.json') as f:
    A = json.load(f)

a_prompts = {(q['category'], q['prompt'].strip()) for q in A}

replacements = {
    'sum_list': {
        "task_id": "B-gen-5a", "category": "code_generation",
        "prompt": "Write a Python function named average_list that takes a list of numbers and returns their average, as a float.",
        "gold": {
            "function": "average_list",
            "check_code": "def check(candidate):\n    assert candidate([1, 2, 3, 4]) == 2.5\n    assert candidate([10]) == 10.0\n    assert candidate([]) == 0.0\n    assert candidate([-5, 5]) == 0.0",
            "context": "def average_list(nums):\n    \"\"\"Return the average of the numbers in the list nums. Return 0.0 if empty.\"\"\"",
            "_reference": "def average_list(nums):\n    if not nums:\n        return 0.0\n    return sum(nums) / len(nums)"
        }
    },
    'is_even': {
        "task_id": "B-gen-5b", "category": "code_generation",
        "prompt": "Write a Python function named is_positive that takes an integer and returns True if it is greater than zero, False otherwise.",
        "gold": {
            "function": "is_positive",
            "check_code": "def check(candidate):\n    assert candidate(5) == True\n    assert candidate(-3) == False\n    assert candidate(0) == False\n    assert candidate(1000) == True",
            "context": "def is_positive(n):\n    \"\"\"Return True if n is positive (> 0), False otherwise.\"\"\"",
            "_reference": "def is_positive(n):\n    return n > 0"
        }
    },
    'reverse_string': {
        "task_id": "B-gen-5c", "category": "code_generation",
        "prompt": "Write a Python function named count_consonants that takes a string and returns the number of consonants (letters that are not a, e, i, o, u) in it, case-insensitively.",
        "gold": {
            "function": "count_consonants",
            "check_code": "def check(candidate):\n    assert candidate('hello') == 3\n    assert candidate('aeiou') == 0\n    assert candidate('') == 0\n    assert candidate('XYZ') == 3",
            "context": "def count_consonants(s):\n    \"\"\"Return the number of consonants in s (case-insensitive). Vowels are a, e, i, o, u.\"\"\"",
            "_reference": "def count_consonants(s):\n    vowels = 'aeiou'\n    count = 0\n    for c in s.lower():\n        if c.isalpha() and c not in vowels:\n            count += 1\n    return count"
        }
    },
    'count_words': {
        "task_id": "B-gen-5d", "category": "code_generation",
        "prompt": "Write a Python function named has_unique_chars that takes a string and returns True if all characters in it are unique, False otherwise.",
        "gold": {
            "function": "has_unique_chars",
            "check_code": "def check(candidate):\n    assert candidate('abc') == True\n    assert candidate('aabc') == False\n    assert candidate('') == True\n    assert candidate('12321') == False",
            "context": "def has_unique_chars(s):\n    \"\"\"Return True if all characters in s appear only once.\"\"\"",
            "_reference": "def has_unique_chars(s):\n    return len(s) == len(set(s))"
        }
    }
}

replaced = 0
for i, q in enumerate(B):
    key = (q['category'], q['prompt'].strip())
    if key in a_prompts and q['category'] == 'code_generation':
        prompt_lower = q['prompt'].lower()
        if 'sum_list' in prompt_lower:
            B[i] = replacements['sum_list']
            replaced += 1
        elif 'is_even' in prompt_lower:
            B[i] = replacements['is_even']
            replaced += 1
        elif 'reverse_string' in prompt_lower:
            B[i] = replacements['reverse_string']
            replaced += 1
        elif 'count_words' in prompt_lower:
            B[i] = replacements['count_words']
            replaced += 1

print(f"Replaced {replaced} duplicates")

# Verify
a_prompts2 = {(q2['category'], q2['prompt'].strip()) for q2 in A}
b_prompts2 = {(q2['category'], q2['prompt'].strip()) for q2 in B}
remaining = a_prompts2 & b_prompts2
print(f"Remaining cross-set duplicates: {len(remaining)}")
for cat, p in remaining:
    print(f"  [{cat}] {p[:80]}")

with open('/home/artem/dev/amd-hackathon/data/eval/generated/build-B-40.json', 'w') as f:
    json.dump(B, f, indent=2, ensure_ascii=False)
print("Saved patched build-B-40.json")
