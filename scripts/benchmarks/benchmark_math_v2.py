#!/usr/bin/env python3
"""
Benchmark SymPy math solver improvements implemented in deterministic.py and tools.py.

Tests:
1. dev_40 math questions (5 questions) — verify no regression
2. 60_medium_hard math questions (8 questions) — verify new coverage
3. Additional targeted tests for each archetype
"""
import json
import re
import sys
sys.path.insert(0, '/home/artem/dev/amd-hackathon')

from agent.solvers.deterministic import solve_arithmetic
from agent.solvers.tools import sympy_solve


def norm(s):
    """Normalize a result for comparison."""
    if s is None:
        return ""
    s = str(s)
    # Remove surrounding [brackets] from sympy list results like [5/2]
    s = re.sub(r'^\[(.+)\]$', r'\1', s)
    s = re.sub(r'\.0$', '', s)
    s = s.strip().lower()
    s = s.replace('$', '')
    return s


def frac_to_float(s):
    """Convert a fraction string to float."""
    if '/' in s:
        try:
            parts = s.split('/')
            return float(parts[0]) / float(parts[1])
        except:
            return None
    return None


def check_match(result, expected):
    """Check if result matches expected using multiple comparison strategies."""
    if result is None:
        return False
    
    r = norm(result)
    e = norm(expected)
    
    # Direct string match
    if r == e:
        return True
    
    # Float comparison
    try:
        r_f = float(r)
        e_f = float(e)
        if abs(r_f - e_f) < 0.001:
            return True
    except (ValueError, TypeError):
        pass
    
    # Result is a fraction, expected is a number
    r_frac = frac_to_float(r)
    e_frac = frac_to_float(e)
    if r_frac is not None:
        try:
            e_f = float(e)
            if abs(r_frac - e_f) < 0.001:
                return True
        except:
            pass
    if e_frac is not None:
        try:
            r_f = float(r)
            if abs(e_frac - r_f) < 0.001:
                return True
        except:
            pass
    
    # Both are fractions
    if r_frac is not None and e_frac is not None:
        if abs(r_frac - e_frac) < 0.001:
            return True
        # Check simplified form
        from fractions import Fraction
        try:
            if Fraction(r) == Fraction(e):
                return True
        except:
            pass
    
    # Check if result contains a number that matches expected
    # e.g., result [5] should match "5"
    r_nums = re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', r)
    e_nums = re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', e)
    for rn in r_nums:
        for en in e_nums:
            if rn == en:
                return True
            try:
                if abs(float(rn) - float(en)) < 0.001:
                    return True
            except:
                pass
    
    return False


print("=" * 80)
print("SYMPY MATH SOLVER IMPROVEMENTS — BENCHMARK")
print("=" * 80)

# ========== TEST 1: dev_40 math questions ==========
print("\n--- DEV_40 MATH QUESTIONS ---")
with open('/home/artem/dev/amd-hackathon/input/dev_40.json') as f:
    dev_data = json.load(f)

math_qs_dev = [q for q in dev_data if q.get('category', '').startswith('math')]
print(f"Found {len(math_qs_dev)} math questions in dev_40")

dev_correct = 0
for q in math_qs_dev:
    prompt = q['prompt']
    expected = q['gold']['answer']
    expected_str = str(expected).strip()
    
    result = solve_arithmetic(prompt, q['category'])
    match = check_match(result, expected_str)
    if match:
        dev_correct += 1
    status = "✓" if match else "✗"
    print(f"  [{status}] {q['task_id']}: {prompt[:55]} -> {result} (expected: {expected_str})")

print(f"\n  dev_40 accuracy: {dev_correct}/{len(math_qs_dev)} ({dev_correct*100//len(math_qs_dev)}%)")

# ========== TEST 2: 60_medium_hard math questions ==========
print("\n--- 60_MEDIUM_HARD MATH QUESTIONS ---")
with open('/home/artem/dev/amd-hackathon/data/eval/primary/eval_60_medium_hard.json') as f:
    eval_data = json.load(f)

math_qs = [q for q in eval_data['questions'] if q.get('category') == 'math']
print(f"Found {len(math_qs)} math questions in 60_medium_hard")

# What each question tests and our target behavior
q_targets = {
    0: {"can_solve": False, "reason": "Law of Sines — needs geometry/trig beyond current scope"},
    1: {"can_solve": True,  "reason": "Mean/median word problem"},
    2: {"can_solve": True,  "reason": "Geometric series sum"},
    3: {"can_solve": False, "reason": "Absolute value inequality — needs case analysis"},
    4: {"can_solve": True,  "reason": "Matrix determinant"},
    5: {"can_solve": True,  "reason": "Inclusion-exclusion probability"},
    6: {"can_solve": True,  "reason": "Log equation"},
    7: {"can_solve": False, "reason": "Perpendicular line — needs coordinate geometry"},
}

correct_solved = 0
correct_skipped = 0
wrong_answers = 0

for i, q in enumerate(math_qs):
    prompt = q['prompt']
    target = q_targets[i]
    
    # Extract numeric/symbolic expected value from the explanation
    full_expected = q['expected_answer']
    
    # The expected_answer field is explanatory; extract the key numeric result
    expected_num = None
    if i == 0: expected_num = "5.177"
    elif i == 1: expected_num = "20"
    elif i == 2: expected_num = "3069"
    elif i == 3: expected_num = "(-4, 5)"
    elif i == 4: expected_num = "3"
    elif i == 5: expected_num = "5/6"  # from expected_answer: "25/30 = 5/6"
    elif i == 6: expected_num = "4"
    elif i == 7: expected_num = "(-1/2)x + 1"
    
    result = solve_arithmetic(prompt, 'math')
    match = check_match(result, expected_num) if result and expected_num else False
    
    if target["can_solve"]:
        if match:
            correct_solved += 1
            status = "✓ SOLVED"
        elif result is not None:
            wrong_answers += 1
            status = "✗ WRONG"
        else:
            wrong_answers += 1
            status = "✗ MISS"
    else:
        # Should skip
        if result is None:
            correct_skipped += 1
            status = "✓ SKIP"
        else:
            wrong_answers += 1
            status = "✗ FALSE POSITIVE"
    
    print(f"  [{status}] Q{i}: {prompt[:65]}")
    print(f"           result={result} expected={expected_num} target={'solve' if target['can_solve'] else 'skip'}")
    print()

print(f"\n  60_medium_hard: {correct_solved}/5 correctly solved + {correct_skipped}/3 correctly skipped")
print(f"  Wrong answers: {wrong_answers}")

# ========== TEST 3: Targeted tests for each archetype ==========
print("\n\n--- TARGETED ARCHETYPE TESTS ---")

targeted_tests = [
    # (prompt, category, expected, archetype_key)
    ("Solve 2x = 5", "math", "2.5", "fix_implicit"),
    ("solve 3x + 7 = 22", "math", "5", "fix_implicit"),
    ("The mean of 5 numbers is 12. If one value is removed, the new mean is 10. What is the removed value?", "math", "20", "mean"),
    ("The average of 4 numbers is 12. If a fifth number 20 is added, what is the new average?", "math", "13.6", "mean"),
    ("Compute the determinant of [[1,2],[3,4]]", "math", "-2", "det"),
    ("Find the determinant of [[2,1,3],[1,0,2],[3,2,1]]", "math", "3", "det"),
    ("Solve log2(x) + log2(x-3) = 2", "math", "4", "log"),
    ("Solve log_2(x) = 3", "math", "8", "log"),
    ("In a class of 30 students, 18 play soccer, 15 play basketball, and 8 play both. If a student is selected at random, what is the probability they play at least one?", "math", "5/6", "inclusion"),
    ("A geometric sequence has first term a = 3 and common ratio r = 2. What is the sum of the first 10 terms?", "math", "3069", "geometric"),
    # Regression tests
    ("What is 17 * 24?", "math_reasoning", "408", "regression"),
    ("What is the square root of 144?", "math_arithmetic", "12", "regression"),
    ("What is 15% of 240?", "math_reasoning", "36", "regression"),
    ("If x + 7 = 3x - 5, solve for x.", "math_reasoning", "6", "regression"),
]

archetype_results = {}
for prompt, cat, expected, key in targeted_tests:
    result = solve_arithmetic(prompt, cat)
    match = check_match(result, expected)
    if key not in archetype_results:
        archetype_results[key] = {"correct": 0, "total": 0}
    archetype_results[key]["total"] += 1
    if match:
        archetype_results[key]["correct"] += 1
    
    status = "✓" if match else "✗"
    print(f"  [{status}] {prompt[:55]:55s} -> {str(result):15s} (expected: {expected})")

# Print archetype summary
print("\n--- ARCHETYPE SUMMARY ---")
labels = {
    "fix_implicit": "1. Fix implicit_application bug in sympy_solve",
    "mean": "2. Mean/median word problems",
    "det": "3. Matrix determinant extraction",
    "log": "4. Log equation solving",
    "inclusion": "5a. Inclusion-exclusion",
    "geometric": "5b. Geometric series",
    "regression": "Regression (existing patterns preserved)",
}

for key in ["fix_implicit", "mean", "det", "log", "inclusion", "geometric", "regression"]:
    if key in archetype_results:
        r = archetype_results[key]
        print(f"  {labels[key]}: {r['correct']}/{r['total']}")

# Combined IE + geometric
ie_g = archetype_results.get("inclusion", {"correct": 0})["correct"] + \
       archetype_results.get("geometric", {"correct": 0})["correct"]
ie_g_t = archetype_results.get("inclusion", {"total": 0})["total"] + \
         archetype_results.get("geometric", {"total": 0})["total"]
print(f"  5. Inclusion-exclusion + geometric series (combined): {ie_g}/{ie_g_t}")

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"Dev_40 math:                    {dev_correct}/{len(math_qs_dev)} (no regression expected)")
print(f"60_medium_hard correctly solved: {correct_solved}/5 archetype questions")
print(f"60_medium_hard correctly skipped: {correct_skipped}/3 non-target questions")
print(f"Wrong/FPs: {wrong_answers}")
print(f"SymPy ceiling? {'Yes — all 5 archetypes work' if correct_solved == 5 else 'No — round 3 could help'}")
print()
