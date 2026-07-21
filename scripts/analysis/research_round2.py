#!/usr/bin/env python3
"""
Round 2 SymPy Math Solver Improvements — Research Prototype.

Tests all proposed improvements against actual benchmark data.
"""
import json
import math
import re
import sys
sys.path.insert(0, '/home/artem/dev/amd-hackathon')

from agent.solvers.tools import sympy_solve as original_sympy_solve
from agent.solvers.deterministic import solve_arithmetic, _extract_equation

# ============================================================
# IMPROVEMENT 1: Fix implicit_application issue + pre-strip
# ============================================================

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication,
    function_exponentiation,
)

_FIXED_TRANSFORMS = standard_transformations + (implicit_multiplication, function_exponentiation)

def sympy_solve_v2(expr_str: str) -> str | None:
    """
    Improved SymPy solver:
    1. Pre-strip command words like 'solve', 'calculate', 'find', etc.
    2. Use implicit_multiplication only (not implicit_multiplication_application)
    3. Preprocess log_2(x) -> log(x, 2) notation
    4. Preprocess matrix determinant notation
    """
    if not expr_str or not expr_str.strip():
        return None
    
    s = expr_str.strip()
    
    # Pre-strip leading command words
    s = re.sub(r'^(?:solve|calculate|compute|find|evaluate|simplify|determine)\s+(?:for\s+\w+\s*[:\s]+)?', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s+(?:solve|calculate|compute|find|evaluate|simplify|determine)\s+for\s+\w+', '', s, flags=re.IGNORECASE)
    
    # Preprocess log_2(x) -> log(x, 2)
    s = re.sub(r'log[_\u2082]\(', 'log(', s)
    s = re.sub(r'\blog_(\d+)\(', r'log(,\1) ', s)  # log_2(x) -> log(x, 2) — handled later
    
    # Preprocess matrix notation: [[a,b],[c,d]]
    # Will be handled by detecting matrix patterns
    
    def _parse(s: str):
        try:
            return parse_expr(s, local_dict={}, transformations=_FIXED_TRANSFORMS)
        except Exception:
            return sp.sympify(s, dict())
    
    # Try parsing as a regular expression first
    try:
        expr = _parse(s)
    except Exception:
        expr = None
    
    if expr is not None and hasattr(expr, 'is_Number'):
        try:
            if expr.is_Number or (hasattr(expr, 'is_constant') and expr.is_constant()):
                val = sp.N(expr)
                if val.is_Float:
                    fval = float(val)
                    if abs(fval - round(fval)) < 1e-12:
                        return str(int(round(fval)))
                    return f"{fval:.10f}".rstrip('0').rstrip('.')
                return str(val)
            return None  # Has variables
        except Exception:
            return None
    
    # Try equation with "=" sign
    if "=" in s:
        try:
            left, right = s.split("=", 1)
            left_expr = _parse(left.strip())
            right_expr = _parse(right.strip())
            equation = sp.Eq(left_expr, right_expr)
            solution = sp.solve(equation)
            if solution:
                return str(solution)
            return str(equation)
        except Exception:
            pass
    
    return None


# ============================================================
# IMPROVEMENT 2: Matrix determinant from text
# ============================================================

def _extract_matrix(text: str) -> sp.Matrix | None:
    """Extract a matrix from text like [[2,1,3],[1,0,2],[3,2,1]] or
    'matrix: [2 1 3; 1 0 2; 3 2 1]'"""
    
    # Pattern 1: Python list-of-lists notation [[a,b,c],[d,e,f],...]
    m = re.search(r'\[\[(\d+(?:\s*,\s*\d+)*)\](?:\s*,\s*\[(\d+(?:\s*,\s*\d+)*)\])+\]', text)
    if m:
        # Extract all rows
        full_match = m.group(0)
        rows_str = re.findall(r'\[([^\]]+)\]', full_match)
        rows = []
        for row_str in rows_str:
            row = [float(x.strip()) for x in row_str.split(',') if x.strip()]
            if row:
                rows.append(row)
        if rows:
            # Check rectangular
            ncols = len(rows[0])
            if all(len(r) == ncols for r in rows):
                return sp.Matrix(rows)
    
    # Pattern 2: Matrix notation with semicolons [2 1 3; 1 0 2; 3 2 1]
    m = re.search(r'\[(\d[\d\s,;.-]*)\]', text)
    if m:
        inner = m.group(1)
        if ';' in inner:
            rows_str = inner.split(';')
            rows = []
            for row_str in rows_str:
                row = [float(x) for x in row_str.split() if x.strip()]
                if row:
                    rows.append(row)
            if rows and all(len(r) == len(rows[0]) for r in rows):
                return sp.Matrix(rows)
    
    return None


# ============================================================
# IMPROVEMENT 3: Log preprocessor
# ============================================================

def _preprocess_log(text: str) -> str:
    """Convert log_2(x), log₂(x), log base 2 of x -> log(x, 2)"""
    # log_2(x) or log₂(x) 
    s = re.sub(r'log[_\\u2082]?(\d+)\(', r'log(\1, ', text)
    # log base 2 of x
    # Actually SymPy: log(x, base). Let's handle cleanly.
    return s


# ============================================================
# IMPROVEMENT 4: Inclusion-Exclusion solver
# ============================================================

_INCLUSION_EXCLUSION_PATTERN = re.compile(
    r'(\d+)\s+(?:play|have|take|study|like|are|enjoy)\s+\w+'
    r'(?:.*?)(\d+)\s+(?:play|have|take|study|like|are|enjoy)\s+\w+'
    r'(?:.*?)(\d+)\s+(?:play|have|take|study|like|are|enjoy)\s+both',
    re.IGNORECASE | re.DOTALL
)

def _solve_inclusion_exclusion(text: str) -> str | None:
    """Solve '18 play soccer, 15 basketball, 8 both' problems."""
    # Pattern: X play A, Y play B, Z play both
    m = _INCLUSION_EXCLUSION_PATTERN.search(text)
    if m:
        set_a = float(m.group(1))
        set_b = float(m.group(2))
        both = float(m.group(3))
        union = set_a + set_b - both
        
        # Check if there's a "total" mentioned
        total_m = re.search(r'(?:out\s+of|in\s+a\s+(?:class|group|set)\s+of|among)\s+(\d+)', text, re.IGNORECASE)
        if total_m:
            total = float(total_m.group(1))
            # Probability question
            prob = union / total
            # Simplify fraction
            numerator = int(union)
            denominator = int(total)
            g = math.gcd(numerator, denominator) if hasattr(math, 'gcd') else 1
            # Try to find gcd manually
            a, b = numerator, denominator
            while b:
                a, b = b, a % b
            g = a
            simple_num = numerator // g
            simple_den = denominator // g
            if simple_den == 1:
                return f"{simple_num}"
            return f"{simple_num}/{simple_den}"
        return str(int(union))
    return None


# ============================================================
# IMPROVEMENT 5: Mean/Median/Average problem solver
# ============================================================

_MEAN_PATTERN = re.compile(
    r'(\d+)\s+(?:values?|numbers?|data\s*set|items?|scores?|grades?)'
    r'.{0,30}?mean\s+(?:of\s+)?(\d+(?:\.\d+)?)',
    re.IGNORECASE
)

def _solve_mean_problem(text: str) -> str | None:
    """
    Solve problems like:
    '5 values with mean of 12. If one removed, mean becomes 10. What was removed?'
    'The average of 4 numbers is 15. If a 5th number is added, average becomes 18. What is the 5th number?'
    """
    # Pattern: "N values with mean of M"
    m = _MEAN_PATTERN.search(text)
    if not m:
        return None
    
    n = float(m.group(1))
    initial_mean = float(m.group(2))
    total_sum = n * initial_mean
    
    # Check for removal pattern: "if one removed, mean becomes M2"
    removed_m = re.search(r'removed|taken\s+out|eliminated|dropped', text, re.IGNORECASE)
    added_m = re.search(r'added|included|included?|appended', text, re.IGNORECASE)
    
    new_mean_m = re.search(r'mean\s+(?:of\s+)?(\d+(?:\.\d+)?)\s+', text, re.IGNORECASE)
    
    if removed_m and new_mean_m:
        # One value removed, new mean given
        new_n = n - 1
        new_mean = float(new_mean_m.group(1))
        new_sum = new_n * new_mean
        removed_value = total_sum - new_sum
        if removed_value == int(removed_value):
            return str(int(removed_value))
        return str(removed_value)
    
    if added_m and new_mean_m:
        # One value added, new mean given
        new_n = n + 1
        new_mean = float(new_mean_m.group(1))
        new_sum = new_n * new_mean
        added_value = new_sum - total_sum
        if added_value == int(added_value):
            return str(int(added_value))
        return str(added_value)
    
    # Check for: "If one more value of X is added, what is the new mean?"
    added_value_m = re.search(r'added\s+(?:a\s+)?(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if added_value_m and 'mean' in text.lower() and 'what' in text.lower():
        added_val = float(added_value_m.group(1))
        new_sum = total_sum + added_val
        new_n = n + 1
        new_mean = new_sum / new_n
        if new_mean == int(new_mean):
            return str(int(new_mean))
        return str(new_mean)
    
    return None


# ============================================================
# IMPROVEMENT 6: Multi-variable equation systems
# ============================================================

def _solve_system(text: str) -> str | None:
    """Detect and solve systems of equations.
    'x + y = 10, x - y = 4'
    """
    # Find multiple equations separated by commas, semicolons, 'and', or newlines
    # Each equation must contain '='
    equations = re.split(r',|;|\band\b|\n', text)
    eqns = []
    for eq in equations:
        eq = eq.strip()
        if '=' in eq and re.search(r'[a-z]', eq, re.IGNORECASE):
            eqns.append(eq)
    
    if len(eqns) >= 2:
        try:
            sympy_eqns = []
            for eq in eqns:
                left, right = eq.split('=', 1)
                l = parse_expr(left.strip(), local_dict={}, transformations=_FIXED_TRANSFORMS)
                r = parse_expr(right.strip(), local_dict={}, transformations=_FIXED_TRANSFORMS)
                sympy_eqns.append(sp.Eq(l, r))
            solution = sp.solve(sympy_eqns)
            if solution:
                return str(solution)
        except Exception:
            pass
    return None


# ============================================================
# IMPROVEMENT 7: Fraction/word problems
# ============================================================

def _solve_fraction_problem(text: str) -> str | None:
    """Solve '2/3 of the students, then 1/4 of the remainder' type problems."""
    # Look for fraction-of patterns
    fractions = re.findall(r'(\d+)/(\d+)\s+of\s+(?:the\s+)?(\w+)', text, re.IGNORECASE)
    if not fractions:
        return None
    
    # Check for total number at start
    total_m = re.search(r'(\d+)\s+(?:students|people|items|members|total)', text, re.IGNORECASE)
    if not total_m:
        return None
    
    total = float(total_m.group(1))
    remaining = total
    
    for frac in fractions:
        num, den, _ = float(frac[0]), float(frac[1]), frac[2]
        taken = remaining * (num / den)
        remaining -= taken
    
    if remaining < 0:
        remaining = 0
    if remaining == int(remaining):
        return str(int(remaining))
    return str(remaining)


# ============================================================
# COMBINED SOLVER V2
# ============================================================

def solve_arithmetic_v2(task: str, category: str) -> str | None:
    """Improved arithmetic solver with Round 2 enhancements."""
    text = task.strip()
    
    # 0. Try SymPy on full text (improved)
    sympy_result = sympy_solve_v2(text)
    if sympy_result and not sympy_result.startswith('Error'):
        return sympy_result
    
    # 1. Try matrix extraction
    matrix = _extract_matrix(text)
    if matrix is not None:
        # Look for "determinant" or "det"
        if re.search(r'\b(det(?:erminant)?|determinant)\b', text, re.IGNORECASE):
            try:
                det = matrix.det()
                if det == int(det):
                    return str(int(det))
                return str(det)
            except Exception:
                pass
    
    # 2. Try equation extraction
    eq = _extract_equation(text)
    if eq:
        sympy_result = sympy_solve_v2(eq)
        if sympy_result and not sympy_result.startswith('Error'):
            return sympy_result
    
    # 3. Try system of equations
    system_result = _solve_system(text)
    if system_result:
        return system_result
    
    # 4. Try inclusion-exclusion
    ie_result = _solve_inclusion_exclusion(text)
    if ie_result is not None:
        return ie_result
    
    # 5. Try mean problems
    mean_result = _solve_mean_problem(text)
    if mean_result is not None:
        return mean_result
    
    # 6. Try fraction problems
    frac_result = _solve_fraction_problem(text)
    if frac_result is not None:
        return frac_result
    
    # 7. Fall back to original solver
    return solve_arithmetic(task, category)


# ============================================================
# BENCHMARK
# ============================================================

def norm(s):
    if s is None:
        return ""
    s = re.sub(r'^\[([^\]]+)\]$', r'\1', str(s))
    s = re.sub(r'\.0$', '', s)
    return s.strip().lower()

print("=" * 80)
print("ROUND 2 RESEARCH — IMPROVEMENT BENCHMARK")
print("=" * 80)

# Load 60_medium_hard math questions
with open('/home/artem/dev/amd-hackathon/data/eval/primary/eval_60_medium_hard.json') as f:
    eval_data = json.load(f)

print("\n--- 60_medium_hard MATH QUESTIONS ---")
math_qs = [q for q in eval_data['questions'] if q.get('category') == 'math']
print(f"Total math questions: {len(math_qs)}")

for q in math_qs:
    prompt = q['prompt']
    expected = q['expected_answer']
    
    original_result = solve_arithmetic(prompt, 'math')
    v2_result = solve_arithmetic_v2(prompt, 'math')
    
    # Check if expected numeric answer is extractable
    expected_nums = re.findall(r'[-]?\d+(?:\.\d+)?', expected)
    original_match = any(norm(original_result) == norm(n) for n in expected_nums) if original_result else False
    v2_match = any(norm(v2_result) == norm(n) for n in expected_nums) if v2_result else False
    
    if original_result != v2_result:
        status = "IMPROVED" if v2_match and not original_match else "REGRESSED"
    elif v2_match:
        status = "SAME ✓"
    else:
        status = "STILL MISSED"
    
    print(f"  [{status}] {prompt[:70]}")
    print(f"    Original: {original_result}")
    print(f"    V2:       {v2_result}")
    print(f"    Expected: {expected[:70]}...")
    print()

# Test dev_40
print("--- DEV_40 MATH QUESTIONS ---")
with open('/home/artem/dev/amd-hackathon/input/dev_40.json') as f:
    dev_data = json.load(f)

math_qs_dev = [q for q in dev_data if q.get('category', '').startswith('math')]
print(f"Total dev_40 math questions: {len(math_qs_dev)}")
correct_original = 0
correct_v2 = 0
for q in math_qs_dev:
    prompt = q['prompt']
    expected = q['gold']['answer']
    
    orig = solve_arithmetic(prompt, q['category'])
    v2 = solve_arithmetic_v2(prompt, q['category'])
    
    if norm(orig) == norm(expected):
        correct_original += 1
    if norm(v2) == norm(expected):
        correct_v2 += 1
    
    if norm(orig) != norm(v2):
        print(f"  DIFF: {prompt[:55]} orig={orig} v2={v2} expected={expected}")

print(f"Original: {correct_original}/{len(math_qs_dev)}")
print(f"V2:       {correct_v2}/{len(math_qs_dev)}")
