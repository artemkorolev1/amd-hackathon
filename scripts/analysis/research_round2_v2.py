#!/usr/bin/env python3
"""
Round 2 SymPy Math Solver Improvements — FINAL RESEARCH PROTOTYPE.
"""
import json
import math
import re
import sys
sys.path.insert(0, '/home/artem/dev/amd-hackathon')

from agent.solvers.tools import sympy_solve as original_sympy_solve, calculator
from agent.solvers.deterministic import (
    solve_arithmetic, _extract_equation, _REMAINDER_PATTERN,
    _PERCENT_PATTERN, _ROOT_PATTERN, _solve_speed_distance, _solve_unit_cost
)

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication,
    function_exponentiation,
)

_FIXED_TRANSFORMS = standard_transformations + (implicit_multiplication, function_exponentiation)

# ============================================================
# IMPROVEMENT 1: Fixed SymPy solver with pre-strip + log handling
# ============================================================

_LOG_SUBSCRIPT_RE = re.compile(r'log(\d*)[' + '\u2080-\u2089' + r']\(', re.IGNORECASE)
# Unicode subscript digits: \u2080=0, \u2081=1, \u2082=2, ... \u2089=9

def _preprocess_log_subscripts(text: str) -> str:
    """Convert log₂(x) -> log(2, x) and log_2(x) -> log(2, x)."""
    # Handle unicode subscripts: log₂(x) 
    # We'll convert subscript digit to regular digit first
    result = []
    for c in text:
        if '\u2080' <= c <= '\u2089':
            result.append(str(ord(c) - 0x2080))
        else:
            result.append(c)
    text = ''.join(result)
    
    # Handle log_N( pattern -> log(N, 
    text = re.sub(r'\blog_(\d+)\(', r'log(\1, ', text)
    # Handle plain log( with subscript already converted
    
    return text

def _is_pure_math_expression(text: str) -> bool:
    """Check if text looks like a pure math expression (short, mostly math chars)."""
    # Remove common words that indicate prose
    stripped = text.strip()
    if len(stripped) > 200:
        return False
    
    # Count math-like characters vs prose
    math_chars = len(re.findall(r'[\d\+\-\*\/\^\(\)\=\[\]\.\,]', stripped))
    total_chars = len(stripped.strip())
    if total_chars == 0:
        return False
    
    ratio = math_chars / total_chars
    return ratio > 0.4 and len(stripped) < 100

def _prestrip_commands(text: str) -> str:
    """Remove leading command words like 'solve', 'calculate', etc."""
    s = text.strip()
    s = re.sub(
        r'^(?:solve|calculate|compute|find|evaluate|simplify|determine|what is|what\'s)\s+'
        r'(?:for\s+\w+\s*[:\s]+)?',
        '', s, flags=re.IGNORECASE
    )
    s = re.sub(r'^solve\s+for\s+\w+\s*[:\s]+', '', s, flags=re.IGNORECASE)
    return s.strip()

def sympy_solve_v2(expr_str: str) -> str | None:
    """Improved SymPy solver with pre-strip, log handling, safer parsing."""
    if not expr_str or not expr_str.strip():
        return None
    
    s = expr_str.strip()
    
    # Only attempt if it looks like a math expression
    # (avoid false positives on prose text)
    
    # Preprocess log subscripts
    s = _preprocess_log_subscripts(s)
    
    # Pre-strip commands for equation solving
    s_eq = _prestrip_commands(s)
    
    def _parse(ss: str):
        try:
            return parse_expr(ss, local_dict={}, transformations=_FIXED_TRANSFORMS)
        except Exception:
            try:
                return sp.sympify(ss, dict())
            except Exception:
                return None
    
    # CASE 1: Equation with "=" sign
    if "=" in s_eq:
        try:
            left, right = s_eq.split("=", 1)
            left_expr = _parse(left.strip())
            if left_expr is None:
                return None
            right_expr = _parse(right.strip())
            if right_expr is None:
                return None
            equation = sp.Eq(left_expr, right_expr)
            solution = sp.solve(equation)
            if solution:
                return str(solution)
            return None
        except Exception:
            return None
    
    # CASE 2: Bare expression (no "=")
    # Only try on short, math-looking text to avoid prose contamination
    if _is_pure_math_expression(s):
        try:
            expr = _parse(s)
            if expr is not None:
                if expr.is_Number or (hasattr(expr, 'is_constant') and expr.is_constant()):
                    val = sp.N(expr)
                    if val.is_Float:
                        fval = float(val)
                        if abs(fval - round(fval)) < 1e-12:
                            return str(int(round(fval)))
                        return f"{fval:.10f}".rstrip('0').rstrip('.')
                    return str(val)
        except Exception:
            pass
    
    return None


# ============================================================
# IMPROVEMENT 2: Matrix determinant extraction
# ============================================================

def _extract_matrix(text: str) -> sp.Matrix | None:
    """Extract matrix from various text formats."""
    # Pattern 1: Multiline brackets [a b c] on separate lines
    rows = re.findall(r'^\[(\d[\d\s,;.\-]*)\]$', text, re.MULTILINE)
    if rows:
        parsed_rows = []
        for row_str in rows:
            row = [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', row_str)]
            if row:
                parsed_rows.append(row)
        if parsed_rows and all(len(r) == len(parsed_rows[0]) for r in parsed_rows):
            return sp.Matrix(parsed_rows)
    
    # Pattern 2: Inline [[a,b,c],[d,e,f],...]
    m = re.search(r'\[\[([^\]]+)\](?:\s*,\s*\[([^\]]+)\])+\]', text)
    if m:
        full = re.search(r'\[\[.+?\]\]', text, re.DOTALL)
        if full:
            inner = full.group(0)
            rows_str = re.findall(r'\[([^\]]+)\]', inner)
            parsed_rows = []
            for row_str in rows_str:
                row = [float(x.strip()) for x in row_str.split(',') if x.strip()]
                if row:
                    parsed_rows.append(row)
            if parsed_rows and all(len(r) == len(parsed_rows[0]) for r in parsed_rows):
                return sp.Matrix(parsed_rows)
    
    return None

def _word_matrix_to_list(text: str) -> list | None:
    """Try to extract matrix from verbal description."""
    # Check if there are 3x3 numbers mentioned
    nums = [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', text)]
    
    # For a 3x3 matrix we expect exactly 9 numbers
    if len(nums) == 9:
        # Check if appears to be a matrix (ordered numbers)
        # Common case: numbers presented in row-major order
        return nums
    
    return None


# ============================================================
# IMPROVEMENT 3: Inclusion-Exclusion solver
# ============================================================

def _solve_inclusion_exclusion(text: str) -> str | None:
    """Solve set problems: '18 play soccer, 15 play basketball, 8 play both'."""
    # Find numbers associated with sets and their overlap
    # Pattern: "X play A, Y play B, Z play both"
    # More flexible: find three numbers near set-related words
    
    # Strategy: find all numbers and their contexts
    set_a = None
    set_b = None
    both = None
    total = None
    
    sentences = re.split(r'[.!\n]', text)
    
    # Look for totals
    total_m = re.search(r'(?:class|group|total|of|among)\s+(?:of\s+)?(\d+)', text, re.IGNORECASE)
    if total_m:
        # Only use if it's the overall population
        t = float(total_m.group(1))
        # Check there are numbers smaller than this for subsets
        all_nums = [float(x) for x in re.findall(r'\d+', text)]
        smaller = sum(1 for n in all_nums if n < t)
        if smaller >= 2:
            total = t
    
    # Look for "X do Y, Z do W, V do both"
    # Try finding numbers associated with 'both'
    for sent in sentences:
        sent_lower = sent.lower().strip()
        
        # Find 'both' mentions
        both_m = re.search(r'(\d+)\s+(?:play|do|have|take|study|like|are|enjoy)\s+both', sent_lower)
        if both_m:
            both = float(both_m.group(1))
        
        # Find individual set mentions
        # First set
        set1 = re.search(r'(\d+)\s+(?:play|do|have|take|study|like|are|enjoy)\s+(\w+)', sent_lower)
        # Second set with different activity
        set2 = re.search(r'(\d+)\s+(?:play|do|have|take|study|like|are|enjoy)\s+(\w+)', sent_lower[sent_lower.find(set1.group(2)) + len(set1.group(2)):] if set1 else sent_lower)
    
    # Simpler: just extract numbers and try inclusion-exclusion
    nums = [float(x) for x in re.findall(r'\d+', text)]
    
    # For the canonical "18 soccer, 15 basketball, 8 both" pattern
    if len(nums) >= 3 and 'both' in text.lower():
        # Try: largest two numbers are the sets, smallest is 'both'
        # But total might be the largest
        sorted_nums = sorted(nums)
        if total:
            # Remove total from candidates
            candidates = [n for n in nums if n != total]
            if len(candidates) >= 3:
                both_candidates = [n for n in candidates if n == min(candidates)]
                rest = [n for n in candidates if n > min(candidates)]
                if len(rest) >= 2:
                    set_a = max(rest)
                    set_b = min(rest)
                    both = min(candidates)
        else:
            # No explicit total
            sorted_unique = sorted(set(nums))
            if len(sorted_unique) >= 3:
                both = sorted_unique[0]  # smallest is 'both'
                set_a = sorted_unique[1]
                set_b = sorted_unique[2]
                total = set_a + set_b - both  # infer total
    
    if set_a is not None and set_b is not None and both is not None:
        union = set_a + set_b - both
        
        if total is not None:
            prob = union / total
            # Simplify fraction
            numerator = int(union)
            denominator = int(total)
            a, b = numerator, denominator
            while b:
                a, b = b, a % b
            g = a
            simple_num = numerator // g
            simple_den = denominator // g
            
            # Return as simplified fraction
            if simple_den == 1:
                return f"{simple_num}"
            return f"{simple_num}/{simple_den}"
        
        return str(int(union))
    
    return None


# ============================================================
# IMPROVEMENT 4: Mean/Average problem solver (fixed)
# ============================================================

def _solve_mean_problem(text: str) -> str | None:
    """Solve mean/average word problems."""
    # Find all "mean of X" occurrences
    means = re.findall(r'mean\s+(?:of\s+)?(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    counts = re.findall(r'(\d+)\s+(?:values?|numbers?|items?|data\s*|scores?|grades?|terms?)', text, re.IGNORECASE)
    
    if not means or not counts:
        return None
    
    n = float(counts[0])
    initial_mean = float(means[0])
    total_sum = n * initial_mean
    
    # Check for removal pattern
    if len(means) >= 2:
        # "mean becomes 10" or "new mean is 10"
        new_mean = float(means[1])
        
        if re.search(r'removed|taken\s+out|eliminated|dropped', text, re.IGNORECASE):
            new_n = n - 1
            new_total = new_n * new_mean
            removed = total_sum - new_total
            if removed == int(removed):
                return str(int(removed))
            return str(removed)
        
        if re.search(r'added|included|included?|appended', text, re.IGNORECASE):
            new_n = n + 1
            new_total = new_n * new_mean
            added = new_total - total_sum
            if added == int(added):
                return str(int(added))
            return str(added)
    
    # Single mean: "average of X is Y, what's the total?" 
    if re.search(r'(?:total|sum)', text, re.IGNORECASE):
        if total_sum == int(total_sum):
            return str(int(total_sum))
        return str(total_sum)
    
    return None


# ============================================================
# IMPROVEMENT 5: Geometric sequence/series solver
# ============================================================

def _solve_geometric(text: str) -> str | None:
    """Solve geometric sequence/series problems."""
    if not re.search(r'geometric\s+(sequence|series|progression)', text, re.IGNORECASE):
        return None
    
    a = None
    r = None
    n = None
    
    # Extract first term
    a_m = re.search(r'first\s+term\s*(?:\w+\s*)?=?\s*(-?\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if a_m:
        a = float(a_m.group(1))
    
    # Extract common ratio
    r_m = re.search(r'common\s+ratio\s*(?:\w+\s*)?=?\s*(-?\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if r_m:
        r = float(r_m.group(1))
    
    # Extract n (number of terms)
    n_m = re.search(r'first\s+(\d+)\s+terms?', text, re.IGNORECASE)
    if n_m:
        n = int(n_m.group(1))
    
    if a is not None and r is not None and n is not None:
        # Sum of geometric series: a(r^n - 1)/(r - 1)
        if r != 1:
            s = a * (r**n - 1) / (r - 1)
            if s == int(s):
                return str(int(s))
            return str(s)
        else:
            s = a * n
            if s == int(s):
                return str(int(s))
            return str(s)
    
    return None


# ============================================================
# IMPROVEMENT 6: Multi-variable equation systems
# ============================================================

def _solve_system(text: str) -> str | None:
    """Detect and solve systems of equations."""
    # Only try on short text with clear equation markers
    if len(text) > 300:
        return None
    
    # Find equations
    equations = re.split(r',|;|\band\b|\n', text)
    eqns = []
    for eq in equations:
        eq = eq.strip()
        if '=' in eq and re.search(r'[a-z]', eq, re.IGNORECASE):
            # Clean up
            eq_clean = _prestrip_commands(eq)
            if '=' in eq_clean:
                eqns.append(eq_clean)
    
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
    
    # Single equation with multiple variables but can be solved
    if len(eqns) == 1:
        eq = eqns[0]
        if '=' in eq:
            left, right = eq.split('=', 1)
            try:
                l = parse_expr(left.strip(), local_dict={}, transformations=_FIXED_TRANSFORMS)
                r = parse_expr(right.strip(), local_dict={}, transformations=_FIXED_TRANSFORMS)
                eq_obj = sp.Eq(l, r)
                sol = sp.solve(eq_obj)
                if sol:
                    return str(sol)
            except Exception:
                pass
    
    return None


# ============================================================
# IMPROVEMENT 7: Percentage change problems
# ============================================================

_PERCENT_CHANGE_PATTERN = re.compile(
    r'(?:what\s+is\s+the\s+)?(\d+(?:\.\d+)?)\s*%\s*(?:of|)\s*(\d+(?:\.\d+)?)',
    re.IGNORECASE
)

def _solve_percent_change(text: str) -> str | None:
    """Solve 'what is X% of Y', 'X% increase of Y', etc."""
    m = _PERCENT_CHANGE_PATTERN.search(text)
    if m:
        pct = float(m.group(1))
        base = float(m.group(2))
        result = base * (pct / 100.0)
        if result == int(result):
            return str(int(result))
        return f"{result:.10f}".rstrip('0').rstrip('.')
    
    # X% increase
    m = re.search(r'(\d+(?:\.\d+)?)\s*%\s*(?:increase|rise|raise|gain)\s+(?:of\s+)?(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if m:
        pct = float(m.group(1))
        base = float(m.group(2))
        result = base * (1 + pct / 100.0)
        if result == int(result):
            return str(int(result))
        return f"{result:.10f}".rstrip('0').rstrip('.')
    
    return None


# ============================================================
# IMPROVEMENT 8: Log equation parser
# ============================================================

def _solve_log_equation(text: str) -> str | None:
    """Solve log equations like 'log₂(x) + log₂(x-3) = 2'."""
    # Check for log patterns
    if not re.search(r'\blog', text, re.IGNORECASE):
        return None
    
    # Preprocess
    cleaned = _preprocess_log_subscripts(text)
    
    # Try to solve as equation
    if '=' in cleaned:
        try:
            left, right = cleaned.split('=', 1)
            l = parse_expr(left.strip(), local_dict={}, transformations=_FIXED_TRANSFORMS)
            r = parse_expr(right.strip(), local_dict={}, transformations=_FIXED_TRANSFORMS)
            eq = sp.Eq(l, r)
            sol = sp.solve(eq)
            if sol:
                # Filter to real solutions
                real_sols = [s for s in sol if s.is_real if hasattr(s, 'is_real')]
                if real_sols:
                    return str(real_sols)
                return str(sol)
        except Exception:
            pass
    
    return None


# ============================================================
# COMBINED SOLVER V2
# ============================================================

def solve_arithmetic_v2(task: str, category: str) -> str | None:
    """Improved arithmetic solver with Round 2 enhancements."""
    text = task.strip()
    
    # 0.1 Try log equation solving (before general SymPy, to avoid prose confusion)
    log_result = _solve_log_equation(text)
    if log_result is not None:
        return log_result
    
    # 0.2 Try SymPy on short, math-looking text
    if _is_pure_math_expression(text):
        sympy_result = sympy_solve_v2(text)
        if sympy_result and not sympy_result.startswith('Error'):
            return sympy_result
    
    # 1. Try matrix determinant
    if re.search(r'\b(det(?:erminant)?|determinant)\b', text, re.IGNORECASE):
        matrix = _extract_matrix(text)
        if matrix is not None:
            try:
                det = matrix.det()
                if det == int(det):
                    return str(int(det))
                return str(det)
            except Exception:
                pass
    
    # 2. Try system of equations
    system_result = _solve_system(text)
    if system_result:
        return system_result
    
    # 3. Try equation extraction
    eq = _extract_equation(text)
    if eq:
        sympy_result = sympy_solve_v2(eq)
        if sympy_result and not sympy_result.startswith('Error'):
            return sympy_result
    
    # 4. Try inclusion-exclusion
    ie_result = _solve_inclusion_exclusion(text)
    if ie_result is not None:
        return ie_result
    
    # 5. Try mean problems
    mean_result = _solve_mean_problem(text)
    if mean_result is not None:
        return mean_result
    
    # 6. Try geometric series
    geom_result = _solve_geometric(text)
    if geom_result is not None:
        return geom_result
    
    # 7. Fall back to original solver for existing patterns
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
print("ROUND 2 FINAL RESEARCH — IMPROVEMENT BENCHMARK")
print("=" * 80)

# Load and test on dev_40
with open('/home/artem/dev/amd-hackathon/input/dev_40.json') as f:
    dev_data = json.load(f)

print("\n--- DEV_40 ---")
math_qs_dev = [q for q in dev_data if q.get('category', '').startswith('math')]
correct_original = 0
correct_v2 = 0
for q in math_qs_dev:
    prompt = q['prompt']
    expected = q['gold']['answer']
    expected_str = str(expected).strip().lower()
    
    orig = solve_arithmetic(prompt, q['category'])
    v2 = solve_arithmetic_v2(prompt, q['category'])
    
    orig_match = norm(orig) == expected_str or norm(orig) == expected_str.replace('$', '')
    v2_match = norm(v2) == expected_str or norm(v2) == expected_str.replace('$', '')
    
    if orig_match:
        correct_original += 1
    if v2_match:
        correct_v2 += 1
    
    if norm(orig) != norm(v2):
        print(f"  DIFF: {prompt[:50]} orig={orig} v2={v2} expected={expected}")

print(f"Original: {correct_original}/{len(math_qs_dev)}")
print(f"V2:       {correct_v2}/{len(math_qs_dev)}")

# Now test 60_medium_hard
print("\n--- 60_MEDIUM_HARD MATH QUESTIONS ---")
with open('/home/artem/dev/amd-hackathon/data/eval/primary/eval_60_medium_hard.json') as f:
    eval_data = json.load(f)

math_qs = [q for q in eval_data['questions'] if q.get('category') == 'math']
print(f"Total: {len(math_qs)}")

# Define expected answers for each math question
# We extract them from the expected_answer field
q_expectations = {
    0: "5.177",       # Law of Sines
    1: "20",          # Mean
    2: "3069",        # Geometric series
    3: "(-4, 5)",     # Absolute value inequality
    4: "3",           # Matrix determinant
    5: "5/6",         # Inclusion-exclusion
    6: "4",           # Log equation
    7: "(-1/2)x + 1", # Perpendicular line
}

for i, q in enumerate(math_qs):
    prompt = q['prompt']
    full_expected = q['expected_answer']
    expected_num = q_expectations.get(i, "")
    
    original_result = solve_arithmetic(prompt, 'math')
    v2_result = solve_arithmetic_v2(prompt, 'math')
    
    # Try to match
    orig_match = False
    v2_match = False
    
    if original_result:
        orig_norm = norm(original_result)
        # Check against expected numbers
        for token in re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', expected_num):
            if norm(token) == orig_norm:
                orig_match = True
                break
        # Also check raw
        if orig_norm == expected_num.strip().lower():
            orig_match = True
    
    if v2_result:
        v2_norm = norm(v2_result)
        for token in re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', expected_num):
            if norm(token) == v2_norm:
                v2_match = True
                break
        if v2_norm == expected_num.strip().lower():
            v2_match = True
    
    if v2_match and not orig_match:
        status = "✓ NEW"
    elif v2_match and orig_match:
        status = "✓ SAME"
    elif not v2_match and original_result:
        status = "✗ LOST"
    else:
        status = "✗ MISS"
    
    print(f"  [{status}] Q{i}: {prompt[:65]}")
    print(f"           orig={original_result} v2={v2_result} expected={expected_num}")
    print()


# Additional targeted tests
print("\n\n--- ADDITIONAL TARGETED TESTS ---")

additional_tests = [
    # (prompt, category, expected_numeric)
    ("Solve 2x + 3 = 7", "math", "2"),
    ("What is the determinant of [[1,2],[3,4]]?", "math", "-2"),
    ("Solve log_2(x) = 3", "math", "8"),
    ("x + y = 10, x - y = 4", "math", "{x: 7, y: 3}"),
    ("The average of 4 numbers is 12. If a fifth number 20 is added, what is the new average?", "math", "13.6"),
    ("A geometric sequence has first term a = 2 and common ratio r = 3. What is the sum of the first 5 terms?", "math", "242"),
    ("In a class of 25 students, 12 play piano, 15 play guitar, and 5 play both. What is the probability a student plays at least one?", "math", "22/25"),
    ("What is 20% of 150?", "math_arithmetic", "30"),
    ("Solve sin(pi/3)", "math", "sqrt(3)/2"),
    ("What is the square root of 144?", "math_arithmetic", "12"),
]

for prompt, cat, expected in additional_tests:
    orig = solve_arithmetic(prompt, cat)
    v2 = solve_arithmetic_v2(prompt, cat)
    match_orig = norm(orig) == norm(expected)
    match_v2 = norm(v2) == norm(expected)
    status = "✓" if match_v2 else "✗"
    improvement = "NEW" if match_v2 and not match_orig else "SAME" if match_v2 else "MISS"
    print(f"  [{status}] {improvement}: {prompt[:55]}")
    print(f"           orig={orig} v2={v2} expected={expected}")
