#!/usr/bin/env python3
"""
Round 2 SymPy Math Solver Improvements — FINAL (v3).
"""
import json
import math
import re
import sys
sys.path.insert(0, '/home/artem/dev/amd-hackathon')

from agent.solvers.tools import sympy_solve as original_sympy_solve, calculator
from agent.solvers.deterministic import (
    solve_arithmetic, _extract_equation,
)

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication,
    function_exponentiation,
)

_FIXED_TRANSFORMS = standard_transformations + (implicit_multiplication, function_exponentiation)

def _preprocess_log_subscripts(text: str) -> str:
    """Convert log₂(x) -> log(x, 2) and log_2(x) -> log(x, 2)."""
    # Step 1: Convert unicode subscript digits to regular digits
    result = []
    for c in text:
        if '\u2080' <= c <= '\u2089':
            result.append(str(ord(c) - 0x2080))
        else:
            result.append(c)
    text = ''.join(result)
    
    # Step 2: Convert log_N(expr) -> log(expr, N)
    # Handles log_2(x), log_2(x-3), etc.
    # Note: this doesn't handle nested parentheses in the argument,
    # but covers the common cases
    text = re.sub(r'log_(\d+)\(([^()]*)\)', r'log(\2, \1)', text)
    
    # Step 3: If any remaining logN( patterns (after unicode conversion), 
    # also handle log2(x) -> log(x, 2)
    text = re.sub(r'\blog(\d+)\(([^()]*)\)', r'log(\2, \1)', text)
    
    return text

def _is_pure_math_expression(text: str) -> bool:
    """Check if text looks like a pure math expression."""
    stripped = text.strip()
    if len(stripped) > 150:
        return False
    math_chars = len(re.findall(r'[\d\+\-\*\/\^\(\)\=\[\]\.\,]', stripped))
    total_chars = len(stripped.strip())
    if total_chars == 0:
        return False
    ratio = math_chars / total_chars
    return ratio > 0.4 and len(stripped) < 80

def _prestrip_commands(text: str) -> str:
    """Remove leading command words. Avoids variable name collisions."""
    s = text.strip()
    s = re.sub(r'^(?:solve|calculate|compute|find|evaluate|simplify|determine|what\s+is|what\'s)\s+'
               r'(?:for\s+\w+\s*[:\s]+)?', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^solve\s+for\s+\w+\s*[:\s]+', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^(?:if|given)\s+', '', s, flags=re.IGNORECASE)
    return s.strip()

def sympy_solve_v2(expr_str: str) -> str | None:
    """Improved SymPy solver."""
    if not expr_str or not expr_str.strip():
        return None
    
    s = expr_str.strip()
    s = _preprocess_log_subscripts(s)
    
    def _parse(ss: str):
        try:
            return parse_expr(ss, local_dict={}, transformations=_FIXED_TRANSFORMS)
        except Exception:
            try:
                return sp.sympify(ss, {})
            except Exception:
                return None
    
    # Try equation
    if "=" in s:
        # Pre-strip, being careful
        cleaned = _prestrip_commands(s)
        if "=" not in cleaned:
            # Some command words removed the = content; use original
            pass
        else:
            s = cleaned
        
        try:
            left, right = s.split("=", 1)
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
    
    # Bare expression
    if _is_pure_math_expression(s):
        try:
            expr = _parse(s)
            if expr is not None and hasattr(expr, 'is_Number'):
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
# Matrix determinant extraction
# ============================================================

def _parse_matrix_rows(rows_str: list[str]) -> sp.Matrix | None:
    """Parse list of bracket contents into a SymPy Matrix."""
    rows = []
    for row_str in rows_str:
        row = [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', row_str)]
        if row:
            rows.append(row)
    if rows and all(len(r) == len(rows[0]) for r in rows):
        return sp.Matrix(rows)
    return None

def _extract_matrix(text: str) -> sp.Matrix | None:
    """Extract matrix from text."""
    # Pattern 1: Multiline brackets
    rows = re.findall(r'^\[(\d[\d\s,;.\-]*)\]$', text, re.MULTILINE)
    if rows:
        result = _parse_matrix_rows(rows)
        if result:
            return result
    
    # Pattern 2: Inline [[a,b,c],[d,e,f],...]  
    # Find the outermost brackets containing inner brackets
    m = re.findall(r'\[\[(.+?)\]\]', text)
    if m:
        inner = m[0]
        # Split by ],[ to get individual rows
        row_strs = inner.split('],[')
        result = _parse_matrix_rows(row_strs)
        if result:
            return result
    
    return None


# ============================================================
# Inclusion-Exclusion solver
# ============================================================

def _solve_inclusion_exclusion(text: str) -> str | None:
    """Solve '18 play soccer, 15 basketball, 8 both' problems."""
    text_lower = text.lower()
    
    if 'both' not in text_lower:
        return None
    
    # Extract all numbers
    nums = [float(x) for x in re.findall(r'\d+', text)]
    
    # Find total if mentioned
    total = None
    total_m = re.search(r'(?:class|group|total|of|among)\s+(?:of\s+)?(\d+)', text_lower)
    if total_m:
        t = float(total_m.group(1))
        candidates = [n for n in nums if n != t]
        if len(candidates) >= 2:
            total = t
            nums = candidates
    
    # Sort to identify set sizes and overlap
    unique_nums = sorted(set(nums))
    
    if len(unique_nums) >= 3:
        # Typically: smallest = both, next two = set A, set B
        both = unique_nums[0]
        set_a = unique_nums[1]
        set_b = unique_nums[2]
        
        union = set_a + set_b - both
        
        if total is not None:
            # Return probability as simplified fraction
            numerator = int(union)
            denominator = int(total)
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
# Mean problem solver (fixed)
# ============================================================

def _solve_mean_problem(text: str) -> str | None:
    """Solve mean/average word problems."""
    text_lower = text.lower()
    
    if 'mean' not in text_lower and 'average' not in text_lower:
        return None
    
    # Find counts (numbers near 'value', 'number', etc.)
    counts = re.findall(r'(\d+)\s+(?:values?|numbers?|items?|scores?|grades?|terms?|marks?)', text_lower)
    if not counts:
        return None
    
    # Find mean values: capture number AND its position in text
    # to maintain correct ordering (first mean mentioned = initial mean)
    means = []  # list of (position, value)
    
    # Pattern: "mean/average of X Y is NUMBER" (e.g., "average of 4 numbers is 12")
    for m in re.finditer(r'(?:mean|average)\s+of\s+\w+\s+\w+\s+is\s+(\d+(?:\.\d+)?)', text_lower):
        means.append((m.start(), m.group(1)))
    
    # Pattern: "mean/average is NUMBER" or "mean/average becomes NUMBER"
    for m in re.finditer(r'(?:mean|average)\s+(?:is|becomes?)\s+(\d+(?:\.\d+)?)', text_lower):
        means.append((m.start(), m.group(1)))
    
    # Pattern: "a mean/average of NUMBER" (not followed by count word)
    for m in re.finditer(r'(?:mean|average)\s+of\s+(\d+(?:\.\d+)?)', text_lower):
        num = m.group(1)
        pos = m.end(1)
        next_chunk = text_lower[pos:pos+15]
        if not re.match(r'\s+(?:values?|numbers?|items?|scores?|grades?|terms?|marks?)', next_chunk):
            means.append((m.start(), num))
    
    # Pattern: "mean ... becomes NUMBER" (handles intervening words)
    for m in re.finditer(r'(?:mean|average)\b[^.]*?(?:becomes?)\s+(\d+(?:\.\d+)?)', text_lower):
        means.append((m.start(), m.group(1)))
    
    # Pattern: "with a mean of NUMBER" or "with an average of NUMBER"
    for m in re.finditer(r'with\s+(?:an?\s+)?(?:mean|average)\s+of\s+(\d+(?:\.\d+)?)', text_lower):
        means.append((m.start(), m.group(1)))
    
    # Sort by position in text to maintain correct order
    means.sort(key=lambda x: x[0])
    # Extract just the values
    means = [v for _, v in means]
    
    # De-duplicate while preserving order
    seen = set()
    unique_means = []
    for v in means:
        if v not in seen:
            seen.add(v)
            unique_means.append(v)
    means = unique_means
    
    if not means:
        return None
    
    n = float(counts[0])
    initial_mean = float(means[0])
    total_sum = n * initial_mean
    
    # Check for removal vs addition
    has_removed = bool(re.search(r'remov|taken\s+out|eliminated|dropped', text_lower))
    has_added = bool(re.search(r'added|included|appended', text_lower))
    
    if has_removed and len(means) >= 2:
        new_mean = float(means[1])
        new_n = n - 1
        new_total = new_n * new_mean
        removed = total_sum - new_total
        if removed == int(removed):
            return str(int(removed))
        return f"{removed:.2f}".rstrip('0').rstrip('.')
    
    if has_added:
        # Find the added value
        added_val_m = re.search(r'(\d+(?:\.\d+)?)\s+is\s+added', text_lower)
        if not added_val_m:
            added_val_m = re.search(r'added\s+(?:a\s+)?(\d+(?:\.\d+)?)', text_lower)
        
        if added_val_m:
            added_val = float(added_val_m.group(1))
            
            if len(means) >= 2:
                # New mean is given, validate against added value
                new_mean = float(means[1])
                new_n = n + 1
                computed_added = (new_n * new_mean) - total_sum
                if abs(computed_added - added_val) < 0.01:
                    if computed_added == int(computed_added):
                        return str(int(computed_added))
                    return f"{computed_added:.2f}".rstrip('0').rstrip('.')
                return None  # Values don't match, can't solve
            
            # No second mean given — compute new average
            if 'new' in text_lower or 'what' in text_lower:
                new_n = n + 1
                new_mean_val = (total_sum + added_val) / new_n
                if new_mean_val == int(new_mean_val):
                    return str(int(new_mean_val))
                return f"{new_mean_val:.10f}".rstrip('0').rstrip('.')
    
    return None


# ============================================================
# Geometric sequence/series solver
# ============================================================

def _solve_geometric(text: str) -> str | None:
    """Solve geometric sequence/series problems."""
    if not re.search(r'geometric\s+(sequence|series|progression)', text, re.IGNORECASE):
        return None
    
    a = None
    r = None
    n = None
    
    a_m = re.search(r'first\s+term\s*(?:\w+\s*)?=?\s*(-?\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if a_m:
        a = float(a_m.group(1))
    
    r_m = re.search(r'common\s+ratio\s*(?:\w+\s*)?=?\s*(-?\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if r_m:
        r = float(r_m.group(1))
    
    n_m = re.search(r'first\s+(\d+)\s+terms?', text, re.IGNORECASE)
    if n_m:
        n = int(n_m.group(1))
    
    if a is not None and r is not None and n is not None:
        if r != 1:
            s = a * (r**n - 1) / (r - 1)
            if s == int(s):
                return str(int(s))
            return str(s)
        else:
            s = a * n
            return str(int(s)) if s == int(s) else str(s)
    
    return None


# ============================================================
# Log equation solver
# ============================================================

def _solve_log_equation(text: str) -> str | None:
    """Solve log equations."""
    if not re.search(r'\blog', text, re.IGNORECASE):
        return None
    
    cleaned = _preprocess_log_subscripts(text)
    
    # Strip prose prefix before the equation
    if ':' in cleaned:
        cleaned = cleaned.split(':', 1)[1].strip()
    # Strip leading prose words more aggressively
    cleaned = re.sub(
        r'^(?:solve|find|calculate|determine|evaluate|what\s+is|what\'s|compute|simplify)'
        r'(?:\s+for\s+\w+)?(?:\s+in\s+(?:the\s+)?)?'
        r'(?:equation|expression|problem|function)?\s*:?\s*',
        '', cleaned, flags=re.IGNORECASE
    ).strip()
    # Clean up any remaining leading colon/spaces
    cleaned = cleaned.lstrip(':').strip()
    
    if '=' in cleaned:
        try:
            left, right = cleaned.split('=', 1)
            l = parse_expr(left.strip(), local_dict={}, transformations=_FIXED_TRANSFORMS)
            r = parse_expr(right.strip(), local_dict={}, transformations=_FIXED_TRANSFORMS)
            eq = sp.Eq(l, r)
            sol = sp.solve(eq)
            if sol:
                return str(sol)
        except Exception:
            pass
    
    return None


# ============================================================
# System of equations solver
# ============================================================

def _solve_system(text: str) -> str | None:
    """Detect and solve systems of equations."""
    if len(text) > 300:
        return None
    
    # Guard: skip if text contains prose elements that look like equations
    # but aren't (e.g., "angle A = 30°", "side c = 10 cm")
    prose_eq_indicators = ['angle', '°', 'side', 'triangle', 'cm', 'mm', 'km', 'kg', 'lbs']
    if any(indicator in text.lower() for indicator in prose_eq_indicators):
        return None
    
    # Split by commas, semicolons, 'and'
    equations = re.split(r',|;|\band\b|\n', text)
    eqns = []
    for eq in equations:
        eq = eq.strip()
        if '=' in eq and re.search(r'[a-z]', eq, re.IGNORECASE):
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
    
    return None


# ============================================================
# COMBINED SOLVER V2
# ============================================================

def solve_arithmetic_v2(task: str, category: str) -> str | None:
    """Improved arithmetic solver with Round 2 enhancements."""
    text = task.strip()
    
    # 0. Check for inequality — return None immediately
    if re.search(r'[<>]', text) and re.search(r'inequalit', text, re.IGNORECASE):
        return None
    
    # 0.1 Log equation
    log_result = _solve_log_equation(text)
    if log_result is not None:
        return log_result
    
    # 0.2 SymPy on short expressions
    if _is_pure_math_expression(text):
        sympy_result = sympy_solve_v2(text)
        if sympy_result and not sympy_result.startswith('Error'):
            return sympy_result
    
    # 1. Matrix determinant
    if re.search(r'\b(det(?:erminant)?)\b', text, re.IGNORECASE):
        matrix = _extract_matrix(text)
        if matrix is not None:
            try:
                det = matrix.det()
                if det == int(det):
                    return str(int(det))
                return f"{float(det):.10f}".rstrip('0').rstrip('.')
            except Exception:
                pass
    
    # 2. System of equations
    system_result = _solve_system(text)
    if system_result:
        return system_result
    
    # 3. Equation extraction
    eq = _extract_equation(text)
    if eq:
        sympy_result = sympy_solve_v2(eq)
        if sympy_result and not sympy_result.startswith('Error'):
            return sympy_result
    
    # 4. Inclusion-exclusion
    ie_result = _solve_inclusion_exclusion(text)
    if ie_result is not None:
        return ie_result
    
    # 5. Mean problems
    mean_result = _solve_mean_problem(text)
    if mean_result is not None:
        return mean_result
    
    # 6. Geometric series
    geom_result = _solve_geometric(text)
    if geom_result is not None:
        return geom_result
    
    # 7. Fall back to original
    return solve_arithmetic(task, category)


# ============================================================
# BENCHMARK
# ============================================================

def norm(s):
    if s is None:
        return ""
    s = re.sub(r'^\[([^\]]+)\]$', r'\1', str(s))
    s = re.sub(r'\.0$', '', s)
    s = s.strip().lower()
    s = re.sub(r'^\[', '', s)
    s = re.sub(r'\]$', '', s)
    return s.strip()

def extract_expected_nums(expected_text: str) -> list[str]:
    """Extract numeric answers from expected answer text."""
    nums = re.findall(r'(?:^|\s|=)(-?\d+(?:\.\d+)?(?:/\d+)?)(?:\s|$|,)', expected_text)
    # Also find simplified fractions like 5/6
    fractions = re.findall(r'(\d+/\d+)', expected_text)
    # Get final numeric answer (last number or fraction)
    candidates = fractions + nums
    if candidates:
        return candidates
    return []

def check_match(result: str | None, expected_num: str) -> bool:
    if result is None:
        return False
    r = norm(result)
    e = norm(expected_num)
    if r == e:
        return True
    # Try numeric comparison
    try:
        r_val = float(r) if '/' not in r else float(r.split('/')[0]) / float(r.split('/')[1])
        e_val = float(e) if '/' not in e else float(e.split('/')[0]) / float(e.split('/')[1])
        return abs(r_val - e_val) < 0.01
    except (ValueError, ZeroDivisionError):
        return False


print("=" * 80)
print("ROUND 2 FINAL RESEARCH — V3 BENCHMARK")
print("=" * 80)

# DEV_40
with open('/home/artem/dev/amd-hackathon/input/dev_40.json') as f:
    dev_data = json.load(f)

print("\n--- DEV_40 ---")
math_qs_dev = [q for q in dev_data if q.get('category', '').startswith('math')]
correct_orig = sum(1 for q in math_qs_dev if norm(solve_arithmetic(q['prompt'], q['category'])) == norm(str(q['gold']['answer'])))
correct_v2 = sum(1 for q in math_qs_dev if norm(solve_arithmetic_v2(q['prompt'], q['category'])) == norm(str(q['gold']['answer'])))
print(f"Original: {correct_orig}/{len(math_qs_dev)}")
print(f"V2:       {correct_v2}/{len(math_qs_dev)}")

for q in math_qs_dev:
    prompt = q['prompt']
    expected = str(q['gold']['answer'])
    orig = solve_arithmetic(prompt, q['category'])
    v2 = solve_arithmetic_v2(prompt, q['category'])
    if norm(orig) != norm(v2):
        print(f"  DIFF: {prompt[:55]} orig={orig} v2={v2} expected={expected}")

# 60_MEDIUM_HARD
print("\n\n--- 60_MEDIUM_HARD MATH ---")
with open('/home/artem/dev/amd-hackathon/data/eval/primary/eval_60_medium_hard.json') as f:
    eval_data = json.load(f)

math_qs = [q for q in eval_data['questions'] if q.get('category') == 'math']

expected_key = {
    0: "5.177",
    1: "20",
    2: "3069",
    3: None,  # inequality
    4: "3",
    5: "5/6",
    6: "4",
    7: None,  # geometry
}

for i, q in enumerate(math_qs):
    prompt = q['prompt']
    expected_num = expected_key.get(i, "")
    
    orig = solve_arithmetic(prompt, 'math')
    v2 = solve_arithmetic_v2(prompt, 'math')
    
    orig_match = check_match(orig, expected_num) if expected_num else False
    v2_match = check_match(v2, expected_num) if expected_num else False
    
    # Special: expected None means should return None (can't handle)
    if expected_num is None:
        orig_match = orig is None
        v2_match = v2 is None
    
    if v2_match and not orig_match:
        status = "✓ NEW!"
    elif v2_match and orig_match:
        status = "✓ SAME"
    elif orig_match and not v2_match:
        status = "✗ REGRESSED"
    else:
        status = "✗ MISS"
    
    print(f"  [{status}] Q{i}")
    print(f"    Expected: {expected_num}")
    print(f"    Original: {orig}")
    print(f"    V2:       {v2}")
    print()


# TARGETED TESTS
print("\n--- TARGETED TESTS ---")
tests = [
    ("Solve 2x + 3 = 7", "math", "2"),
    ("2x + 3 = 7", "math", "2"),
    ("What is the determinant of [[1,2],[3,4]]?", "math", "-2"),
    ("Solve log_2(x) = 3", "math", "[8]"),
    ("x + y = 10, x - y = 4", "math", "{x: 7, y: 3}"),
    ("A geometric sequence has first term a=2, common ratio r=3. Sum of first 5 terms?", "math", "242"),
    ("In a class of 25, 12 play piano, 15 play guitar, 5 play both. Probability at least one?", "math", "22/25"),
    ("If x + 7 = 3x - 5, solve for x.", "math", "[6]"),
    ("What is 17 * 24?", "math_arithmetic", "408"),
    ("What is 15% of 240?", "math_arithmetic", "36"),
    ("Average of 4 numbers is 12. If a 5th number 20 is added, new average?", "math", "13.6"),
    ("5 values mean 12, one removed, new mean 10. What removed?", "math", "20"),
    ("What is the square root of 144?", "math_arithmetic", "12"),
]

for prompt, cat, expected in tests:
    orig = solve_arithmetic(prompt, cat)
    v2 = solve_arithmetic_v2(prompt, cat)
    match_orig = check_match(orig, expected)
    match_v2 = check_match(v2, expected)
    
    tag = ""
    if match_v2 and not match_orig:
        tag = " ✓ NEW!"
    elif match_v2:
        tag = " ✓"
    else:
        tag = " ✗"
    
    print(f"  [{tag}] {prompt[:55]}")
    print(f"         orig={str(orig):20} v2={str(v2):20} expected={expected}")
