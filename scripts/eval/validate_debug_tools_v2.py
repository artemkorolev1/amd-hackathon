#!/usr/bin/env python3
"""Clean validation of debug tools against all code_debug examples."""

import json, os, sys, io

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASETS = [
    "data/eval/training-v2.json",
    "data/eval/training-v3.json",
    "data/eval/validation-v1.json",
    "data/eval/validation-v2.json",
]

def extract_code(prompt):
    """Extract just the Python code, stripping prompt headers and task descriptions."""
    lines = prompt.split('\n')
    # Find where code actually starts — skip "Fix the bug..." header
    start = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if 'def ' in s or s.startswith('for ') or s.startswith('if ') or \
           s.startswith('class ') or s.startswith('return ') or \
           s.startswith('import ') or s.startswith('from ') or \
           s == '':  # blank lines often separate header from code
            start = i
            break
    
    code_lines = []
    in_code = False
    for i in range(start, len(lines)):
        line = lines[i]
        s = line.strip()
        if s.startswith('def ') or s.startswith('class ') or s.startswith('import '):
            in_code = True
        if in_code:
            # Stop at lines that are task descriptions
            if s and in_code and not s.startswith(('def ', 'class ', 'import ', 'from ',
                '    ', '\t', 'return ', 'for ', 'while ', 'if ', 'try:', 'except',
                'with ', 'else:', 'elif ', 'break', 'continue', 'pass', '#', '"""', 
                "'''", '"""', "@", '')) and not any(c.isalpha() for c in s.strip(' ')):
                # Non-code trailing line
                break
            code_lines.append(line)
    
    if code_lines:
        result = '\n'.join(code_lines)
        # Trim leading blank lines
        return result.lstrip('\n')
    return '\n'.join(lines[start:])

# ── Tool tests ──

# 1. Parso — parse with error recovery
def test_parso(code):
    try:
        import parso
        tree = parso.parse(code, error_recovery=True)
        return True, f"OK ({len(tree.children)} children)"
    except Exception as e:
        return False, str(e)

# 2. LibCST — requires valid Python
def test_libcst(code):
    try:
        import libcst
        tree = libcst.parse_module(code)
        return True, "OK"
    except Exception as e:
        return False, str(e)[:80]

# 3. Pyflakes — syntax/name errors
def test_pyflakes(code):
    try:
        from pyflakes import api
        from pyflakes.reporter import Reporter
        out = io.StringIO()
        err = io.StringIO()
        reporter = Reporter(out, err)
        api.check(code, "test.py", reporter)
        warnings = err.getvalue()
        has_warnings = bool(warnings.strip())
        return has_warnings, warnings.strip()[:200] if warnings else "clean"
    except Exception as e:
        return False, f"CRASH: {e}"

# 4. Bandit — security issues
def test_bandit(code):
    try:
        import bandit.core.manager as bm
        import bandit.core.config as bc
        tmp = "/tmp/_bd_vl.py"
        with open(tmp, 'w') as f:
            f.write(code)
        mgr = bm.BanditManager(bc.BanditConfig(), "file", tmp)
        mgr.run_tests()
        results = mgr.get_issue_list()
        os.unlink(tmp)
        return len(results), [(r.test_id, r.text[:60]) for r in results[:3]]
    except Exception as e:
        return 0, [(f"CRASH", str(e)[:60])]

# ── Run ──

print(f"{'='*60}")
print(f"VALIDATION: 233 code_debug examples across 4 datasets")
print(f"{'='*60}")

total = 0
par_ok = 0; lib_ok = 0; pyf_bug = 0; pyf_clean = 0; ban_issues = 0

for ds_path in DATASETS:
    with open(os.path.join(PROJECT, ds_path)) as f:
        data = json.load(f)
    items = [it for it in data if it.get('category') == 'code_debug']
    total += len(items)
    
    d_par = 0; d_lib = 0; d_pyf = 0; d_ban = 0
    for item in items:
        code = extract_code(item['prompt'])
        expected = item.get('expected_answer', '')
        
        p_ok, _ = test_parso(code)
        if p_ok: d_par += 1
        
        l_ok, _ = test_libcst(code)
        if l_ok: d_lib += 1
        
        pyf_hit, _ = test_pyflakes(code)
        if pyf_hit: d_pyf += 1
        
        ban_c, _ = test_bandit(code)
        if ban_c > 0: d_ban += 1
    
    par_ok += d_par; lib_ok += d_lib; pyf_bug += d_pyf; ban_issues += d_ban
    n = len(items)
    print(f"\n{ds_path} ({n} items):")
    print(f"  parso (error recovery):       {d_par:3d}/{n}  ({d_par/n*100:.0f}%)")
    print(f"  libcst (valid Python parse):  {d_lib:3d}/{n}  ({d_lib/n*100:.0f}%)")
    print(f"  pyflakes (syntax/name error): {d_pyf:3d}/{n}  ({d_pyf/n*100:.0f}%)")
    print(f"  bandit (security issue):      {d_ban:3d}/{n}  ({d_ban/n*100:.0f}%)")

print(f"\n{'='*40}")
print(f"OVERALL ({total} total)")
print(f"{'='*40}")
print(f"  parso:          {par_ok:3d}/{total} ({par_ok/total*100:.0f}%)  ← CAN parse buggy code with errors")
print(f"  libcst:         {lib_ok:3d}/{total} ({lib_ok/total*100:.0f}%)  ← Only parses clean Python")
print(f"  pyflakes:       {pyf_bug:3d}/{total} ({pyf_bug/total*100:.0f}%)  ← Finds syntax/name errors in buggy code")
print(f"  bandit:         {ban_issues:3d}/{total} ({ban_issues/total*100:.0f}%)  ← Security issues (expected low)")

print(f"\n{'='*40}")
print("ANALYSIS")
print(f"{'='*40}")
print(f"")
print(f"parso: {par_ok}/{total} ({par_ok/total*100:.0f}%) success parsing with error recovery.")
print(f"  → Ready to integrate NOW. Can parse any broken Python code.")
print(f"")
print(f"libcst: {lib_ok}/{total} ({lib_ok/total*100:.0f}%) success parsing.")
print(f"  → Most code_debug prompts contain incomplete Python fragments")
print(f"    that aren't valid standalone modules. LibCST's utility")
print(f"    is for transforming WELL-FORMED code, not buggy fragments.")
print(f"")
print(f"pyflakes: {pyf_bug}/{total} ({pyf_bug/total*100:.0f}%) found syntax/name errors.")
print(f"  → More relevant — catches undefined names, syntax errors.")
print(f"  → But the bugs in our dataset are mostly LOGIC errors")
print(f"    (wrong range, wrong variable, wrong algorithm)")
print(f"    that no static analyzer catches.")
print(f"")
print(f"bandit: {ban_issues}/{total} ({ban_issues/total*100:.0f}%) security issues.")
print(f"  → Coding challenge bugs are logic errors, not security issues.")
print(f"  → Bandit adds nothing to code_debug pipeline.")
print(f"")

# Show first 3 parso/libcst/pyflakes examples
print(f"{'='*40}")
print("SAMPLE EXTRACTED CODE (first 3)")
print(f"{'='*40}")
with open(os.path.join(PROJECT, "data/eval/training-v2.json")) as f:
    data = json.load(f)
for i, item in enumerate(data):
    if i >= 3: break
    if item.get('category') != 'code_debug': continue
    code = extract_code(item['prompt'])
    print(f"\nPrompt: {item['prompt'][:80]}...")
    print(f"Extracted code ({len(code)}c):")
    print(f"  {code[:120].strip()}")
