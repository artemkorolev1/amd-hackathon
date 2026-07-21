#!/usr/bin/env python3
"""Validate deterministic debug tools (pyflakes, bandit, parso, libcst)
against all code_debug examples across all 4 datasets."""

import json, os, sys, re, subprocess

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASETS = [
    "data/eval/training-v2.json",
    "data/eval/training-v3.json",
    "data/eval/validation-v1.json",
    "data/eval/validation-v2.json",
]

# ── Helpers ──

def extract_code_from_prompt(prompt):
    """Extract Python code from a code_debug prompt."""
    lines = prompt.split('\n')
    code_lines = []
    in_code = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('def ') or stripped.startswith('class '):
            in_code = True
        if in_code:
            code_lines.append(line)
        # Stop at blank line after code that introduces task description
        if in_code and stripped == '' and len(code_lines) > 2:
            remaining = '\n'.join(lines[i:])
            if 'Task:' in remaining or 'Write a' in remaining:
                break
    return '\n'.join(code_lines) if code_lines else prompt


def run_pyflakes(code):
    """Run pyflakes API. Returns (warnings: list, crashed: bool)."""
    try:
        from pyflakes.api import check
        from pyflakes.reporter import Reporter
        warnings = []
        class CapReporter(Reporter):
            def unexpectedError(self, fn, msg):
                warnings.append(("UNEXPECTED", msg, 0))
            def syntaxError(self, fn, msg, lineno, offset, txt):
                warnings.append(("SYNTAX", msg, lineno))
            def flake(self, msg):
                warnings.append(("FLAKE", str(msg.message_args or msg.message), 0))
        check(code, "prog.py", CapReporter())
        return warnings, False
    except Exception as e:
        return [("CRASH", str(e), 0)], True


def run_bandit(code):
    """Run bandit. Returns (issues: list, crashed: bool)."""
    try:
        import bandit.core.manager as bm
        import bandit.core.config as bc
        tmp = "/tmp/_bd_test.py"
        with open(tmp, 'w') as f:
            f.write(code)
        mgr = bm.BanditManager(bc.BanditConfig(), "file", tmp)
        mgr.run_tests()
        results = mgr.get_issue_list()
        os.unlink(tmp)
        return [(r.test_id, r.text, r.line_number) for r in results], False
    except Exception as e:
        return [("CRASH", str(e), 0)], True


def run_parso(code):
    """Check if parso can parse (error recovery). Returns (parsed_ok, has_syntax_errors, detail)."""
    try:
        import parso
        tree = parso.parse(code, error_recovery=True)
        # Check for error nodes
        def _find_errors(node):
            errs = []
            if hasattr(node, 'type') and node.type == 'error_node':
                errs.append(node)
            if hasattr(node, 'children'):
                for c in node.children:
                    errs.extend(_find_errors(c))
            return errs
        errors = _find_errors(tree)
        return True, len(errors) > 0, f"{len(tree.children)} children, {len(errors)} error nodes"
    except Exception as e:
        return False, True, str(e)


def run_libcst(code):
    """Check if libcst can parse as valid Python."""
    try:
        import libcst
        tree = libcst.parse_module(code)
        return True, "OK"
    except Exception as e:
        return False, str(e)


# ── Main ──

print(f"{'='*70}")
print(f"DETERMINISTIC DEBUG TOOL VALIDATION — {233} code_debug examples")
print(f"{'='*70}")
print()

tool_summary = {
    "pyflakes": {"detected": 0, "total": 0, "crash": 0, "fp": 0},
    "bandit": {"detected": 0, "total": 0, "crash": 0},
    "parso": {"parsed": 0, "total": 0, "crash": 0},
    "libcst": {"parsed": 0, "total": 0, "crash": 0},
}

for ds_path in DATASETS:
    with open(os.path.join(PROJECT, ds_path)) as f:
        data = json.load(f)
    
    debug_items = [it for it in data if it.get('category') == 'code_debug']
    print(f"── {ds_path} ({len(debug_items)} items) ──")
    
    ds_pyf = 0; ds_ban = 0; ds_par = 0; ds_lib = 0
    ds_fp = 0
    
    for item in debug_items:
        prompt = item['prompt']
        expected = item.get('expected_answer', '')
        code = extract_code_from_prompt(prompt)
        has_def = 'def ' in code
        
        # pyflakes on buggy code
        pyf_w, pyf_crashed = run_pyflakes(code)
        tool_summary["pyflakes"]["total"] += 1
        if pyf_crashed:
            tool_summary["pyflakes"]["crash"] += 1
        if len(pyf_w) > 0:
            tool_summary["pyflakes"]["detected"] += 1
            ds_pyf += 1
        
        # pyflakes on expected (fixed) — false positive check
        exp_w, _ = run_pyflakes(expected)
        if len(exp_w) > 0:
            tool_summary["pyflakes"]["fp"] += 1
            ds_fp += 1
        
        # bandit
        ban_w, ban_crashed = run_bandit(code)
        tool_summary["bandit"]["total"] += 1
        if ban_crashed:
            tool_summary["bandit"]["crash"] += 1
        if len(ban_w) > 0:
            tool_summary["bandit"]["detected"] += 1
            ds_ban += 1
        
        # parso
        par_ok, par_has_errors, par_detail = run_parso(code)
        tool_summary["parso"]["total"] += 1
        if not par_ok:
            tool_summary["parso"]["crash"] += 1
        if par_ok:
            tool_summary["parso"]["parsed"] += 1
            ds_par += 1
        
        # libcst
        lib_ok, lib_detail = run_libcst(code)
        tool_summary["libcst"]["total"] += 1
        if not lib_ok:
            tool_summary["libcst"]["crash"] += 1
        if lib_ok:
            tool_summary["libcst"]["parsed"] += 1
            ds_lib += 1
    
    n = len(debug_items)
    print(f"  pyflakes:  {ds_pyf}/{n} detected  (FP on expected: {ds_fp}/{n})")
    print(f"  bandit:    {ds_ban}/{n} found issues")
    print(f"  parso:     {ds_par}/{n} parsed with error recovery")
    print(f"  libcst:    {ds_lib}/{n} parsed as valid Python")
    print()

# ── Summary ──
print(f"{'='*70}")
print("OVERALL SUMMARY")
print(f"{'='*70}")
print()
print(f"{'Tool':<20} {'Total':>6} {'Detected':>10} {'Rate':>8} {'Crashes':>8}")
print(f"{'-'*56}")

pyf = tool_summary["pyflakes"]
print(f"{'pyflakes (buggy)':<20} {pyf['total']:>6} {pyf['detected']:>10} {pyf['detected']/max(pyf['total'],1)*100:>7.1f}% {pyf['crash']:>8}")
print(f"{'pyflakes (fixed FP)':<20} {pyf['total']:>6} {pyf['fp']:>10} {pyf['fp']/max(pyf['total'],1)*100:>7.1f}%")

ban = tool_summary["bandit"]
print(f"{'bandit':<20} {ban['total']:>6} {ban['detected']:>10} {ban['detected']/max(ban['total'],1)*100:>7.1f}% {ban['crash']:>8}")

par = tool_summary["parso"]
print(f"{'parso':<20} {par['total']:>6} {par['parsed']:>10} {par['parsed']/max(par['total'],1)*100:>7.1f}% {par['crash']:>8}")

lib = tool_summary["libcst"]
print(f"{'libcst':<20} {lib['total']:>6} {lib['parsed']:>10} {lib['parsed']/max(lib['total'],1)*100:>7.1f}% {lib['crash']:>8}")

print()
# What do pyflakes actually catch? Show some examples
print("─"*70)
print("SAMPLE PYFLAKES DETECTIONS (first few buggy code outputs)")
print("─"*70)

count = 0
for ds_path in DATASETS:
    if count >= 5:
        break
    with open(os.path.join(PROJECT, ds_path)) as f:
        data = json.load(f)
    for item in data:
        if count >= 5:
            break
        if item.get('category') != 'code_debug':
            continue
        code = extract_code_from_prompt(item['prompt'])
        w, _ = run_pyflakes(code)
        if w:
            print(f"\n--- Example {count+1} ---")
            print(f"CODE (first 200c): {code[:200]}")
            for wtype, wmsg, wline in w[:3]:
                print(f"  {wtype}:{wline} {wmsg[:120]}")
            count += 1

print()
print("─"*70)
print("SAMPLE PARSO/LIBCST FAILURES")
print("─"*70)
count = 0
for ds_path in DATASETS:
    if count >= 3:
        break
    with open(os.path.join(PROJECT, ds_path)) as f:
        data = json.load(f)
    for item in data:
        if count >= 3:
            break
        if item.get('category') != 'code_debug':
            continue
        code = extract_code_from_prompt(item['prompt'])
        par_ok, _, par_d = run_parso(code)
        lib_ok, lib_d = run_libcst(code)
        if not par_ok or not lib_ok:
            print(f"\n--- Example {count+1} ---")
            print(f"CODE (first 150c): {code[:150]}")
            print(f"  parso:  {'OK' if par_ok else 'FAIL: '+par_d[:80]}")
            print(f"  libcst: {'OK' if lib_ok else 'FAIL: '+lib_d[:80]}")
            count += 1
