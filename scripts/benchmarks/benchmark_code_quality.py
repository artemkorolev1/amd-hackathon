"""
Benchmark for format_and_lint — Black + Ruff code validation.
Tests on code_gen and code_debug expected answers from eval sets.
"""
import json
import sys
import time

sys.path.insert(0, '/home/artem/dev/amd-hackathon')
from agent.solvers.verify import format_and_lint, _valid_python, _extract_code

# ── Load eval questions ──
with open('/home/artem/dev/amd-hackathon/data/eval/training-v3.json') as f:
    data = json.load(f)

code_questions = [q for q in data if q['category'] in ('code_gen', 'code_debug')]
gen = [q for q in code_questions if q['category'] == 'code_gen']
debug = [q for q in code_questions if q['category'] == 'code_debug']

# Also load from primary eval sets
primary_sets = [
    '/home/artem/dev/amd-hackathon/data/eval/generated/build-A-40.json',
    '/home/artem/dev/amd-hackathon/data/eval/generated/build-B-40.json',
    '/home/artem/dev/amd-hackathon/data/eval/primary/eval_60_medium_hard.json',
    '/home/artem/dev/amd-hackathon/data/eval/primary/eval_hard_218.json',
]
extra_code_qs = []
for path in primary_sets:
    try:
        with open(path) as f:
            raw = json.load(f)
        # Handle different formats: dict with 'questions', or list directly
        if isinstance(raw, dict) and 'questions' in raw:
            items = raw['questions']
        elif isinstance(raw, list):
            items = raw
        else:
            continue
        for q in items:
            if isinstance(q, dict):
                cat = q.get('category', '').lower()
                # Normalize category names (code_debugging -> code_debug, code_generation -> code_gen)
                if cat in ('code_debugging', 'code_debug'):
                    q['category'] = 'code_debug'
                    extra_code_qs.append(q)
                elif cat in ('code_generation', 'code_gen'):
                    q['category'] = 'code_gen'
                    extra_code_qs.append(q)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  [skip] {path}: {e}")

all_code_qs = code_questions + extra_code_qs

# ── Test data: hand-crafted correct/buggy examples ──
handcrafted = {
    "correct_simple": {
        "code": "def add(a, b):\n    return a + b",
        "expect_syntax_ok": True,
    },
    "correct_class": {
        "code": "class Counter:\n    def __init__(self):\n        self.count = 0\n    def increment(self):\n        self.count += 1",
        "expect_syntax_ok": True,
    },
    "fragment_return": {
        "code": "return [s for s in text if s.lower() not in ['a', 'e', 'i', 'o', 'u']]",
        "expect_syntax_ok": True,
        "note": "return-only fragment"
    },
    "fragment_if": {
        "code": "if not numbers:\n    return []",
        "expect_syntax_ok": True,
        "note": "if fragment"
    },
    "buggy_missing_colon": {
        "code": "def foo(x)\n    return x",
        "expect_syntax_ok": False,
        "expect_error": True,
    },
    "buggy_unclosed_paren": {
        "code": "result = foo(x, y(",
        "expect_syntax_ok": False,
        "expect_error": True,
    },
    "wrong_indentation": {
        "code": "def bar():\n  return 1\n    extra = 2",
        "expect_syntax_ok": False,
        "expect_error": True,
    },
    "unused_variable": {
        "code": "def process(items):\n    x = 42\n    return [i * 2 for i in items]",
        "expect_syntax_ok": True,
        "expect_lint": True,
        "note": "ruff should catch unused variable"
    },
}

results = {}

print("=" * 70)
print("BENCHMARK: format_and_lint on code questions")
print("=" * 70)

# 1. Handcrafted tests
print("\n--- 1. Handcrafted examples ---")
for name, test in handcrafted.items():
    code = test['code']
    expect_ok = test['expect_syntax_ok']
    t0 = time.time()
    r = format_and_lint(code, relaxed=True)
    elapsed = time.time() - t0
    passed = (r['syntax_ok'] == expect_ok)
    status = "PASS" if passed else "FAIL"
    print(f"  {status:5s} | {name:30s} | syntax_ok={r['syntax_ok']} | lint={len(r['lint_errors'])} | {elapsed*1000:.0f}ms")
    if not passed:
        print(f"         error: {r.get('error', 'none')}")
    results[name] = {
        "status": status,
        "syntax_ok": r['syntax_ok'],
        "ast_ok": r['ast_ok'],
        "lint_count": len(r['lint_errors']),
        "has_formatted": r['formatted'] is not None,
        "latency_ms": round(elapsed * 1000, 1),
    }

# 2. code_gen expected answers
print("\n--- 2. code_gen expected answers (training-v3.json) ---")
gen_syntax_ok = 0
gen_lint_ok = 0
gen_total = len(gen)
gen_results = []
t0 = time.time()
for q in gen:
    r = format_and_lint(q['expected_answer'], relaxed=False)
    gen_results.append({
        "task_id": q.get('task_id', '?'),
        "syntax_ok": r['syntax_ok'],
        "lint_count": len(r['lint_errors']),
        "formatted": r['formatted'] != q['expected_answer'],
    })
    if r['syntax_ok']:
        gen_syntax_ok += 1
    if not r['lint_errors']:
        gen_lint_ok += 1
gen_elapsed = time.time() - t0
print(f"  Total: {gen_total} | Syntax OK: {gen_syntax_ok}/{gen_total} ({100*gen_syntax_ok//gen_total}%)")
print(f"  Lint OK: {gen_lint_ok}/{gen_total} | Avg latency: {gen_elapsed/gen_total*1000:.0f}ms/q")
if gen_syntax_ok < gen_total:
    for i, q in enumerate(gen):
        if not gen_results[i]['syntax_ok']:
            print(f"    FAIL syntax: {q['expected_answer'][:60]}...")

# 3. code_debug expected answers
print("\n--- 3. code_debug expected answers (training-v3.json) ---")
debug_syntax_ok = 0
debug_lint_ok = 0
debug_total = len(debug)
debug_results = []
t0 = time.time()
for q in debug:
    r = format_and_lint(q['expected_answer'], relaxed=False)
    debug_results.append({
        "task_id": q.get('task_id', '?'),
        "syntax_ok": r['syntax_ok'],
        "lint_count": len(r['lint_errors']),
        "formatted": r['formatted'] != q['expected_answer'],
        "error": r.get('error'),
    })
    if r['syntax_ok']:
        debug_syntax_ok += 1
    if not r['lint_errors']:
        debug_lint_ok += 1
debug_elapsed = time.time() - t0
print(f"  Total: {debug_total} | Syntax OK: {debug_syntax_ok}/{debug_total} ({100*debug_syntax_ok//debug_total}%)")
print(f"  Lint OK: {debug_lint_ok}/{debug_total} | Avg latency: {debug_elapsed/debug_total*1000:.0f}ms/q")

# 4. Extra code questions from other eval sets
print(f"\n--- 4. Extra code questions ({len(extra_code_qs)} from primary sets) ---")
extra_done = 0
extra_ok = 0
for q in extra_code_qs[:10]:
    exp = (q.get('expected_answer') or q.get('gold') or q.get('answer', ''))
    if isinstance(exp, (dict, list)):
        exp = str(exp)
    r = format_and_lint(exp)
    extra_done += 1
    if r['syntax_ok']:
        extra_ok += 1
    status = "PASS" if r['syntax_ok'] else "FAIL"
    print(f"  {status:5s} | {q['category']:12s} | syntax={r['syntax_ok']} | lint={len(r['lint_errors'])} | {str(exp)[:50]}")
print(f"\n  Syntax OK: {extra_ok}/{extra_done}")

# 5. Strict vs Relaxed comparison
print("\n--- 5. Strict vs Relaxed comparison (sampled 5 questions) ---")
for i, q in enumerate(code_questions[:5]):
    r_strict = format_and_lint(q['expected_answer'], relaxed=False)
    r_relaxed = format_and_lint(q['expected_answer'], relaxed=True)
    print(f"  Q{i}: strict_lint={len(r_strict['lint_errors'])} | relaxed_lint={len(r_relaxed['lint_errors'])} | {q['expected_answer'][:50]}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Handcrafted: {sum(1 for v in results.values() if v['status']=='PASS')}/{len(handcrafted)}")
print(f"  code_gen:     {gen_syntax_ok}/{gen_total} syntax OK, {gen_lint_ok}/{gen_total} lint-free")
print(f"  code_debug:   {debug_syntax_ok}/{debug_total} syntax OK, {debug_lint_ok}/{debug_total} lint-free")
print(f"  Avg latency:  {(gen_elapsed + debug_elapsed) / (gen_total + debug_total) * 1000:.0f}ms per question")
