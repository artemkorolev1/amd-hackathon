"""
Validate code_tool_router cascade — trains on training-v2, validates on validation-v2.
"""
import json, os, sys, contextlib, io

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, "../.."))
sys.path.insert(0, _PROJECT)
os.environ['LOG_LEVEL'] = 'ERROR'
os.environ['MODEL_PATH'] = '/dev/null'

from scripts.grade_answer import fuzzy_match
from agent.solvers.code_tool_router import route_code, CodeRoutingResult


def run_route(prompt, cat):
    """Run route with output suppression."""
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        return route_code(prompt, cat)


for label, path in [
    ('TRAINING-v2', os.path.join(_PROJECT, 'data/eval/training-v2.json')),
    ('VALIDATION-v2', os.path.join(_PROJECT, 'data/eval/validation-v2.json')),
]:
    with open(path) as f:
        data = json.load(f)

    print(f'=== {label} - Code Tool Router Cascade ===')
    print()

    for cat in ('code_gen', 'code_debug'):
        items = [it for it in data if it['category'] == cat]
        if not items:
            continue

        # Track routing decisions
        route_counts = {}
        correct_by_route = {}
        total_correct = 0
        total = len(items)

        for item in items:
            p = item['prompt']
            e = item.get('expected_answer', '')

            result = run_route(p, cat)
            tool = result.tool
            ans = result.answer

            route_counts[tool] = route_counts.get(tool, 0) + 1
            if tool not in correct_by_route:
                correct_by_route[tool] = {'c': 0, 't': 0}
            correct_by_route[tool]['t'] += 1

            if ans and fuzzy_match(ans, e):
                correct_by_route[tool]['c'] += 1
                total_correct += 1

        print(f'  {cat} ({total} items): {total_correct}/{total} = {total_correct/total*100:.1f}% correct')
        for tool, count in sorted(route_counts.items()):
            s = correct_by_route[tool]
            pct = s['c'] / s['t'] * 100 if s['t'] else 0
            print(f'    {tool:30s}: {count:4d} items routed, {s["c"]:3d}/{s["t"]:3d} correct ({pct:5.1f}%)')
        print()

    # Deep dive: per-step analysis for code_gen
    cg = [it for it in data if it['category'] == 'code_gen']
    if cg:
        print(f'  Code_gen cascade step-by-step:')
        step_counts = {
            'step1_exact_match': 0,
            'step2_algorithm_name': 0,
            'step3_spec_with_tests': 0,
            'step4_llm': 0,
        }
        step_correct = {k: 0 for k in step_counts}
        for item in cg:
            p = item['prompt']
            e = item.get('expected_answer', '')
            result = run_route(p, 'code_gen')
            step = result.tool
            if step == 'template_exact_match':
                step_counts['step1_exact_match'] += 1
            elif step == 'template_algorithm_match':
                step_counts['step2_algorithm_name'] += 1
            elif step == 'llm_complex':
                step_counts['step3_spec_with_tests'] += 1
            elif step == 'llm_simple':
                step_counts['step4_llm'] += 1

        for step, count in sorted(step_counts.items()):
            pct = count / len(cg) * 100
            print(f'    {step:30s}: {count:4d}/{len(cg)} ({pct:5.1f}%)')
    print()
