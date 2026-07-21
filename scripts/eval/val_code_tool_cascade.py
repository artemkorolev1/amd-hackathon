"""
Validate binary cascade tree on both code datasets.
"""
import json, os, sys, csv, contextlib, io

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, "../.."))
sys.path.insert(0, _PROJECT)
os.environ['LOG_LEVEL'] = 'ERROR'

from scripts.grade_answer import fuzzy_match
from agent.solvers.code_tool_cascade import (
    route_code, RoutingResult,
    has_structured_io, is_data_structure, is_dp_problem, is_sort_search_math,
    fn_name_matches_template, known_algorithm_named,
)

def run(prompt, cat):
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        return route_code(prompt, cat)

def show_tree(prompt):
    """Trace the binary tree path for a code_gen prompt."""
    path = []
    # N0
    path.append(f'N0(structured_io)={"Y" if has_structured_io(prompt) else "N"}')
    if has_structured_io(prompt):
        path.append(f'N1(ds)={"Y" if is_data_structure(prompt) else "N"}')
        if not is_data_structure(prompt):
            path.append(f'N2(dp)={"Y" if is_dp_problem(prompt) else "N"}')
            if not is_dp_problem(prompt):
                path.append(f'N3(sort)={"Y" if is_sort_search_math(prompt) else "N"}')
    else:
        path.append(f'N4(fn_match)={"Y" if fn_name_matches_template(prompt) else "N"}')
        if not fn_name_matches_template(prompt):
            path.append(f'N5(alg_name)={"Y" if known_algorithm_named(prompt) else "N"}')
    return ' → '.join(path)


# ── training-v2 (HumanEval) ──
print('=== TRAINING-v2 (HumanEval, 200 code_gen) ===')
with open(os.path.join(_PROJECT, 'data/eval/training-v2.json')) as f:
    data = json.load(f)

cg = [it for it in data if it['category'] == 'code_gen']
route_counts = {}
correct_by_route = {}
for item in cg:
    p = item['prompt']
    e = item.get('expected_answer', '')
    result = run(p, 'code_gen')
    rt = result.tool
    route_counts[rt] = route_counts.get(rt, 0) + 1
    if rt not in correct_by_route:
        correct_by_route[rt] = {'c': 0, 't': 0}
    correct_by_route[rt]['t'] += 1
    ans = result.solve()
    if ans and fuzzy_match(ans, e):
        correct_by_route[rt]['c'] += 1

for tool in sorted(route_counts):
    s = correct_by_route[tool]
    pct = s['c']/s['t']*100 if s['t'] else 0
    print(f'  {tool:25s}: {route_counts[tool]:4d} routed  {s["c"]:3d}/{s["t"]:3d} correct ({pct:5.1f}%)')

print()
print('Tree path distribution:')
paths = {}
for item in cg:
    p = show_tree(item['prompt'])
    paths[p] = paths.get(p, 0) + 1
for path, count in sorted(paths.items(), key=lambda x: -x[1]):
    print(f'  {count:4d}x → {path}')
print()


# ── Kaggle dataset ──
print('=== KAGGLE (616 formal coding challenges) ===')
with open('/tmp/kaggle_coding_q.csv') as f:
    reader = csv.DictReader(f)
    kaggle_rows = list(reader)

# Build a prompt from each Kaggle row
kaggle_prompts = []
for r in kaggle_rows:
    prompt = f"{r['title']}: {r['description']}\n\nExamples: {r['examples'][:300]}\n\nConstraints: {r['constraints'][:200]}"
    kaggle_prompts.append(prompt)

# Test tree coverage
paths_k = {}
for p in kaggle_prompts:
    path = show_tree(p)
    paths_k[path] = paths_k.get(path, 0) + 1

for path, count in sorted(paths_k.items(), key=lambda x: -x[1]):
    print(f'  {count:4d}x → {path}')
print(f'  Total: {len(kaggle_prompts)}')
