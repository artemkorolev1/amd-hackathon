"""
Validate the cascade determination classifier.
Trains on training-v2, validates on validation-v2.
"""
import json, os, sys, contextlib, io

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, "../.."))
sys.path.insert(0, _PROJECT)

os.environ['MODEL_PATH'] = '/dev/null'
os.environ['LOG_LEVEL'] = 'ERROR'

from scripts.grade_answer import fuzzy_match
from agent.solvers.cascade_router import route

def evaluate(data_path, label):
    with open(data_path) as f:
        data = json.load(f)

    print(f'=== {label} ({len(data)} Q) ===')
    print('  Category        Correct Solved Total  Acc% MinScore')
    print('  ' + '-' * 50)
    
    total_correct = 0
    total_solved = 0
    total_q = 0
    
    for cat in ['factual', 'sentiment', 'ner', 'code_gen', 'code_debug',
                'logic', 'math', 'summarization']:
        items = [it for it in data if it['category'] == cat]
        if not items:
            continue
        c = 0
        s = 0
        for item in items:
            prompt = item['prompt']
            expected = item.get('expected_answer', '')
            try:
                with contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(io.StringIO()):
                    ans = route(cat, prompt)
                if ans:
                    s += 1
                    if fuzzy_match(ans, expected):
                        c += 1
            except:
                pass
        total_correct += c
        total_solved += s
        total_q += len(items)
        acc = c / len(items) * 100
        # Show the match_score threshold for this category
        from agent.solvers.cascade_router import ROUTE_TABLE
        entries = ROUTE_TABLE.get(cat, [])
        ms = entries[0].min_score if entries else '-'
        ms_str = str(ms) if ms != '-' else '-'
        print(f'  {cat:15s} {c:5d} {s:5d} {len(items):5d} {acc:5.1f}% {ms_str:>8s}')
    
    print(f'  {"TOTAL":15s} {total_correct:5d} {total_solved:5d} {total_q:5d} {total_correct/total_q*100:5.1f}%')
    print()
    return total_correct, total_solved, total_q

# Train on training-v2
evaluate(os.path.join(_PROJECT, 'data/eval/training-v2.json'), 'TRAINING-v2')
# Validate on validation-v2
evaluate(os.path.join(_PROJECT, 'data/eval/validation-v2.json'), 'VALIDATION-v2')

# Also test the code_gen function-name guard specifically
print('=== CODE_GEN DEEP DIVE ===')
with open(os.path.join(_PROJECT, 'data/eval/training-v2.json')) as f:
    data = json.load(f)
cg = [it for it in data if it['category'] == 'code_gen']

fn_matches = 0
keyword_matches = 0
no_match = 0
correct_when_route = 0

for item in cg:
    p = item['prompt']
    e = item.get('expected_answer', '')
    score = None
    from agent.solvers.cascade_router import match_code_gen_template
    score = match_code_gen_template(p)
    
    if score >= 1.0:
        fn_matches += 1
    elif score >= 0.3:
        keyword_matches += 1
    else:
        no_match += 1
    
    if score >= 0.5:  # Would route to solver
        ans = route('code_gen', p)
        if ans and fuzzy_match(ans, e):
            correct_when_route += 1

print(f'  Function name matches (score=1.0): {fn_matches}/200')
print(f'  Keyword matches (score=0.3-0.9): {keyword_matches}/200')
print(f'  No match (score < 0.3): {no_match}/200')
print(f'  Correct when routed to solver: {correct_when_route}/{fn_matches+keyword_matches}')
print(f'  Correct when function name match: ?/{fn_matches}')
