"""
Tool Routing Validator — tests optimal per-category solver chains and validates on held-out data.
"""

import json, os, sys, contextlib
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, "../.."))
sys.path.insert(0, _PROJECT)

from agent.solvers.deterministic import (
    solve_arithmetic, solve_logic, solve_sentiment, solve_ner as solve_ner_old,
    solve_factual_qa, solve_code_debugging, solve_code_generation, solve_summarization,
)
from agent.solvers.prototype_ner_v3 import solve_ner as solve_ner_v3
from agent.solvers.prototype_zebra_v2 import solve_zebra_puzzle
from agent.solvers.logic_reasoning import solve_logical_reasoning

# ── Proposed optimal routing (trained on training-v2 data) ──
ROUTING_TABLE = {
    "math": {
        "solver_chain": [],  # Deterministics useless for word problems — straight to LLM
        "consensus": True,
        "model": "qwen2.5-1.5b-instruct",
        "note": "0% fire/17% correct on training-v2. Skip deterministics entirely.",
    },
    "logic": {
        "solver_chain": [
            ("solve_logical_reasoning", solve_logical_reasoning),
            ("solve_logic", solve_logic),
        ],
        "consensus": False,
        "model": "qwen2.5-1.5b-instruct",
        "note": "28% correct on logic when fired. Keep zebra solver for v3-style prompts.",
    },
    "sentiment": {
        "solver_chain": [
            ("solve_sentiment", solve_sentiment),
        ],
        "consensus": False,
        "model": "qwen2.5-1.5b-instruct",
        "note": "73% correct — VADER handles strong signals. LLM for ambiguous cases.",
    },
    "ner": {
        "solver_chain": [
            ("solve_ner_v3", solve_ner_v3),
            ("solve_ner_old", solve_ner_old),
        ],
        "consensus": False,
        "model": "qwen2.5-coder-1.5b",
        "note": "v3 handles 93% fire but format mismatch (4% correct). Old regex has better match (20%). Keep both, prefer old for final answer.",
    },
    "factual": {
        "solver_chain": [
            ("solve_factual_qa", solve_factual_qa),
        ],
        "consensus": False,
        "model": "qwen2.5-1.5b-instruct",
        "note": "97% correct on 100% fire rate. FactDB is the star solver.",
    },
    "code_debug": {
        "solver_chain": [
            ("solve_code_debugging", solve_code_debugging),
        ],
        "consensus": False,
        "model": "qwen2.5-coder-1.5b",
        "note": "11% fire rate but 11-40% correct when it fires. Keep as free pre-filter.",
    },
    "code_gen": {
        "solver_chain": [],  # 0% correct from templates — straight to LLM
        "consensus": False,
        "model": "gemma-3-1b-it",
        "note": "62% fire but 0% correct — format mismatch with HumanEval. Skip deterministics.",
    },
    "summarization": {
        "solver_chain": [],  # 0% correct from Sumy — straight to LLM
        "consensus": False,
        "model": "qwen2.5-1.5b-instruct",
        "note": "100% fire but 0% correct — Sumy extractive doesn't match expected answer format. Skip deterministics.",
    },
}


def grade(expected: str, actual: str) -> bool:
    if not expected or not actual:
        return False
    e, a = expected.strip().lower(), actual.strip().lower()
    return e in a or a in e


def evaluate_chain(dataset_path: str, label: str):
    with open(dataset_path) as f:
        data = json.load(f)

    cat_stats = defaultdict(lambda: {"total": 0, "det_solved": 0, "names_fired": set()})

    for item in data:
        cat = item["category"]
        prompt = item["prompt"]
        expected = item.get("expected_answer", "")
        routing = ROUTING_TABLE.get(cat)
        if not routing:
            continue

        cat_stats[cat]["total"] += 1
        chain = routing["solver_chain"]

        # Try deterministic chain
        for name, fn in chain:
            try:
                with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
                    ans = fn(prompt, cat)
                if ans and grade(expected, ans):
                    cat_stats[cat]["det_solved"] += 1
                    cat_stats[cat]["names_fired"].add(name)
                    break  # First-in-chain wins
            except:
                pass

    print(f"=== {label} ({len(data)} Q) ===")
    print()
    total_solved = 0
    total_questions = 0
    for cat in sorted(cat_stats):
        s = cat_stats[cat]
        pct = s["det_solved"] / s["total"] * 100 if s["total"] else 0
        total_solved += s["det_solved"]
        total_questions += s["total"]
        chain = ROUTING_TABLE[cat]["solver_chain"]
        chain_names = ", ".join(n for n, _ in chain) if chain else "*(LLM ONLY)*"
        note = ROUTING_TABLE[cat]["note"]
        print(f"  {cat:15s}: {s['det_solved']:4d}/{s['total']:4d} = {pct:5.1f}%  {chain_names}")
        print(f"    {note}")
        print()
    print(f"  {'TOTAL':15s}: {total_solved:4d}/{total_questions:4d} = {total_solved/total_questions*100:.1f}%")
    print()
    return cat_stats


if __name__ == "__main__":
    train_stats = evaluate_chain(os.path.join(_PROJECT, "data/eval/training-v2.json"), "TRAINING-v2")
    val_stats = evaluate_chain(os.path.join(_PROJECT, "data/eval/validation-v2.json"), "VALIDATION-v2")
