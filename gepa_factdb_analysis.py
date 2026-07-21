#!/usr/bin/env python3
"""Analysis: FactDB coverage + detailed breakdown of factual question types."""

import json
import sys
import os

sys.path.insert(0, "/home/artem/dev/amd-hackathon")
from agent.solvers.fact_db import FactDB

DB_PATH = "/home/artem/dev/amd-hackathon/data/facts/facts.db"

def load_questions():
    questions = []
    with open("/home/artem/dev/amd-hackathon/data/eval/training-v3.json") as f:
        data = json.load(f)
    for d in data:
        if d.get("category") == "factual":
            questions.append({
                "prompt": d["prompt"],
                "expected": d["expected_answer"],
                "source": d.get("source", "training-v3"),
            })
    with open("/home/artem/dev/amd-hackathon/data/eval/factual_combined_80.json") as f:
        data = json.load(f)
    for d in data:
        questions.append({
            "prompt": d["prompt"],
            "expected": d["expected_answer"],
            "source": d.get("source", "factual_combined_80"),
        })
    return questions

def classify_question(q):
    """Classify question type."""
    p = q["prompt"]
    if p.startswith("Context:") or p.startswith("Context :"):
        return "counterfactual_w_context"
    if len(p) > 200:
        return "long_form"
    return "short_trivia"

def main():
    db = FactDB(DB_PATH)
    print(f"FactDB has {db.fact_count()} facts\n")
    
    questions = load_questions()
    
    # Classify
    types = {}
    for q in questions:
        t = classify_question(q)
        types.setdefault(t, 0)
        types[t] += 1
    print("=== Question Type Breakdown ===")
    for t, c in sorted(types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {t}: {c}")
    print(f"  TOTAL: {len(questions)}")
    print()
    
    # FactDB coverage
    print("=== FactDB Coverage ===")
    factdb_hits = 0
    factdb_high_conf = 0
    factdb_correct = 0
    factdb_incorrect = 0
    no_match = 0
    
    for q in questions:
        results = db.query(q["prompt"], k=3)
        if results:
            best_score, best_q, best_answer, best_source = results[0]
            factdb_hits += 1
            
            # Check if FactDB answer matches expected
            exp_lower = q["expected"].lower().strip()
            ans_lower = best_answer.lower().strip()
            is_correct = exp_lower in ans_lower or ans_lower in exp_lower
            
            if best_score >= 6.0:
                factdb_high_conf += 1
                if is_correct:
                    factdb_correct += 1
                else:
                    factdb_incorrect += 1
                    print(f"  HIGH CONF WRONG: [{q['prompt'][:60]}...]")
                    print(f"    Expected: {q['expected'][:80]}")
                    print(f"    FactDB:   {best_answer[:80]} (score={best_score:.1f})")
        else:
            no_match += 1
    
    print(f"\n  Total questions: {len(questions)}")
    print(f"  FactDB had any match: {factdb_hits}/{len(questions)} ({100*factdb_hits/len(questions):.1f}%)")
    print(f"  FactDB high conf (>=6.0): {factdb_high_conf}/{len(questions)} ({100*factdb_high_conf/len(questions):.1f}%)")
    print(f"  FactDB high conf + correct: {factdb_correct}/{factdb_high_conf}")
    print(f"  FactDB high conf + wrong: {factdb_incorrect}/{factdb_high_conf}")
    print(f"  No FactDB match: {no_match}/{len(questions)} ({100*no_match/len(questions):.1f}%)")
    
    # Analyze what types FactDB covers
    print("\n=== FactDB Coverage by Type ===")
    for t in sorted(types.keys()):
        type_qs = [q for q in questions if classify_question(q) == t]
        type_hits = sum(1 for q in type_qs if db.query(q["prompt"], k=3))
        type_high = sum(1 for q in type_qs if db.query(q["prompt"], k=3) and db.query(q["prompt"], k=3)[0][0] >= 6.0)
        print(f"  {t}: {type_hits}/{len(type_qs)} any match, {type_high}/{len(type_qs)} high conf")
    
    # Show examples of what FactDB matches well
    print("\n=== FactDB High-Conf Matches (Good) ===")
    shown = 0
    for q in questions:
        results = db.query(q["prompt"], k=3)
        if results:
            best_score, best_q, best_answer, best_source = results[0]
            if best_score >= 6.0:
                exp_lower = q["expected"].lower().strip()
                ans_lower = best_answer.lower().strip()
                is_correct = exp_lower in ans_lower or ans_lower in exp_lower
                if is_correct and shown < 10:
                    print(f"  [{best_score:.1f}] {q['prompt'][:60]}... -> {best_answer[:80]}")
                    shown += 1
    
    db.close()

if __name__ == "__main__":
    main()
