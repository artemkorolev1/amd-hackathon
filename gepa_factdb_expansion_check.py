#!/usr/bin/env python3
"""Check how many factual training entries exist across all training/validation sets."""

import json
import os

EVAL_DIR = "/home/artem/dev/amd-hackathon/data/eval"

files = {
    "training-v1.json": "NQ/NaturalQuestions",
    "training-v2.json": "NQ/NaturalQuestions", 
    "training-v3.json": "NQ/NaturalQuestions",
    "validation-v1.json": "NQ/NaturalQuestions",
    "validation-v2.json": "NQ/NaturalQuestions",
    "validation-v3.json": "NQ/NaturalQuestions",
}

# Also check common_knowledge.jsonl
ck_path = "/home/artem/dev/amd-hackathon/data/facts/common_knowledge.jsonl"
if os.path.exists(ck_path):
    with open(ck_path) as f:
        ck_count = sum(1 for line in f if line.strip())
    print(f"common_knowledge.jsonl: {ck_count} facts currently loaded")
else:
    print("common_knowledge.jsonl: NOT FOUND")

print("\n=== Factual entries in eval files that COULD be loaded into FactDB ===")
total_factual = 0
for fn, source in files.items():
    path = os.path.join(EVAL_DIR, fn)
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        factual = [d for d in data if d.get("category") == "factual"]
        total_factual += len(factual)
        # Show first 3 prompts as samples
        print(f"  {fn:30s}: {len(factual):3d} factual entries (source: {source})")
        for d in factual[:2]:
            print(f"    Example: {d['prompt'][:70]}... -> {d['expected_answer'][:50]}")
    else:
        print(f"  {fn}: NOT FOUND")

print(f"\n  TOTAL unloaded factual entries: {total_factual}")

# Also check training-v3 for ALL factual entries' sources
print("\n=== training-v3.json factual source breakdown ===")
with open(os.path.join(EVAL_DIR, "training-v3.json")) as f:
    data = json.load(f)
factual = [d for d in data if d.get("category") == "factual"]
sources = {}
for d in factual:
    s = d.get("source", "unknown")
    sources[s] = sources.get(s, 0) + 1
for s, c in sorted(sources.items(), key=lambda x: x[1], reverse=True):
    print(f"  {s}: {c}")

# Check if these are already in FactDB (duplicate detection)
print("\n=== Duplicate check vs existing FactDB ===")
from agent.solvers.fact_db import FactDB
db = FactDB("/home/artem/dev/amd-hackathon/data/facts/facts.db")
print(f"Current FactDB has {db.fact_count()} facts")

# Try querying one of the training factual questions to see if it matches
test_q = factual[0] if factual else None
if test_q:
    results = db.query(test_q["prompt"], k=1)
    if results:
        score, q, a, src = results[0]
        print(f"Test query: '{test_q['prompt'][:60]}...'")
        print(f"  Expected: {test_q['expected_answer'][:60]}")
        print(f"  FactDB: {a[:60]} (score={score:.1f}, source={src})")
    else:
        print(f"Test query had NO FactDB match")

db.close()
