#!/usr/bin/env python3
"""Check if validation/training factual entries are already in FactDB."""
import json
import sys
sys.path.insert(0, "/home/artem/dev/amd-hackathon")
from agent.solvers.fact_db import FactDB

db = FactDB("/home/artem/dev/amd-hackathon/data/facts/facts.db")

for fn in ["training-v1.json", "training-v2.json", "validation-v1.json", "validation-v2.json", "validation-v3.json"]:
    path = f"/home/artem/dev/amd-hackathon/data/eval/{fn}"
    with open(path) as f:
        data = json.load(f)
    factual = [d for d in data if d.get("category") == "factual"]
    
    # Check a sample of 5
    in_db = 0
    not_in_db = 0
    for d in factual[:20]:
        results = db.query(d["prompt"], k=1)
        if results and results[0][0] >= 3.0:
            in_db += 1
        else:
            not_in_db += 1
            if not_in_db <= 3:
                print(f"  NOT IN DB: [{fn}] {d['prompt'][:60]}... -> {d['expected_answer'][:40]}")
    
    total = len(factual)
    print(f"\n{fn}: {total} factual entries, sample {in_db}/20 in DB, {not_in_db}/20 NOT in DB")

# Also check what sources exist
print("\n=== Distinct sources in FactDB ===")
cursor = db.conn.execute("SELECT DISTINCT source FROM facts LIMIT 20")
for row in cursor:
    print(f"  {row[0]}")
cursor = db.conn.execute("SELECT COUNT(*) as c, source FROM facts GROUP BY source ORDER BY c DESC LIMIT 10")
for row in cursor:
    print(f"  {row[1]}: {row[0]} facts")

db.close()
