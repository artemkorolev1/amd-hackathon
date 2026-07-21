#!/usr/bin/env python3
"""
Load pop culture facts from JSONL into FactDB.
Usage:
    python scripts/load_popculture_facts.py
"""

import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.solvers.fact_db import FactDB

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

FACTS_JSONL = os.path.join(DATA_DIR, "facts", "pop_culture_facts_v1.jsonl")
DB_PATH = os.path.join(DATA_DIR, "facts", "facts.db")

def main():
    if not os.path.exists(FACTS_JSONL):
        print(f"ERROR: Facts file not found at {FACTS_JSONL}")
        print("Run `python3 data/facts/build_popculture_facts.py` first.")
        sys.exit(1)

    # Connect to existing FactDB (do NOT delete it)
    db = FactDB(DB_PATH)
    before = db.fact_count()
    print(f"Current fact count before loading: {before}")

    # Read facts from JSONL
    batch = []
    with open(FACTS_JSONL, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            fact_id = rec.get("id", "")
            category = rec.get("category", "general")
            question = rec.get("question", "")
            answer = rec.get("answer", "")
            source = rec.get("source", "pop-culture-v1")
            if not question or not answer:
                continue
            batch.append((fact_id, category, question, answer, source))

    # Insert in sub-batches of 500
    total_added = 0
    sub_batch_size = 500
    for i in range(0, len(batch), sub_batch_size):
        sub = batch[i:i + sub_batch_size]
        db.add_facts_batch(sub)
        total_added += len(sub)
        print(f"  Added {total_added}/{len(batch)}...")

    after = db.fact_count()
    db.close()

    print(f"\n=== Load Complete ===")
    print(f"Facts loaded: {total_added}")
    print(f"Fact count before: {before}")
    print(f"Fact count after:  {after}")
    print(f"Net increase: {after - before}")


if __name__ == "__main__":
    main()
