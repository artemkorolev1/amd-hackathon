#!/usr/bin/env python3
"""
Build SQLite FTS5 fact database from Dolly 15K + common knowledge facts.

Usage:
    python scripts/build_fact_db.py

This script:
1. Loads Dolly 15K closed_qa and open_qa entries
2. Generates ~500 curated common knowledge facts
3. Builds the SQLite FTS5 database at data/facts/facts.db
"""

import json
import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.solvers.fact_db import FactDB

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

DOLLY_PATH = os.path.join(DATA_DIR, "raw", "prompt_data", "databricks-dolly-15k.jsonl")
COMMON_KNOWLEDGE_PATH = os.path.join(DATA_DIR, "facts", "common_knowledge.jsonl")
DB_PATH = os.path.join(DATA_DIR, "facts", "facts.db")


def load_dolly_qa(db: FactDB) -> int:
    """Load Dolly 15K closed_qa and open_qa entries into the database."""
    if not os.path.exists(DOLLY_PATH):
        logger.warning(f"Dolly data not found at {DOLLY_PATH}, skipping")
        return 0

    count = 0
    batch = []
    
    with open(DOLLY_PATH, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            cat = rec.get("category", "")
            if cat not in ("closed_qa", "open_qa"):
                continue
            
            question = rec.get("instruction", "")
            answer = rec.get("response", "")
            if not question or not answer:
                continue
            
            # Check for context-based questions — use the context-aware answer
            # For closed_qa, the context is often important
            context = rec.get("context", "")
            if context:
                # Some Dolly entries have context-dependent answers;
                # store a version that includes context for better matching
                if len(context) < 500:  # Keep context short for DB space
                    augmented_q = f"{context} {question}"
                    fact_id = f"dolly-ctx-{count}"
                    batch.append((fact_id, "factual", augmented_q, answer, "dolly-15k"))
            
            fact_id = f"dolly-{count}"
            batch.append((fact_id, "factual", question, answer, "dolly-15k"))
            count += 1
            
            if len(batch) >= 500:
                db.add_facts_batch(batch)
                logger.info(f"Loaded {count} Dolly facts so far...")
                batch = []
    
    if batch:
        db.add_facts_batch(batch)
    
    logger.info(f"Loaded {count} Dolly QA facts total")
    return count


def load_common_knowledge(db: FactDB) -> int:
    """Load common knowledge facts from JSONL into the database."""
    if not os.path.exists(COMMON_KNOWLEDGE_PATH):
        logger.warning(f"Common knowledge data not found at {COMMON_KNOWLEDGE_PATH}, skipping")
        return 0
    
    count = 0
    batch = []
    
    with open(COMMON_KNOWLEDGE_PATH, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            fact_id = rec.get("id", f"ck-{count}")
            category = rec.get("category", "general")
            question = rec.get("question", "")
            answer = rec.get("answer", "")
            
            if not question or not answer:
                continue
            
            batch.append((fact_id, category, question, answer, "common-knowledge"))
            count += 1
            
            if len(batch) >= 500:
                db.add_facts_batch(batch)
                batch = []
    
    if batch:
        db.add_facts_batch(batch)
    
    logger.info(f"Loaded {count} common knowledge facts")
    return count


def main():
    # Clean existing DB if present
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        logger.info(f"Removed existing database at {DB_PATH}")
    
    db = FactDB(DB_PATH)
    
    # Load Dolly QA facts
    dolly_count = load_dolly_qa(db)
    
    # Load common knowledge facts
    ck_count = load_common_knowledge(db)
    
    total = db.fact_count()
    db_size = os.path.getsize(DB_PATH)
    
    logger.info(f"=== Build Complete ===")
    logger.info(f"Dolly QA facts: {dolly_count}")
    logger.info(f"Common knowledge facts: {ck_count}")
    logger.info(f"Total facts: {total}")
    logger.info(f"Database size: {db_size:,} bytes ({db_size/1024/1024:.2f} MB)")
    
    # Run a quick sanity check
    test_queries = [
        "What is the capital of France?",
        "Which planet is the largest?",
        "What is the chemical symbol for gold?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?",
    ]
    
    logger.info("=== Sanity Checks ===")
    for q in test_queries:
        results = db.query(q, k=1)
        if results:
            score, question, answer, source = results[0]
            logger.info(f"  Q: {q}")
            logger.info(f"  A: {answer} (score={score:.2f}, source={source})")
        else:
            logger.warning(f"  Q: {q} -> NO MATCH")
    
    db.close()


if __name__ == "__main__":
    main()
