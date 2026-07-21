"""
Factual QA database using SQLite FTS5.
Zero external dependencies — uses Python's built-in sqlite3 module.
"""

import sqlite3
import json
import os
import re
import threading
from pathlib import Path
from typing import Optional


class FactDB:
    """
    Lightweight deterministic factual QA database.
    
    Usage:
        db = FactDB("data/facts/facts.db")
        db.load_facts("data/facts/dolly_facts.jsonl")  # one-time index
        results = db.query("What is the capital of France?")  # returns top-3 matches
    """
    
    def __init__(self, db_path: str = "data/facts/facts.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")  # faster writes
        self.conn.execute("PRAGMA cache_size=-8000")  # 8MB page cache
        self._create_tables()
    
    def _create_tables(self):
        """Create FTS5 virtual table + metadata table."""
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS facts USING fts5(
                id, category, question, answer, source,
                tokenize='porter unicode61'
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS fact_meta (
                id TEXT PRIMARY KEY,
                category TEXT,
                source TEXT,
                confidence REAL DEFAULT 1.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
    
    def add_fact(self, fact_id: str, category: str, question: str, 
                  answer: str, source: str = ""):
        """Add a single fact."""
        self.conn.execute(
            "INSERT INTO facts VALUES (?, ?, ?, ?, ?)",
            (fact_id, category, question, answer, source)
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO fact_meta (id, category, source) VALUES (?, ?, ?)",
            (fact_id, category, source)
        )
        self.conn.commit()
    
    def add_facts_batch(self, facts: list):
        """Add multiple facts in a batch."""
        self.conn.executemany(
            "INSERT INTO facts VALUES (?, ?, ?, ?, ?)",
            facts
        )
        self.conn.executemany(
            "INSERT OR REPLACE INTO fact_meta (id, category, source) VALUES (?, ?, ?)",
            [(f[0], f[1], f[4]) for f in facts]
        )
        self.conn.commit()
    
    def query(self, question: str, k: int = 3) -> list:
        """..."""
        with self._lock:
            return self._query_unsafe(question, k)
    
    def _query_unsafe(self, question: str, k: int = 3) -> list:
        """
        Search for facts matching the question.

        Returns list of (score, question, answer, source) tuples.
        Uses FTS5 AND query for precision; falls back to less strict
        matching only when needed.
        Score is negative BM25 rank (higher positive after conversion = better match).
        Curated facts (common-knowledge) are boosted by 25% to prefer them over
        noisy Dolly entries when scores are comparable.
        """
        # Strip "Context: ... Question: " prefix if present (SQuAD-style prompts)
        q_match = re.search(r'Question:\s*(.+?)$', question, re.IGNORECASE | re.DOTALL)
        if q_match:
            question = q_match.group(1).strip()
        # Clean the query for FTS5
        clean_q = self._clean_query(question)
        if not clean_q:
            return []
        
        # Always fetch enough raw results for proper re-scoring
        # Use a larger fetch to capture long-answer entries penalized by BM25
        fetch_k = max(k * 10, 50)
        
        # Strategy 1: Standard AND query (all terms must match) — highest precision
        try:
            cursor = self.conn.execute("""
                SELECT bm25(facts, 2.0, 0.75, 0.5, 0.5, 10.0, 5.0, 1.0) AS rank,
                       question, answer, source
                FROM facts
                WHERE facts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, [clean_q, fetch_k])
            results = cursor.fetchall()
            if results:
                return self._score_results(results, clean_q)[:k]
        except sqlite3.OperationalError:
            # FTS5 syntax error in query — fall through to next strategy
            results = []

        # Strategy 2: If no AND results, try prefix AND (each term with * wildcard)
        terms = [t for t in clean_q.split() if len(t) >= 2]
        if terms and not results:
            try:
                prefix_q = ' '.join(t + '*' for t in terms)
                cursor = self.conn.execute("""
                    SELECT bm25(facts, 2.0, 0.75, 0.5, 0.5, 10.0, 5.0, 1.0) AS rank,
                           question, answer, source
                    FROM facts
                    WHERE facts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """, [prefix_q, fetch_k])
                results = cursor.fetchall()
                if results:
                    return self._score_results(results, clean_q)[:k]
            except sqlite3.OperationalError:
                results = []

        # Strategy 3: Last resort — find facts where any key term matches
        if len(terms) >= 2 and not results:
            try:
                key_terms = sorted(terms, key=len, reverse=True)[:3]  # longest 3 terms
                or_q = ' OR '.join(t + '*' for t in key_terms)
                cursor = self.conn.execute("""
                    SELECT bm25(facts, 2.0, 0.75, 0.5, 0.5, 10.0, 5.0, 1.0) AS rank,
                           question, answer, source
                    FROM facts
                    WHERE facts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """, [or_q, fetch_k])
                results = cursor.fetchall()
                if results:
                    # Penalize OR match scores (halve them) to distinguish from exact AND matches
                    scored = [(max(-r[0], 0) * 0.5 * self._source_boost(r[3]), r[1], r[2], r[3]) for r in results]
                    scored.sort(key=lambda x: x[0], reverse=True)
                    return scored[:k]
            except sqlite3.OperationalError:
                pass

        return []
    
    def _score_results(self, results, query=""):
        """Convert FTS5 rank results to positive scores with source boosting.
        
        Also applies a question-relevance bonus: entries whose question field
        contains a higher proportion of query tokens get a boost. This helps
        comprehensive entries (with long answers) compete against short entries.
        """
        query_terms = set(query.lower().split()) if query else set()
        scored = []
        for r in results:
            rank, question, answer, source = r
            raw_score = max(-rank, 0)
            boost = self._source_boost(source)
            
            # Question relevance bonus: how many query terms appear in the question
            if query_terms:
                q_words = set(re.sub(r'[^\w\s]', '', question.lower()).split())
                overlap = len(query_terms & q_words)
                # Moderate bonus: 0.8x per matching term, capped at 4.0x total
                # This helps entries whose question closely matches the query
                # (e.g. "ranks" vs "rank" in navy queries) without distorting scores
                relevance = 1.0 + (overlap * 0.8)
                relevance = min(relevance, 4.0)
                boost *= relevance
            
            scored.append((raw_score * boost, question, answer, source))
        # Re-sort by boosted score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored
    
    def _source_boost(self, source: str) -> float:
        """Return a score multiplier based on source reliability."""
        boosts = {
            "common-knowledge": 1.25,  # Boost curated facts
            "pop-culture-v1": 1.30,    # Boost pop culture facts (longer answers penalized by BM25)
            "dolly-15k": 1.0,          # Neutral for Dolly
        }
        return boosts.get(source, 1.0)
    
    def _clean_query(self, text: str) -> str:
        """Clean question text for FTS5 query syntax."""
        # Remove common stop words that don't help matching
        text = re.sub(r'\b(what|who|when|where|why|how|which|is|are|was|were|do|does|did|has|have|had|in|the|a|an|of|for|to|at|by|with|on|it|be|been|being)\b', '', text, flags=re.I)
        # Remove punctuation (including hyphens which break FTS5 parsing)
        text = re.sub(r'[?,.!;:\'"-]', ' ', text)
        # Remove FTS5 special characters and problematic symbols
        text = re.sub(r'[()*"^%~@#$&=/+]', ' ', text)
        # Collapse whitespace
        text = ' '.join(text.split())
        return text[:200]  # safety limit
    
    def fact_count(self) -> int:
        cursor = self.conn.execute("SELECT COUNT(*) FROM facts")
        return cursor.fetchone()[0]
    
    def close(self):
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
