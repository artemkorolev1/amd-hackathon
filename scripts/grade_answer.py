#!/usr/bin/env python3
"""Reusable answer grading logic — pure functions, no I/O, no CLI.

Extracted from scripts/evaluate.py for shared use by runner/evaluate.py,
runner/instrumented_evaluate.py, and runner/regression.py.

Usage:
    from scripts.grade_answer import fuzzy_match, grade_answer, summarization_grade
"""

from __future__ import annotations

import re
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Fuzzy matching helpers
# ---------------------------------------------------------------------------


def extract_numbers(s: str) -> List[float]:
    """Extract all decimal numbers from a string."""
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", s)]


def tokenize(s: str) -> set:
    """Lowercase, split on non-alphanumeric, return non-empty tokens."""
    return set(tok for tok in re.split(r"[^a-zA-Z0-9.]+", s.lower()) if tok)


def fuzzy_match(answer: str, expected: str) -> bool:
    """Check if *answer* matches *expected* — 4-strategy cascade.

    Strategies applied in order:
      1. Exact (case-insensitive) match
      2. Substring match (expected is substring of answer, or vice versa for
         short answers)
      3. Numeric comparison within 1% tolerance
      4. Token overlap — at least half of expected tokens appear in answer
         (for short expected answers) or a third (for longer ones)
    """
    a = answer.strip()
    e = expected.strip()
    if not e:
        return bool(a)

    a_low = a.lower()
    e_low = e.lower()

    # --- 1. Exact (case-insensitive) ---
    if a_low == e_low:
        return True

    # --- 2. Substring ---
    # Expected is contained in answer
    if e_low in a_low:
        return True
    # Short answer contained in expected (answer is a fragment of ground truth)
    if len(a) >= 3 and a_low in e_low:
        return True

    # --- 3. Numeric comparison within 1% ---
    na = extract_numbers(a)
    ne = extract_numbers(expected)
    if na and ne:
        # Same-length list of numbers: pairwise compare
        if len(na) == len(ne):
            if all(
                abs(na[i] - ne[i]) <= 0.01
                if ne[i] == 0
                else abs(na[i] - ne[i]) / abs(ne[i]) <= 0.01
                for i in range(len(ne))
            ):
                return True
        # Expected is a single number: any number in answer within 1%
        if len(ne) == 1:
            target = ne[0]
            for n in na:
                if target == 0:
                    if abs(n - target) <= 0.01:
                        return True
                elif abs(n - target) / abs(target) <= 0.01:
                    return True

    # --- 4. Token overlap ---
    a_tokens = tokenize(a)
    e_tokens = tokenize(e)

    # Strip very common words that add noise
    stopwords = {"the", "a", "an", "is", "to", "of", "in", "and", "that",
                 "for", "it", "on", "with", "as", "at", "by", "or", "be"}
    e_tokens -= stopwords
    a_tokens -= stopwords

    if not e_tokens:
        return False

    overlap = e_tokens & a_tokens
    # For short expected answers (< 50 chars, fewer tokens): require >= 50% overlap
    # For longer: require >= 30% overlap
    threshold = 0.5 if len(e) < 50 else 0.3
    if len(overlap) >= len(e_tokens) * threshold:
        return True

    return False


def grade_answer(answer: str, expected: str) -> Tuple[bool, str]:
    """Grade a single answer. Returns (passed, reason_string)."""
    a = answer.strip()
    e = expected.strip()

    if not a:
        return False, "Empty answer"

    if a.startswith("[ERROR]"):
        return False, f"Agent error: {a}"

    if fuzzy_match(a, e):
        return True, "Passed"

    # Build a helpful diagnostic
    reason_parts = []
    na = extract_numbers(a)
    ne = extract_numbers(e)
    if na and ne and len(na) == len(ne):
        for i in range(len(ne)):
            if ne[i] == 0:
                if abs(na[i] - ne[i]) > 0.01:
                    reason_parts.append(f"numeric mismatch {na[i]} vs {ne[i]}")
            elif abs(na[i] - ne[i]) / abs(ne[i]) > 0.01:
                reason_parts.append(f"numeric mismatch {na[i]} vs {ne[i]}")

    if not reason_parts:
        reason_parts.append(f"expected: {e[:120]}, got: {a[:120]}")

    return False, "; ".join(reason_parts)


# ---------------------------------------------------------------------------
# Summarization-specific grading
# ---------------------------------------------------------------------------


def summarization_grade(output: str, expected: str) -> bool:
    """Grade summarization output using entity + keyword overlap (3-signal cascade).

    Signals (tried in order):
      1. fuzzy_match cascade (catches near-exact)
      2. Entity recall (capitalized named entities)
      3. Keyword overlap (significant content words ≥4 chars)
      4. Numeric overlap (shared numbers)
    
    This is more lenient than fuzzy_match for free-form summaries where
    different vocabulary can express the same meaning.
    """
    # 1. Standard fuzzy_match cascade (catches near-exact)
    if fuzzy_match(output, expected):
        return True

    # 2. Extract capitalized entities (names, orgs, places)
    def extract_entities(text):
        return set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))

    exp_entities = extract_entities(expected)
    out_entities = extract_entities(output)
    if len(exp_entities) > 0:
        overlap = exp_entities & out_entities
        recall = len(overlap) / len(exp_entities)
        # Entity recall >= 50% or at least 2 entities match
        if recall >= 0.5 or len(overlap) >= 2:
            return True

    # 3. Keyword overlap: significant shared content words (4+ chars)
    exp_words = set(re.findall(r'[a-zA-Z]{4,}', expected.lower()))
    out_words = set(re.findall(r'[a-zA-Z]{4,}', output.lower()))
    if len(exp_words) > 0:
        word_overlap = len(exp_words & out_words) / len(exp_words)
        if word_overlap >= 0.4:
            return True

    # 4. Extract numbers and check overlap
    exp_nums = set(re.findall(r'\d+(?:\.\d+)?', expected))
    out_nums = set(re.findall(r'\d+(?:\.\d+)?', output))
    if exp_nums and exp_nums & out_nums:
        return True

    return False
