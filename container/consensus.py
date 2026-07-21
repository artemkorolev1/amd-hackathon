"""
Consensus engine — voting, fuzzy matching, quality gates, and judge prompt.

Four answers per question → one final answer.

Pipeline:
  1. fuzzy_match pairwise → equivalence classes
  2. Largest class >= 3 → use it (high confidence)
  3. Largest class == 2 or all differ → judge prompt decides
  4. Quality check (empty/degenerate) → Fireworks fallback

Judge prompt uses the same model (one slot borrowed during judging phase).
"""

import json
import re
import math
import os
import sys
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))


# ── Fuzzy match (adapted from scripts/evaluate.py for self-containment) ─────

def _extract_numbers(s: str):
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", s)]


def _tokenize(s: str) -> set:
    return set(t for t in re.split(r"[^a-zA-Z0-9.]+", s.lower()) if t)


def fuzzy_match(answer: str, expected: str) -> bool:
    """Cascade: exact → substring → numeric 1% → token overlap."""
    a, e = answer.strip(), expected.strip()
    if not e:
        return bool(a)
    a_low, e_low = a.lower(), e.lower()

    # 1. Exact
    if a_low == e_low:
        return True
    # 2. Substring
    if e_low in a_low or (len(a) >= 3 and a_low in e_low):
        return True
    # 3. Numeric 1%
    na, ne = _extract_numbers(a), _extract_numbers(e)
    if na and ne:
        if len(na) == len(ne):
            if all(abs(na[i] - ne[i]) <= 0.01 if ne[i] == 0
                   else abs(na[i] - ne[i]) / abs(ne[i]) <= 0.01
                   for i in range(len(ne))):
                return True
        if len(ne) == 1:
            target = ne[0]
            for n in na:
                if target == 0:
                    if abs(n - target) <= 0.01:
                        return True
                elif abs(n - target) / abs(target) <= 0.01:
                    return True
    # 4. Token overlap
    a_tok, e_tok = _tokenize(a), _tokenize(e)
    stopwords = {"the", "a", "an", "is", "to", "of", "in", "and",
                 "that", "for", "it", "on", "with", "as", "at", "by", "or", "be"}
    e_tok -= stopwords
    a_tok -= stopwords
    if not e_tok:
        return False
    overlap = e_tok & a_tok
    threshold = 0.5 if len(e) < 50 else 0.3
    return len(overlap) >= len(e_tok) * threshold


# ── Quality checks ──────────────────────────────────────────────────────────

_DEGENERATE_PATTERNS = [
    r"\bi don'?t know\b", r"\bi do not know\b", r"\bi cannot\b",
    r"\bi can'?t\b", r"\bas an ai\b", r"\bunable to\b",
    r"\bno information\b", r"\bcannot answer\b", r"\bnot enough information\b",
    r"\binsufficient\b", r"\bsorry\b", r"\bthe text does not\b",
]


def is_degenerate(text: str) -> bool:
    """Check if answer is degenerate (empty, hedge, self-talk, repetition)."""
    t = text.strip()
    if not t:
        return True
    # Short numeric answers (e.g. "42", "6.0") are valid
    if re.match(r"^-?\d+(?:\.\d+)?$", t):
        return False
    if len(t) < 3:
        return True
    low = t.lower()
    for pat in _DEGENERATE_PATTERNS:
        if re.search(pat, low):
            return True
    words = t.split()
    if len(set(words)) / max(len(words), 1) < 0.25:
        return True  # heavy repetition
    return False


# ── Equivalence clustering ──────────────────────────────────────────────────

def equivalence_classes(answers: list[str]) -> list[set[int]]:
    """Group answer indices into equivalence classes using fuzzy_match."""
    groups: list[set[int]] = []
    assigned = set()
    for i in range(len(answers)):
        if i in assigned:
            continue
        group = {i}
        for j in range(i + 1, len(answers)):
            if j in assigned:
                continue
            if fuzzy_match(answers[i], answers[j]):
                group.add(j)
        assigned.update(group)
        groups.append(group)
    return groups


# ── Consensus decision ──────────────────────────────────────────────────────

def resolve_consensus(answers: list[str]) -> tuple[str, str, float]:
    """
    Given 4 answer strings, return (best_answer, method, confidence).

    Methods:
      'consensus'    — 3+ answers equivalent → high confidence
      'majority'     — 2 equivalent, judge picked one → medium confidence
      'judge'        — all differ, judge picked best → medium confidence
      'degenerate'   — all answers failed quality → low confidence (triggers fallback)
    """
    if not answers or all(is_degenerate(a) for a in answers):
        # Pick the least degenerate
        scored = sorted(answers, key=lambda a: (
            0 if not is_degenerate(a) else 1, -len(a.strip())
        ))
        return (scored[0] if scored else "", "degenerate", 0.1)

    groups = equivalence_classes(answers)
    largest = max(groups, key=len)
    largest_idx = list(largest)
    best_answer = answers[largest_idx[0]]

    if len(largest) >= 3:
        return (best_answer, "consensus", 0.95)
    elif len(largest) == 2:
        return (best_answer, "majority", 0.7)
    else:
        # All different — will need the judge
        return (best_answer, "no_consensus", 0.3)


# ── Judge prompt builder ────────────────────────────────────────────────────

def build_judge_prompt(
    question: str,
    answers: list[str],
    judge_template: str,
) -> str:
    """Format the judge prompt with question and anonymized candidates."""
    if len(answers) < 4:
        answers = answers + [""] * (4 - len(answers))
    return judge_template.format(
        question=question,
        answer_a=answers[0],
        answer_b=answers[1],
        answer_c=answers[2],
        answer_d=answers[3],
    )


def parse_judge_output(raw: str) -> tuple[int, str]:
    """
    Parse 'Best: C' or 'Best: C\nReason: ...' → (index 2, reason).
    Returns (best_index, reason_string). On failure: (-1, "parse_failed").
    """
    m = re.search(r"Best:\s*([A-Da-d])", raw)
    if m:
        idx = ord(m.group(1).upper()) - ord("A")
        reason = ""
        rm = re.search(r"Reason:\s*(.+)", raw, re.DOTALL)
        if rm:
            reason = rm.group(1).strip()
        return (idx, reason)
    return (-1, "parse_failed")


# ── Final merge ─────────────────────────────────────────────────────────────

def merge_answers(
    question: str,
    answers: list[str],
    judge_template: str,
    call_judge_fn=None,  # async callable that returns raw judge text
) -> dict:
    """
    Full merge pipeline. Returns:
      {
        "answer": str,
        "method": "consensus" | "majority" | "judge" | "fallback",
        "confidence": float,
        "judge_reason": str,
        "raw_answers": list[str],
        "degenerate": list[bool],
      }
    """
    result = {
        "answer": "",
        "method": "fallback",
        "confidence": 0.0,
        "judge_reason": "",
        "raw_answers": answers,
        "degenerate": [is_degenerate(a) for a in answers],
    }

    best, method, conf = resolve_consensus(answers)

    if method in ("consensus", "majority"):
        result["answer"] = best
        result["method"] = method
        result["confidence"] = conf
        return result

    # No consensus → use judge if callable provided
    if call_judge_fn and not all(is_degenerate(a) for a in answers):
        judge_text = build_judge_prompt(question, answers, judge_template)
        try:
            raw = call_judge_fn(judge_text)
            idx, reason = parse_judge_output(raw)
            if 0 <= idx < len(answers) and not is_degenerate(answers[idx]):
                result["answer"] = answers[idx]
                result["method"] = "judge"
                result["confidence"] = 0.7
                result["judge_reason"] = reason
                return result
            # Judge picked a degenerate one — fall through
        except Exception as e:
            result["judge_reason"] = f"judge_error: {e}"

    # Fallback: use the non-degenerate one with highest confidence from resolve
    non_degen = [(answers[i], i) for i in range(len(answers))
                 if not is_degenerate(answers[i])]
    if non_degen:
        result["answer"] = non_degen[0][0]
        result["method"] = "fallback_best"
        result["confidence"] = 0.3
    else:
        # All degenerate — return first non-empty or empty
        first_nonempty = next((a for a in answers if a.strip()), "")
        result["answer"] = first_nonempty
        result["method"] = "degenerate"
        result["confidence"] = 0.05

    return result
