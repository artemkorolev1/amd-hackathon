#!/usr/bin/env python3
"""Deterministic format post-processor for LLM sentiment output.

Strips markdown, explanations, and normalisation noise to extract clean
sentiment labels ("positive", "negative", "neutral", "mixed").

Usage:
    from agent.solvers.format_normalizer import normalize_sentiment_output
    label, confidence = normalize_sentiment_output("**Positive!!**")
    # → ("positive", "high")
"""

from __future__ import annotations

import re
from typing import Optional

# ── Known labels (all lower-case) ─────────────────────────────────────────────

VALID_LABELS = frozenset({"positive", "negative", "neutral", "mixed"})

ABBREVIATIONS = {
    "pos": "positive",
    "neg": "negative",
    "neut": "neutral",
    "mix": "mixed",
}

# ── Helpers ───────────────────────────────────────────────────────────────────


def _strip_markdown(text: str) -> str:
    """Remove markdown / reStructuredText formatting characters."""
    text = re.sub(r"\*\*+", "", text)
    text = re.sub(r"__+", "", text)
    text = re.sub(r"\*", "", text)
    text = re.sub(r"_", "", text)
    text = re.sub(r"`", "", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^-{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"```[\w]*\s*", "", text)
    text = re.sub(r"```", "", text)
    return text


def _strip_punctuation(text: str) -> str:
    """Collapse non-alphabetic chars to spaces so we can find clean words."""
    text = re.sub(r"[^a-zA-Z\s\-/]+", " ", text)
    return text


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings (iterative)."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr.append(min(curr[-1] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def _find_label_via_levenshtein(word: str) -> Optional[str]:
    """If word is a short typo of a valid label (distance <= 2), return label."""
    if len(word) < 3:
        return None
    best_label = None
    best_dist = 3  # only accept ≤ 2
    for label in VALID_LABELS:
        dist = _levenshtein_distance(word, label)
        if dist < best_dist:
            best_dist = dist
            best_label = label
    return best_label


def _find_keywords(text: str) -> list[str]:
    """Return all valid-label keywords found in text, in order of appearance.

    Handles exact matches and common inflections (plural, -ing, -ed).
    """
    lower = text.lower()
    matches: list[tuple[int, str]] = []

    # 1. Exact word-boundary match
    for label in VALID_LABELS:
        for m in re.finditer(r"\b" + re.escape(label) + r"\b", lower):
            matches.append((m.start(), label))

    # 2. Prefix match for inflected forms (e.g. "negatives" → "negative")
    words = re.findall(r"[a-zA-Z]+", lower)
    for w in words:
        for label in VALID_LABELS:
            if w == label:
                continue  # already caught above
            # Check if word starts with label and has common suffix
            if w.startswith(label) and len(w) > len(label) and len(w) <= len(label) + 3:
                suffix = w[len(label):]
                if suffix in {"s", "es", "ed", "ing", "ly", "ion", "ions"}:
                    matches.append((lower.index(w), label))

    matches.sort(key=lambda x: x[0])
    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for _, label in matches:
        if label not in seen:
            seen.add(label)
            result.append(label)
    return result


def _has_contrastive(text: str) -> bool:
    """Check if text contains contrastive words that indicate mixed sentiment."""
    contrastive = {
        "but", "however", "although", "though", "yet", "nevertheless",
        "on the other hand", "conversely", "while", "whereas",
    }
    lower = text.lower()
    for word in contrastive:
        if word in lower:
            return True
    return False


def _apply_negation_rules(text: str) -> Optional[str]:
    """Check negation patterns and return the inverted label if matched."""
    lower = text.lower()
    # "not X" inversion
    if re.search(r"\bnot\s+positive\b", lower):
        return "negative"
    if re.search(r"\bnot\s+negative\b", lower):
        return "positive"
    if re.search(r"\bnot\s+neutral\b", lower):
        return "mixed"
    # "neither X nor Y"
    if re.search(r"\bneither\s+positive\s+nor\s+negative\b", lower):
        return "neutral"
    if re.search(r"\bneither\s+negative\s+nor\s+positive\b", lower):
        return "neutral"
    if re.search(r"\bneither\s+positive\s+", lower):
        return "negative"
    if re.search(r"\bneither\s+negative\s+", lower):
        return "positive"
    return None


def _apply_mixed_rules(text: str, clean: str) -> Optional[str]:
    """Check if the text should be classified as mixed."""
    lower_clean = clean.lower()
    text_lower = text.lower()

    # "both X and Y" pattern
    if re.search(r"\bboth\s+positive\s+and\s+negative\b", lower_clean):
        return "mixed"
    if re.search(r"\bboth\s+negative\s+and\s+positive\b", lower_clean):
        return "mixed"

    # "X/Y" pattern
    if re.search(r"\bpositive\s*/\s*negative\b", text_lower):
        return "mixed"
    if re.search(r"\bnegative\s*/\s*positive\b", text_lower):
        return "mixed"

    # Both positive and negative keywords appear with contrastive words
    keywords = _find_keywords(clean)
    has_positive = "positive" in keywords
    has_negative = "negative" in keywords
    if has_positive and has_negative and _has_contrastive(text):
        return "mixed"

    return None


def normalize_sentiment_output(raw_output: Optional[str]) -> tuple[str, str]:
    """Normalize LLM sentiment output to a clean label.

    Args:
        raw_output: Raw text from LLM (could be markdown, explanation, etc.)

    Returns:
        (normalized_label, confidence)
        label: "positive", "negative", "neutral", or "mixed"
               (or "unknown" if no label can be extracted)
        confidence: "high", "medium", "low" based on extraction certainty
    """
    # ── Step 0: Handle empty / None ───────────────────────────────────────────
    if not raw_output or not raw_output.strip():
        return "unknown", "low"

    text = raw_output.strip()
    lower = text.lower()

    # ── Step 1: Direct label match ────────────────────────────────────────────
    if lower in VALID_LABELS:
        return lower, "high"

    # ── Step 2: Abbreviation direct match ─────────────────────────────────────
    if lower in ABBREVIATIONS:
        return ABBREVIATIONS[lower], "medium"

    # ── Step 3: Strip markdown and re-check ───────────────────────────────────
    stripped = _strip_markdown(text).strip()
    lower_stripped = stripped.lower()
    if lower_stripped in VALID_LABELS:
        return lower_stripped, "high"

    # ── Step 4: Strip punctuation and re-check ────────────────────────────────
    clean = _strip_punctuation(stripped).strip()
    # Normalise whitespace after punctuation removal
    clean = re.sub(r"\s+", " ", clean).strip()
    lower_clean = clean.lower()
    if lower_clean in VALID_LABELS:
        return lower_clean, "high"

    # ── Step 5: Negation detection (before keyword extraction) ────────────────
    neg_result = _apply_negation_rules(text)
    if neg_result:
        return neg_result, "medium"

    # ── Step 6: Mixed detection (before single-label extraction) ──────────────
    mixed_result = _apply_mixed_rules(text, clean)
    if mixed_result:
        return mixed_result, "high"

    # ── Step 7: Keyword extraction — single keyword ───────────────────────────
    keywords = _find_keywords(clean)
    if len(keywords) == 1:
        return keywords[0], "high"

    # ── Step 8: Multiple keywords — return the LAST one (final verdict) ───────
    if len(keywords) > 1:
        return keywords[-1], "medium"

    # ── Step 9: Sentence-by-sentence extraction with Levenshtein ─────────────
    sentences = re.split(r"[.!?]+", clean)
    for sent in sentences:
        sent = sent.strip().lower()
        if not sent:
            continue
        words = re.findall(r"[a-zA-Z]+", sent)
        for w in words:
            w_clean = w.strip("-/")
            if w_clean in VALID_LABELS:
                return w_clean, "high"

    # ── Step 10: Typo correction via Levenshtein ──────────────────────────────
    words = re.findall(r"[a-zA-Z]+", lower_clean)
    for w in words:
        result = _find_label_via_levenshtein(w)
        if result:
            return result, "medium"

    # ── Step 11: Abbreviation in words ────────────────────────────────────────
    for w in words:
        if w in ABBREVIATIONS:
            return ABBREVIATIONS[w], "medium"

    # ── Step 12: Nothing worked ───────────────────────────────────────────────
    return "unknown", "low"


# ── Test block ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        # (raw_input, expected_label, expected_confidence)
        ("positive", "positive", "high"),
        ("**Positive**", "positive", "high"),
        ("The tone is positive.", "positive", "high"),
        ("I would classify this as negative.", "negative", "high"),
        ("**Negative!!**", "negative", "high"),
        ("positive!", "positive", "high"),
        ("It's neutral.", "neutral", "high"),
        ("Both positive and negative.", "mixed", "high"),
        ("", "unknown", "low"),
        ("NEGATIVE", "negative", "high"),
        ("Positiv", "positive", "medium"),  # typo via Levenshtein
        ("negative.", "negative", "high"),
        ("**Positive**", "positive", "high"),
        ("The sentiment of this review is positive.", "positive", "high"),
        ("I think it's negative, but I'm not sure.", "negative", "high"),
        ("**Neutral**", "neutral", "high"),
        ("positive/negative", "mixed", "high"),
        ("It's positive!!!", "positive", "high"),
        ("NEGATIVE!!", "negative", "high"),
        ("Both positive and negative aspects.", "mixed", "high"),
        ("neither positive nor negative", "neutral", "medium"),
        ("Not positive", "negative", "medium"),
        ("Not negative", "positive", "medium"),
        ("I would say this is neutral.", "neutral", "high"),
        ("pos", "positive", "medium"),
        ("neg", "negative", "medium"),
        ("neut", "neutral", "medium"),
        ("The tone is overwhelmingly positive.", "positive", "high"),
        ("## Negative", "negative", "high"),
        ("> Positive", "positive", "high"),
        ("`mixed`", "mixed", "high"),
        ("___Negative___", "negative", "high"),
        ("   ", "unknown", "low"),
        (None, "unknown", "low"),
        ("Positive and negative but mostly balanced", "mixed", "high"),
        ("I think it's positive, however there are negatives", "mixed", "high"),
        ("Positive. But also negative.", "mixed", "high"),
        ("Negativ", "negative", "medium"),  # typo
        ("Nutral", "neutral", "medium"),  # typo
        ("Positivo", "positive", "medium"),  # typo
        ("It's Positive!", "positive", "high"),
        ("positive.", "positive", "high"),
        ("***Mixed***", "mixed", "high"),
        ("## Positive result", "positive", "high"),
        (">>> negative <<<", "negative", "high"),
        ("I am neutral on this.", "neutral", "high"),
        ("The outcome is negative.", "negative", "high"),
    ]

    passed = 0
    failed = 0
    print(f"{'Status':<6} {'Input':<50} {'Got':<12} {'Expected':<12} {'Conf':<8}")
    print("-" * 90)
    for raw, expected_label, expected_conf in test_cases:
        label, conf = normalize_sentiment_output(raw)
        status = "✅" if (label == expected_label) else "❌"
        if label == expected_label:
            passed += 1
        else:
            failed += 1
        print(f"{status:<6} {repr(raw):<50} {label:<12} {expected_label:<12} {conf:<8}")

    print(f"\n{'='*50}")
    print(f"Passed: {passed}/{len(test_cases)}")
    print(f"Failed: {failed}/{len(test_cases)}")
    if failed:
        print("❌ SOME TESTS FAILED")
    else:
        print("✅ ALL TESTS PASSED")
