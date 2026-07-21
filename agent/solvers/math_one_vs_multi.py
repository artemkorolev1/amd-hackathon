"""
Binary classifier for 1-step vs multi-step GSM8K problems.

Uses deterministic keyword/regex rules to identify problems whose answers
require only a single computation step (one <<expr=result>> marker).

Strategy:
  Stage 1 - Hard reject problems containing strong multi-step keywords
            (each, per, then, next, also, another, remaining, after, finally).
  Stage 2 - Score remaining questions on positive (1-step) and negative
            (multi-step) linguistic features. Classify as 1-step only when
            the net score >= 2 (conservative, high-precision regime).

Reference statistics (GSM8K train, n=7,473):
  1-step problems: 404 (5.4%)
  Multi-step problems: 6,974 (93.3%)
  Zero-step (no <<...>> markers): 95 (1.3%)

Performance (train / test):
  Precision: 13.4% / 18.2%  (2.5-3.7x above baseline)
  Recall:    4.7%  / 6.2%
  F1:        7.0%  / 9.2%
"""

import re


def classify_one_vs_multi(question: str) -> bool:
    """Return True if *question* is likely a 1-step problem, else False.

    Conservative — only predicts True when reasonably confident.
    False negatives (predicting multi-step for a true 1-step) are acceptable;
    the downstream multi-step pipeline handles those anyway.
    """
    q = question.lower()

    # ------------------------------------------------------------------
    # STAGE 1 — Hard multi-step exclusions
    # These keywords almost always signal sequential / multi-stage reasoning
    # ------------------------------------------------------------------
    _HARD_REJECT = re.compile(
        r"\b(each|per|then|next|also|another|remaining|after|finally)\b"
    )
    if _HARD_REJECT.search(q):
        return False

    # ------------------------------------------------------------------
    # STAGE 2 — Feature scoring
    # ------------------------------------------------------------------
    score = 0

    nums = re.findall(r"\d+", q)
    num_count = len(nums)

    # Split on sentence boundaries (., !, ?) and count non-empty pieces
    sentences = [s for s in re.split(r"[.!?]", q) if s.strip()]
    sentence_count = len(sentences)

    # ----- negative features (multi-step indicators) -------------------

    # Multiplicative / comparison words → multi-step
    if re.search(r"\b(twice|double|triple|half|times|than)\b", q):
        score -= 2
    if re.search(r"\b(more|less|fewer)\b", q):
        score -= 2

    # Ordinal indicators → narrative / sequential
    if re.search(r"\b(first|second|third|last)\b", q):
        score -= 2

    # Rate or periodic expressions → unit-rate / multi-stage
    if re.search(r"\b(every)\b", q):
        score -= 3
    if re.search(
        r"\ban? (hour|day|week|month|year|ounce|pound|kilogram|gallon|liter|mile|meter)\b",
        q,
    ):
        score -= 2
    if re.search(r"\b(monthly|weekly|yearly|daily|hourly)\b", q):
        score -= 3
    if re.search(r"\bin \d+ (minutes?|hours?|days?|weeks?|months?)\b", q):
        score -= 2

    # Fractions and percentages → often multi-step
    if re.search(r"\d+/\d+", q):
        score -= 2
    if re.search(r"%|percent", q):
        score -= 2

    # Multiple information-carrying sentences → narrative / multi-step
    if sentence_count >= 3:
        score -= 3
    elif sentence_count >= 2:
        score -= 1

    # Too many numbers → more than one computation
    if num_count >= 5:
        score -= 3
    elif num_count >= 4:
        score -= 1

    # Length correlates with complexity
    if len(q) > 250:
        score -= 2
    elif len(q) > 200:
        score -= 1

    # Spelled-out numbers → usually more verbose / multi-entity
    if re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\b", q):
        score -= 2

    # Unit words — usually signal unit conversions or multi-entity counts
    if re.search(
        r"\b(dozen|legs?|wheels?|inches?|feet|yards?|pounds?|ounces?|ml|liters?|meters?|grams?)\b",
        q,
    ):
        score -= 2

    # ----- positive features (1-step indicators) -----------------------

    # Short questions with few numbers → single computation
    if 2 <= num_count <= 3 and len(q) < 200:
        score += 2
    if num_count == 2 and len(q) < 150:
        score += 2

    # Simple question openings
    if re.search(r"^what (is|was|are)\b", q):
        score += 2
    if re.search(r"^if ", q) and 2 <= num_count <= 3:
        score += 1
    if re.search(r"(how many|how much)", q) and 2 <= num_count <= 3:
        score += 1

    # Single sentence → fewer narrative stages
    if sentence_count == 1:
        score += 1

    # Bonus for having no remaining negative keywords at all
    _REMAINING_NEGATIVES = re.compile(
        r"\b(every|half|twice|double|triple|times|than|"
        r"more|less|fewer|first|second|last|"
        r"one|two|three|four|five)\b"
    )
    if not _REMAINING_NEGATIVES.search(q):
        score += 1

    # Conservative threshold
    return score >= 2
