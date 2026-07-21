"""
Secondary factual↔math binary resolver — catches simple arithmetic word
problems misrouted as factual by the primary 8-way classifier.

These are GSM8K-style problems that use word-form arithmetic (half, twice,
double, triple, "how many") without explicit digit operators or calculation
keywords like "calculate", "solve", or "compute".

Architecture:
  resolve_mathfact(category_8way, prompt) -> str

  Pure deterministic — zero model calls, zero imports beyond stdlib.

  Only fires when primary="factual":
    → returns "math" when word-problem arithmetic patterns are detected
    → returns "factual" otherwise (no change)
"""

import re
from typing import Optional

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# ── Word-form arithmetic operators ──
# These are implicit arithmetic operations written in plain English
_WORD_ARITHMETIC_RE = re.compile(
    r"\b(?:"
    r"half|halve|halved|"
    r"twice|double|doubled|doubling|"
    r"triple|tripled|tripling|"
    r"quadruple|"
    r"(?:four|five|six|seven|eight|nine|ten)\s*times|"
    r"times\s+as\s+(?:many|much|fast|slow|old|long|tall|big|small|large|heavy|wide|deep)|"
    r"each\s+of\s+their|"
    r"apiece|per\s+(?:day|week|hour|minute|second|year|month|pound|ounce|gallon|liter|mile|kilometer)"
    r")\b",
    re.IGNORECASE,
)

# ── Ratio / rate / speed patterns ──
_RATIO_RE = re.compile(
    r"\b(?:"
    r"ratio\s+(?:of\s+)?\d+\s*:\s*\d+|"
    r"\d+\s*(?:mph|km/h|kmh|m/s|kph|per\s+(?:hour|minute|second|day|week|month|year))|"
    r"at\s+a\s+rate\s+of|"
    r"at\s+the\s+same\s+(?:speed|rate|pace)|"
    r"times\s+faster\s+than|times\s+slower\s+than|"
    r"half\s+as\s+(?:fast|long|many|much|old|big|large|heavy|wide|deep)"
    r")\b",
    re.IGNORECASE,
)

# ── Calculation question starters ──
# These are "how many/much/fast/far/long/old" questions that ask for
# computation rather than factual lookup
_CALC_QUESTION_RE = re.compile(
    r"\b(?:"
    r"how\s+(?:"
    r"  many\s+(?:"
    r"    (?:bolts|eggs|cups|pounds|ounces|gallons|liters|miles|feet|inches|yards|"
    r"     meters|kilometers|grams|kilos|dollars|cents|tickets|boxes|"
    r"     times|hours|minutes|days|weeks|months|years|pages|books|"
    r"     slices|pieces|parts|sections|portions|candles|socks|chickens|cows|"
    r"     dogs|cats|fish|birds|trees|flowers|plants|rooms|floors|"
    r"     bottles|cans|jars|bags|boxes|crates|cases|"
    r"     miles|kilometers|yards|feet|inches|acres|"
    r"     calories|grams|ounces|pounds|tons|carats|"
    r"     gallons|quarts|pints|cups|liters|mL|"
    r"     dozen|dozens|"
    r"     total|altogether|combined|remaining|"
    r"     left|more|less|each"
    r"    )|"
    r"    \w+\s+(?:does|can|will|would|are|were)"
    r"  )|"
    r"  much\s+(?:does|do|will|would|is|are|was|were)|"
    r"  far|long|old|fast|slow|tall|heavy|big|large|wide|deep|often"
    r")"
    r")\b",
    re.IGNORECASE | re.VERBOSE,
)

# ── Calculation question starters ──
# These are "how many/much/fast/far/long/old" or "what's the" questions
# that ask for computation rather than factual lookup
_CALC_QUESTION_RE_SIMPLE = re.compile(
    r"\b(?:"
    r"how\s+(?:"
    r"many\s+\w+|"
    r"much\s+(?:does|do|will|would|is|are|was|were)|"
    r"far|long|old|fast|slow|tall|heavy|big|large|wide|deep|often"
    r")|"
    r"what(?:'s| is)\s+(?:the\s+)?(?:"
    r"distance|area|volume|speed|time|age|length|width|height|depth|"
    r"total|sum|difference|product|average|mean|result|value|amount|"
    r"remaining|missing|combined"
    r")"
    r")\b",
    re.IGNORECASE,
)

# ── Numeric quantity expression (implicit arithmetic) ──
# Matches phrases like "80 miles", "150 miles", "3 hours" that signal
# measurement quantities used in math problems
_NUMERIC_QUANTITY_RE = re.compile(
    r"\d+\s+(?:"
    r"miles|feet|inches|yards|meters|centimeters|millimeters|kilometers|"
    r"ounces|pounds|tons|grams|kilograms|milligrams|"
    r"gallons|quarts|pints|cups|liters|milliliters|"
    r"degrees|minutes|seconds|hours|days|weeks|months|years|"
    r"dollars|cents|euros|pounds\s+sterling|"
    r"mph|km/h|m/s|knots|knot|"
    r"percent|%|"
    r"times|dozen|dozens|"
    r"people|persons?|children|adults|students|workers|"
    r"miles\s+(?:per|a|an|each)|"
    r"feet\s+(?:per|a|an|each)|"
    r"hours?\s+(?:per|a|an|each|a\s+day|a\s+week)|"
    r"days?\s+(?:per|a|an|each|a\s+week|a\s+month)|"
    r"miles\s+(?:per|a|an)\s+(?:hour|day|week)|"
    r"kilometers\s+(?:per|a|an)\s+(?:hour|day)"
    r")",
    re.IGNORECASE,
)
_FACTUAL_GUARD_RE = re.compile(
    r"\b(?:"
    r"who\s+(?:is|was|are|were)|"
    r"when\s+(?:was|did|is|were)|"
    r"where\s+(?:is|was|are|were)|"
    r"why\s+(?:did|does|is|are)|"
    r"what\s+(?:is|was|are|were)\s+(?:the|a|an)\s+"
    r"(?:definition|meaning|capital|population|name|title|"
    r"  purpose|function|role|reason|cause|effect|result|"
    r"  difference|similarity|advantage|disadvantage)|"
    r"define\s+"
    r")\b",
    re.IGNORECASE,
)

# ── Measurement units (numeric quantities that need arithmetic) ──
_MEASUREMENT_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"(?:feet|inches|yards|miles|meters|centimeters|millimeters|"
    r" kilometers|ounces|pounds|tons|grams|kilograms|milligrams|"
    r" gallons|quarts|pints|cups|liters|milliliters|"
    r" degrees|minutes|seconds|hours|days|weeks|months|years|"
    r" dollars|cents|miles\s+per|miles|mph|mph|km/h|m/s)"
    r")\b",
    re.IGNORECASE,
)

# ── Core logic ──

def _is_word_problem_math(prompt: str) -> bool:
    """
    Returns True if the prompt looks like a word-problem math question
    disguised as factual (has numbers + word-form arithmetic + no factual guard).
    """
    lower = prompt.lower()
    nums = _NUM_RE.findall(prompt)

    # Must have numbers
    if not nums:
        return False

    # Guard: if it's a pure factual lookup question, return False
    if _FACTUAL_GUARD_RE.search(lower):
        return False

    # Score based on signal strength
    score = 0

    # Word-form arithmetic operators
    if _WORD_ARITHMETIC_RE.search(lower):
        score += 3.0

    # Ratio / rate / speed patterns
    if _RATIO_RE.search(lower):
        score += 3.0

    # Calculation question starter ("how many X", "how far", "what's the distance")
    if _CALC_QUESTION_RE_SIMPLE.search(lower):
        score += 2.0

    # Numeric quantities with measurement units
    if _NUMERIC_QUANTITY_RE.search(prompt):
        score += 1.5

    # Multiple numbers (stronger signal)
    if len(nums) >= 2:
        score += 1.5
    elif len(nums) >= 1 and score >= 2.0:
        score += 0.5  # Weak: one number + some math signal

    # Measurement context with numbers
    if _MEASUREMENT_CONTEXT_RE.search(lower) and len(nums) >= 1:
        score += 1.0

    # Small numbers (1-100) in a calculation context — more likely math
    # than number-heavy factual
    small_nums = sum(1 for n in nums if n.isdigit() and 1 <= int(n) <= 100)
    if small_nums >= 2:
        score += 1.0

    return score >= 4.0


def resolve_mathfact(category_8way: str, prompt: str) -> str:
    """
    Override factual→math when a word-problem arithmetic prompt is
    misclassified as factual.

    Only fires when primary category is "factual". Returns "math" if the
    prompt has strong word-problem arithmetic patterns, otherwise returns
    the original category unchanged.
    """
    if category_8way != "factual":
        return category_8way

    if _is_word_problem_math(prompt):
        return "math"

    return "factual"
