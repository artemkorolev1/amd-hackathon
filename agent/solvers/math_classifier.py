"""
agent/solvers/math_classifier.py — Deterministic keyword-based classifier
for math problem types.

Returns one of the following problem-type strings:
    simple_arithmetic
    narrative_entity
    narrative_remaining
    comparison_ratio
    rate_per_unit
    multiplication_chain
    fraction_percentage
    speed_distance
    unit_conversion
    age_ratio
    money_shopping
    profit_cost
    multi_step_complex

Design mirrors the 8-way primary classifier in agent/category_filter.py:
keyword-based, deterministic (no ML), fast (no import overhead beyond re).
"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Keyword / pattern definitions (ordered by specificity)
# ---------------------------------------------------------------------------

# 1. Simple arithmetic: direct expressions and equations
_RE_SIMPLE_ARITH = re.compile(
    r'\b(?:what\s+is|what\'s|calculate|compute)\s+'
    r'\d+\s*[+\-*/]',
    re.IGNORECASE,
)

# 2. Entity-based narrative: "X has N, Y has M" patterns with entity verbs
_RE_NARRATIVE_ENTITY = re.compile(
    r'\b(?:'
    r'(?:has|have|had|buys|bought|needs|wants|gets|collects|makes|makes?\s*$)'
    r'|(?:earns?|sells?|produces?|contains?|owns?|receives?|takes?|pays?|paid)'
    r'|(?:bakes?|grows?|plants?|cuts?|spends?|spent)'
    r')\s+\d+',
    re.IGNORECASE,
)

# 3. Remaining / leftover problems
_RE_REMAINING = re.compile(
    r'\b(?:'
    r'remain(?:ing|s|ed)?|left\s*(?:over)?|'
    r'how\s+many\s+(?:pieces|liters|cookies|apples|miles|hours|'
    r'students|people|cars|books|pages|units|bags|boxes|'
    r'chickens|eggs|candles|roses|flowers|thorns|raspberries|'
    r'pets|toys|dollars|coins|bolts)'
    r'(?:\s+(?:are|were|is)\s+)?(?:remaining|left)'
    r'|left\s+over|unoccupied|untaken|uneaten|unsold|unused|'
    r'how\s+many\s+were\s+(?:taken|removed|eaten|sold)'
    r')\b',
    re.IGNORECASE,
)

# 4. Comparison / ratio: "more than", "less than", "times as many", etc.
_RE_COMPARISON = re.compile(
    r'\b(?:'
    r'(?:how\s+)?(?:many|much)\s+more\b'
    r'|(?:more|less|fewer)\s+(?:than|that)'
    r'|times\s+(?:as\s+)?(?:many|much|\w+(?:\s+\w+){0,3}\s+as)'
    r'|how\s+many\s+(?:more|fewer|less)'
    r'|ratio|proportion'
    r'|(?:twice|thrice)\s+(?:as\s+)?(?:many|much|\w+(?:\s+\w+){0,2})'
    r'|(?:older|younger)\s+than'
    r'|per\s+(?:head|person|student|child|adult)'
    r')\b',
    re.IGNORECASE,
)

# 5. Rate per unit: "X per Y", "X a day", "X every Y", "X for Z dollars"
_RE_RATE = re.compile(
    r'\b(?:'
    r'(?:\d+)\s+(?:\w+\s+){0,3}(?:per|a|an|each|every)\s+'
    r'(?:day|week|month|hour|minute|second|year|dozen|pound|ounce|'
    r'mile|kilometer|kilometre|student|person|class|session)'
    r'|how\s+much\s+(?:per|a|an|each|every)'
    r'|per\s+(?:day|week|month|hour|minute|second|dozen|pound|ounce)'
    r')\b',
    re.IGNORECASE,
)

# 6. Multiplication chain: "each X has Y, each Y has Z"
_RE_MULT_CHAIN = re.compile(
    r'\b(?:'
    r'each\s+\w+\s+(?:has|have|contain|contains?|includes?|produces?|makes?|grows?|is|are)\s+\d+'
    r'|each\s+(?:of\s+)?(?:the\s+)?\w+\s+has'
    r')\b',
    re.IGNORECASE,
)

# 7. Fraction / percentage
_RE_FRACTION_PCT = re.compile(
    r'\b(?:'
    r'\d+\s*(?:%|percent)'
    r'|\d+\s*/\s*\d+'
    r'|half|third|quarter|fifth'
    r'|fraction|percentage|percent'
    r'|reduced\s+by\s+\d+\s*%'
    r'|increased\s+by\s+\d+\s*%'
    r'|discount|interest'
    r')\b',
    re.IGNORECASE,
)

# 8. Speed / distance / time
_RE_SPEED = re.compile(
    r'\b(?:'
    r'(?:\d+)\s*(?:km/h|mph|m/s|knots?|miles\s+per\s+hour|km\s+per\s+hour)'
    r'|speed|velocity|distance|travel'
    r'|how\s+(?:far|fast|long)\s+(?:will|does|can|did)'
    r'|drives?\s+(?:for|at)\s+\d+'
    r')\b',
    re.IGNORECASE,
)

# 9. Unit conversion — ONLY actual measurement units, NOT generic containers.
_RE_UNIT_CONV = re.compile(
    r'\b(?:'
    r'(?:how\s+)?many\s+(?:inches|feet|yards|miles|cm|meters|centimeters|'
    r'pounds|ounces|gallons|quarts|pints|cups|liters|milliliters)'
    r'|convert|conversion|change\s*(to|into)'
    r'|(?:how\s+)?many\s+(?:meters|feet|inches|yards|cm|pounds|ounces|gallons|'
    r'quarts|pints|cups|liters)\s+(?:are|is|in)'
    r')\b',
    re.IGNORECASE,
)

# 10. Age / ratio problems
_RE_AGE = re.compile(
    r'\b(?:'
    r'years?\s+old|age|aged'
    r'|years?\s+(?:older|younger)\s+than'
    r'|how\s+old'
    r')\b',
    re.IGNORECASE,
)

_RE_RATIO = re.compile(
    r'\bratio\b',
    re.IGNORECASE,
)

# 11. Money / shopping
_RE_MONEY = re.compile(
    r'\b(?:'
    r'\$\s*\d+\.?\d*\s*'
    r'|(?:costs?|cost|price|priced?|paid|spent?|spend|'
    r'sells?\s+for|buys?\s+for|bought\s+for|'
    r'earns?|wages?|salary|income|revenue|profit)'
    r'|how\s+much\s+(?:money|does|did|will|would)'
    r'|(?:dollars?|cents?)\s+(?:each|per|a|an)'
    r')\b',
    re.IGNORECASE,
)

# 12. Profit / cost / net
_RE_PROFIT = re.compile(
    r'\b(?:'
    r'profit|revenue|cost|expense|net|margin|markup|discount|savings?|save'
    r')\b',
    re.IGNORECASE,
)

# 13. Multi-step complex: combination of multiple patterns or compound sentences
_RE_MULTI_STEP = re.compile(
    r'(?:'
    r'(?:then|after\s+that|next|finally|now)\s+'
    r'|first\s+.*?(?:then|afterwards|subsequently)'
    r'|over\s+\d+\s+(?:days?|weeks?|months?|years?)'
    r')',
    re.IGNORECASE,
)


def classify_math(prompt: str) -> str:
    """Classify a math problem into a problem type.

    Uses deterministic keyword/pattern matching. Returns one of:
        simple_arithmetic, narrative_entity, narrative_remaining,
        comparison_ratio, rate_per_unit, multiplication_chain,
        fraction_percentage, speed_distance, unit_conversion,
        age_ratio, money_shopping, profit_cost, multi_step_complex

    Args:
        prompt: The math problem text.

    Returns:
        Problem type string.
    """
    text = prompt.strip()

    # --- Quick detection of simple arithmetic ---
    if _RE_SIMPLE_ARITH.search(text):
        return "simple_arithmetic"

    # Count various signals
    has_entity = bool(_RE_NARRATIVE_ENTITY.search(text))
    has_comparison = bool(_RE_COMPARISON.search(text))
    has_remaining = bool(_RE_REMAINING.search(text))
    has_rate = bool(_RE_RATE.search(text))
    has_mult_chain = bool(_RE_MULT_CHAIN.search(text))
    has_fraction = bool(_RE_FRACTION_PCT.search(text))
    has_speed = bool(_RE_SPEED.search(text))
    has_unit = bool(_RE_UNIT_CONV.search(text))
    has_age = bool(_RE_AGE.search(text))
    has_ratio = bool(_RE_RATIO.search(text))
    has_money = bool(_RE_MONEY.search(text))
    has_profit = bool(_RE_PROFIT.search(text))
    has_multi_step = bool(_RE_MULTI_STEP.search(text))
    has_question = "?" in text or "how many" in text.lower() or "how much" in text.lower()

    # --- Detection logic (specific → general) ---

    # Speed/distance is very specific
    if has_speed:
        return "speed_distance"

    # Unit conversion
    if has_unit:
        return "unit_conversion"

    # Age problems (often have comparison too)
    if has_age:
        return "age_ratio"

    # Ratio specific
    if has_ratio:
        return "comparison_ratio"

    # Remaining / leftover — standalone check (doesn't need entity)
    if has_remaining:
        return "narrative_remaining"

    # Profit-specific (money + profit keywords)
    if has_profit and has_money:
        return "profit_cost"

    # Rate per unit
    if has_rate:
        return "rate_per_unit"

    # Multiplication chain — also check for "each" with entity
    if has_mult_chain:
        return "multiplication_chain"

    # Comparison ratio
    if has_comparison:
        return "comparison_ratio"

    # Entity-based narrative
    if has_entity and has_question:
        return "narrative_entity"

    # Fraction/percentage
    if has_fraction:
        return "fraction_percentage"

    # Money shopping
    if has_money:
        return "money_shopping"

    # Multi-step complex
    if has_multi_step:
        return "multi_step_complex"

    # Default: if it looks like it has numbers and asks a question, it's narrative
    has_numbers = bool(re.search(r'\d+', text))
    if has_numbers and has_question:
        return "narrative_entity"

    return "multi_step_complex"
