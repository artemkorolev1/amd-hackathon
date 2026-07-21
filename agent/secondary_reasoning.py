"""
Secondary deterministic classifier for logic vs math.

Resolves ambiguity when the 8-way category_filter.py confuses the two
(especially logic puzzles with numbers getting classified as math).

Uses pure regex/heuristic scoring — zero model calls.
"""

import re
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Logic puzzle signals (stronger → higher weights)
# ---------------------------------------------------------------------------

# Explicit logic puzzle declarations (NOT bare "puzzle" — that matches jigsaw/sudoku)
_LOGIC_PUZZLE_RE = re.compile(
    r"\b(logic puzzle|logical puzzle|logical reasoning|logical deduction|"
    r"lateral thinking|brain.teaser|riddle)\b",
    re.IGNORECASE,
)

# Syllogism / formal logic patterns
_SYLLOGISM_RE = re.compile(
    r"\b(syllogism|knight|knave|kna ve|"
    r"deduce|deduction|infer|inference|therefore|hence|"
    r"implies|implication|iff\b|if and only if|"
    r"must be (true|false|the case)|"
    r"necessarily|necessarily true|necessarily false)\b",
    re.IGNORECASE,
)

# Conditional / constraint puzzle patterns
_CONSTRAINT_RE = re.compile(
    r"\b(if\s+.{0,60}\bthen\b|"
    r"\beither\s+\w+\s+or\b|neither\s+\w+\s+nor\b|"
    r"exactly one\b|at most one\b|at least one\b|"
    r"must be the case|cannot be the case|"
    r"each\s+\w+\s+(has|is|owns|likes|works|drives|sits|lives|gets)\b|"
    r"different\s+(from|than)\s+each\b|"
    r"each\s+(of|person|one)\b.{0,40}(who|which)\b|"
    r"ordered\s+by|arranged\s+in\s+(a|order)|ranked\s+by|"
    r"sits?\s+in\s+a\s+row|seated\s+in\s+a\s+row|"
    r"adjacent\s+to|to\s+the\s+(left|right)\s+of|"
    r"between\s+\w+\s+and\b)",
    re.IGNORECASE,
)

# Named-entity logic puzzle setup (people initials or proper names)
_ENTITY_PUZZLE_RE = re.compile(
    r"(?:\b(?:friends?|colleagues?|neighbors?|people|team|members?|"
    r"classmates?|candidates?|students?|recruits?|employees?|"
    r"judges?|scientists?|patients?|students?|workers?|"
    r"outstanding\s+students?|candidates?|recruits?)\b"
    r".{0,60}"
    r"(?:each|all|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"different|assigned|allocated|distributed|placed|selected|"
    r"choosen|chosen|divided|grouped|paired)\b)",
    re.IGNORECASE,
)

# List of capitalized names (people initials like "F, G, H, I" or names like "Zhang Lin, Zhao Qiang")
_NAME_LIST_RE = re.compile(
    r"\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)?(?:\s*[,，、]\s*[A-Z][a-z]*(?:\s+[A-Z][a-z]*)?){2,}"
)

# "Which of the following" — very common in logic multiple-choice
_WHICH_FOLLOWING_RE = re.compile(
    r"\bwhich\s+of\s+the\s+following\b", re.IGNORECASE
)

# Options pattern: numbered options typically in logic questions
_OPTIONS_RE = re.compile(
    r"(?:^|\n)\s*(?:[Oo]ptions?|[Aa]nswer|[Cc]hoices?|[Ss]elect|[Ss]ection)\s*:?\s*"
    r"(?:\n\s*(?:\d+[.)]\s*|[-*]\s*)[A-Za-z])",
    re.DOTALL,
)

# Statement-level reasoning patterns (NOT generic "reasoning"/"conclusion" which
# appear in math answer templates like "Structure your answer as: (1) ...")
_STATEMENT_REASONING_RE = re.compile(
    r"\b(logical|argument|premise|"
    r"fallacy|assumption|weaken|strengthen|flaw|parallel|"
    r"justify|justification|evaluate|"
    r"most\s+(similar\s+to|strongly\s+(support|weaken)|"
    r"reasonably\s+(be\s+)?(inferred|concluded|drawn))|"
    r"cannot\s+be\s+(true|inferred|concluded)|"
    r"follows\s+from|contradict|consistent\s+with|"
    r"principle|generalization|analogy|"
    r"inconsisten(cy|cies)|reasonably\s+explained|"
    r"all\s+of\s+the\s+following\s+(are\s+)?(true|false)|"
    r"which\s+of\s+the\s+(above|following|statements)|"
    r"if\s+(the\s+)?(above|following|statement)\b)\b",
    re.IGNORECASE,
)

# All/some/no quantifier statements (weaker logic signal)
_QUANTIFIER_RE = re.compile(
    r"\b(all\s+\w+\s+(are|is|have|has)|"
    r"some\s+\w+\s+(are|is|have|has)|"
    r"no\s+\w+\s+(are|is|have|has)|"
    r"conclude|conclusion|statement|reasoning)\b",
    re.IGNORECASE,
)

# Chinese logic exam patterns (very common in training data)
_CHINESE_LOGIC_RE = re.compile(
    r"[\u4e00-\u9fff].{0,30}(?:推理|逻辑|论证|前提|结论|"
    r"如果.{0,20}那么|要么|或者|必须|所有|有些|没有|"
    r"以下哪项|以下哪个|最能|最不能|除了|"
    r"基于|根据|假设|断定|陈述|"
    r"质疑|反驳|支持|削弱|解释|评价|"
    r"匹配|对应|排列|组合|分配|"
    r"相邻|左右|顺序|名次|位置)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Math problem signals
# ---------------------------------------------------------------------------

# Explicit "Solve:" prefix and calculation commands
_SOLVE_PREFIX_RE = re.compile(
    r"^(?:Solve|Calculate|Compute|Find|Evaluate|Determine)\b",
    re.IGNORECASE | re.MULTILINE,
)

# Math word problem keywords
_MATH_OPERATION_RE = re.compile(
    r"\b(calculate|compute|solve|equation|formula|algebra|geometry|"
    r"trigonometry|calculus|derivative|integral|probability|statistics|"
    r"percentage|percent|quotient|remainder|square root|factorial|modulo|"
    r"perimeter|area|volume|circumference|diameter|radius|velocity|"
    r"acceleration|matrix|determinant|logarithm|log\s+base|"
    r"how\s+many|how\s+much|how\s+fast|how\s+far|how\s+long|"
    r"what\s+is\s+the\s+(value|sum|difference|product|area|perimeter|volume|"
    r"speed|distance|time|probability|total|remainder|quotient)|"
    r"what\s+is\s+\d|find\s+the\s+(value|number|sum|difference|product|area|"
    r"perimeter|volume|speed|distance|time|total)|"
    r"total\s+number|total\s+amount|"
    r"x\s*=\s*\d+)\b",
    re.IGNORECASE,
)

# Arithmetic expressions with numbers — exclude dates (e.g. 2007/2008) and
# indices (e.g. "1 to 5", "numbered 1-5")
# We match actual computation patterns, not date-like or index-like patterns.
_ARITHMETIC_RE = re.compile(
    r"\b\d{1,3}\s*[+*×]\s*\d{1,3}\b"  # small-number multiplication/addition
    r"|\b\d{1,3}\s*/\s*(?:[2-9]|1[0-9]|20)\b"  # division with small denominator (not dates)
    r"|\b\d{1,2}\s*-\s*\d{1,2}\b(?!\s*[-/\d])"  # simple subtraction (but not year ranges)
    r"|\b\d+\.\d+(?!\s*[-\d/])"  # decimal numbers (not decimal ranges)
    r"|\b\d+\s*percent"
    r"|\b\d+\s*%\s*of\s*\d+",
)

# Money / rate / time math patterns
_MONEY_RATE_RE = re.compile(
    r"\$\s*\d+\.?\d*|\d+\s*(dollars?|euros?|pounds?|cents?|per\s+(hour|day|week|month|year)|"
    r"mph|km/h|kmh|miles?\s+per|minutes?\s+per)",
    re.IGNORECASE,
)

# "Step-by-step" instruction
_STEP_BY_STEP_RE = re.compile(
    r"\b(step.by.step|show your work|show\s+your\s+reasoning|"
    r"output\s+the\s+answer|respond\s+in\s+JSON|"
    r"provide\s+your\s+answer|structure\s+your\s+answer)\b",
    re.IGNORECASE,
)

# Fraction patterns — NOT dates like "2007/2008"
_FRACTION_RE = re.compile(
    r"(?<!\d/)\b\d+\s*/\s*(?:[2-9]|1[0-9]|20)\b(?!\s*/\s*\d)|"  # simple fractions (1/2, 3/4, not 2007/2008)
    r"\\frac\{\d+\}\{\d+\}|\bhalf\b|\bquarter\b|\bthird\b|\bdouble\b|\btriple\b"
)

# ---------------------------------------------------------------------------
# Scoring function
# ---------------------------------------------------------------------------


def _score_logic(prompt: str) -> float:
    """Score how much the prompt looks like a logic puzzle."""
    score = 0.0

    # ── Strong signals (+4 each) ──
    if _LOGIC_PUZZLE_RE.search(prompt):
        score += 4.0

    if _WHICH_FOLLOWING_RE.search(prompt):
        score += 3.0

    if _SYLLOGISM_RE.search(prompt):
        score += 4.0

    if _STATEMENT_REASONING_RE.search(prompt):
        score += 3.0

    # ── Medium signals (+2-3 each) ──
    if _CONSTRAINT_RE.search(prompt):
        score += 3.0

    if _ENTITY_PUZZLE_RE.search(prompt):
        score += 3.0

    # Check for named-entity list with constraint context
    names = _NAME_LIST_RE.findall(prompt)
    has_constraint_context = bool(
        re.search(
            r"\b(each|different|assigned|allocated|must|"
            r"condition|rule|requirement|"
            r"older|younger|taller|shorter|faster|slower|"
            r"left|right|between|adjacent|before|after|"
            r"not\s+(the\s+)?same|no\s+two|"
            r"if\s|then\s|either|neither|nor)\b",
            prompt,
            re.IGNORECASE,
        )
    )
    if names and has_constraint_context:
        score += 3.0

    if _CHINESE_LOGIC_RE.search(prompt):
        score += 3.0

    # Options list (multiple-choice style)
    if _OPTIONS_RE.search(prompt):
        score += 2.0

    # ── Weaker signals (+1 each) ──
    # "Question:" followed by reasoning about statements
    if re.search(r"\b(?:Question|question|Q:|Q\.)\s*\:?", prompt) and _STATEMENT_REASONING_RE.search(
        prompt
    ):
        score += 1.0

    # Quantifier statements about logical relationships (weaker)
    if _QUANTIFIER_RE.search(prompt):
        score += 1.0

    return score


def _score_math(prompt: str) -> float:
    """Score how much the prompt looks like a math problem."""
    score = 0.0

    # ── Strong signals ──
    if _SOLVE_PREFIX_RE.search(prompt):
        score += 4.0

    if _STEP_BY_STEP_RE.search(prompt):
        score += 3.0

    if _MATH_OPERATION_RE.search(prompt):
        score += 3.0

    if _MONEY_RATE_RE.search(prompt):
        score += 2.0

    # Arithmetic expressions (genuine computation, not labels)
    arith_matches = _ARITHMETIC_RE.findall(prompt)
    if arith_matches and len(arith_matches) >= 2:
        score += 3.0
    elif arith_matches:
        score += 2.0

    # Fraction patterns
    if _FRACTION_RE.search(prompt):
        score += 2.0

    # Numeric density — lots of numbers indicating computation
    nums = re.findall(r"\b\d{1,4}(?:,\d{3})*(?:\.\d+)?\b", prompt)
    if len(nums) >= 5:
        # Check if these are used in computation context (not just dates/indices)
        # Counting patterns like "1 to 5", "numbered 1 to" are logic indicators
        indexing_pattern = re.search(
            r"(?:numbered|from|to|row|column|house|room|floor|"
            r"stage|level|rank|position)\s+\d+\s*(?:to|through|-)\s*\d+",
            prompt,
            re.IGNORECASE,
        )
        if not indexing_pattern:
            score += min(len(nums) * 0.3, 2.0)

    # "how many/much" questions that ask for a quantity (as opposed to "which of the following")
    if re.search(
        r"\b(how\s+many|how\s+much)\b", prompt, re.IGNORECASE
    ) and not _WHICH_FOLLOWING_RE.search(prompt):
        score += 2.0

    # Equation patterns
    if re.search(r"\d+\s*[xX×]\s*\d+\s*[=＝]\s*\d*|\w+\s*=\s*\d+", prompt):
        score += 2.0

    return score


def resolve_reasoning(category_8way: str, prompt: str) -> str:
    """
    Determine whether a prompt is a logic puzzle or a math problem.

    Args:
        category_8way: The category assigned by the 8-way classifier
                       ("logic", "math", or any other from CATEGORIES_8WAY).
        prompt: The original prompt text.

    Returns:
        The corrected category string ("logic", "math", or original if uncertain).
    """
    logic_score = _score_logic(prompt)
    math_score = _score_math(prompt)

    # Thresholds for override
    LOGIC_THRESHOLD = 3.0
    MATH_THRESHOLD = 3.0
    DECISIVE_MARGIN = 1.0

    # ── Detect formatting-only usage of logic-like words ──
    # E.g. "Structure your answer as: (1) summary, (2) reasoning, (3) conclusion"
    # These are answer templates, not logical reasoning content.
    _is_formatting_answer = bool(
        re.search(
            r"\b(?:structure|format|organize)\s+your\s+answer\b",
            prompt,
            re.IGNORECASE,
        )
    )
    _has_numbered_list = bool(
        re.search(r"\(\d+\)\s+\w+", prompt)
    )
    is_answer_template = _is_formatting_answer or (
        _has_numbered_list
        and bool(re.search(r"\b(?:summary|reasoning|conclusion|answer|output)\b", prompt, re.IGNORECASE))
        and not _STATEMENT_REASONING_RE.search(prompt)
        and not _SYLLOGISM_RE.search(prompt)
    )

    # If the prompt is mostly an answer template (no real logic content),
    # suppress the logic score from generic quantifier matches
    if is_answer_template:
        # Reduce logic score by the quantifier contribution (removes the
        # "+1" from "conclusion", "reasoning", "statement" etc.)
        if _QUANTIFIER_RE.search(prompt) and not (
            _STATEMENT_REASONING_RE.search(prompt)
            or _SYLLOGISM_RE.search(prompt)
            or _CONSTRAINT_RE.search(prompt)
            or _ENTITY_PUZZLE_RE.search(prompt)
            or _WHICH_FOLLOWING_RE.search(prompt)
        ):
            logic_score -= 1.0

    # If the original category is neither logic nor math, we only override
    # if the signal is extremely clear
    if category_8way not in ("logic", "math"):
        if logic_score >= 6.0 and logic_score > math_score + DECISIVE_MARGIN:
            return "logic"
        if math_score >= 6.0 and math_score > logic_score + DECISIVE_MARGIN:
            return "math"
        return category_8way  # stay with original

    # ── Logic check ──
    # Special case: explicit "logic puzzle" declaration always wins
    if _LOGIC_PUZZLE_RE.search(prompt):
        return "logic"

    # Named-entity puzzle + constraint context → logic
    names = _NAME_LIST_RE.findall(prompt)
    has_constraint_context = bool(
        re.search(
            r"\b(each|different|assigned|allocated|must|"
            r"condition|rule|requirement|"
            r"older|younger|taller|shorter|"
            r"left|right|between|adjacent|before|after|"
            r"not\s+(the\s+)?same|no\s+two|"
            r"if\s|then\s|either|neither|nor)\b",
            prompt,
            re.IGNORECASE,
        )
    )
    if names and has_constraint_context and logic_score >= 3.0:
        return "logic"

    # Syllogism patterns → logic
    if _SYLLOGISM_RE.search(prompt) and logic_score >= 3.0:
        return "logic"

    # "Which of the following" with statement reasoning → logic
    if _WHICH_FOLLOWING_RE.search(prompt) and _STATEMENT_REASONING_RE.search(prompt):
        return "logic"

    # Chinese logic exam patterns → logic
    if _CHINESE_LOGIC_RE.search(prompt) and logic_score >= 3.0:
        return "logic"

    # Statement reasoning verbs → logic
    if _STATEMENT_REASONING_RE.search(prompt) and logic_score >= 3.0 and logic_score > math_score:
        return "logic"

    # ── Constraint/zebra puzzle guard (hoisted) ──
    # These are logic puzzles even with "Solve:" prefix or high math scores
    _has_constraint_structure = bool(
        re.search(r'(?:There (?:are|is|exists)|numbered)\s+\d+\s+\w+', prompt, re.IGNORECASE)
        and re.search(r'(?:each|unique|different)\s', prompt, re.IGNORECASE)
    )
    _has_name_list = bool(
        _NAME_LIST_RE.search(prompt)
        or re.search(r'`[A-Z][a-z]+`', prompt)
        or re.search(r'\b[A-Z][a-z]+\b.{0,10}\b[A-Z][a-z]+\b.{0,10}\b[A-Z][a-z]+\b', prompt)
    )
    _is_constraint_puzzle = _has_constraint_structure and _has_name_list

    # ── Solve: prefix math check ──
    # "Solve:" prefix with calculation context → math (unless constraint puzzle)
    if _SOLVE_PREFIX_RE.search(prompt) and math_score >= 3.0 and math_score > logic_score:
        if not _is_constraint_puzzle:
            return "math"

    # Step-by-step / structured output instructions → math
    # BUT not when there's a "which of the following" logic context
    if (
        _STEP_BY_STEP_RE.search(prompt)
        and math_score >= 3.0
        and not _LOGIC_PUZZLE_RE.search(prompt)
        and not _WHICH_FOLLOWING_RE.search(prompt)
        and not _CHINESE_LOGIC_RE.search(prompt)
    ):
        return "math"

    # Arithmetic expressions with money/rate → math
    if _MONEY_RATE_RE.search(prompt) and _ARITHMETIC_RE.search(prompt) and math_score >= logic_score:
        return "math"

    # ── Default: use score comparison with margin ──
    # Constraint/zebra puzzles defer to original regardless of score
    if _is_constraint_puzzle:
        return category_8way

    if logic_score >= LOGIC_THRESHOLD and math_score >= MATH_THRESHOLD:
        # Both active — the one with a clear margin wins
        if logic_score >= math_score + DECISIVE_MARGIN:
            return "logic"
        if math_score >= logic_score + DECISIVE_MARGIN:
            return "math"
        # Close call — defer to original classifier
        return category_8way

    if logic_score >= LOGIC_THRESHOLD and logic_score > math_score:
        return "logic"

    if math_score >= MATH_THRESHOLD and math_score > logic_score:
        return "math"

    # ── Uncertain — return original ──
    return category_8way


def classify_reasoning(prompt: str) -> Tuple[str, float, float]:
    """
    Direct classification without needing 8-way input.

    Returns:
        (category: str, logic_score: float, math_score: float)
    """
    logic_score = _score_logic(prompt)
    math_score = _score_math(prompt)
    cat = resolve_reasoning("logic" if logic_score >= math_score else "math", prompt)
    return cat, logic_score, math_score
