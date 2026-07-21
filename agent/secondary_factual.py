"""
Secondary factual QA detector — resolves 8-way confusion between
factual vs logic (113 errors) and factual vs math (96 errors).

Architecture:
  resolve_factual(category_8way, prompt) -> str

  Pure deterministic — zero model calls, zero imports beyond stdlib.

  Directions overridden:
    primary="factual"  → overrides to logic/math when reasoning/calc signals dominant
    primary="logic"    → overrides to factual when SQuAD/Choices/definition patterns
    primary="math"     → overrides to factual when SQuAD/Choices/definition patterns
"""

import re
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Scoring primitives (shared)
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# ---------------------------------------------------------------------------
# ---- STRONG FACTUAL SIGNALS ----
# ---------------------------------------------------------------------------

# SQuAD / reading-comprehension format: "Context: ... Question: ..."
_FACTUAL_QA_FORMAT_RE = re.compile(
    r"(?:"
    r"(?:context|passage|text|document|article|story|paragraph|excerpt)"
    r"\s*:\s*.{20,}.*?"
    r"(?:question|q)\s*:"
    r")",
    re.DOTALL | re.IGNORECASE,
)

# Source markers (document header style)
_FACTUAL_SOURCE_RE = re.compile(
    r"^(?:Context|Passage|Article|Text|Document|SOURCE\s+\d+):",
    re.MULTILINE | re.IGNORECASE,
)

# MMLU / MCQ knowledge: "Choices:" is the single strongest factual signal
# (124/200 factual samples have Choices:, 0/200 logic, 0/200 math)
_FACTUAL_CHOICES_RE = re.compile(r"^Choices?:", re.MULTILINE | re.IGNORECASE)

# MMLU-style knowledge questions
_FACTUAL_KNOWLEDGE_QUESTIONS = re.compile(
    r"(?:"
    r"(?:^|[.?!;:\n])\s*(?:what|who|when|where|why|how)\s+(?:is|was|were|are|does|do|did)|"
    r"\b(?:define|describe|explain|tell\s+me\s+about|"
    r"facts?\s+about|history\s+of|capital\s+of|"
    r"population\s+of|meaning\s+of|definition\s+of|"
    r"what\s+does\s+\w+\s+mean|invented\s+by|discovered\s+by|"
    r"located\s+in|known\s+for|famous\s+for|"
    r"referred\s+to\s+as|also\s+known\s+as|"
    r"completes?\s+the\s+following\s+statement|"
    r"best\s+describes|best\s+matches|"
    r"purpose\s+of|function\s+of|role\s+of)"
    r")\b",
    re.IGNORECASE,
)

# Factual-knowledge-specific "which of the following" (followed by knowledge content)
# NOT the same as constraint-logic "which of the following must be true"
_FACTUAL_WOTF_RE = re.compile(
    r"which\s+of\s+the\s+following\s+(?:"
    r"is|are|was|were|best|correct|incorrect|true|false|not|"
    r"completes|describes|matches|identifies|statements|"
    r"substances|options|choices|factors|would|could"
    r")",
    re.IGNORECASE,
)

# Domain-specific factual lookup patterns
_FACTUAL_DOMAIN_RE = re.compile(
    r"\b(?:"
    r"capital\s+of|population\s+of|area\s+of|"
    r"president\s+of|prime\s+minister\s+of|king\s+of|queen\s+of|"
    r"located\s+in|situated\s+in|found\s+in|"
    r"invented\s+by|discovered\s+by|created\s+by|founded\s+by|"
    r"written\s+by|authored\s+by|directed\s+by|"
    r"also\s+known\s+as|also\s+called|referred\s+to\s+as|"
    r"consists?\s+of|composed\s+of|made\s+up\s+of|"
    r"known\s+for|famous\s+for|notable\s+for|"
    r"first\s+to|one\s+of\s+the\s+(?:first|most|only)|"
    r"refers?\s+to|is\s+a\s+type\s+of|is\s+a\s+form\s+of|"
    r"is\s+the\s+(?:study|practice|process|method|technique|theory)\s+of|"
    r"concept\s+of|principle\s+of|theory\s+of|law\s+of|"
    r"difference\s+between\s+\w+\s+and\s+\w+"
    r")\b",
    re.IGNORECASE,
)

# Definition patterns: "what does X mean", "what is the definition of X"
_FACTUAL_DEF_RE = re.compile(
    r"\b(?:"
    r"what\s+(?:does|is)\s+\w+\s+mean|"
    r"what\s+is\s+the\s+(?:definition|meaning|purpose|function|role|concept)|"
    r"define\s+\w+|"
    r"definition\s+of"
    r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# ---- STRONG LOGIC SIGNALS ----
# ---------------------------------------------------------------------------

# Constraint / reasoning words (appear in 159/200 logic, only 22/200 factual)
_LOGIC_CONSTRAINT_RE = re.compile(
    r"\b(?:"
    r"each\s+\w+\s+(?:has|is|works|owns|lives|sits|drives|likes|takes|gets|says|thinks|knows|wants)|"
    r"different\s+(?:department|role|job|position|color|type|item|one|kind|sort)|"
    r"must\s+be\s+(?:true|false|correct|incorrect)|"
    r"conclu(?:de|ded|ding|sion|sive)|therefore|hence|thus|"
    r"implies?|implication|infer|inference|"
    r"premise|assumption|syllogism|"
    r"deduce|deduc(?:ed|ing|tion)|logical|reasoning|"
    r"if\s+\w+\s+is\s+\w+\s+then|"
    r"if\s+.{0,40}then|"
    r"either|neither|exactly\s+one|exactly\s+two|"
    r"knight|knave|lying|liar|truth\s+teller|"
    r"ordered|arrangement|arranged\s+in|seated\s+in|"
    r"sits?\s+(?:in|on|at|next|between|beside|across)|"
    r"adjacent\s+to|immediately\s+(?:to\s+the|after|before|left|right)|"
    r"consistency|consistent\s+with|"
    r"all\s+\w+\s+are\s+\w+"
    r")\b",
    re.IGNORECASE,
)

# Logic multiple-choice: "which of the following must be true/cannot be true"
_LOGIC_WOTF_RE = re.compile(
    r"which\s+of\s+the\s+following\s+(?:"
    r"must|can|cannot|could|could\s+not|may|might|"
    r"would|would\s+not|is\s+most|is\s+least"
    r")\s+(?:be\s+)?(?:true|false|correct|incorrect|supported|weaken|strengthen|justify|conclude)",
    re.IGNORECASE,
)

# Puzzle structure: "Four friends — X, Y, Z — each have..."
_LOGIC_PUZZLE_STRUCTURE_RE = re.compile(
    r"\b(?:"
    r"(?:four|five|six|seven|eight|nine|ten|\d+)\s+"
    r"(?:friends|colleagues|neighbors|people|students|candidates|members|workers|classmates|"
    r"teams?\s+of\s+\w+|persons?|judges|scientists?|doctors?|lawyers?|players?|"
    r"employees?|managers?|directors|teachers?|children|boys|girls)"
    r")",
    re.IGNORECASE,
)

# Logic puzzle constraint density: many capitalized proper names + constraint words
_LOGIC_NAME_DENSITY_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")

# Reasoning explanation markers
_LOGIC_EXPLAIN_RE = re.compile(
    r"\b(?:"
    r"explain\s+(?:step\s+by\s+step|why|how|your\s+reasoning|the\s+reasoning)|"
    r"reason\s+step\s+by\s+step|"
    r"what\s+is\s+the\s+(?:reasoning|logic|conclusion|inference|assumption)|"
    r"which\s+of\s+the\s+following\s+can\s+be\s+inferred|"
    r"which\s+of\s+the\s+following\s+is\s+an\s+assumption"
    r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# ---- STRONG MATH SIGNALS ----
# ---------------------------------------------------------------------------

# Explicit calculation keywords
_MATH_CALC_RE = re.compile(
    r"\b(?:"
    r"solve|calculate|compute|evaluate|"
    r"equation|formula|formulae|algebra|geometry|trigonometry|"
    r"derivative|integral|differential|calculus|"
    r"matrix|determinant|vector|logarithm|log\s+base|"
    r"quadratic|polynomial|binomial|factorial|"
    r"perimeter|area|volume|radius|diameter|"
    r"circumference|velocity|acceleration|speed|distance|"
    r"probability|permutation|combination|"
    r"integer|integers|positive\s+integer|prime\s+number|"
    r"divisible|divisibility|factor|multiple|"
    r"quotient|remainder|modulo|modulus|"
    r"sum\s+of|product\s+of|difference\s+between"
    r")\b",
    re.IGNORECASE,
)

# Arithmetic operators (real ones, not dates/ages/patents)
_MATH_OPERATORS_RE = re.compile(
    r"\d+\s*[+/*]\s*\d+|\d+\s*-\s*\d+(?!\s*(?:year|old|month|day|hour|minutes|sec))"
)

# Word-problem math patterns
_MATH_WORDPROB_RE = re.compile(
    r"\b(?:"
    r"solve\s+(?:for|the\s+following|step)|"
    r"how\s+(?:many|much|fast|far|long|old|tall|heavy|big|large|wide|deep|often)|"
    r"what\s+(?:is\s+the\s+)?(?:value|sum|total|product|difference|area|volume|"
    r"perimeter|result|solution|answer|number|amount|cost|price|"
    r"probability|percentage|fraction|ratio|average|mean|median|mode)|"
    r"find\s+(?:the\s+)?(?:value|number|sum|area|volume|distance|speed|age|length|width|height|"
    r"total|remaining|missing|cost|price|interest|probability)|"
    r"what\s+(?:distance|speed|time|age|length|width|height|depth)|"
    r"step-by-step|step\s+by\s+step"
    r")\b",
    re.IGNORECASE,
)

# Numeric density - used as weak signal
def _numeric_density(prompt: str) -> float:
    """Numbers per 100 chars."""
    nums = _NUM_RE.findall(prompt)
    if not prompt:
        return 0.0
    return (len(nums) * 100) / len(prompt)


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

def _score_factual(prompt: str, lower: str) -> float:
    """Score how strongly this looks like a factual QA prompt."""
    score = 0.0

    # +++ Strongest: SQuAD QA format
    if _FACTUAL_QA_FORMAT_RE.search(prompt):
        score += 6.0

    # Source markers
    if _FACTUAL_SOURCE_RE.search(prompt):
        score += 4.0

    # Choices: indicator (MMLU-style MCQ)
    if _FACTUAL_CHOICES_RE.search(prompt):
        score += 5.0

    # Knowledge question starters
    kq = _FACTUAL_KNOWLEDGE_QUESTIONS.findall(lower)
    score += len(kq) * 1.5

    # Factual "which of the following" (knowledge content)
    if _FACTUAL_WOTF_RE.search(lower):
        score += 2.0

    # Domain-specific lookup patterns
    factual_domain = _FACTUAL_DOMAIN_RE.findall(lower)
    score += len(factual_domain) * 2.0

    # Definition patterns
    if _FACTUAL_DEF_RE.search(lower):
        score += 3.0

    # "Context:" or "Passage:" without "Question:" still counts
    has_context_marker = bool(re.search(
        r"\b(context|passage|article|excerpt|paragraph)\s*:",
        lower,
    ))
    has_any_q = bool(re.search(
        r"\b(question\s*:|q\s*:|query\s*:)",
        lower,
    ))
    if has_context_marker and has_any_q:
        score += 3.0  # Strong structured QA format

    return score


def _score_logic(prompt: str, lower: str) -> float:
    """Score how strongly this looks like a logic/reasoning prompt."""
    score = 0.0

    # Constraint words (most indicative)
    constraints = _LOGIC_CONSTRAINT_RE.findall(prompt)
    score += len(constraints) * 2.0

    # If-then pattern (very strong)
    if re.search(r"\bif\b.{0,60}\bthen\b", lower, re.DOTALL):
        score += 3.0

    # Logic "which of the following must be true"
    if _LOGIC_WOTF_RE.search(lower):
        score += 3.0

    # Puzzle structure
    if _LOGIC_PUZZLE_STRUCTURE_RE.search(prompt):
        score += 2.0

    # Named entity density + constraints = logic puzzle
    names = _LOGIC_NAME_DENSITY_RE.findall(prompt)
    unique_names = len(set(names))
    if unique_names >= 3 and len(constraints) >= 1:
        score += unique_names * 0.5

    # Reasoning explanation patterns
    if _LOGIC_EXPLAIN_RE.search(lower):
        score += 2.0

    # Constraint word density: 3+ constraint words → strong logic
    constraint_words = {"each", "every", "all", "none", "no", "neither",
                        "either", "both", "only", "unless", "except",
                        "must", "if", "then", "hence", "thus", "therefore"}
    constraint_count = sum(1 for w in constraint_words if re.search(rf"\b{w}\b", lower))
    if constraint_count >= 4:
        score += 3.0
    elif constraint_count >= 3:
        score += 1.5

    return score


def _score_math(prompt: str, lower: str) -> float:
    """Score how strongly this looks like a math prompt."""
    score = 0.0

    # Calculation keywords
    calc_matches = _MATH_CALC_RE.findall(lower)
    score += len(calc_matches) * 2.0

    # Arithmetic operators (not age/date ranges)
    if _MATH_OPERATORS_RE.search(prompt):
        score += 2.5

    # Word problem patterns
    if _MATH_WORDPROB_RE.search(lower):
        score += 2.0

    # "Solve" prefix (GSM8K/SVAMP style)
    if re.search(r"^(?:solve|problem|question)\b", lower):
        score += 2.0

    # Numeric density > 5 per 100 chars (with at least some math signal)
    nd = _numeric_density(prompt)
    if nd > 5 and (score > 0 or len(_NUM_RE.findall(prompt)) >= 3):
        score += 2.0
    elif nd > 3 and score > 0:
        score += 1.0

    # LaTeX math notation
    if re.search(r"\$.*?[\\=+*/^_{}\d].*?\$|\\\\[a-zA-Z]+", prompt):
        score += 2.0

    return score


# ---------------------------------------------------------------------------
# Contextual guards (suppress signals that are incidental)
# ---------------------------------------------------------------------------

def _has_squad_structure(prompt: str, lower: str) -> bool:
    """Full SQuAD: both context and question markers."""
    return bool(re.search(
        r"\b(context|passage|text|document|article|story)\s*:.{30,}"
        r"\b(question|q)\s*:",
        lower, re.DOTALL,
    ))


def _has_mmlu_structure(prompt: str, lower: str) -> bool:
    """MMLU-style: 'Choices:' present."""
    return bool(_FACTUAL_CHOICES_RE.search(prompt))


def _is_calculation_intent(prompt: str, lower: str) -> bool:
    """True if the prompt's primary intent is calculation/math (not factual numbers).

    Requires stronger signals than a single incidental math word like 'area'
    appearing in a non-math context (e.g. 'surface area in organs').
    """
    # Strong explicit signals (unambiguous)
    has_solve_prefix = bool(re.search(r"^(?:solve|calculate|compute)\b", lower))
    has_step_by_step = bool(re.search(r"step[-\s]by[-\s]step", lower))
    has_operators = bool(re.search(r"\d+\s*[+/*]\s*\d+", prompt))
    has_numeric_op = bool(re.search(
        r"(?:"
        r"\d+\s*[+/*-]\s*\d+|"
        r"(?:[=<>]\s*\d+|plus|minus|times|divided\s+by)"
        r")",
        lower,
    ))

    # If it has SQuAD or MMLU structure, shouldn't be math despite numbers
    if _has_squad_structure(prompt, lower) or _has_mmlu_structure(prompt, lower):
        if not has_solve_prefix and not has_numeric_op:
            return False

    # Strong enough: explicit solve/compute or arithmetic operators
    if has_solve_prefix or has_step_by_step or has_operators:
        return True

    # Count strong math keywords (not the weak ones like area/volume/radius alone)
    strong_math_keywords = re.findall(
        r"\b(?:"
        r"solve|calculate|compute|evaluate|"
        r"equation|formula|algebra|geometry|trigonometry|"
        r"derivative|integral|differential|calculus|"
        r"matrix|determinant|vector|logarithm|log\s+base|"
        r"quadratic|polynomial|binomial|factorial|"
        r"perimeter|circumference|velocity|acceleration|"
        r"probability|permutation|combination|"
        r"integer|integers|prime\s+number|"
        r"divisible|divisibility|factor|multiple|"
        r"quotient|remainder|modulo|modulus|"
        r"sum\s+of|product\s+of|difference\s+between"
        r")\b",
        lower,
    )
    weak_math_keywords = re.findall(
        r"\b(?:area|volume|radius|diameter|speed|distance)\b",
        lower,
    )

    num_count = len(_NUM_RE.findall(prompt))

    # 2+ strong keywords → clear math
    if len(strong_math_keywords) >= 2:
        return True

    # 1 strong keyword + numbers → moderate math
    if len(strong_math_keywords) >= 1 and num_count >= 2:
        return True

    # 1 weak keyword + operators + numbers → moderate math
    if len(weak_math_keywords) >= 1 and has_numeric_op and num_count >= 2:
        return True

    # 1 weak keyword + 3+ numbers → possible word problem
    if len(weak_math_keywords) >= 1 and num_count >= 3:
        return True

    # Numeric density > 8 per 100 chars (very number-heavy)
    nd = _numeric_density(prompt)
    if nd > 8 and num_count >= 3:
        return True

    return False


def _is_reasoning_intent(prompt: str, lower: str) -> bool:
    """True if the prompt's primary intent is logical reasoning.

    Key: excludes output formatting instructions like
    'Structure your answer as: (1) reasoning, (2) conclusion'
    which are NOT reasoning intent.
    """
    # First check: if the only logic/reasoning words appear in output formatting,
    # it's not reasoning intent.
    has_output_format = bool(re.search(
        r"\b(?:structure|format|organize|present)\s+your\s+(?:answer|response|explanation)\s+as\b",
        lower,
    ))
    has_output_structure = bool(re.search(
        r"\(\d\)\s+\w+|\(\w+\)\s+\w+",
        prompt,
    ))

    reasoning_words_in_text = bool(re.search(
        r"\b(reasoning|conclusion|step|analysis)\b",
        lower,
    ))

    # If reasoning words only appear in output formatting context, skip them
    if has_output_format and has_output_structure and reasoning_words_in_text:
        # Check if ALL reasoning words are part of formatting
        # Simple heuristic: if there are < 3 reasoning/logic signals and output
        # formatting is present, it's probably not reasoning intent
        logic_signal_count = len(_LOGIC_CONSTRAINT_RE.findall(prompt))
        if logic_signal_count <= 1:
            return False

    # Must have strong constraint/reasoning patterns
    constraints = _LOGIC_CONSTRAINT_RE.findall(prompt)
    constraint_count = sum(1 for w in {"each", "every", "must", "if", "then",
                                       "either", "neither", "all", "none",
                                       "unless", "except", "therefore", "hence",
                                       "thus", "conclusion", "implies", "infer",
                                       "premise", "assumption"}
                           if re.search(rf"\b{w}\b", lower))

    has_if_then = bool(re.search(r"\bif\b.{0,60}\bthen\b", lower, re.DOTALL))
    has_logic_wotf = bool(_LOGIC_WOTF_RE.search(lower))
    has_puzzle = bool(_LOGIC_PUZZLE_STRUCTURE_RE.search(prompt))
    names = len(set(_LOGIC_NAME_DENSITY_RE.findall(prompt)))

    # If it has MMLU structure (Choices:), it's factual even with some logic words
    if _has_mmlu_structure(prompt, lower):
        # BUT check if it's actually a logic puzzle with choices
        # Logic puzzles never have domain-knowledge choices
        if constraint_count >= 5 and has_puzzle:
            return True
        return False

    # If-then is a strong standalone reasoning signal
    if has_if_then and (len(constraints) >= 1 or names >= 2):
        return True

    # Logic-specific WOTF
    if has_logic_wotf:
        return True

    # Puzzle structure with names
    if has_puzzle and names >= 3:
        return True

    # Multiple constraints + names
    if len(constraints) >= 2 and constraint_count >= 3:
        return True

    # High constraint density
    if constraint_count >= 4 and names >= 2:
        return True

    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_factual(category_8way: str, prompt: str) -> str:
    """
    Override the primary 8-way category when factual vs logic/math
    is misclassified.

    Pure deterministic — zero model calls, zero imports beyond stdlib.

    Args:
        category_8way: Primary classifier output ("factual", "logic", "math", ...)
        prompt: The user prompt / question text

    Returns:
        Corrected category string (one of "factual", "logic", "math", or
        the original category unchanged)
    """
    if category_8way not in ("factual", "logic", "math"):
        return category_8way

    lower = prompt.lower()

    # Compute scores for the three overlapping categories
    factual_score = _score_factual(prompt, lower)
    logic_score = _score_logic(prompt, lower)
    math_score = _score_math(prompt, lower)

    # ---- STEP 1: Strong factual override ----
    # If the prompt has clear factual structure (SQuAD, MMLU Choices, definitions),
    # return factual regardless of what the primary said.

    # SQuAD / reading comprehension format (overwhelmingly strong)
    if _has_squad_structure(prompt, lower):
        return "factual"

    # MMLU-style choices (124/200 factual, 0/200 logic/math)
    if _has_mmlu_structure(prompt, lower):
        # Exception: if it's ALSO a heavy logic puzzle with many constraints
        # (unlikely but defensive)
        constraints = _LOGIC_CONSTRAINT_RE.findall(prompt)
        if len(constraints) < 3:
            return "factual"
        # Check if this is a logic puzzle pretending to have choices
        constraint_words = {"each", "every", "all", "must", "if", "then",
                            "either", "neither", "unless"}
        ccount = sum(1 for w in constraint_words if re.search(rf"\b{w}\b", lower))
        if ccount < 4:
            return "factual"
        # Falls through to scoring below

    # Source markers (Context:/Passage:/Article:/Text:/Document:)
    source_match = _FACTUAL_SOURCE_RE.search(prompt)
    if source_match and factual_score >= 3.0:
        return "factual"

    # Definition patterns
    if _FACTUAL_DEF_RE.search(lower) and factual_score >= 3.0:
        return "factual"

    # ---- STEP 2: Primary said "factual" — check for logic/math override ----
    if category_8way == "factual":
        # Strong reasoning intent → override to logic
        if _is_reasoning_intent(prompt, lower) and logic_score > factual_score + 1.0:
            return "logic"

        # Strong calculation intent → override to math
        if _is_calculation_intent(prompt, lower) and math_score > factual_score + 1.0:
            return "math"

        # Closer check: if logic beats factual by a big margin
        if logic_score > factual_score + 3.0 and _is_reasoning_intent(prompt, lower):
            return "logic"
        if math_score > factual_score + 3.0 and _is_calculation_intent(prompt, lower):
            return "math"

        # When scores are close, stay with factual
        return "factual"

    # ---- STEP 3: Primary said "logic" — check for factual override ----
    if category_8way == "logic":
        # LogiQA guard: if there's "Question:" that asks about inference/
        # conclusion/assumption, this is a reasoning task, not factual QA.
        _logiqa_reasoning_question = re.search(
            r"question\s*:.*?(?:"
            r"infer(?:red|ence)?|conclu(?:de|ded|ding|sion|sive)|"
            r"deduc(?:e|ed|ing|tion)|"
            r"imply|implied|implication|"
            r"assum(?:e|ed|ing|ption)|"
            r"weaken|strengthen|justify|support|"
            r"must\s+be\s+(?:true|false)|"
            r"can\s+be\s+(?:inferred|concluded|deduced)|"
            r"refu(?:te|ting|tal)|"
            r"explain(?:\s+the\s+above|\s+this|\s+the\s+seemingly)|"
            r"anomal(y|ies)|"
            r"closest\s+to\s+the\s+meaning|"
            r"best\s+(?:explain|refute|describes?|characterizes?|account|argument)|"
            r"argument(?:\s+against|\s+take\s+place|s\s+above)?|"
            r"reasoning|logically|"
            r"raise\s+(?:the\s+most\s+)?doubts?|"
            r"opinions?\s+of\s+the\s+above|"
            r"above\s+argument|"
            r"above\s+(?:reasoning|conclusion|speculation|point)"
            r")",
            lower, re.DOTALL,
        )
        if _logiqa_reasoning_question:
            return "logic"

        # If factual score is significantly higher, override
        if factual_score >= 4.0 and factual_score > logic_score + 1.0:
            return "factual"

        # Strong knowledge indicators + no strong reasoning
        has_knowledge = (
            bool(_FACTUAL_KNOWLEDGE_QUESTIONS.search(lower))
            or bool(_FACTUAL_DOMAIN_RE.search(lower))
        )
        has_weak_reasoning = not _is_reasoning_intent(prompt, lower)
        if has_knowledge and has_weak_reasoning and factual_score >= 2.0:
            return "factual"

        return "logic"

    # ---- STEP 4: Primary said "math" — check for factual override ----
    if category_8way == "math":
        # If factual score is significantly higher, override
        if factual_score >= 4.0 and factual_score > math_score + 1.0:
            return "factual"

        # Strong knowledge indicators + no calculation intent
        has_knowledge = (
            bool(_FACTUAL_KNOWLEDGE_QUESTIONS.search(lower))
            or bool(_FACTUAL_DOMAIN_RE.search(lower))
        )
        has_no_calc = not _is_calculation_intent(prompt, lower)
        if has_knowledge and has_no_calc and factual_score >= 2.0:
            return "factual"

        return "math"

    # ---- STEP 5: Fallback (shouldn't reach here) ----
    return category_8way


# ---------------------------------------------------------------------------
# Diagnostic / CLI
# ---------------------------------------------------------------------------

def diagnose(prompt: str) -> dict:
    """Return detailed scores for debugging."""
    lower = prompt.lower()
    return {
        "factual_score": _score_factual(prompt, lower),
        "logic_score": _score_logic(prompt, lower),
        "math_score": _score_math(prompt, lower),
        "has_squad": _has_squad_structure(prompt, lower),
        "has_mmlu": _has_mmlu_structure(prompt, lower),
        "has_source_marker": bool(_FACTUAL_SOURCE_RE.search(prompt)),
        "is_reasoning_intent": _is_reasoning_intent(prompt, lower),
        "is_calculation_intent": _is_calculation_intent(prompt, lower),
    }


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) > 1 and sys.argv[1] == "--validate":
        # Validate against training-v2.json
        with open("data/eval/training-v2.json") as f:
            data = json.load(f)

        total = 0
        correct = 0
        confusion = {"factual": {"factual": 0, "logic": 0, "math": 0},
                     "logic": {"factual": 0, "logic": 0, "math": 0},
                     "math": {"factual": 0, "math": 0, "logic": 0}}

        for item in data:
            true_cat = item["category"]
            if true_cat not in ("factual", "logic", "math"):
                continue
            predicted = resolve_factual(true_cat, item["prompt"])
            total += 1
            if predicted == true_cat:
                correct += 1
            confusion[true_cat][predicted] = confusion[true_cat].get(predicted, 0) + 1

        print(json.dumps({
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 4),
            "confusion": confusion,
        }, indent=2))

    elif len(sys.argv) > 1:
        prompt = sys.argv[1]
        result = diagnose(prompt)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python -m agent.secondary_factual --validate")
        print("       python -m agent.secondary_factual '<prompt>'")
