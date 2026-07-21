"""
Deterministic (regex/heuristic) fallback for code_debug vs code_gen.

Used when the ML model (code_debug_vs_gen.pkl) is unavailable or for consistency.
Pure regex-based scoring — no sklearn dependency.

Interface matches hierarchical_classifier.py resolve_code():
    resolve_code(category_8way: str, prompt: str) -> str

Also provides:
    deterministic_resolve_code(category_8way: str, prompt: str) -> str | None
        Returns 'code_debug', 'code_gen', or None if uncertain.
"""

import re


# ── Debug signal patterns ──

# "fix the bug" — strongest debug signal (appears in all HumanEvalPack debug prompts)
_FIX_THE_BUG_RE = re.compile(r"fix\s+the\s+bug", re.IGNORECASE)
# "Fix this Python function" or "Fix this code" or "Fix this function"
_FIX_THIS_RE = re.compile(r"fix\s+this\s+(?:python\s+)?(?:function|code|program|script|method)", re.IGNORECASE)
# Standalone "bug"
_BUG_RE = re.compile(r"\bbug\b", re.IGNORECASE)
# "debug"
_DEBUG_RE = re.compile(r"\bdebug\b", re.IGNORECASE)
# Error/failure related — separate patterns so multiple can stack
_ERROR_RE = re.compile(r"\berror\b", re.IGNORECASE)
_TRACEBACK_RE = re.compile(r"\btraceback\b", re.IGNORECASE)
_EXCEPTION_RE = re.compile(r"\bexception\b", re.IGNORECASE)
_CRASH_RE = re.compile(r"\bcrash\b", re.IGNORECASE)
_BROKEN_RE = re.compile(r"\bbroken\b", re.IGNORECASE)
_INCORRECT_RE = re.compile(r"\bincorrect\b", re.IGNORECASE)
_WRONG_RE = re.compile(r"\bwrong\b", re.IGNORECASE)
_FAILING_RE = re.compile(r"\bfailing\b", re.IGNORECASE)
# "not working" / "doesn't work" / "isn't working"
_NOT_WORKING_RE = re.compile(
    r"(?:not\s+work(?:ing|s)?|doesn['’]?t\s+work|isn['’]?t\s+work(?:ing)?)",
    re.IGNORECASE,
)
# "output is"
_OUTPUT_IS_RE = re.compile(r"\boutput\s+is\b", re.IGNORECASE)
# "should be"
_SHOULD_BE_RE = re.compile(r"\bshould\s+be\b", re.IGNORECASE)
# "expected" / "actual"
_EXPECTED_ACTUAL_RE = re.compile(r"\b(?:expected|actual)\b", re.IGNORECASE)
# Standalone "fix" (caught after more specific patterns)
_FIX_RE = re.compile(r"\bfix\b", re.IGNORECASE)  # standalone "fix"


# ── Generation signal patterns ──

# "Write a Python function" or "Write a function"
_WRITE_FUNCTION_RE = re.compile(r"write\s+a\s+(?:python\s+)?function", re.IGNORECASE)
# "Write a ..." in coding context
_WRITE_CODE_RE = re.compile(
    r"write\s+(?:a|the|this|your|some)\s+(?:function|program|code|script|method|algorithm|class)",
    re.IGNORECASE,
)
# "Write" at start of prompt
_WRITE_START_RE = re.compile(r"^\s*write", re.IGNORECASE | re.MULTILINE)
# "def " at start of a line (function definition)
_DEF_RE = re.compile(r"^def\s+|^\s+def\s+", re.MULTILINE)
# "implement a function/program/code/..."
_IMPLEMENT_CODE_RE = re.compile(
    r"implement\s+(?:a|the|this|your)\s+(?:function|program|code|script|method|algorithm|class|solution)",
    re.IGNORECASE,
)
# Generic "implement" (lower weight)
_IMPLEMENT_RE = re.compile(r"\bimplement\b", re.IGNORECASE)
# "create a function/program" — allow extra words between "a" and the noun
_CREATE_CODE_RE = re.compile(
    r"(?:create|generate|build)\s+(?:a|the|this)\s+.*?(?:function|program|code|script|module)",
    re.IGNORECASE,
)
# "function that" / "program that"
_FUNCTION_THAT_RE = re.compile(r"\b(?:function|program)\s+that\b", re.IGNORECASE)
# "in python" / "using python"
_IN_PYTHON_RE = re.compile(r"\b(?:using|in)\s+python\b", re.IGNORECASE)


def deterministic_resolve_code(category_8way: str, prompt: str) -> str | None:
    """
    Pure regex/deterministic scoring for code_debug vs code_gen.

    Scores the prompt for debug signals and generation signals,
    returns a corrected category ONLY when confident.

    Args:
        category_8way: The primary 8-way category prediction.
        prompt: The raw text prompt.

    Returns:
        'code_debug' — confident it's a debug/fix task.
        'code_gen'   — confident it's a code generation task.
        None         — uncertain; caller should fall back to primary or ML.
    """
    if category_8way not in ("code_debug", "code_gen"):
        # Not our lane — return None so the caller doesn't override
        return None

    lower = prompt.lower()

    # ── Score debug signals ──
    debug_score = 0.0

    if _FIX_THE_BUG_RE.search(lower):
        debug_score += 5.0
    if _FIX_THIS_RE.search(lower):
        debug_score += 4.0
    if _BUG_RE.search(lower):
        debug_score += 3.0
    if _DEBUG_RE.search(lower):
        debug_score += 3.0
    if _ERROR_RE.search(lower):
        debug_score += 3.0
    if _TRACEBACK_RE.search(lower):
        debug_score += 3.0
    if _EXCEPTION_RE.search(lower):
        debug_score += 3.0
    if _CRASH_RE.search(lower):
        debug_score += 3.0
    if _BROKEN_RE.search(lower):
        debug_score += 3.0
    if _INCORRECT_RE.search(lower):
        debug_score += 3.0
    if _WRONG_RE.search(lower):
        debug_score += 3.0
    if _FAILING_RE.search(lower):
        debug_score += 3.0
    if _NOT_WORKING_RE.search(lower):
        debug_score += 3.0
    if _OUTPUT_IS_RE.search(lower):
        debug_score += 2.0
    if _SHOULD_BE_RE.search(lower):
        debug_score += 1.0
    if _EXPECTED_ACTUAL_RE.search(lower):
        debug_score += 2.0
    if _FIX_RE.search(lower):
        debug_score += 2.0  # standalone "fix"

    # ── Score generation signals ──
    gen_score = 0.0

    if _WRITE_FUNCTION_RE.search(lower):
        gen_score += 3.0
    elif _WRITE_CODE_RE.search(lower) or _WRITE_START_RE.search(lower):
        gen_score += 2.0

    if _DEF_RE.search(lower):
        gen_score += 1.0

    if _IMPLEMENT_CODE_RE.search(lower):
        gen_score += 3.0
    elif _IMPLEMENT_RE.search(lower):
        gen_score += 1.0  # generic "implement", low weight

    if _CREATE_CODE_RE.search(lower):
        gen_score += 2.0

    if _FUNCTION_THAT_RE.search(lower):
        gen_score += 1.0

    if _IN_PYTHON_RE.search(lower):
        gen_score += 1.0

    # ── Decision ──
    # Return override only when confident:
    #   - dominant score must meet a minimum threshold
    #   - dominant must exceed the other by at least 1.4x
    DEBUG_THRESHOLD = 5.0
    GEN_THRESHOLD = 3.0
    RATIO = 1.4

    if debug_score >= DEBUG_THRESHOLD and debug_score >= gen_score * RATIO:
        return "code_debug"

    if gen_score >= GEN_THRESHOLD and gen_score >= debug_score * RATIO:
        return "code_gen"

    return None


def resolve_code(category_8way: str, prompt: str) -> str:
    """
    Determinstic fallback for code_debug vs code_gen.

    Matches the interface of the ML-based resolve_code in
    hierarchical_classifier.py so it can be used as a drop-in replacement.

    Args:
        category_8way: The primary 8-way category ('code_debug' or 'code_gen').
        prompt: The raw text prompt.

    Returns:
        The corrected category, or the original if uncertain.
    """
    result = deterministic_resolve_code(category_8way, prompt)
    if result is not None:
        return result
    return category_8way
