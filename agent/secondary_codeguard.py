"""
secondary_codeguard.py — Code context guard.

Suppresses code_gen and code_debug when the prompt looks like MCQ formatting
or educational choices rather than actual coding. Fires before code_secondary.

Pure regex/heuristic — zero model calls.
"""

import re

# ── MCQ choice patterns (NOT code) ──
_LETTERED_CHOICES = re.compile(
    r'(?:^[A-Za-z][\.\)]\s+\w+\s*$[\s]*){2,}', re.MULTILINE
)
_NUMERIC_CHOICES = re.compile(
    r'(?:^\d+[\.\)]\s+\w+\s*$[\s]*){2,}', re.MULTILINE
)
_CHOICES_HEADER = re.compile(
    r'(?:Choices|Options|Answers|Select|Choose|Answer choices|'
    r'Response choices|Multiple choice)(?:\s*:)?',
    re.IGNORECASE,
)
_ANSWER_LETTER = re.compile(
    r'\b[A-Z][\.\)]\s+\w+'
)

# ── Genuine code signals ──
_CODE_DEF = re.compile(
    r'\b(def |class |import |from \w+ import|lambda |return |print\()',
)
_CODE_STRUCTURE = re.compile(
    r'\b(if\s+__name__|try:|except:|finally:|elif |else:|for \w+ in |while )'
)
_CODE_BLOCK_FENCE = re.compile(
    r'```(?:python|py)?'
)
_CODE_KEYWORDS = re.compile(
    r'\b(function|method|variable|parameter|argument|return value|'
    r'syntax error|compile|runtime|debug|bug|fix|algorithm|'
    r'implement|write a (function|program|class))\b',
    re.IGNORECASE,
)

# ── MCQ educational subjects that look like code but aren't ──
_EDUCATIONAL_MCQ = re.compile(
    r'\b(multiple.?choice|select the|choose the|which of the following|'
    r'best describes|best explains|refers to|all of the following|'
    r'identify the|classify the|label the|match the following)\b',
    re.IGNORECASE,
)

# ── True code project structure ──
_CODE_PROJECT_RE = re.compile(
    r'(?:\.py|\.js|\.ts|\.rs|\.go|main\.|src/|tests/|requirements\.txt|package\.json)',
    re.IGNORECASE,
)


def resolve_codeguard(category: str, prompt: str) -> str:
    """Detect false-positive code classifications.

    Suppresses code_gen/code_debug when the content is really an MCQ
    or educational test question with choice formatting.

    Args:
        category: Current category (code_gen, code_debug, or other)
        prompt: Original prompt text

    Returns:
        Corrected category or original if uncertain
    """
    if category not in ("code_gen", "code_debug"):
        return category

    has_mcq_format = bool(
        _LETTERED_CHOICES.search(prompt)
        or _NUMERIC_CHOICES.search(prompt)
    )
    has_choices_header = bool(_CHOICES_HEADER.search(prompt))
    has_code_structure = bool(
        _CODE_DEF.search(prompt)
        or _CODE_STRUCTURE.search(prompt)
        or _CODE_BLOCK_FENCE.search(prompt)
    )

    # ── If no actual code structure but has MCQ formatting → not code ──
    if not has_code_structure and (has_mcq_format or has_choices_header):
        return "factual"

    # ── Educational MCQ with answer letters but no code ──
    if (
        has_choices_header
        and not has_code_structure
        and _EDUCATIONAL_MCQ.search(prompt)
    ):
        return "factual"

    # ── Medical/biological choices (numeric + clinical terms) ──
    if (
        has_numeric_mcq(prompt)
        and not has_code_structure
        and not _CODE_KEYWORDS.search(prompt)
    ):
        return "factual"

    # ── Genuinely has code — keep original category ──
    return category


def has_numeric_mcq(prompt: str) -> bool:
    """Check for numeric MCQ patterns (common in bio/med)."""
    return bool(
        re.search(
            r'(?:^\d+\.\s+\w+\s*$[\s]*){2,}',
            prompt, re.MULTILINE,
        )
    )
