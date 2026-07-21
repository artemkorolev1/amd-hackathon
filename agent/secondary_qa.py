"""
secondary_qa.py — Factual QA disambiguator.

Resolves factual↔summarization and factual↔code_gen confusions by detecting
SQuAD-style reading comprehension structure (Context + Question), MCQ choice
formatting, and factual lookup patterns that the primary scorer misroutes.

Pure regex/heuristic — zero model calls.
"""

import re

# ── SQuAD / reading comprehension structure ──
_SQUAD_CONTEXT_RE = re.compile(
    r'^(Context|Passage|Article|Text|Document|Story|Paragraph|Excerpt)\s*:',
    re.IGNORECASE | re.MULTILINE,
)
_SQUAD_QUESTION_RE = re.compile(
    r'(?:^|\n)\s*(Question|Query|Q)\s*[\.:]\s',
    re.IGNORECASE | re.MULTILINE,
)
# MCQs that start with an A) option — indicates reading comprehension
_MCQ_RC_RE = re.compile(
    r'^[A-Za-z]\)\s', re.MULTILINE
)
# "Choices:" or "Options:" line followed by lettered options
_CHOICES_BLOCK_RE = re.compile(
    r'(?:Choices|Options|Answers|Select|Choose)(?:\s*:)?\s*\n\s*(?:[A-Za-z][\.\)]\s)',
    re.IGNORECASE | re.DOTALL,
)
# Narrative ending in a question (period+capitalized+question mark pattern)
_NARRATIVE_QUESTION_RE = re.compile(
    r'\.\s+[A-Z][a-z]+.{5,60}\?\s*$', re.DOTALL
)

# ── BIO/MEDICAL MCQs (high factual density) ──
_BIO_MED_MCQ_RE = re.compile(
    r'\b(which of the following|what is the most likely|the patient|diagnosis|'
    r'treatment|symptoms?|syndrome|disease|disorder|infection|'
    r'clinical|patholog|etiology|prognosis|therapy)\b',
    re.IGNORECASE,
)

# ── CODE_GEN GUARD patterns ──
# MCQ with lettered choices that triggers code_gen scorer but isn't code
_CODE_GEN_FP = re.compile(
    # Format like "A) homologous structures\nB) analogous...\nC)..."
    r'(?:Choices|Options|Select|Choose)(?:\s*:)?\s*\n\s*[A-Za-z][\.\)]\s',
    re.IGNORECASE,
)
# NOT code_gen if it has no function/class/import
_NO_CODE_STRUCTURE = re.compile(
    r'\b(def |class |import |from \w+ import|lambda |return |print\()',
)
# Code-gen FPs have answer choices
_ANSWER_CHOICES = re.compile(
    r'\b(?:[A-Za-z][\.\)]\s+\w+)', re.MULTILINE,
)
# Multiple consecutive lettered options = MCQ, not code
_MCQ_LETTERED_LINES = re.compile(
    r'(?:^[A-Za-z][\.\)]\s+\w+\s*$[\s]*){2,}', re.MULTILINE,
)
# Numeric choices like "5)" or "1."
_NUMERIC_CHOICES = re.compile(
    r'(?:^\d+[\.\)]\s+\w+\s*$[\s]*){2,}', re.MULTILINE,
)

# ── "Choices:" appearing anywhere in the prompt ──
_CHOICES_ANYWHERE = re.compile(
    r'(?:^|\n)\s*(?:Choices|Options|Answers|Select|Choose|Best answer|'
    r'Correct answer|Answer choices)\s*(?::|$)',
    re.IGNORECASE | re.MULTILINE,
)

# ── Long narrative guard (suppress summarization for factual QA) ──
_LONG_FACTUAL_QA_RE = re.compile(
    r'(?:^.{100,})(?:which of the following|what is the|what can be|'
    r'how many|identify the|explain why|describe the|determine the)',
    re.IGNORECASE | re.DOTALL,
)


def resolve_qa(category: str, prompt: str) -> str:
    """Resolve factual↔summarization and factual↔code_gen confusions.

    Args:
        category: Current category from primary classifier
        prompt: Original prompt text

    Returns:
        Corrected category or original if uncertain
    """
    lower = prompt.lower()

    # ── STEP 1: SQuAD reading comprehension → factual ──
    has_squad_context = bool(_SQUAD_CONTEXT_RE.search(prompt))
    has_squad_question = bool(_SQUAD_QUESTION_RE.search(prompt))
    has_narrative_question = bool(_NARRATIVE_QUESTION_RE.search(prompt))
    has_choices_block = bool(_CHOICES_BLOCK_RE.search(prompt))
    is_long_narrative = len(prompt) > 200

    # If it has Context+Question structure, strongly prefer factual
    if has_squad_context and (has_squad_question or has_choices_block):
        return "factual"

    # If it has a narrative with a factual question at the end
    if is_long_narrative and has_narrative_question and not has_squad_context:
        if category == "summarization":
            return "factual"

    # Reading comprehension MCQ: narrative + "Choices:" block
    if is_long_narrative and has_choices_block and not has_squad_context:
        if category in ("summarization", "code_gen"):
            return "factual"

    # ── STEP 2b: Long narrative MCQ → factual (overrides summarization) ──
    # Long passage (200+ chars) + "Choices:" header → factual reading comprehension
    has_choices_header = bool(_CHOICES_ANYWHERE.search(prompt))
    is_long_passage = len(prompt) > 200

    if is_long_passage and has_choices_header:
        if category in ("summarization", "code_gen"):
            return "factual"

    # ── STEP 3: Bio/medical MCQ → factual (override any summarization/code_gen) ──
    if _BIO_MED_MCQ_RE.search(lower) and has_choices_block:
        if category in ("summarization", "code_gen"):
            return "factual"

    # ── MCQ choice formatting → not code_gen ──
    if category == "code_gen":
        is_mcq = (
            bool(_MCQ_LETTERED_LINES.search(prompt))
            or bool(_NUMERIC_CHOICES.search(prompt))
            or bool(_CODE_GEN_FP.search(prompt))
        )
        has_code = bool(
            re.search(
                r'\b(def |class |import |lambda |return |print\()', prompt
            )
        )
        if is_mcq and not has_code:
            return "factual"
    else:
        has_code = False

    # ── STEP 4: Code-gen-like prompts without code structure → factual ──
    if category == "code_gen" and has_choices_block and not has_code:
        return "factual"

    # ── STEP 5: Sentiment MCQs → factual ──
    if category == "sentiment" and has_choices_block:
        return "factual"

    # ── Uncertain — return original ──
    return category
