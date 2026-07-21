"""
Secondary summarization detector — resolves 8-way confusion between
summarization vs math (47 errors), code_gen (40 errors), and logic (69 errors).

Architecture:
  resolve_summarization(category_8way, prompt) -> str

  Pure deterministic — zero model calls, zero imports beyond stdlib.

  Directions overridden:
    primary="logic"    → overrides to summarization when document/reading-comprehension structure
    primary="math"     → overrides to summarization when narrative prose with incidental numbers
    primary="code_gen" → overrides to summarization when document-header structure, no real code
"""

import re
from typing import Dict


# ---------------------------------------------------------------------------
# STRONG DOCUMENT-STRUCTURE PATTERNS (summarization signals)
# ---------------------------------------------------------------------------

# Source attribution headers: "SOURCE 1 (Name, Year):" at line start
_SUMMARY_SOURCE_RE = re.compile(
    r"^(?:SOURCE|STUDY)\s+\d+\s*\(", re.MULTILINE,
)

# News-style dateline: "On January 15, 2024, ..." or "In 2023, ..." at prompt start
_SUMMARY_NEWS_OPENING_RE = re.compile(
    r"^(?:On|In)\s+(?:\w+\s+\d{1,2},?\s+)?\d{4}[,\s]",
)

# Formal document headers in ALL CAPS
_SUMMARY_DOC_HEADER_RE = re.compile(
    r"^(?:LEGAL BRIEF|PRESS RELEASE|STATEMENT BY THE|EXECUTIVE SUMMARY|"
    r"WHITE PAPER|POLICY BRIEF|MEMORANDUM)\b",
    re.MULTILINE,
)

# "Read / Consider / Review / Analyze the following [text/article/passage/document]"
_SUMMARY_READ_RE = re.compile(
    r"\b(?:Read|Consider|Review|Analyze)\s+(?:the following|this\s+(?:text|article|passage|source|document|report|brief))",
    re.IGNORECASE,
)

# According to / published in / reported by / as stated by
_SUMMARY_ATTRIBUTION_RE = re.compile(
    r"\b(?:According to|reported by|published in|as reported by|"
    r"as stated by|according to a\s+(?:report|study|article|analysis))\b",
    re.IGNORECASE,
)

# "presents/provides/offers a [adj] analysis|overview|summary|report"
_SUMMARY_PRESENTS_RE = re.compile(
    r"\b(presents|provides|offers|delivers)\s+(?:a|an|the)\s+"
    r"(?:\w+\s+)?(analysis|overview|summary|report|study|assessment|review|guide|introduction|breakdown)\b",
    re.IGNORECASE,
)

# "A new study" / "The report / study / document" / "This study"
_SUMMARY_STUDY_RE = re.compile(
    r"\b(?:A new study|The report|The document|The article|This study|Our analysis)\b",
)

# SQuAD / reading comprehension context marker
_SQUAD_CONTEXT_RE = re.compile(
    r"^(?:Context|Passage|Article|Text|Document|Story|Paragraph|Excerpt)\s*:",
    re.IGNORECASE | re.MULTILINE,
)

# "Choices:" MCQ formatting (factual reading comprehension, not summarization)
_CHOICES_MCQ_RE = re.compile(
    r"(?:^|\n)\s*(?:Choices|Options|Answers|Select|Choose|Best answer|"
    r"Correct answer|Answer choices)\s*(?::|$)",
    re.IGNORECASE | re.MULTILINE,
)

# HEADLINE / DATELINE / BREAKING markers
_SUMMARY_HEADLINE_RE = re.compile(
    r"^(?:HEADLINE|DATELINE|BREAKING|BRIEF)\s*:", re.MULTILINE,
)

# All-caps headline followed by quote or colon
_SUMMARY_ALLCAPS_HEADLINE_RE = re.compile(
    r"(?:^|\n)\s*[A-Z][A-Z\s'\"'\\u201C\\u2018]{8,}\s*:?\s*[\"'\\u201C\\u2018]",
)

# "Summarize: / Summary:" as a prefix (line-start)
_SUMMARY_PREFIX_RE = re.compile(
    r"^(?:Summarize|Summary):\s*", re.MULTILINE | re.IGNORECASE,
)

# Report/document title pattern: "The [Org]'s [Report Name/Title]"
_SUMMARY_REPORT_TITLE_RE = re.compile(
    r"\bThe\s+\w+(?:\s+\w+){0,4}'\s*s\s+(?:[A-Z][a-z]+\s+){1,4}(?:Report|Outlook|Analysis|Review|Survey|Study|Brief|Update|Assessment|Index)",
)

# Document attribution pattern: "The report, titled '...'," or "The document, titled"  
_SUMMARY_REPORT_TITLED_RE = re.compile(
    r"\b(?:The|A)\s+(?:report|study|article|document|analysis),?\s+titled\b",
    re.IGNORECASE,
)

# Explicit summarization keywords
_SUMMARY_EXPLICIT_RE = re.compile(
    r"\b(summarize|summary|tl;?dr|tldr|condense|recap|gist|"
    r"key\s+points|main\s+idea|boil down|shorten|compress|"
    r"bullet(?:ed)?\s+(?:list|point|summary)|bullet\s+point\s+summary)\b",
    re.IGNORECASE,
)

# "in X sentences/words/bullets" constraint
_SUMMARY_CONSTRAINT_RE = re.compile(
    r"\bin\s+(?:one|two|three|a\s+few|\d+)\s+(?:sentence|word|bullet|point|line|paragraph)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# STRONG NON-SUMMARIZATION PATTERNS (counter-signals)
# ---------------------------------------------------------------------------

# Actual code structure — not prose about code
_CODE_STRUCTURE_RE = re.compile(
    r"(?:```\s*(?:python|js|ts|rust|go|java)?\s*$|"
    r"def\s+\w+\s*\(|class\s+\w+\s*[:\(]|"
    r"from\s+\w+\s+import|import\s+(?:os|sys|json|re|numpy|pandas|typing|collections))",
    re.MULTILINE,
)

# True arithmetic operations (not range expressions or dates)
_MATH_ARITHMETIC_RE = re.compile(
    r"\d+\s*[+*/]\s*\d+",
)

# Solve/compute/calculate at start or with equation
_MATH_EXPLICIT_INTENT_RE = re.compile(
    r"^(?:Solve|Calculate|Compute|Evaluate|Find)\s+(?:the|this|for)",
    re.IGNORECASE,
)

# Logic puzzle / syllogism keywords (unambiguous)
_LOGIC_PUZZLE_RE = re.compile(
    r"\b(?:knight|knave|syllogism|I am thinking of|cryptarithm|"
    r"must be true|can we conclude|what can we conclude|"
    r"who (?:lives|works|owns|sits|drives|likes) in|"
    r"in a row|sits? next to|adjacent to|same order|"
    r"truth.teller|always tells|always lies)\b",
    re.IGNORECASE,
)

# Counter-signals: patterns that indicate this is NOT summarization
# despite having document-like structure
def _is_analytical_prompt(prompt: str, lower: str) -> bool:
    """True if prompt asks for analysis, explanation, or comparison — not summarization."""
    if re.search(
        r"\b(Solve the following|Prove (that|the|a)|"
        r"Explain why|Explain how|Explain step by step|"
        r"Compare .+ and .+|"
        r"Analyze the|Analyze how|"
        r"What is the (difference|relationship|effect|impact|"
        r"connection|link between)|"
        r"Why (do|does|is|are|would|did)|"
        r"How (do|does|is|are|would|can) a|"
        r"Describe the (process|mechanism|relationship|"
        r"difference|similarities|characteristics))", lower
    ):
        return True
    # Bare "Solve:" prefix (not "Solve the following" — already matched above)
    if re.search(r"^Solve\b", prompt, re.IGNORECASE) or \
       re.search(r"^Solve\s", prompt, re.IGNORECASE):
        return True
    # Logic puzzle instructions: "Solve the following logic puzzle"
    if "logic puzzle" in lower and "solve" in lower:
        return True
    return False


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def _document_structure_score(prompt: str, lower: str) -> float:
    """Score how strongly this looks like a document-summarization prompt."""
    score = 0.0

    # +++ Explict summarization keyword +++
    if _SUMMARY_EXPLICIT_RE.search(lower):
        score += 6.0
    if _SUMMARY_PREFIX_RE.search(prompt):
        score += 4.0
    if _SUMMARY_CONSTRAINT_RE.search(lower):
        score += 3.0

    # +++ Read/consider/review the following +++
    if _SUMMARY_READ_RE.search(prompt):
        score += 4.0

    # +++ Document headers +++
    if _SUMMARY_DOC_HEADER_RE.search(prompt):
        score += 4.0
    if _SUMMARY_SOURCE_RE.search(prompt):
        score += 3.5
    if _SUMMARY_NEWS_OPENING_RE.search(prompt):
        score += 3.0

    # +++ Headline markers +++
    if _SUMMARY_HEADLINE_RE.search(prompt):
        score += 3.0
    if _SUMMARY_ALLCAPS_HEADLINE_RE.search(prompt):
        score += 2.0

    # +++ Report/document title attribution +++
    if _SUMMARY_REPORT_TITLE_RE.search(prompt):
        score += 2.0
    if _SUMMARY_REPORT_TITLED_RE.search(lower):
        score += 2.0

    # +++ Attribution / study patterns +++
    if _SUMMARY_ATTRIBUTION_RE.search(lower):
        score += 2.5
    if _SUMMARY_PRESENTS_RE.search(lower):
        score += 2.0
    if _SUMMARY_STUDY_RE.search(prompt):
        score += 2.0

    # +++ Length-based (prose length) +++
    word_count = len(prompt.split())
    if word_count > 100:
        score += 1.0
    if word_count > 180:
        score += 0.5

    # Paragraph breaks
    para_count = prompt.count("\n\n")
    if para_count >= 2:
        score += 1.5
    elif para_count >= 1:
        score += 0.5

    # Narrative connecting words (signals prose, not task)
    if word_count > 60 and re.search(
        r"\b(however|therefore|meanwhile|furthermore|moreover|"
        r"nevertheless|consequently|additionally|in addition|"
        r"as a result|for example|for instance)\b", lower
    ):
        score += 0.5

    return score


def _math_intent_score(prompt: str, lower: str) -> float:
    """Score how strongly the prompt has genuine math intent."""
    score = 0.0
    if _MATH_EXPLICIT_INTENT_RE.search(prompt):
        score += 5.0
    if _MATH_ARITHMETIC_RE.search(prompt):
        score += 4.0
    # Calculation keywords
    calc_words = re.findall(
        r"\b(?:calculate|compute|solve|equation|formula|derivative|integral|"
        r"algebra|geometry|trig|calculus|factorial|permutation|combination|"
        r"probability|matrix|vector|quotient|remainder|modulo|divided by|"
        r"multiplied by|power of|exponent|square root|logarithm|log)\b",
        lower,
    )
    score += len(calc_words) * 2.0
    return score


def _code_intent_score(prompt: str, lower: str) -> float:
    """Score actual code-generation intent, not prose that mentions code words."""
    score = 0.0
    if _CODE_STRUCTURE_RE.search(prompt):
        score += 5.0
    # Write/implement/create a function/class/program
    if re.search(r"\b(write|implement|create|generate)\b.{0,30}\b(function|class|program|script|algorithm|implementation)\b", lower):
        score += 3.0
    return score


def _logic_intent_score(prompt: str, lower: str) -> float:
    """Score actual logic-puzzle intent vs prose comprehension."""
    score = 0.0
    if _LOGIC_PUZZLE_RE.search(lower):
        score += 5.0
    # Constraint word density
    constraint_words = {"each", "every", "must", "if", "then",
                        "either", "neither", "all", "none",
                        "unless", "except", "therefore", "hence",
                        "thus", "implies", "infer", "premise", "assumption"}
    constraint_count = sum(1 for w in constraint_words if re.search(rf"\b{w}\b", lower))
    if constraint_count >= 4:
        score += 3.0
    elif constraint_count >= 3:
        score += 1.5
    return score


def _is_actual_code(prompt: str) -> bool:
    """True if prompt contains actual code fences or definitions."""
    return bool(_CODE_STRUCTURE_RE.search(prompt))


def _has_low_calc_density(prompt: str, lower: str) -> bool:
    """True when numbers appear in prose context (not calculation context)."""
    word_count = len(prompt.split())
    if word_count < 60:
        return False
    # Count numbers that are followed by measurement/prose nouns
    incidental = len(re.findall(
        r"\d+\s*(?:percent|%|million|billion|trillion|GW|MW|kW|kg|km|"
        r"miles?|years?|months?|days?|dollars?|euros?|pounds?|"
        r"participants|patients|adults|countries|cities|companies|"
        r"employees|users|customers|GW|MW|W|reduction|increase|decrease)",
        lower,
    ))
    # Count actual calculation operators
    operators = len(re.findall(r"\d+\s*[+*/]\s*\d+", prompt))
    return incidental > operators * 2 and incidental >= 2


# ---------------------------------------------------------------------------
# Main resolution function
# ---------------------------------------------------------------------------

def resolve_summarization(category: str, prompt: str) -> str:
    """
    Re-classify if primary 8-way got summarization wrong.

    Returns corrected category or original if uncertain.

    Edge-triggered — only fires when document-structure signals strongly
    indicate summarization and the competing signal is weak.
    """
    if not prompt or not prompt.strip():
        return category

    lower = prompt.lower()

    doc_score = _document_structure_score(prompt, lower)
    math_score = _math_intent_score(prompt, lower)
    code_score = _code_intent_score(prompt, lower)
    logic_score = _logic_intent_score(prompt, lower)
    word_count = len(prompt.split())

    # ── CASE 1: Primary says logic, actually document summarization ──
    if category == "logic":
        # DO NOT override if the prompt is asking for analysis/explanation/proof
        if _is_analytical_prompt(prompt, lower):
            return category
        # Strong document structure + weak logic puzzle signal
        if doc_score >= 4.0 and logic_score < 3.0:
            return "summarization"
        # Read/consider/review text with summarization constraint
        if _SUMMARY_READ_RE.search(prompt) and logic_score < 2.0:
            return "summarization"
        # Explicit summarization keyword + long text
        if _SUMMARY_EXPLICIT_RE.search(lower) and word_count > 60:
            return "summarization"

    # ── CASE 2: Primary says math, actually document text with incidental numbers ──
    if category == "math":
        # DO NOT override if the prompt is asking for math/calculation
        if _is_analytical_prompt(prompt, lower):
            return category
        # Explicit summarization keyword trumps everything
        if _SUMMARY_EXPLICIT_RE.search(lower):
            return "summarization"
        # Source/news structure + weak math intent
        if doc_score >= 3.0 and math_score < 2.0:
            return "summarization"
        # News dateline opening + any prose structure
        if _SUMMARY_NEWS_OPENING_RE.search(prompt) and word_count > 80:
            return "summarization"
        # Study/article attribution with incidental numbers
        if _SUMMARY_STUDY_RE.search(prompt) and _has_low_calc_density(prompt, lower):
            return "summarization"

    # ── CASE 3: Primary says code_gen, actually document with header ──
    if category == "code_gen":
        # Strong document structure + no actual code
        if doc_score >= 4.0 and not _is_actual_code(prompt):
            return "summarization"
        # Explicit summarization keyword
        if _SUMMARY_EXPLICIT_RE.search(lower):
            return "summarization"
        # Document header + summarization constraint
        if _SUMMARY_DOC_HEADER_RE.search(prompt) and not _is_actual_code(prompt):
            return "summarization"

    # ── CASE 4: Primary says factual, has explicit summarization keyword or document header ──
    if category == "factual":
        # MCQ / reading comprehension guard: if the prompt has "Choices:" or MCQ
        # option formatting with long text, it's factual QA, not summarization —
        # even if the answer format instruction contains "bullet list" or "summary" words.
        if _SQUAD_CONTEXT_RE.search(prompt) or \
           (len(prompt) > 200 and _CHOICES_MCQ_RE.search(prompt)):
            return category
        if _SUMMARY_EXPLICIT_RE.search(lower):
            return "summarization"
        if _SUMMARY_READ_RE.search(prompt):
            return "summarization"
        if _SUMMARY_HEADLINE_RE.search(prompt):
            return "summarization"
        if _SUMMARY_DOC_HEADER_RE.search(prompt):
            return "summarization"
        if _SUMMARY_REPORT_TITLE_RE.search(prompt):
            return "summarization"

    return category
