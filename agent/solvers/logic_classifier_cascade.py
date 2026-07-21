"""
Binary reject cascade for routing logic prompts to the correct solver tool.

Each level is a high-precision binary classifier that checks if a prompt
matches a specific logic subtype. If yes, it routes to the corresponding
solver; if no, it falls through to the next level.

Cascade structure:
  Level 1: truth-teller/liar puzzles     → solve_logic (truth-teller solver)
  Level 2: number/letter sequences       → solve_logic (sequence solver)
  Level 3: syllogisms                     → solve_logic (syllogism solver)
  Level 4: constraint/zebra puzzles      → solve_logic (constraint solver)
  Level 5: argument analysis (LogiQA)    → solve_logical_reasoning
  Fallback: None                           → LLM handles it

Each classifier returns (bool, confidence) with confidence 0.0-1.0.
All classifiers are pure deterministic regex/heuristic — zero model calls.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

from agent.solvers.deterministic import solve_logic
from agent.solvers.logic_reasoning import solve_logical_reasoning
from agent.solvers.prototype_zebra_solver import solve_zebra_puzzle

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Level 1: Truth-teller / liar puzzles
# ═══════════════════════════════════════════════════════════════════════════

# High-precision patterns — must mention a truth-telling/lying attribute
_TRUTH_TELLER_KEYWORDS = [
    r"\bknight[s]?\b",
    r"\bknave[s]?\b",
    r"\btruth[-\s]?teller\b",
    r"\btruth[-\s]?tellers\b",
    r"\balways\s+tells?\s+(?:the\s+)?truth\b",
    r"\balways\s+lies?\b",
    r"\b(?:tells?\s+the\s+truth|tells?\s+truth)\b",
    r"\bis\s+(?:a\s+)?liar\b",
]

# Negative guards — match on generic truth/lie mentions that aren't puzzles
_TRUTH_TELLER_NEGATIVE = [
    r"\btruth\s+is\b",            # "the truth is that..."
    r"\btruth\s+of\b",            # "the truth of the matter"
    r"\btell\s+(?:you|me|us)\s+the\s+truth\b",  # "to tell you the truth"
    r"\bground\s+truth\b",        # ML terminology
    r"\btruth\s+table\b",         # logic gates
    r"\bmathematical\s+truth\b",
    r"\bobjective\s+truth\b",
    r"\btruth\s+condition[s]?\b",
    r"\bin\s+truth\b",
    r"\btruth\s+be\s+told\b",
    r"\blie\s+(?:in|within|at|about|on)\b",  # "the problem lies in..."
]


def is_truth_teller(prompt: str) -> Tuple[bool, float]:
    """Detect truth-teller/liar puzzles (knights and knaves).

    Returns (True, confidence) if the prompt matches.
    High precision (>90%) — may miss some edge cases.
    """
    text = prompt.lower()

    # Check negative guards first — if any match, reject immediately
    for neg in _TRUTH_TELLER_NEGATIVE:
        if re.search(neg, text):
            return False, 0.0

    # Count matching keywords
    matches = 0
    for pat in _TRUTH_TELLER_KEYWORDS:
        if re.search(pat, text):
            matches += 1

    if matches == 0:
        return False, 0.0

    # Also require at least one named character (capitalized word) or
    # a statement pattern like "X said" or "X says"
    has_speaker = bool(
        re.search(r"\b[A-Z][a-z]+\s+(?:says?|said|stated?|claimed?)\b", prompt)
        or re.search(r"\b[A-Z][a-z]+\s*:\s*[\"']", prompt)
    )

    confidence = min(0.5 + matches * 0.2, 0.95)
    if has_speaker and matches >= 2:
        confidence = min(confidence + 0.2, 0.98)
    elif matches >= 2 and not has_speaker:
        # Two keywords but no named speaker — still likely but slightly lower
        confidence = min(0.5 + matches * 0.15, 0.9)

    return confidence >= 0.6, confidence


# ═══════════════════════════════════════════════════════════════════════════
# Level 2: Number/letter sequence puzzles
# ═══════════════════════════════════════════════════════════════════════════

_SEQUENCE_PATTERNS = [
    # Explicit keywords
    r"\bwhat\s+comes\s+next\b",
    r"\bnext\s+(?:number|term|item|element|value|figure|letter)\b",
    r"\bfind\s+the\s+(?:next|pattern|missing)\b",
    r"\b(?:number|letter)\s+sequence\b",
    r"\bsequence\s+(?:of\s+)?(?:numbers|letters|digits)\b",
    r"\bcomplete\s+the\s+sequence\b",
    r"\bmissing\s+(?:number|term)\s+in\s+(?:the\s+)?sequence\b",
    r"\bpattern\s+:\s*\d+",
    r"\bseries\s+:\s*\d+",
    # Raw number sequence ending with ? — 3+ numbers then ?
    r"\d+\s*,\s*\d+\s*,\s*\d+.*\?\s*$",
    # Alternating / Fibonacci pattern indicators
    r"\beach\s+term\s+(?:doubles|triples|adds|multiplied)\b",
    r"\bpattern\s+is\s+to\s+(?:double|triple|add|multiply)\b",
]

# Negative guards — things that look like sequences but aren't
_SEQUENCE_NEGATIVE = [
    r"\bsequence\s+(?:of\s+)?events?\b",     # historical sequence
    r"\bsequence\s+(?:of\s+)?steps?\b",       # step sequence
    r"\bsequence\s+diagram\b",
    r"\bDNA\s+sequence\b",
    r"\bgene\s+sequence\b",
    r"\bprotein\s+sequence\b",
    r"\btimeline\b",
    r"\bchronological\b",
    r"\bordering\s+of\s+events\b",
    r"\bsequence\s+(?:of\s+)?operations?\b",
    # Math word problem patterns — lists of scores/numbers in context
    r"\bscores?\s+of\b",
    r"\btests?\s+(?:scores?|grades?|results?)\b",
    r"\baverage\s+(?:of|score|grade)\b",
    r"\bmean\s+(?:of|score|value)\b",
    r"\bhow\s+many\s+(?:does|do|points)\b",
    r"\btotal\s+(?:cost|price|amount|distance|time|weight)\b",
    r"\b(?:add|subtract|multiply|divide)\s",
    r"\bsolve\s*:\s*\w",
    r"\bfind\s+(?:the\s+)?(?:value|number|sum|difference|product|quotient)\b",
    r"\bwhat\s+is\s+the\s+(?:value|sum|difference|product|average|mean|total)\b",
]


def is_sequence(prompt: str) -> Tuple[bool, float]:
    """Detect number/letter sequence puzzles (what comes next).

    Returns (True, confidence) if the prompt asks about finding the next
    element in a numeric or alphabetic sequence.
    """
    text = prompt.lower()

    # Negative guards
    for neg in _SEQUENCE_NEGATIVE:
        if re.search(neg, text):
            return False, 0.0

    # Check explicit patterns
    matches = 0
    for pat in _SEQUENCE_PATTERNS:
        if re.search(pat, text):
            matches += 1

    # Also check for 3+ numbers separated by commas/spaces + question mark
    has_raw_sequence = bool(
        re.search(r"\d+\s*[,\s]\s*\d+\s*[,\s]\s*\d+", text)
        and "?" in text
    )

    if matches >= 1:
        confidence = min(0.5 + matches * 0.2, 0.95)
        if has_raw_sequence:
            confidence = min(confidence + 0.2, 0.98)
        return True, confidence

    if has_raw_sequence:
        return True, 0.7

    return False, 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Level 3: Syllogisms (All/No/Some X are Y)
# ═══════════════════════════════════════════════════════════════════════════

_SYLLOGISM_PATTERN = re.compile(
    r"(?:all|no|some|every|not\s+all|most|few)\s+\w+\s+(?:are|is|have|has)\s+",
    re.IGNORECASE,
)

# Syllogism-specific negative guard — exclude general categorization
_SYLLOGISM_NEGATIVE = [
    r"\ball\s+(?:of|the)\s+(?:us|them|you|these|those)\b",
    r"\b(?:all|no|some)\s+(?:people|men|women)\s+(?:are|have)\s+",
    r"\bnot\s+all\s+of\b",
    r"\bno\s+(?:one|body|thing)\b",
    r"\bevery\s+(?:time|day|week|month|year|morning|evening|night)\b",
    r"\ball\s+(?:right|set|done|good|bad)\b",
    # Zebra puzzle preamble — "Each house has a unique attribute"
    r"\beach\s+(?:house|person|one|member|player|item|entity)\s+(?:is|has|was|occupied)\b",
    r"\beach\s+of\s+(?:the\s+)?(?:houses?|persons?|members?|players?|items?|entities?)\b",
    r"\bThere\s+are\s+\d+\s+houses?\b",
    r"\bnumbered\s+\d+\s+to\s+\d+\b",
    r"\battributes?\s*:",
    r"\bcharacteristics?\s*:",
]


def is_syllogism(prompt: str) -> Tuple[bool, float]:
    """Detect syllogisms — "All X are Y", "No X are Y", "Some X are Y" patterns.

    Requires at least 2 premise statements using syllogistic quantifiers
    to avoid false positives on single statements of fact.
    """
    text = prompt.lower()

    # Negative guard: generic statements about people
    for neg in _SYLLOGISM_NEGATIVE:
        if re.search(neg, text):
            return False, 0.0

    # Count premise-like statements
    sentences = text.replace("?", ".").replace("!", ".").split(".")
    premise_count = 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if _SYLLOGISM_PATTERN.search(sent):
            premise_count += 1

    if premise_count >= 3:
        return True, min(0.7 + premise_count * 0.1, 0.95)
    if premise_count == 2:
        return True, 0.7

    # Also detect single premise + answer choices (A-D) for simple syllogisms
    if premise_count == 1:
        has_choices = bool(re.search(r"\b[A-D][\.\)]\s", prompt))
        if has_choices:
            return True, 0.6

    return False, 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Level 4: Constraint puzzles (zebra-style, named entities)
# ═══════════════════════════════════════════════════════════════════════════

_CONSTRAINT_PATTERNS = [
    # Zebra puzzle format: "Solve the following logic puzzle: There are N houses..."
    r"Solve\s+the\s+following\s+logic\s+puzzle",
    r"There\s+are\s+\d+\s+houses?",
    r"Each\s+house\s+is\s+occupied\s+by\s+a\s+different\s+person",
    r"numbered\s+\d+\s+to\s+\d+\s+from\s+left\s+to\s+right",
    # Named entities + constraints
    r"each\s+\w+\s+has\s+a\s+unique\s+",
    r"Each\s+person\s+has\s+a\s+unique\s+",
    r"\b(?:names?|attributes?|characteristics?)\s*:",
    # Constraint keywords
    r"\bmust\s+be\s+(?:before|after|next\s+to|between|first|second|third|fourth|last)\b",
    r"\bdifferent\s+(?:from|than)\s+each\s+other\b",
    r"\bat\s+least\s+one\s+of\s+(?:\w+\s+){1,3}(?:is|has|was)\b",
]

_CONSTRAINT_NEGATIVE = [
    r"\bconstraint\s+(?:satisfaction|programming|optimization)\b",
    r"\b(?:SQL|database)\s+constraint\b",
    r"\bconstraint\s+on\b",
    r"\budge\s+constraint\b",
    r"\btime\s+constraint\b",
    r"\bconstraints\s+of\b",
    r"\b(?:physical|financial|resource)\s+constraint\b",
]


def is_constraint_puzzle(prompt: str) -> Tuple[bool, float]:
    """Detect constraint puzzles with named entities (zebra-style).

    Looks for:
    - "Solve the following logic puzzle..." preamble
    - "N houses" with positions
    - Named entities with attributes and constraints
    - "must be" positional constraints
    """
    text = prompt.lower()

    # Negative guards
    for neg in _CONSTRAINT_NEGATIVE:
        if re.search(neg, text):
            return False, 0.0

    matches = 0
    for pat in _CONSTRAINT_PATTERNS:
        if re.search(pat, text):
            matches += 1

    if matches == 0:
        return False, 0.0

    # Check for named entities (3+ capitalized words = people/places/things)
    names = re.findall(r"\b[A-Z][a-z]{2,}\b", prompt)
    # Filter out common words that are capitalized in prompts
    skip_names = {
        "The", "This", "That", "These", "Those", "There", "Here", "What",
        "Which", "When", "Where", "Who", "How", "Why", "One", "Two", "Three",
        "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "First",
        "Second", "Third", "Last", "Next", "Each", "Every", "Some", "Many",
        "Both", "Neither", "Either", "All", "Any", "None", "Please", "Help",
        "Solve", "Find", "Given", "Using", "Assume", "Let", "True", "False",
        "Note", "Hint", "Answer", "Question", "Options", "Option", "Task",
        "Example", "Input", "Output", "Result", "Conclusion", "Reasoning",
        "Because", "However", "Therefore", "Thus", "Hence", "Although",
        "Also", "Only", "Just", "Not", "And", "But", "For", "With",
    }
    real_names = [n for n in names if n not in skip_names and len(n) > 1]

    confidence = min(0.5 + matches * 0.15, 0.95)

    if len(real_names) >= 3:
        confidence = min(confidence + 0.2, 0.98)

    # Strong indicator: zebra puzzle format with "houses" + "unique"
    if matches >= 2 and re.search(r"houses?", text) and re.search(r"unique", text):
        confidence = max(confidence, 0.85)

    return confidence >= 0.6, confidence


# ═══════════════════════════════════════════════════════════════════════════
# Level 5: Argument analysis (LSAT / LogiQA style)
# ═══════════════════════════════════════════════════════════════════════════

_ARGUMENT_ANALYSIS_PATTERNS = [
    # Question type indicators
    r"\b(?:most|best)\s+(?:strengthen|weaken|support|undermine)\b",
    r"\bwhich\s+of\s+the\s+following\b",
    r"\bargument\s+(?:depends|relies|assumes|presupposes)\b",
    r"\bcan\s+be\s+(?:properly\s+)?inferred\b",
    r"\bmust\s+be\s+true\b",
    r"\blogically\s+(?:follows|implied|deduced|completes)\b",
    r"\b(?:identifies|describe)\s+a\s+flaw\b",
    r"\breasoning\s+is\s+flawed\b",
    r"\bmain\s+(?:point|conclusion|idea)\b",
    r"\bbest\s+explains\b",
    r"\bresolv(?:e|ing)\s+(?:the\s+)?(?:paradox|discrepancy|inconsistency)\b",
    r"\bassumption\s+(?:required|needed|made)\s+by\s+the\s+argument\b",
    # Chinese-style LogiQA patterns
    r"\bis\s+correct\b",
    r"\bcan\s+be\s+(?:concluded|drawn|derived|obtained)\b",
    r"\bcannot\s+be\s+(?:derived|inferred|concluded|determined)\b",
    r"\bthe\s+above\s+(?:argument|statement|passage|text)\b",
    r"\bthe\s+following\s+(?:conclusion|statement|inference|deduction)\b",
    r"\bcan\s+we\s+(?:conclude|infer|draw)\b",
    r"\bdoes\s+not\s+(?:support|weaken|undermine|contradict)\b",
    r"\bif\s+the\s+above\s+is\s+true\b",
    r"\bwhich\s+of\s+the\s+following\s+can\b",
    r"\bwhich\s+statement\b",
]

_ARGUMENT_ANALYSIS_NEGATIVE = [
    r"\bargument\s+(?:about|over|between|with)\b",  # "we had an argument"
    r"\bfollowing\s+(?:the\s+)?(?:rule|instruction|step|guide|tutorial|recipe)\b",
    r"\b(?:is|are)\s+(?:the|a)\s+(?:following\s+)?(?:list|table|set|example|code|function|program|script|text|passage|article|paragraph)\b",
    r"\bfollowing\s+(?:code|function|program|script)\b",
    r"\bfollowing\s+(?:text|passage|article|paragraph)\b",
    r"\bexplain\s+(?:the\s+)?(?:concept|theory|phenomenon|process|mechanism)\b",
    # MMLU-style factual questions: "Choices:" followed by comma-separated list
    r"\bchoices?\s*:",
]


def is_argument_analysis(prompt: str) -> Tuple[bool, float]:
    """Detect LSAT / LogiQA style argument analysis questions.

    Returns (True, confidence) if the prompt contains argument analysis
    patterns: question stem keywords, answer choices, and a paragraph of
    reasoning.
    """
    text = prompt.lower()

    # Negative guards
    for neg in _ARGUMENT_ANALYSIS_NEGATIVE:
        if re.search(neg, text):
            return False, 0.0

    # Check for question patterns
    matches = 0
    for pat in _ARGUMENT_ANALYSIS_PATTERNS:
        if re.search(pat, text):
            matches += 1

    if matches == 0:
        # Final strong signal: "Question:" prefix + numbered choices (0-4 or A-E)
        has_question_prefix = bool(
            re.search(r"(?:^|\n)\s*(?:Q|Question)\s*[:\\.]\s*", prompt, re.MULTILINE)
        )
        has_choices = bool(
            re.search(r"(?:^|\n)\s*(?:[A-E][\.\)]|\([A-E]\)|[0-4][\.\)])\s", prompt, re.MULTILINE)
        )
        if has_question_prefix and has_choices:
            return True, 0.75
        return False, 0.0

    # Check for answer choices
    has_choices = bool(
        re.search(r"(?:^|\n)\s*(?:[A-E][\.\)]|\([A-E]\)|[0-4][\.\)])\s", prompt, re.MULTILINE)
    )

    # Check for "Question:" or "Q:" prefix
    has_question_prefix = bool(re.search(r"(?:^|\n)\s*(?:Q|Question)\s*[:\\.]\s*", prompt, re.MULTILINE))

    confidence = min(0.5 + matches * 0.15, 0.9)

    if has_choices:
        confidence = min(confidence + 0.15, 0.95)
    if has_question_prefix:
        confidence = min(confidence + 0.1, 0.95)

    # Strong indicator: both question stem + answer choices
    if matches >= 2 and (has_choices or has_question_prefix):
        confidence = max(confidence, 0.85)

    return confidence >= 0.6, confidence


# ═══════════════════════════════════════════════════════════════════════════
# Routing function
# ═══════════════════════════════════════════════════════════════════════════

# Routes tracked for reporting
_ROUTE_NAMES = {
    1: "truth_teller",
    2: "sequence",
    3: "syllogism",
    4: "constraint_puzzle",
    5: "argument_analysis",
}


def route_logic(prompt: str) -> Optional[str]:
    """Walk the binary reject cascade and return the first solver's output.

    Args:
        prompt: The full prompt text.

    Returns:
        Solver output string, or None if no solver matched (LLM fallback).
    """
    # Level 1: Truth-teller / liar puzzles
    matched, conf = is_truth_teller(prompt)
    if matched:
        logger.debug(f"Cascade Level 1 (truth_teller, conf={conf:.2f})")
        result = solve_logic(prompt, "logic")
        if result is not None:
            return result
        # Solver didn't produce an answer — fall through to next level
        # to give other tools a chance

    # Level 2: Number/letter sequences
    matched, conf = is_sequence(prompt)
    if matched:
        logger.debug(f"Cascade Level 2 (sequence, conf={conf:.2f})")
        result = solve_logic(prompt, "logic")
        if result is not None:
            return result

    # Level 3: Syllogisms
    matched, conf = is_syllogism(prompt)
    if matched:
        logger.debug(f"Cascade Level 3 (syllogism, conf={conf:.2f})")
        result = solve_logic(prompt, "logic")
        if result is not None:
            return result

    # Level 4: Constraint puzzles (zebra-style)
    matched, conf = is_constraint_puzzle(prompt)
    if matched:
        logger.debug(f"Cascade Level 4 (constraint_puzzle, conf={conf:.2f})")
        # Try the dedicated zebra solver first (handles named entities, multiple attributes)
        result = solve_zebra_puzzle(prompt, "logic")
        if result is not None:
            return result
        # Fall through to the generic constraint solver
        result = solve_logic(prompt, "logic")
        if result is not None:
            return result

    # Level 5: Argument analysis (LogiQA/LSAT)
    matched, conf = is_argument_analysis(prompt)
    if matched:
        logger.debug(f"Cascade Level 5 (argument_analysis, conf={conf:.2f})")
        result = solve_logical_reasoning(prompt, "logic")
        if result is not None:
            return result

    # Fallback — return None for LLM to handle
    return None
