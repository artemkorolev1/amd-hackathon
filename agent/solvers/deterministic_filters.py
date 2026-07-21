"""Deterministic pre-filters for routing tasks to the right solver tool.

Each filter examines a prompt and category, then decides:
1. Can this tool handle it? (returns True/False or tool_name/None)
2. Which specific tool variant to use?

These filters run BEFORE the tool is called — they prevent wasting time
on prompts the tool clearly can't handle.
"""
import re
from typing import Optional


# ═════════════════════════════════════════════════════════════════════════════
# MATH pre-filter — can math_solve handle this?
# ═════════════════════════════════════════════════════════════════════════════

# Raw math expression patterns (math_solve can handle these)
_HAS_RAW_EXPRESSION = re.compile(
    r'(?:\d+\s*[+\-*/%^]\s*\d+)'          # 2 + 3, 5 * 7
    r'|\b(sin|cos|tan|sqrt|log|ln)\s*\('    # sin(30), sqrt(144)
    r'|\b\d+\s*=\s*'                        # 2x = 5, x + 3 = 7
    r'|[,;:]\s*(?:find|what|solve|compute)'  # ... ; find x
    r'|\b(?:solve|compute|evaluate|simplify|calculate)\b'  # solve keywords
    r'|\b(?:remainder|modulo)\b'             # remainder
    r'|\b(?:mean|median|average|determinant|geometric)\b'  # word math we handle
    r'|\b(?:inclusion.exclusion|sum\s+of\s+first)\b'       # Set/sequence math
    r'|\d+\s*[+\-*/%^]\s*\d+',              # bare expressions
    re.I,
)

# Word problem markers (math_solve CANNOT handle these directly)
_HAS_WORD_PROBLEM = re.compile(
    r'(?:how\s+many|how\s+much|what\s+is\s+the\s+(?:total|sum|value|number))'
    r'|(?:if|when|given)\s+.{10,}(?:what|find|determine)'
    r'|(?:store|shop|buy|purchase|sell|sold|price|cost|dollar|pound|euro)'
    r'|(?:train|car|bike|walk|run|travel|speed|distance|time|hour|minute)'
    r'|(?:garden|field|pool|tank|pipe|work|job|together|alone)'
    r'|(?:ratio|fraction|percent|percentage)\s+of',
    re.I,
)


def can_solve_math(prompt: str) -> str:
    """Decide how to handle a math prompt.

    Returns:
        'direct' — math_solve can handle it (raw expression/equation)
        'word_problem' — needs narrative math solver (not built yet)
        'skip' — neither, let LLM handle it
    """
    # Check for raw expression first (high precision)
    if _HAS_RAW_EXPRESSION.search(prompt):
        # Verify it has actual math content (digits + operators or vars)
        has_digits = bool(re.search(r'\d', prompt))
        has_ops = bool(re.search(r'[+\-*/^=]', prompt))
        if has_digits and has_ops:
            return 'direct'
        # Check for symbolic math: "solve for x", "x + 5 = 10"
        if re.search(r'\b(solve|find)\s+for\s+\w+', prompt, re.I) or \
           re.search(r'\w+\s*[+\-*/]\s*\w+\s*=', prompt):
            return 'direct'
        # Pure word problem — check for word-problem markers
        if _HAS_WORD_PROBLEM.search(prompt):
            return 'word_problem'
        # Expression-like but not obviously math — skip
        return 'skip'
    return 'skip'


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARIZATION pre-filter — can summarize handle this?
# ═════════════════════════════════════════════════════════════════════════════

def can_summarize_extractive(prompt: str) -> bool:
    """Check if extractive summarization (Sumy) can handle this prompt.

    Sumy extractive works when:
    - The prompt contains the text to summarize (not a URL/reference)
    - The text is long enough (>100 chars)
    - There's a clear source passage (not a conversational request)
    - The expected output is shorter than the input
    """
    text = prompt.strip()

    # Must be substantial text
    if len(text) < 100:
        return False

    # Must have clear source text (not just a query)
    if not any(marker in text for marker in [
        'text', 'passage', 'article', 'paragraph', 'story',
        'following', 'below', 'document', 'report', 'email',
    ]):
        # Check if it starts with a query-like pattern
        if re.match(r'^(?:summarize|summarise|sum up|tl;dr|condense)\b', text, re.I):
            # If it's just a command without source text, skip
            if len(text) < 200:
                return False
        else:
            # No clear source text indicator
            return False

    # Check it has actual content words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    if len(words) < 20:
        return False

    return True


# ═════════════════════════════════════════════════════════════════════════════
# LOGIC pre-filter — which logic tool to use?
# ═════════════════════════════════════════════════════════════════════════════

def detect_logic_type(prompt: str) -> Optional[str]:
    """Detect which logic tool can handle this prompt.

    Returns tool_name or None if no logic tool applies.
    """
    text = prompt.lower()

    # 1. Truth-teller / Liar puzzles (knights and knaves)
    if any(kw in text for kw in ['knight', 'knave', 'always tells the truth',
                                   'always lies', 'truth-teller', 'liar']):
        # Check for statements/quotes indicating a puzzle
        if re.search(r'\b(says?|states?|claims?)\b', text):
            return 'solve_truth_teller_liar'

    # 2. Number/letter sequence puzzles
    if re.search(r'(?:sequence|pattern|series|what\s+comes?\s+next)\b', text):
        nums = re.findall(r'\d+', text)
        if len(nums) >= 3:
            return 'solve_number_sequence'
        # Letter sequences: A, C, E, G, ?
        if re.search(r'\b[A-Z]\s*,\s*[A-Z]\s*,\s*[A-Z]', prompt):
            return 'solve_number_sequence'

    # 3. Syllogisms
    if re.search(r'(?:all\s+\w+\s+are|no\s+\w+\s+are|some\s+\w+\s+are)\b', text):
        return 'solve_syllogism'

    # 4. Constraint puzzles (seating, ordering, scheduling, attribute matching)
    has_names = bool(re.search(r'\b([A-Z][a-z]{1,10})\b', prompt))
    has_constraint_keywords = any(kw in text for kw in [
        'sits', 'sitting', 'seat', 'arrange', 'order', 'ranking',
        'next to', 'to the left', 'to the right', 'between',
        'taller than', 'shorter than', 'older than', 'younger than',
        'must be', 'cannot be', 'not the same', 'different',
        'assigned', 'matched', 'paired',
    ])
    if has_names and has_constraint_keywords:
        return 'solve_logic_puzzle'

    return None


# ═════════════════════════════════════════════════════════════════════════════
# NER pre-filter — can NER extract entities from this?
# ═════════════════════════════════════════════════════════════════════════════

def can_ner(prompt: str) -> bool:
    """Check if spaCy NER can extract entities from this prompt.

    NER works when the text contains named entities like persons,
    organizations, locations, dates, etc.
    """
    text = prompt.strip()

    # Must have text to extract from
    if len(text) < 20:
        return False

    # Check for known entity patterns
    # Capitalised words (potential named entities)
    capitalised = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
    if len(capitalised) >= 2:
        return True

    # Check for numbers (dates, percentages, money)
    if re.search(r'\b\d{4}\b', text) or re.search(r'\$\d+', text):
        return True

    # Check for NER task keywords
    if any(kw in text.lower() for kw in [
        'entity', 'entities', 'named', 'person', 'organisation',
        'location', 'date', 'extract.*entit',
    ]):
        return True

    return False


# ═════════════════════════════════════════════════════════════════════════════
# Sentiment pre-filter — can VADER handle this?
# ═════════════════════════════════════════════════════════════════════════════

def can_solve_sentiment(prompt: str) -> bool:
    """Check if VADER sentiment analysis can handle this prompt.

    VADER works on any text with sentiment-bearing words.
    """
    text = prompt.strip()
    if len(text) < 3:
        return False
    # VADER can handle basically any natural language text
    # Check for sentiment-bearing content
    has_sentiment_words = bool(re.search(
        r'\b(good|bad|great|terrible|love|hate|amazing|awful|'
        r'wonderful|horrible|excellent|poor|beautiful|ugly|'
        r'fantastic|dreadful|brilliant|disgusting|perfect|wrong|'
        r'nice|nasty|happy|sad|angry|calm|positive|negative|'
        r'enjoy|dislike|recommend|avoid|best|worst)\b',
        text, re.I
    ))
    if not has_sentiment_words:
        # For very short texts, still try VADER
        return len(text) > 10
    return True


# ═════════════════════════════════════════════════════════════════════════════
# Factual pre-filter — can the fact DB answer this?
# ═════════════════════════════════════════════════════════════════════════════

def can_answer_factual(prompt: str) -> bool:
    """Check if the FTS5 fact DB is likely to answer this question.

    Fact DB works on direct factual questions (who, what, when, where, etc.)
    when the question matches one of 16K indexed facts.
    """
    text = prompt.strip()

    # Must be a question
    if not any(marker in text.lower() for marker in [
        'what', 'who', 'when', 'where', 'why', 'how', 'which',
        '?',
    ]):
        return False

    # Must be a factual knowledge question (not opinion/instruction)
    if re.search(r'\b(?:your\s+opinion|you\s+think|should\s+we|do\s+you\s+like)\b', text, re.I):
        return False

    if len(text) < 5:
        return False

    return True
