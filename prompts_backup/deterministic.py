"""
Deterministic solvers that run BEFORE the local model, producing answers
at zero tokens for arithmetic, logic, sentiment, NER, factual QA, and
code debugging tasks.

Each solver returns a string answer or None (meaning "can't solve, let the
model handle it"). Returns None when uncertain — never guesses.
"""

import itertools
import logging
import math
import re
from typing import Optional

from agent.solvers.tools import calculator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Arithmetic solver
# ---------------------------------------------------------------------------

# Patterns that trigger arithmetic extraction, ordered most-specific first
_ARITH_PATTERNS = [
    # "what is X?" or "What is X"
    re.compile(r"(?:what\s+is|what's)\s+(.+?)\s*\?", re.IGNORECASE),
    # "calculate X"
    re.compile(r"calculate\s+(.+)", re.IGNORECASE),
    # "compute X"
    re.compile(r"compute\s+(.+)", re.IGNORECASE),
    # "solve X"
    re.compile(r"solve\s+(.+)", re.IGNORECASE),
    # "X = ?" or "X=?"
    re.compile(r"(.+?)\s*=\s*\?\s*", re.IGNORECASE),
    # "find X"
    re.compile(r"find\s+(.+)", re.IGNORECASE),
]

# Root extraction patterns: "square root of 144", "sqrt(144)", "cube root of 8"
_ROOT_PATTERN = re.compile(
    r"(?:square\s+)?root\s+(?:of|is)?\s*(\d+)", re.IGNORECASE
)

# Percentage-of pattern: "15% of 240" or "15 percent of 240"
_PERCENT_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:of|)\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Speed/distance/time patterns
# " ... km/h for X hours" or " ... mph for X hours"
_SPEED_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:km/h|mph|m/s|knots?|km per hour|miles per hour)\s+"
    r"(?:for|over|during)\s+(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h|minutes?|mins?|m)\b",
    re.IGNORECASE,
)

# Simple numeric expression: only digits, operators, parens, decimals
_SIMPLE_EXPR = re.compile(
    r"^[\d\s\+\-\*\/\(\)\.\%]+$"
)


def _normalize_expression(raw: str) -> Optional[str]:
    """Clean up a raw math expression string for evaluation."""
    expr = raw.strip().rstrip("?.,!;:")
    # Replace common words with operators
    expr = re.sub(r"\bplus\b", "+", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bminus\b", "-", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\btimes\b", "*", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bmultiplied\s+by\b", "*", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bdivided\s+by\b", "/", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bover\b", "/", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bto\s+the\s+power\s+of\b", "**", expr, flags=re.IGNORECASE)
    # Remove remaining non-math text
    expr = re.sub(r"[^\d\s\+\-\*\/\(\)\.\%]+", "", expr).strip()
    return expr if expr else None


def _solve_speed_distance(task: str) -> Optional[str]:
    """Try to solve speed/distance/time word problems."""
    m = _SPEED_PATTERN.search(task)
    if not m:
        return None

    speed = float(m.group(1))
    time_val = float(m.group(2))
    time_unit = m.group(3).lower()

    # Convert time to hours
    if time_unit.startswith(("minute", "min", "m")):
        time_hrs = time_val / 60.0
    else:
        time_hrs = time_val

    distance = speed * time_hrs
    # Round to reasonable precision
    result = round(distance, 2)
    if result == int(result):
        return str(int(result))
    return f"{result:.2f}".rstrip("0").rstrip(".")


def solve_arithmetic(task: str, category: str) -> Optional[str]:
    """
    Solve arithmetic tasks deterministically.

    Uses category as a hint but also checks the text directly for
    arithmetic patterns, since misclassification is common.

    Handles:
    - "What is X?" / "calculate X" / "X = ?"
    - Square root of N
    - Percentage calculations ("15% of 240")
    - Speed/distance/time word problems
    - Bare numeric expressions with operators

    Returns the answer string, or None if no expression was found
    or evaluation fails.
    """

    text = task.strip()

    # 0. Try root extraction first (e.g., "square root of 144")
    #    Check text directly — category may be factual/math_reasoning
    m = _ROOT_PATTERN.search(text)
    if m:
        num = float(m.group(1))
        result = math.sqrt(num)
        if result == int(result):
            return str(int(result))
        return f"{result:.10f}".rstrip("0").rstrip(".")

    # 1. Try speed/distance/time first (most structured)
    result = _solve_speed_distance(text)
    if result is not None:
        logger.debug(f"Deterministic arithmetic (speed): {text} -> {result}")
        return result

    # 2. Try percentage-of pattern — check text directly
    m = _PERCENT_PATTERN.search(text)
    if m:
        pct = float(m.group(1))
        num = float(m.group(2))
        result = num * (pct / 100.0)
        result_rounded = round(result, 10)
        if result_rounded == int(result_rounded):
            return str(int(result_rounded))
        return f"{result_rounded:.10f}".rstrip("0").rstrip(".")

    # 3. For the softer patterns, use category as a gate to reduce false positives
    if category != "math_arithmetic":
        return None

    # 4. Try extracting expression from question patterns
    expr = None
    for pattern in _ARITH_PATTERNS:
        m = pattern.search(text)
        if m:
            expr = _normalize_expression(m.group(1))
            if expr:
                break

    # 5. If no pattern matched, try treating the whole text as an expression
    if not expr and _SIMPLE_EXPR.match(text):
        expr = _normalize_expression(text)

    if not expr:
        return None

    # 6. Evaluate
    logger.debug(f"Deterministic arithmetic: {text} -> expr={expr}")
    result = calculator(expr)
    if result and not result.startswith("Error"):
        return result

    return None


# ---------------------------------------------------------------------------
# Simple logic solver
# ---------------------------------------------------------------------------

# Syllogism patterns
_SYLLOGISM_PATTERN = re.compile(
    r"(?:all|no|some|every|not\s+all)\s+\w+\s+(?:are|is|have|has)\s+",
    re.IGNORECASE,
)

_OPTION_PATTERN = re.compile(r"^[A-D][\)\.\:]\s*", re.MULTILINE)


def _is_syllogism(task: str) -> bool:
    """Detect if a task is a syllogism puzzle."""
    sentences = task.replace("?", ".").replace("!", ".").split(".")
    premise_count = 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if _SYLLOGISM_PATTERN.search(sent):
            premise_count += 1
    return premise_count >= 2


def _solve_syllogism(task: str) -> Optional[str]:
    """
    Solve syllogism puzzles using Venn-like set logic.

    Works by evaluating each option against the premises using
    simple set relationships: subset (all X are Y), overlap (some X are Y),
    disjoint (no X are Y).

    Returns the correct option text, or None if can't determine.
    """
    # Parse premises and options
    sentences = [s.strip() for s in task.replace("?", ".").replace("!", ".").split(".")]
    sentences = [s for s in sentences if len(s) > 3]

    premises = []
    options = []
    in_options = False

    for sent in sentences:
        # Check if this looks like an option (starts with A), B), etc. or is preceded by "?"
        if re.match(r"^[A-D][\)\.\:\\)\s]", sent) or re.match(r"^[A-D]\s*\)", sent):
            in_options = True

        if in_options:
            options.append(sent.strip())
        elif _SYLLOGISM_PATTERN.search(sent):
            premises.append(sent.strip())

    if len(premises) < 2 or not options:
        return None

    # Build relationships from premises
    subsets = {}  # smaller -> set of supersets
    disjoint = set()  # pairs that are disjoint (no overlap)
    overlap = set()  # pairs that have some overlap

    for premise in premises:
        p_lower = premise.lower().strip()

        # "All X are Y"
        m = re.match(r"all\s+(\w+)\s+(?:are|is|have|has)\s+(\w+)", p_lower)
        if m:
            x, y = m.group(1), m.group(2)
            subsets.setdefault(x, set()).add(y)
            continue

        # "No X are Y"
        m = re.match(r"no\s+(\w+)\s+(?:are|is|have|has)\s+(\w+)", p_lower)
        if m:
            x, y = m.group(1), m.group(2)
            disjoint.add((x, y))
            disjoint.add((y, x))
            continue

        # "Some X are Y"
        m = re.match(r"some\s+(\w+)\s+(?:are|is|have|has)\s+(\w+)", p_lower)
        if m:
            x, y = m.group(1), m.group(2)
            overlap.add((x, y))
            overlap.add((y, x))
            continue

        # "Not all X are Y" (some X are not Y)
        m = re.match(r"not\s+all\s+(\w+)\s+(?:are|is|have|has)\s+(\w+)", p_lower)
        if m:
            x, y = m.group(1), m.group(2)
            disjoint.add((x, y))
            continue

    def _is_subset_of(a, b):
        """Check if all A are B based on premises and transitive closure."""
        visited = set()
        stack = [a]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            if cur == b:
                return True
            for superset in subsets.get(cur, set()):
                if superset not in visited:
                    stack.append(superset)
        return False

    def _has_overlap(a, b):
        """Check if some A are B."""
        if (a, b) in overlap:
            return True
        for c in subsets.get(a, set()):
            if (c, b) in overlap:
                return True
        return False

    def _is_disjoint(a, b):
        """Check if no A are B."""
        if (a, b) in disjoint:
            return True
        for c in subsets.get(a, set()):
            if (c, b) in disjoint:
                return True
        return False

    # Evaluate each option and pick the one that follows
    for option in options:
        opt_clean = re.sub(r"^[A-D][\)\.\:\s]+", "", option).strip()
        opt_lower = opt_clean.lower()

        # "All X are Y" check
        m = re.match(r"all\s+(\w+)\s+(?:are|is|have|has)\s+(\w+)", opt_lower)
        if m:
            x, y = m.group(1), m.group(2)
            if _is_subset_of(x, y):
                return option.strip()

        # "No X are Y" check
        m = re.match(r"no\s+(\w+)\s+(?:are|is|have|has)\s+(\w+)", opt_lower)
        if m:
            x, y = m.group(1), m.group(2)
            if _is_disjoint(x, y):
                return option.strip()

        # "Some X are Y" check
        m = re.match(r"some\s+(\w+)\s+(?:are|is|have|has)\s+(\w+)", opt_lower)
        if m:
            x, y = m.group(1), m.group(2)
            if _has_overlap(x, y):
                return option.strip()

    # If we get here, pick "cannot be determined" if present
    for option in options:
        opt_clean = re.sub(r"^[A-D][\)\.\:\s]+", "", option).strip()
        if "cannot be determined" in opt_clean.lower() or "cannot determine" in opt_clean.lower():
            return option.strip()

    return None


def _solve_constraint_puzzle(task: str) -> Optional[str]:
    """
    Solve small constraint puzzles via brute-force search.

    Only attempts when the search space is tiny (< 1000 possibilities).

    Looks for patterns like:
    - "A, B, C, D are ... Each has a different ..."
    - "Arrange X, Y, Z such that ..."
    """
    items = re.findall(r"\b([A-Z])\b(?:\s+and\s+|\s*,\s*)?", task)
    items = list(dict.fromkeys(items))

    if len(items) < 3 or len(items) > 5:
        return None

    n = len(items)
    if n > 5:
        return None

    clues_text = task.lower()

    if not any(kw in clues_text for kw in ["order", "arrange", "rank", "sequence",
                                            "assign", "who", "which", "where",
                                            "position", "first", "second", "third",
                                            "last", "next to", "between",
                                            "different", "each"]):
        return None

    best_answer = None
    best_count = 0

    for perm in itertools.permutations(range(1, n + 1)):
        assignment = dict(zip(items, perm))

        valid = True
        constraint_found = False

        lines = clues_text.replace("?", ".").replace("!", ".").split(".")
        for line in lines:
            line = line.strip()
            if not line or "must be" not in line:
                continue

            constraint_found = True
            # "X must be first"
            m = re.match(r"(\w+)\s+must be\s+(\w+)", line)
            if m:
                name, position = m.group(1), m.group(2)
                pos_map = {"first": 1, "second": 2, "third": 3,
                           "fourth": 4, "fifth": 5, "last": n}
                expected = pos_map.get(position)
                if expected and assignment.get(name.upper()) != expected:
                    valid = False
                    break

            # "X must be before Y"
            m = re.match(r"(\w+)\s+must be\s+before\s+(\w+)", line)
            if m:
                a, b = m.group(1).upper(), m.group(2).upper()
                if a in assignment and b in assignment:
                    if assignment[a] >= assignment[b]:
                        valid = False
                        break

            # "X must be after Y"
            m = re.match(r"(\w+)\s+must be \s+after\s+(\w+)", line)
            if m:
                a, b = m.group(1).upper(), m.group(2).upper()
                if a in assignment and b in assignment:
                    if assignment[a] <= assignment[b]:
                        valid = False
                        break

            # "X must be next to Y"
            m = re.match(r"(\w+)\s+must be\s+next\s+to\s+(\w+)", line)
            if m:
                a, b = m.group(1).upper(), m.group(2).upper()
                if a in assignment and b in assignment:
                    if abs(assignment[a] - assignment[b]) != 1:
                        valid = False
                        break

        if valid and constraint_found:
            sorted_items = sorted(assignment.items(), key=lambda x: x[1])
            answer = ", ".join(f"{item}={pos}" for item, pos in sorted_items)
            best_answer = answer
            best_count += 1

    if best_count == 1:
        return best_answer

    return None


def solve_logic(task: str, category: str) -> Optional[str]:
    """
    Solve simple logic tasks deterministically.

    Handles:
    - Syllogisms ("All X are Y. Some Y are Z...")
    - Small constraint puzzles (< 1000 possible arrangements)

    Returns the answer string, or None for complex puzzles.
    """
    if category != "logical_reasoning":
        return None

    text = task.strip()

    # Try syllogism detection and solving
    if _is_syllogism(text):
        result = _solve_syllogism(text)
        if result is not None:
            logger.debug(f"Deterministic logic (syllogism): solved")
            return result
        logger.debug("Deterministic logic (syllogism): could not determine, deferring to model")
        return None

    # Try constraint puzzle detection and solving
    result = _solve_constraint_puzzle(text)
    if result is not None:
        logger.debug(f"Deterministic logic (constraint): solved")
        return result

    # Neither pattern matched — let the model handle it
    return None


# ===========================================================================
# SENTIMENT ANALYSIS SOLVER
# ===========================================================================

# Strong positive keywords with weights (higher = stronger signal)
_POSITIVE_WORDS = {
    # Strong positive
    "love": 3, "loved": 3, "loving": 3, "heartfelt": 3, "mesmerizing": 3,
    "amazing": 2, "wonderful": 2, "fantastic": 2, "excellent": 2,
    "beautiful": 2, "brilliant": 2, "outstanding": 2, "exceptional": 2,
    "phenomenal": 3, "magnificent": 3, "superb": 2, "splendid": 2,
    "marvelous": 2, "delightful": 2, "charming": 2, "elegant": 2,
    "perfect": 2, "glorious": 2, "spectacular": 2, "stunning": 2,
    "breathtaking": 3, "incredible": 2, "extraordinary": 2,
    "masterpiece": 3, "genius": 2, "inspiring": 2, "uplifting": 2,
    "impressive": 1, "enjoyable": 1, "pleasurable": 1,
    "great": 1, "good": 1, "nice": 1, "pleasant": 1,
    "happy": 1, "joyful": 1, "glad": 1, "cheerful": 1,
    "thrilled": 2, "ecstatic": 3, "elated": 2,
    "fabulous": 2, "terrific": 2, "awesome": 2,
    "recommend": 1, "recommended": 1, "recommending": 1,
    "gem": 1, "masterfully": 2, "best": 1, "finest": 1,
    "remarkable": 2, "notable": 1, "admirable": 1,
    "refreshing": 1, "revitalizing": 1, "rejuvenating": 1,
}

_NEGATIVE_WORDS = {
    # Strong negative
    "terrible": 3, "awful": 3, "horrible": 3, "disgusting": 3,
    "worst": 3, "dreadful": 3, "appalling": 3, "atrocious": 3,
    "abysmal": 3, "pathetic": 2, "miserable": 2, "wretched": 2,
    "horrendous": 3, "hideous": 2, "repulsive": 2, "revolting": 2,
    "loathe": 3, "loathed": 3, "despise": 3, "despised": 3,
    "hate": 2, "hated": 2, "hating": 2, "detest": 2,
    "boring": 2, "dull": 2, "tedious": 2, "monotonous": 2,
    "disappointing": 2, "disappointed": 2, "underwhelming": 2,
    "mediocre": 1, "subpar": 1, "inferior": 1, "lousy": 1,
    "frustrating": 1, "annoying": 1, "irritating": 1,
    "sad": 1, "angry": 1, "upset": 1, "depressing": 1,
    "poor": 1, "bad": 1, "worse": 2, "ugly": 1,
    "nasty": 2, "vile": 2, "disgraceful": 2,
    "broken": 1, "faulty": 1, "defective": 1,
    "waste": 1, "wasted": 1, "useless": 1, "worthless": 1,
    "regret": 1, "regrettable": 1, "unfortunate": 1,
    "overpriced": 1, "overrated": 1, "ridiculous": 1,
    "cringe": 1, "cringey": 1, "embarrassing": 1,
    "painful": 1, "agonizing": 2, "torturous": 2,
    "sucks": 2, "stinks": 2, "garbage": 2, "trash": 2,
    "clunky": 1, "slow": 1, "laggy": 1, "buggy": 1,
}

# Negation modifiers that flip sentiment
_NEGATORS = {"not", "no", "never", "neither", "nor", "nothing", "nobody",
             "nowhere", "hardly", "barely", "scarcely", "doesn't", "don't",
             "didn't", "won't", "wouldn't", "can't", "cannot", "isn't",
             "wasn't", "weren't", "aren't", "ain't", "shouldn't"}

# Intensifiers that boost word weight
_INTENSIFIERS = {"very", "really", "extremely", "absolutely", "completely",
                 "utterly", "totally", "highly", "incredibly", "remarkably",
                 "exceptionally", "truly", "quite", "so", "deeply",
                 "thoroughly", "immensely", "enormously", "vastly"}


def _tokenize_sentiment(text: str) -> list[str]:
    """Tokenize text for sentiment analysis, preserving negations."""
    # Split on non-alphanumeric but keep apostrophes
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    return tokens


def _classify_sentiment(text: str) -> Optional[str]:
    """
    Classify sentiment using weighted keyword counting with negation handling.

    Returns "positive", "negative", or None (if signal is too weak or balanced).
    """
    tokens = _tokenize_sentiment(text)
    if len(tokens) < 3:
        return None

    pos_score = 0
    neg_score = 0
    negated = False
    intensified = False

    for i, token in enumerate(tokens):
        # Check for negators in previous position(s)
        if i >= 1 and tokens[i - 1] in _NEGATORS:
            negated = True
        elif i >= 2 and tokens[i - 2] in _NEGATORS and tokens[i - 1] in _INTENSIFIERS:
            negated = True
        else:
            negated = False

        # Check for intensifiers
        if i >= 1 and tokens[i - 1] in _INTENSIFIERS:
            intensified = True
        else:
            intensified = False

        multiplier = 1.5 if intensified else 1.0

        if token in _POSITIVE_WORDS:
            weight = _POSITIVE_WORDS[token] * multiplier
            if negated:
                neg_score += weight
            else:
                pos_score += weight
        elif token in _NEGATIVE_WORDS:
            weight = _NEGATIVE_WORDS[token] * multiplier
            if negated:
                pos_score += weight
            else:
                neg_score += weight

    # Require a clear signal — lower bar for short texts (<10 tokens)
    diff = abs(pos_score - neg_score)
    text_len = len(tokens)
    if text_len < 10:
        if diff >= 2 and pos_score != neg_score:
            return "positive" if pos_score > neg_score else "negative"
        return None
    if diff < 2:
        return None  # Too close to call
    if diff < 4 and max(pos_score, neg_score) < 6:
        return None  # Weak signal overall

    return "positive" if pos_score > neg_score else "negative"


def solve_sentiment(task: str, category: str) -> Optional[str]:
    """
    Solve sentiment analysis tasks deterministically using keyword counting.

    Only handles explicit sentiment questions or reviews. Returns None if
    the text doesn't clearly express sentiment or if the category isn't
    sentiment-related.

    Handles:
    - Review classification (positive/negative)
    - "What is the sentiment of this text?"
    - "Is this review positive or negative?"
    """
    if category not in ("sentiment",):
        return None

    text = task.strip()

    # If there's a clear "Review:" marker or it looks like a review text
    # embedded in a question, try to extract the actual review content
    review_match = re.search(
        r'(?:review|text|passage|sentence)[\s:]*[:\n]+(.+)',
        text, re.IGNORECASE | re.DOTALL
    )
    if review_match:
        target = review_match.group(1).strip()
    elif len(text.split()) > 30:
        # Long text without a question — probably the review itself
        target = text
    else:
        # Short question — look for quoted text or just use the whole thing
        quoted = re.findall(r'"([^"]+)"', text)
        if quoted:
            target = quoted[0]
        else:
            target = text

    result = _classify_sentiment(target)
    if result is not None:
        logger.debug(f"Deterministic sentiment: {result}")
        return result

    return None


# ===========================================================================
# NAMED ENTITY RECOGNITION (NER) SOLVER
# ===========================================================================

# Disease-related suffixes and patterns for biomedical NER
_DISEASE_SUFFIXES = re.compile(
    r'\b(\w+(?:itis|osis|oma|emia|pathy|penia|plasia|uria|algia|'
    r'rrhagia|rrhea|sclerosis|malacia|necrosis|ptosis|spasm|'
    r'ectasis|stenosis|plegia|phasia|phagia|mania|phobia|'
    r'sarcoma|carcinoma|myeloma|lymphoma|leukemia))\b',
    re.IGNORECASE
)

# Known disease names (common biomedical entities)
_KNOWN_DISEASES = {
    "diabetes", "cancer", "asthma", "tuberculosis", "malaria",
    "influenza", "pneumonia", "hepatitis", "cholera", "typhoid",
    "measles", "mumps", "rubella", "polio", "rabies", "tetanus",
    "diphtheria", "pertussis", "meningitis", "encephalitis",
    "alzheimer", "parkinson", "huntington", "multiple sclerosis",
    "als", "lou gehrig", "cystic fibrosis", "sickle cell",
    "hemophilia", "anemia", "leukemia", "lymphoma", "melanoma",
    "carcinoma", "sarcoma", "glioblastoma", "neuroblastoma",
    "osteoporosis", "arthritis", "osteoarthritis", "rheumatoid",
    "lupus", "fibromyalgia", "chronic fatigue", "celiac",
    "crohn", "ulcerative colitis", "ibs", "gerd", "copd",
    "emphysema", "bronchitis", "hypertension", "stroke",
    "atherosclerosis", "arrhythmia", "cardiomyopathy",
    "endocarditis", "psoriasis", "eczema", "dermatitis",
    "glaucoma", "cataract", "macular degeneration",
    "schizophrenia", "bipolar", "depression", "anxiety",
    "ptsd", "ocd", "adhd", "autism", "dyslexia",
    "ebola", "zika", "dengue", "yellow fever", "chikungunya",
    "covid", "sars", "mers", "hiv", "aids", "herpes",
    "gonorrhea", "syphilis", "chlamydia", "hpv",
    "pancreatitis", "appendicitis", "diverticulitis", "colitis",
    "gastritis", "nephritis", "cystitis", "prostatitis",
    "tonsillitis", "sinusitis", "otitis", "conjunctivitis",
    "phlebitis", "vasculitis", "myocarditis", "pericarditis",
    "cellulitis", "folliculitis", "impetigo",
}

# Capitalized entity pattern: sequences of capitalized words
_CAPITALIZED_ENTITY = re.compile(
    r'\b([A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|of|de|van|der|von|the|and|&))+)'
)

# Date patterns
_DATE_PATTERNS = [
    re.compile(r'\b\d{4}\b'),                          # 2024
    re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'),
    re.compile(r'\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b'),
    re.compile(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2},? \d{4}\b'),
    re.compile(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'),
    re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),
]

# Gene/protein patterns (common in biomedical text)
_GENE_PATTERN = re.compile(
    r'\b([A-Z]{2,}[0-9]+[A-Z]?|[A-Z][a-z]{2,}[0-9]+)\b'
)

# Protein patterns
_PROTEIN_PATTERN = re.compile(
    r'\b(p53|BRCA[12]|EGFR|TP53|TNF|IL-\d+|'
    r'CD\d+|HER2|VEGF|PD-?L?1|CTLA-?4|'
    r'KRAS|NRAS|BRAF|ALK|ROS1|MET|RET|NTRK|'
    r'IDH[12]|FLT3|KIT|PDGFR[AB]|FGFR)\b',
    re.IGNORECASE
)


def _extract_diseases(text: str) -> list[str]:
    """Extract disease names from biomedical text."""
    found = set()
    lower_text = text.lower()

    # Check known disease names
    for disease in _KNOWN_DISEASES:
        # Use word boundary check for multi-word diseases
        if disease in lower_text:
            found.add(disease)

    # Check disease suffixes
    for match in _DISEASE_SUFFIXES.finditer(text):
        word = match.group(1).lower()
        if len(word) > 5:  # Skip very short matches
            found.add(word)

    # Also look for "X disease" or "X syndrome" patterns
    for match in re.finditer(r'(\w+(?:\s+\w+){0,3})\s+(?:disease|syndrome|disorder|condition|infection)',
                              text, re.IGNORECASE):
        found.add(match.group(1).strip().lower())

    return sorted(found)


def _extract_capitalized_entities(text: str) -> list[str]:
    """Extract capitalized named entities (people, orgs, locations)."""
    entities = set()

    for match in _CAPITALIZED_ENTITY.finditer(text):
        entity = match.group(1).strip()
        # Filter out sentence-initial words and common false positives
        if len(entity.split()) >= 2 and len(entity) > 3:
            # Skip common non-entity phrases
            lower = entity.lower()
            if lower in ("the first", "the second", "the last", "the same",
                         "each other", "one another", "for example",
                         "in addition", "in fact", "as well", "such as",
                         "the following", "due to", "based on",
                         "while the", "when the", "after the", "before the"):
                continue
            entities.add(entity)

    return sorted(entities)


def _extract_dates(text: str) -> list[str]:
    """Extract date expressions from text."""
    dates = set()
    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            dates.add(match.group(0).strip())
    return sorted(dates)


def _extract_genes_proteins(text: str) -> list[str]:
    """Extract gene and protein names from biomedical text."""
    entities = set()

    # Known protein patterns
    for match in _PROTEIN_PATTERN.finditer(text):
        entities.add(match.group(1).upper())

    # Gene-like patterns: uppercase + numbers
    for match in _GENE_PATTERN.finditer(text):
        gene = match.group(1)
        # Filter out obvious non-genes (all-numeric strings, common words in caps)
        if not gene.isdigit() and len(gene) >= 3:
            if gene.upper() not in ("THE", "AND", "FOR", "WITH", "THIS", "THAT",
                                    "FROM", "HAVE", "HAS", "WERE", "WILL"):
                entities.add(gene.upper())

    return sorted(entities)


def solve_ner(task: str, category: str) -> Optional[str]:
    """
    Solve NER tasks deterministically using regex patterns.

    Handles:
    - Biomedical disease extraction
    - Capitalized entity extraction (people, organizations, locations)
    - Date extraction
    - Gene/protein extraction (biomedical)

    Returns a comma-separated list of entities, or None if no entities found
    or category mismatch.
    """
    if category not in ("named_entity_recognition",):
        return None

    text = task.strip()

    # Try to extract the target text (after "Context:" or "Text:" markers)
    target_match = re.search(
        r'(?:context|text|passage|document|abstract)[\s:]*[:\n]+(.+)',
        text, re.IGNORECASE | re.DOTALL
    )
    target = target_match.group(1).strip() if target_match else text

    # Determine what kind of entities to extract
    lower_all = task.lower()
    entities = []

    # Check for biomedical NER (diseases, genes, proteins)
    is_biomedical = any(kw in lower_all for kw in (
        "disease", "gene", "protein", "biomedical", "clinical",
        "medical", "patient", "diagnosis", "cancer", "tumor",
        "mutation", "genomic", "pathway", "cell", "tissue",
    ))

    if is_biomedical:
        # Extract diseases
        diseases = _extract_diseases(target)
        if diseases:
            entities.extend(diseases)

        # Extract genes/proteins
        genes_proteins = _extract_genes_proteins(target)
        if genes_proteins:
            entities.extend(genes_proteins)

    # Check for general NER (people, orgs, locations, dates)
    is_general_ner = any(kw in lower_all for kw in (
        "person", "people", "organization", "location", "date",
        "named entity", "entity recognition", "extract", "find all",
        "people mentioned", "organizations", "locations",
        "who", "where", "when",
    )) or (not is_biomedical and category == "named_entity_recognition")

    if is_general_ner:
        # Extract capitalized entities
        cap_entities = _extract_capitalized_entities(target)
        if cap_entities:
            entities.extend(cap_entities)

        # Extract dates
        dates = _extract_dates(target)
        if dates:
            entities.extend(dates)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for e in entities:
        if e.lower() not in seen:
            seen.add(e.lower())
            unique.append(e)

    if not unique:
        return None

    # Limit to reasonable output size
    if len(unique) > 20:
        unique = unique[:20]

    result = ", ".join(unique)
    logger.debug(f"Deterministic NER: found {len(unique)} entities")
    return result


# ===========================================================================
# FACTUAL QA SOLVER
# ===========================================================================

# Common factual question patterns with known answers
# Maps lowercase question patterns → answer
_KNOWN_FACTS = {
    # Science & Technology
    "who developed the theory of relativity": "Albert Einstein",
    "who discovered penicillin": "Alexander Fleming",
    "who invented the telephone": "Alexander Graham Bell",
    "who invented the light bulb": "Thomas Edison",
    "who developed the polio vaccine": "Jonas Salk",
    "what is the speed of light": "299,792,458 meters per second",
    "what is the chemical symbol for gold": "Au",
    "what is the chemical symbol for water": "H2O",
    "what is dna": "deoxyribonucleic acid",
    "what does dna stand for": "deoxyribonucleic acid",
    "what is the powerhouse of the cell": "mitochondria",
    "what planet is closest to the sun": "Mercury",
    "what is the largest planet": "Jupiter",
    "what is the smallest planet": "Mercury",
    "how many planets are in the solar system": "8",
    "how many continents are there": "7",
    "what is the largest continent": "Asia",
    "what is the smallest continent": "Australia",
    "what is the largest ocean": "Pacific Ocean",
    "what is the largest country by area": "Russia",
    "what is the most populous country": "India",
    "what is the capital of france": "Paris",
    "what is the capital of germany": "Berlin",
    "what is the capital of japan": "Tokyo",
    "what is the capital of china": "Beijing",
    "what is the capital of the united kingdom": "London",
    "what is the capital of the united states": "Washington, D.C.",
    "what is the capital of canada": "Ottawa",
    "what is the capital of australia": "Canberra",
    "what is the capital of brazil": "Brasília",
    "what is the capital of india": "New Delhi",
    "what is the capital of russia": "Moscow",
    "what is the capital of italy": "Rome",
    "what is the capital of spain": "Madrid",
    "who wrote romeo and juliet": "William Shakespeare",
    "who wrote hamlet": "William Shakespeare",
    "who wrote the great gatsby": "F. Scott Fitzgerald",
    "who wrote to kill a mockingbird": "Harper Lee",
    "who wrote 1984": "George Orwell",
    "who painted the mona lisa": "Leonardo da Vinci",
    "who was the first president of the united states": "George Washington",
    "who was the first man on the moon": "Neil Armstrong",
    "when did world war ii end": "1945",
    "when did world war i end": "1918",
    "when was the declaration of independence signed": "1776",
    "what year did the titanic sink": "1912",
    "what is the longest river in the world": "Nile",
    "what is the tallest mountain in the world": "Mount Everest",
    "what is the largest desert in the world": "Antarctic Desert",
    "how many bones are in the human body": "206",
    "what is the boiling point of water": "100 degrees Celsius",
    "what is the freezing point of water": "0 degrees Celsius",
    "how many elements are in the periodic table": "118",
    "what is the most abundant element in the universe": "Hydrogen",
    "what is h2o": "Water",
    "what is the formula for water": "H2O",
    "who is the current us president": "Joe Biden",
    "who discovered gravity": "Isaac Newton",
    "who developed calculus": "Isaac Newton and Gottfried Wilhelm Leibniz",
    "what is pi": "3.14159",
    "what is the value of pi": "3.14159",
    "how many seconds in a minute": "60",
    "how many minutes in an hour": "60",
    "how many hours in a day": "24",
    "how many days in a year": "365",
    "how many days in a leap year": "366",
    "what is the largest animal": "Blue whale",
    "what is the fastest land animal": "Cheetah",
    "what is the national language of brazil": "Portuguese",
    "what language is spoken in brazil": "Portuguese",
}

# Context-based QA patterns
_QA_CONTEXT_PATTERN = re.compile(
    r'(?:context|passage|text|paragraph|article|document|abstract)[\s:]*[:\n]+(.+)',
    re.IGNORECASE | re.DOTALL
)

_QUESTION_PATTERN = re.compile(
    r'(?:question|q|query)[\s:]*[:\n]*(.+?)(?:\n|$|context:|passage:|text:)',
    re.IGNORECASE | re.DOTALL
)


def _extract_context_and_question(task: str) -> tuple[Optional[str], Optional[str]]:
    """
    Extract context and question from SQuAD-style formatted tasks.
    Returns (context, question) tuple.
    """
    # Try explicit Q&A format: "Question: ... Context: ..."
    q_match = re.search(
        r'question[\s:]*[:\n]*(.+?)\s*(?:context|passage|text|paragraph|answer)[\s:]*[:\n]',
        task, re.IGNORECASE | re.DOTALL
    )
    c_match = re.search(
        r'(?:context|passage|text|paragraph)[\s:]*[:\n]+(.+)',
        task, re.IGNORECASE | re.DOTALL
    )

    question = None
    context = None

    if q_match:
        question = q_match.group(1).strip().rstrip("?") + "?"

    if c_match:
        context = c_match.group(1).strip()
        # If context also has a question after it, split
        q_in_context = re.search(
            r'(?:\n|\.)\s*(.+?\?)',
            context
        )
        if q_in_context and not question:
            question = q_in_context.group(1).strip()

    # If no explicit format, try treating the whole text
    if not question and task.endswith("?"):
        # Find the last sentence ending with ?
        sentences = re.split(r'(?<=[.!?])\s+', task)
        for s in reversed(sentences):
            if s.strip().endswith("?"):
                question = s.strip()
                context = " ".join(sentences[:-1]) if sentences[:-1] else None
                break

    return context, question


def _keyword_match_qa(question: str, context: str) -> Optional[str]:
    """
    Answer a question by finding the most relevant sentence in the context.

    Uses keyword overlap scoring. Returns None if no good match.
    """
    if not question or not context:
        return None

    # Extract question keywords (ignore stop words)
    stop_words = {"what", "is", "the", "a", "an", "of", "in", "to", "for",
                  "with", "on", "at", "by", "from", "are", "was", "were",
                  "be", "been", "being", "have", "has", "had", "do", "does",
                  "did", "will", "would", "could", "should", "may", "might",
                  "can", "shall", "i", "you", "he", "she", "it", "we", "they",
                  "me", "him", "her", "us", "them", "my", "your", "his", "its",
                  "our", "their", "this", "that", "these", "those", "who",
                  "whom", "whose", "which", "where", "when", "why", "how",
                  "and", "but", "or", "not", "if", "then", "else", "than",
                  "too", "very", "just", "about", "also", "so", "as", "into",
                  "through", "during", "before", "after", "above", "below",
                  "between", "out", "off", "over", "under", "again", "further",
                  "once", "here", "there", "all", "both", "each", "few",
                  "more", "most", "other", "some", "such", "no", "only",
                  "own", "same", "up", "down", "?",
                  }

    q_keywords = []
    for word in re.findall(r'\b\w+\b', question.lower()):
        if word not in stop_words and len(word) > 2:
            q_keywords.append(word)

    if not q_keywords:
        return None

    # Split context into sentences
    sentences = re.split(r'(?<=[.!?])\s+', context)
    if not sentences:
        return None

    # Score each sentence by keyword overlap
    best_score = 0
    best_sentence = None
    scored = []

    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent) < 10:
            continue
        sent_lower = sent.lower()
        score = sum(1 for kw in q_keywords if kw in sent_lower)
        # Bonus for exact phrase matches
        for i in range(len(q_keywords)):
            for j in range(i + 1, min(i + 4, len(q_keywords))):
                phrase = " ".join(q_keywords[i:j])
                if phrase in sent_lower:
                    score += 2
        scored.append((score, sent))

    if not scored:
        return None

    # Sort by score
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_sentence = scored[0]

    # Require at least 2 keyword matches or a high relative score
    n_keywords = len(q_keywords)
    if best_score < 2 and n_keywords > 2:
        return None

    # If the best sentence is just repeating the question, skip
    if best_score >= n_keywords * 0.8:
        # This might be the question restated, check next best
        if len(scored) > 1 and scored[1][0] >= 2:
            best_sentence = scored[1][1]

    # Clean up the answer: remove leading "The answer is" etc.
    answer = best_sentence
    answer = re.sub(r'^(?:the answer is|answer:|thus|therefore|so|in conclusion)[,:]?\s*',
                    '', answer, flags=re.IGNORECASE)
    answer = answer.strip(" ,.")

    if len(answer) > 300:
        answer = answer[:297] + "..."

    return answer if answer else None


def solve_factual_qa(task: str, category: str) -> Optional[str]:
    """
    Solve simple factual QA tasks deterministically.

    Try these strategies in order:
    1. Exact match against a built-in fact database
    2. Fuzzy question matching against known facts
    3. Context-based keyword matching (SQuAD-style Q&A)

    Returns the answer string, or None if no match found.
    """
    if category not in ("factual_knowledge", "question_answering"):
        return None

    text = task.strip()

    # Strategy 1: Try exact match against known facts
    # Normalize question for matching
    q_norm = re.sub(r'[^\w\s]', '', text.lower()).strip()
    q_norm = re.sub(r'\s+', ' ', q_norm)

    # Direct lookup
    if q_norm in _KNOWN_FACTS:
        result = _KNOWN_FACTS[q_norm]
        logger.debug(f"Deterministic factual (exact): {result[:80]}")
        return result

    # Fuzzy: try matching by removing "what is", "who is" etc. and checking known facts
    q_stripped = re.sub(
        r'^(?:what\s+is|who\s+is|when\s+did|where\s+is|how\s+many|'
        r'how\s+much|how\s+long|which|define|what\s+are|who\s+are|'
        r'what\s+was|what\s+were)\s+',
        '', q_norm, flags=re.IGNORECASE
    )
    if q_stripped in _KNOWN_FACTS:
        result = _KNOWN_FACTS[q_stripped]
        logger.debug(f"Deterministic factual (prefix-stripped): {result[:80]}")
        return result

    # Strategy 2: Context-based keyword matching
    context, question = _extract_context_and_question(text)
    if context and question:
        result = _keyword_match_qa(question, context)
        if result is not None:
            logger.debug(f"Deterministic factual (context QA): {result[:80]}")
            return result

    return None


# ===========================================================================
# CODE DEBUGGING SOLVER
# ===========================================================================

def _fix_common_bugs(code: str, task_hint: str = "") -> Optional[str]:
    """
    Apply common bug-fix patterns to code. Returns fixed code string,
    or None if no bugs were detected/fixable.
    """
    original = code
    fixed = code
    fixes_applied = 0

    # Pattern 1: Off-by-one in len() - 1 when iterating
    # "for i in range(len(arr) - 1):" → "for i in range(len(arr)):"
    # Only fix if the code uses arr[i] and doesn't use i+1
    off_by_one_patterns = [
        (r'range\s*\(\s*len\s*\(\s*(\w+)\s*\)\s*-\s*1\s*\)',
         lambda m: f'range(len({m.group(1)}))'),
        (r'range\s*\(\s*len\s*\(\s*(\w+)\s*\)\s*-\s*1\s*,\s*-1\s*,\s*-1\s*\)',
         lambda m: f'range(len({m.group(1)}) - 1, -1, -1)'),
    ]

    for pattern, replacement in off_by_one_patterns:
        new_code = re.sub(pattern, replacement, fixed)
        if new_code != fixed:
            # Verify the fix makes sense: check that i+1 is not used in the loop
            # Simple heuristic: if "+1" appears after the range, skip this fix
            found_range = re.search(pattern, fixed)
            if found_range:
                post_range = fixed[fixed.index(found_range.group(0)):]
                if "i + 1" not in post_range and "i+1" not in post_range:
                    fixed = new_code
                    fixes_applied += 1
                    logger.debug("Code bugfix: off-by-one in len()-1 range")

    # Pattern 2: Product initialized to 0 instead of 1
    # "prod = 0" or "product = 0" → change to 1, but only if multiplication follows
    var_match = re.search(r'(\bprod(?:uct)?\w*)\s*=\s*0', fixed)
    if var_match:
        var_name = var_match.group(1)
        if re.search(rf'{re.escape(var_name)}\s*\*=', fixed) or \
           re.search(rf'{re.escape(var_name)}\s*=\s*{re.escape(var_name)}\s*\*', fixed):
            fixed = re.sub(
                rf'(\b{re.escape(var_name)}\s*=\s*)0',
                rf'\g<1>1',
                fixed
            )
            fixes_applied += 1
            logger.debug(f"Code bugfix: product variable '{var_name}' initialized to 0")

    # Pattern 3: Assignment instead of equality in condition
    # "if x = y:" → "if x == y:"
    # "while x = y:" → "while x == y:"
    for keyword in ("if", "while", "elif"):
        cond_pattern = re.compile(
            rf'\b{keyword}\s+(\w+)\s*=\s*([^=])',
        )
        fixed, n = cond_pattern.subn(
            rf'{keyword} \g<1> == \g<2>',
            fixed
        )
        if n > 0:
            fixes_applied += n
            logger.debug(f"Code bugfix: assignment-to-comparison in {keyword}")

    # Pattern 4: Missing abs() for distance/difference calculations
    # "diff = a - b" → "diff = abs(a - b)" when "distance" or "difference" in context
    if re.search(r'\b(distance|difference|absolute|abs_diff)\b',
                 fixed + " " + task_hint, re.IGNORECASE):
        # Find diff/distance/difference assignments using subtraction
        missing_abs = re.finditer(
            r'(diff\w*|distance|difference|dist|d)\s*=\s*(\w+)\s*-\s*(\w+)',
            fixed, re.IGNORECASE
        )
        for m in missing_abs:
            var = m.group(1)
            a = m.group(2)
            b = m.group(3)
            # Only fix if there's no abs() already
            if f"abs({a} - {b})" not in fixed and f"abs({a}-{b})" not in fixed:
                old = f"{var} = {a} - {b}"
                new = f"{var} = abs({a} - {b})"
                fixed = fixed.replace(old, new)
                fixes_applied += 1
                logger.debug(f"Code bugfix: missing abs() in distance calculation")

    # Pattern 5: Incorrect division in percentage calculations
    # "percentage = part / total * 100" → "percentage = (part / total) * 100" or
    # "percentage = part / total" → "percentage = (part / total) * 100"
    if re.search(r'\b(percentage|pct|percent)\b', fixed + " " + task_hint,
                 re.IGNORECASE):
        # Fix "percentage = part/total" missing *100
        pct_missing_mul = re.sub(
            r'(percentage|pct|percent)\s*=\s*(\w+)\s*/\s*(\w+)\s*$',
            r'\g<1> = (\g<2> / \g<3>) * 100',
            fixed,
            flags=re.IGNORECASE | re.MULTILINE
        )
        if pct_missing_mul != fixed:
            fixed = pct_missing_mul
            fixes_applied += 1
            logger.debug("Code bugfix: percentage missing *100")

    # Pattern 6: Incorrect sum pattern: sum = arr[0] + arr[1] for only 2 elements
    # This is not a bug, skip. But "total = arr[0]" then loop from 1 → fine.

    # Pattern 7: Missing colon after if/for/while/def
    colon_fixes = re.sub(
        r'^(if|for|while|def|elif|else|class)\s+([^:\n]+?)$',
        r'\g<1> \g<2>:',
        fixed,
        flags=re.MULTILINE
    )
    if colon_fixes != fixed:
        fixed = colon_fixes
        fixes_applied += 1
        logger.debug("Code bugfix: missing colon")

    # Pattern 8: Python 2 print statement → Python 3
    # Only fix if clearly a print statement (not a function call with parens)
    print_fixes = re.sub(
        r'^print\s+([^(].+?)$',
        r'print(\g<1>)',
        fixed,
        flags=re.MULTILINE
    )
    if print_fixes != fixed:
        fixed = print_fixes
        fixes_applied += 1
        logger.debug("Code bugfix: Python 2 print statement")

    # Pattern 9: List index out of bounds (off-by-one access)
    # "return arr[len(arr)]" → "return arr[len(arr) - 1]"
    idx_fix = re.sub(
        r'(\w+)\[len\((\w+)\)\]',
        r'\g<1>[len(\g<2>) - 1]',
        fixed
    )
    if idx_fix != fixed:
        # Verify context: if it's already len(arr)-1, don't fix
        fixed = idx_fix
        fixes_applied += 1
        logger.debug("Code bugfix: index out of bounds")

    # Pattern 10: Uninitialized variable in accumulation
    # "for x in arr: total += x" without "total = 0"
    # Detect this by looking for += without preceding initialization
    plus_equal_vars = re.findall(r'(\w+)\s*\+=', fixed)
    for var in set(plus_equal_vars):
        # Check if variable is initialized before use
        if not re.search(rf'\b{re.escape(var)}\s*=', fixed.split(var)[0] if var in fixed.split(var, 1) else ""):
            # Don't auto-fix — too risky. Just note it.
            pass

    if fixes_applied == 0:
        return None

    return fixed


def solve_code_debugging(task: str, category: str) -> Optional[str]:
    """
    Solve code debugging tasks deterministically using pattern-based bug fixes.

    Detects common bug patterns:
    - Off-by-one errors (len()-1 in range)
    - Product initialized to 0
    - Assignment (=) instead of comparison (==)
    - Missing abs() for distance/difference
    - Percentage missing *100
    - Missing colons, Python 2 print
    - Index out of bounds

    Returns the fixed code string, or None if no fixable bugs found.
    """
    if category not in ("code_debugging",):
        return None

    text = task.strip()

    # Extract code blocks from the task
    code = None

    # Try fenced code blocks first
    code_match = re.search(r'```(?:python|py)?\s*\n?(.+?)```', text, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()

    # Try indented code after a description
    if not code:
        # Look for lines that look like Python code (start with def, import, for, if, etc.)
        code_lines = []
        in_code = False
        for line in text.split('\n'):
            stripped = line.rstrip()
            if re.match(r'^(def |class |import |from |for |if |while |print |return |#|    |\t)', stripped):
                in_code = True
                code_lines.append(stripped)
            elif in_code and (stripped.startswith(' ') or stripped.startswith('\t')):
                code_lines.append(stripped)
            elif in_code and not stripped.strip():
                code_lines.append('')  # preserve blank lines
            elif in_code:
                break  # end of code block

        if code_lines:
            code = '\n'.join(code_lines)

    if not code or len(code) < 10:
        return None

    # Try to fix common bugs
    fixed = _fix_common_bugs(code, task)
    if fixed is None:
        return None

    logger.debug(f"Deterministic code debugging: applied fixes")
    return fixed
