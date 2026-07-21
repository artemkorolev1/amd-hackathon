"""
Stage 2 — 8-way deterministic category classifier.

Uses the winning-repo scorer pattern: each of 8 categories gets a
`_score_<cat>(prompt) -> float` function, highest score wins, ties
resolved by category priority.

Output: (category_8way: str, confidence: float, scores: dict)

All pure regex/heuristic — zero model calls, zero imports beyond stdlib.
"""

import re
from typing import Callable, Dict, List, Optional, Tuple


# ── Inline CATEGORY_REGISTRY (self-contained) ──
CATEGORIES_8WAY = [
    "code_debug", "code_gen", "factual", "logic",
    "math", "ner", "sentiment", "summarization",
]
PRIORITY = {
    "code_debug": 8, "code_gen": 7, "math": 6, "logic": 5,
    "sentiment": 4, "ner": 3, "summarization": 2, "factual": 1,
}
SHORT_TO_CLASSIFIER = dict(zip(
    CATEGORIES_8WAY,
    ["code_debugging", "code_generation", "factual_knowledge",
     "logical_reasoning", "math_reasoning", "named_entity_recognition",
     "sentiment_classification", "text_summarisation"]
))
CLASSIFIER_TO_SHORT = {v: k for k, v in SHORT_TO_CLASSIFIER.items()}
SHORT_TO_FOUR_WAY = {
    "code_debug": "code", "code_gen": "code",
    "math": "reasoning", "logic": "reasoning",
    "factual": "knowledge",
    "sentiment": "text", "ner": "text", "summarization": "text",
}
ALT_CLASSIFIER_TO_SHORT = {
    "code_debugging": "code_debug", "code_generation": "code_gen",
    "factual": "factual", "factual_knowledge": "factual",
    "logical_reasoning": "logic", "math_reasoning": "math",
    "named_entity_recognition": "ner", "sentiment_classification": "sentiment",
    "text_summarisation": "summarization",
}
HUMAN_NAMES = {
    "code_debug": "Code Debugging", "code_gen": "Code Generation",
    "factual": "Factual Knowledge", "logic": "Logical Reasoning",
    "math": "Math Reasoning", "ner": "Named Entity Recognition",
    "sentiment": "Sentiment", "summarization": "Summarisation",
}
CATEGORY_4WAY = {
    "code_debug": "code", "code_gen": "code",
    "logic": "reasoning", "math": "reasoning",
    "factual": "knowledge",
    "sentiment": "text", "ner": "text", "summarization": "text",
}

def get_short_name(name):
    n = name.strip().lower().replace(" ", "_").replace("-", "_")
    if n in CATEGORIES_8WAY:
        return n
    return CLASSIFIER_TO_SHORT.get(n, ALT_CLASSIFIER_TO_SHORT.get(n, "factual"))

def get_classifier_name(name):
    return SHORT_TO_CLASSIFIER.get(get_short_name(name), "factual_knowledge")

def get_four_way_name(name):
    return SHORT_TO_FOUR_WAY.get(get_short_name(name), "knowledge")

def get_human_name(name):
    return HUMAN_NAMES.get(get_short_name(name), name)

def get_4way(name):
    return CATEGORY_4WAY.get(get_short_name(name), "knowledge")

ALL_CATEGORIES_SHORT = CATEGORIES_8WAY
# ── END REGISTRY ──

# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_CODE_FENCE_RE = re.compile(r"```|\bdef\s|\breturn\b|\bfunction\b|\bimport\s")

# ---------------------------------------------------------------------------
# Per-category scoring functions
# ---------------------------------------------------------------------------

def _score_math(prompt: str) -> float:
    """Math / arithmetic tasks — numbers, operators, calculation keywords."""
    lower = prompt.lower()
    nums = _NUM_RE.findall(prompt)
    s = 0.0

    # Strong explicit math instruction signals
    explicit = bool(re.search(
        r"\b(calculate|compute|solve|equation|formula|algebra|geometry|"
        r"trigonometry|calculus|derivative|integral|probability|statistics|"
        r"what\s+is\s+\d|how many|how much|percent|percentage|quotient|"
        r"remainder|square root|factorial|modulo|triangle|angle|vertex|"
        r"perimeter|circumference|diameter|radius|velocity|acceleration|"
        r"derivative|integral|matrix|determinant|logarithm|log\s+base)\b", lower
    ))
    # Arithmetic operators — but NOT range expressions (10-20), patent codes (G06N10/00),
    # dates (5/12/2024), or times (6:30). Only match REAL arithmetic.
    raw_ops = re.findall(r"\d+\s*([+/*-])\s*\d+", prompt)
    num_op = bool(raw_ops) and any(
        op in {"+", "*", "/"} or
        (op == "-" and prompt.count("-") >= 1 and 
         not re.search(r"[A-Za-z]+\d+\s*-\s*\d+", prompt) and  # patent/code ranges
         not re.search(r"\d+-year|-year-old|aged?\s+\d+-", prompt.lower()))  # age ranges
        for op in raw_ops
    )
    # LaTeX math notation ($...$ containing equations)
    latex_math = bool(re.search(r"\$\s*[A-Za-z0-9_\\{}]+\s*\$", prompt)) and bool(
        re.search(r"\\[a-zA-Z]+|[+*/=^_{}]|\d", prompt)
    )
    word_prob = bool(re.search(
        r"\b(solve for|find\s+(the\s+)?(value|number|sum|difference|product|"
        r"area|perimeter|volume|speed|distance|time|interest|discount|ratio|"
        r"proportion|half of|quarter of|dozen|double|triple)|"
        r"maximum value|minimum|minimum value|evaluate|compute the|"
        r"what is the (value|area|sum|product|probability|distance|speed|time|"
        r"volume|perimeter|length|width|height|total|average|mean|result)|"
        r"what\s+(distance|speed|time|volume|area|perimeter|probability|fraction)|"
        r"how\s+(fast|far|long|many|much)|^(?:problem|question)\s*\d*[:.]?\s)\b",
        lower
    ))
    # Competition math: factorial, Asymptotic, geometry diagrams, $...$ LaTeX
    competition = bool(re.search(
        r"\b\d+!|\\\\circ|\\\\triangle|\\\\sqrt|\\\\frac|\\\\log|\\\\sin|\\\\cos|\\\\tan|"
        r"\\\\angle|^\$|\$.*?\$|\\\\begin\{|\\\\[a-zA-Z]+\{|\[asy\]|"
        r"\b(integer|integers|positive integer|prime\s+number|prime\s+factor|prime\s+divisor|"
        r"prime\s+pair|co-prime|relatively prime|factor|divisible|digit)\b",
        prompt
    ))

    if num_op:
        s += 2.0
    if explicit:
        s += 2.0
    if word_prob:
        s += 2.0
    if latex_math or competition:
        s += 2.0

    # Negative guard: suppress math score for SQuAD-style QA or long factual passages
    is_squad = ("context:" in lower or "passage:" in lower)
    has_qa = "question:" in lower
    if is_squad or (has_qa and not (explicit or num_op)):
        return s  # SQuAD/reading comprehension — numbers are incidental

    # Negative guard: suppress math for factual-lookup prompts with numbers but no calculation intent
    # Catches: "How many continents are there?", "What is the population of France?",
    # "Who discovered penicillin?", "What is the capital of Australia?"
    is_factual_lookup = bool(re.search(
        r"\b(what is|who (is|was|are|were)|when (was|did|is)|where is|"
        r"define |explain |describe |meaning of|capital of|population of|"
        r"history of|invented by|discovered by|how many)\b", lower
    ))
    has_calc_verbs = bool(re.search(
        r"\b(calculate|compute|solve|equation|formula|derivative|integral|"
        r"algebra|geometry|trig|calculus|factorial|permutation|combination|"
        r"probability|matrix|vector|quotient|remainder|modulo|divided by|"
        r"multiplied by|power of|exponent|square root|logarithm|log|"
        r"ways to|arrange|choose|select|"
        r"sum of|difference of|product of|"
        r"distance|speed|velocity|time|area|perimeter|volume|"
        r"average|mean|median|total|length|width|height|radius|diameter|"
        r"circumference|cost|price|rate|interest|discount|profit|loss)\b", lower
    ))
    if is_factual_lookup and not has_calc_verbs and not num_op and not nums:
        return 0.0  # Factual lookup with no numbers/digits — not math

    # Negative guard: suppress math for long non-math text (news, summaries, NER)
    words = lower.split()
    if len(words) > 80 and not (explicit or word_prob):
        return s  # Long text without any math instruction — numbers are incidental

    # Negative guard: suppress math for explicit NER tasks (extract dates/numbers)
    has_ner_pattern = bool(re.search(
        r"\b(extract|identify|find|list|tag|pull|pick)\b[\s\S]{0,60}\b(entities?|names|people|persons|organizations|companies|cities|countries|dates?|diseases|genes?|inventors|authors|institutions|locations|places|teams|players|positions|currencies|patents)\b", lower, re.DOTALL
    ))
    if has_ner_pattern and not explicit:
        return s

    # Numeric density — only scores if there's basic math context
    if len(nums) >= 2 and (explicit or word_prob or num_op):
        s += 2.0
    elif len(nums) >= 2 and re.search(
        r"%|percent|total|sum|difference|product|average|mean|[-+*/]|\bper\b|cost|price", lower
    ):
        s += 0.5  # weak signal — numbers with incidental math words

    # ── GUARD: suppress math when strong logic patterns are detected ──
    # Prevents math from stealing logic puzzles that happen to contain numbers.
    # Only fires on unambiguous logic-specific patterns unlikely in math problems.
    if s > 0:
        # Check for named-entity constraint puzzles (3+ names + each/different)
        name_list = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?", prompt)
        _skip_names = {
            "the", "this", "that", "these", "those", "what", "which", "when",
            "where", "how", "why", "who", "whom", "whose", "if", "then", "else",
            "each", "all", "some", "any", "none", "one", "two", "three", "four",
            "five", "six", "seven", "eight", "nine", "ten", "first", "second",
            "third", "last", "next", "many", "much", "few", "several", "both",
            "every", "either", "neither", "such", "only", "same", "other",
            "another", "no", "not", "but", "and", "or", "for", "nor", "yet",
            "true", "false", "yes",
        }
        name_count = sum(
            1 for n in name_list
            if len(n) > 2 and n.lower() not in _skip_names
        )
        is_logic_puzzle = (
            # Strong named-entity logic puzzles (3+ distinct names + constraints)
            (name_count >= 3 and re.search(
                r"\b(each\s+.*\b(different|distinct|has|is|works|owns|lives|sits|drives|likes)|"
                r"different\s+\w+)\b", lower
            ))
            # Syllogism conclusion questions (unambiguous)
            or re.search(r"\b(can we conclude|what can we conclude)\b", lower)
            # Truth-teller puzzles (unambiguous)
            or re.search(r"\b(knight|knave|liar|tells the truth|always tells|always lies)\b", lower)
            # Clue-based puzzles (unambiguous)
            or (re.search(r"\b(clue[s]?|hint[s]?)\b", lower) and
                re.search(r"\b(digit|number|different|distinct|each)\b", lower))
        )
        if is_logic_puzzle:
            s *= 0.2  # Heavily suppress — this is a logic puzzle, not math

    return s


def _score_logic(prompt: str) -> float:
    """Logical reasoning — syllogisms, puzzles, constraints."""
    lower = prompt.lower()
    s = 0.0

    # ── Strong explicit logic signals ──
    if re.search(r"\b(knight|knave|lying|liar|syllogism|deduce|infer)\b", lower):
        s += 3.0
    if re.search(r"\beach\b.{0,30}\b(who|which)\b", lower):
        s += 2.5
    if re.search(r"\bdifferent\b.{0,30}\b(who|which)\b", lower):
        s += 2.0
    if re.search(r"\bif\b.{0,60}\bthen\b", lower):
        s += 2.0
    if re.search(r"\b(either|neither|exactly one|must be (true|false))\b", lower):
        s += 2.0
    if re.search(r"\b(logical|conclusion|implies|therefore|hence|iff)\b", lower):
        s += 1.5
    if re.search(r"\bwhich of the following\b", lower):
        s += 1.5
    if re.search(r"\b(order|rank|arrangement|seat|adjacent)\b", lower):
        s += 1.5
    # Truth-teller patterns
    if re.search(
        r"\b(tells the truth|always tells|always lies|truth.?teller|is lying|is telling)\b", lower
    ):
        s += 3.0

    # ── Mathematical proof / justification patterns ──
    if re.search(r"\b(prove|proof|justify|justification)\b", lower):
        s += 3.0
    if re.search(r"\b(convergence|converge|divergence|diverge)\b", lower):
        s += 2.5

    # ── Algorithm / complexity analysis patterns ──
    if re.search(
        r"\b(time\s+complexity|space\s+complexity|big.\s*O|O\s*\(|\bO\b\s*\(n|"
        r"algorithmic|asymptotic|worst.case|best.case|average.case|"
        r"runtime\s+(analysis|complexity))\b", lower
    ):
        s += 3.0
    if re.search(r"\b(analyze|analysis)\b.{0,40}\b(algorithm|complexity|function)\b", lower):
        s += 2.0

    # ── Physics / scientific reasoning patterns ──
    if re.search(
        r"\b(refraction|refract|prism|spectral|spectrum|wavelength|"
        r"gradient\s+descent|gradient\s+boost|backpropagation)\b", lower
    ):
        s += 2.5

    # ── Constraint-puzzle patterns ──
    if re.search(r"(?:sit|sits|seated|are|is|placed|arranged)\s+in a row", lower):
        s += 2.5

    # ── SYLLOGISM PATTERNS ──
    # Classic Aristotelian syllogisms: all/no/some X are Y
    if re.search(r"\ball\s+\w+\s+are\s+\w+", lower):
        s += 2.0
    if re.search(r"\bno\s+\w+\s+(is|are)\s+\w+", lower):
        s += 2.0
    if re.search(r"\bsome\s+\w+\s+(are\s+not|are\s+\w+|is\s+\w+)", lower):
        s += 2.0
    if re.search(r"\b(not all|all\s+\w+\s+are\s+not)\b", lower):
        s += 2.0
    # "can we conclude" / "what can we conclude" / "what can be determined"
    if re.search(r"\b(can we conclude|what can we|what can be determined|it follows that)\b", lower):
        s += 2.5

    # ── People-group logic puzzles ──
    if re.search(
        r"\b(friends?|colleagues?|neighbors?|people|team|members?|classmates?|"
        r"candidates?|students?|persons?)\s.{0,30}(each|all|one|two|three|four|"
        r"five|six|seven|eight|nine|ten)\b", lower
    ):
        s += 2.0

    # ── NAMED-ENTITY PUZZLE DETECTION ──
    # 3+ capitalized proper names (filtering out common English capitals)
    name_list = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?", prompt)
    _skip_names = {
        "the", "this", "that", "these", "those", "what", "which", "when",
        "where", "how", "why", "who", "whom", "whose", "if", "then", "else",
        "each", "all", "some", "any", "none", "one", "two", "three", "four",
        "five", "six", "seven", "eight", "nine", "ten", "first", "second",
        "third", "last", "next", "many", "much", "few", "several", "both",
        "every", "either", "neither", "such", "only", "same", "other",
        "another", "no", "not", "but", "and", "or", "for", "nor", "yet",
        "with", "without", "within", "into", "onto", "upon", "from", "than",
        "also", "very", "just", "more", "most", "less", "least", "true",
        "false", "yes", "no",
    }
    name_count = sum(
        1 for n in name_list
        if len(n) > 2 and n.lower() not in _skip_names
    )
    has_constraint_words = bool(re.search(
        r"\b(each\s+(has|is|works|owns|lives|sits|drives|likes|gets|receives|"
        r"takes|plays|studies|teaches|bought|sold|wears|eats|drinks|reads|"
        r"writes|knows|thinks|says|tells|gives|makes|does|goes|comes|stays|"
        r"stands|waits|brings)|"
        r"different\s+(department|role|job|position|color|item|type|one|kind|"
        r"sort|brand|flavor|size|shape|number|digit|letter|symbol|animal|"
        r"plant|food|drink|country|city|state|team|sport|hobby|subject|class)|"
        r"who\s+(is|has|works|likes|owns|sits|lives|drives|gets|does|takes|plays))",
        lower
    ))
    if name_count >= 3 and has_constraint_words:
        s += 4.0  # Strong boost — named-entity constraint puzzle
    elif name_count >= 2 and has_constraint_words:
        s += 2.0  # Moderate boost — minimal names but constraint words

    # ── CLUE-BASED LOGIC PUZZLES ──
    # Digit/number puzzles with clues
    if re.search(r"\b(clue[s]?|hint[s]?)\b", lower) and \
       re.search(r"\b(different|distinct|each|either|neither|all different)\b", lower):
        s += 3.0
    # "I am thinking of" + number/digit → classic logic puzzle
    if re.search(r"\bi\.?\s*am\s+thinking\s+of\b", lower) and \
       re.search(r"\b(digit|number|word|code|lock|combination|password)\b", lower):
        s += 3.0

    # ── Explanation / proof / comparison patterns ──
    if re.search(
        r"\b(Explain|explain)\s+(why|how|the|step|what|this|that)\b", prompt
    ):
        s += 2.0
    if re.search(
        r"\b(Prove|prove)\s+(that|the|a|this)\b", prompt
    ):
        s += 2.5
    if re.search(
        r"\b(Compare|compare)\s+(?:\w+\s+){0,5}and\s+\w+\b", prompt
    ):
        s += 2.0
    # Standalone imperative "Compare X" (not "Compare X and Y" — those above)
    if re.search(r"^Compare\s+\w+", prompt.strip()) and not re.search(
        r"\bCompare\s+\w+\s+and\s+\w+\b", prompt
    ):
        s += 1.5
    if re.search(
        r"\b(Why|why)\s+(do|does|is|are|would|did|can|could)\b", prompt
    ) and not re.search(
        r"\b(?:what|who|when|where)\s+(?:is|are|was|were)\b", prompt.lower()
    ):
        s += 1.5

    # ── CONDITIONAL REASONING BOOST ──
    if re.search(
        r"\b(must be|cannot be|can not be|could be|might be)\b", lower
    ) and re.search(
        r"\b(if|unless|provided|assuming|given that|suppose)\b", lower
    ):
        s += 2.0

    # ── CONSTRAINT SATISFACTION BOOST ──
    # Detect multiple constraint signals for a compounding boost
    cs_count = 0
    if name_count >= 2 and re.search(r"\beach\b.{0,50}\b(different|distinct|unique|own|separate)\b", lower):
        cs_count += 1
    if re.search(r"\b(all\s+different|all\s+distinct|no\s+two|different\s+from\s+each)\b", lower):
        cs_count += 1
    if name_count >= 2 and re.search(
        r"\beach\b.{0,40}\b(has|is|works|likes|owns|sits|lives|drives|gets)\b", lower
    ):
        cs_count += 1
    if re.search(r"\b(each|every)\b.{0,30}\b(different|distinct)\b", lower):
        cs_count += 1
    if re.search(r"\bexactly one|at most|at least|no more than|no less than\b", lower):
        cs_count += 1
    if cs_count >= 2:
        s += 3.0
    elif cs_count >= 1:
        s += 1.5

    return s


def _score_factual(prompt: str) -> float:
    """Factual knowledge / QA — who/what/when/where/why questions."""
    lower = prompt.lower()
    s = 0.0

    # ── SQuAD-style QA format (strong) ──
    # Detect "Context:" + "Question:" pairs even when words are in any order
    has_squad_ctx = bool(re.search(r"(?:^|\n)\s*context\s*:", lower))
    has_squad_q = bool(re.search(r"(?:^|\n)\s*question\s*:", lower))
    has_passage = bool(re.search(r"(?:^|\n)\s*(?:passage|text)\s*:", lower))

    if has_squad_q and (has_squad_ctx or has_passage):
        s += 3.0
    elif "question:" in lower and ("context:" in lower or "passage:" in lower):
        s += 2.5  # Looser — words present anywhere in prompt

    # ── Question-starter patterns ──
    q_start = bool(re.search(
        r"\b(what is|who (is|was|are|were)|when (did|was|is)|where is|"
        r"why (did|does|is|are)|how (does|do|did|many|much|long|far|old)|"
        r"which of the following is|"
        r"which\s+\w+\s+(is|are|was|were|does|do|did|has|have|had|would|could|should|can|will))\b", lower
    ))
    if q_start:
        s += 1.5

    # ── Knowledge lookup patterns (stronger weighting) ──
    if re.search(r"\b(define|explain|describe|tell me about|facts? about|"
                 r"history of|capital of|population of|meaning of)\b", lower):
        s += 1.5
    # Explicit knowledge/invention/discovery patterns
    if re.search(
        r"\b(invented by|created by|discovered by|founded by|"
        r"composed by|written by|known for|famous for|"
        r"located in|situated in|referred to as|"
        r"what does\s+\w+\s+mean|what is\s+\w+\s+known for)\b", lower
    ):
        s += 2.0

    # ── "How many" factual-vs-calculation detection ──
    how_many_match = re.search(r"how many\b", lower)
    if how_many_match:
        # Look at the rest of the sentence / phrase after "how many"
        after = lower[how_many_match.end():].split(".")[0].split("?")[0]

        # Factual/lookup nouns — "How many continents are there?"
        is_factual_hm = bool(re.search(
            r"\b(continents?|countries?|cities?|states?|presidents?|"
            r"planets?|oceans?|seas?|rivers?|mountains?|languages?|"
            r"people|species|elements?|bones?|muscles?|organs?|"
            r"chapters?|books?|volumes?|pages?|members?|players?|"
            r"teams?|gods?|kings?|queens?|wars?|battles?|treaties?|"
            r"amendments?|laws?|rights?|days?|months?|weeks?|years?|"
            r"decades?|centuries?|millennia?|inches|feet|miles|"
            r"kilometers|meters|grams|pounds|ounces|liters|gallons|"
            r"degrees?|children|siblings|parents|cousins?|aunts?|uncles?|"
            r"sides?|corners?|edges?|faces?|vertices|angles?|"
            r"colors?|flavors?|types?|kinds?|groups?|classes?|"
            r"ranks?|levels?|letters?|words?|numbers?|digits?)\b", after
        ))

        # Calculation/math nouns — "How many ways can 5 people sit?"
        is_calc_hm = bool(re.search(
            r"\b(ways?|combinations?|permutations?|arrangements?|"
            r"different ways|possible ways?|subsets?|orders?|"
            r"sequences?|strings?|trees?|paths?|choices?|"
            r"solutions?|roots?|factors?|divisors?|multiples?|"
            r"times?|possibilities?)\b", after
        ))

        if is_factual_hm:
            s += 3.0  # Strong factual — lookup question
        elif not is_calc_hm and not re.search(
            r"\b(calculate|solve|equation|formula|probability|factorial|"
            r"permutation|combination|ways to|combinations of|"
            r"arrange|choose|select)\b", lower
        ):
            s += 2.0  # Likely factual — "how many" without math indicators

    # ── Guard: numbers in question-word-started prompts with no calculation verbs ──
    nums = _NUM_RE.findall(prompt)
    if len(nums) >= 1 and q_start and not re.search(
        r"\b(calculate|solve|compute|equation|formula|derivative|integral|"
        r"algebra|geometry|trig|calculus|factorial|permutation|combinations?|"
        r"probability|matrix|vector|sum of|difference of|product of|"
        r"quotient|remainder|modulo|divided by|multiplied by|"
        r"plus|minus|times|power of|exponent|square root|logarithm|log)\b", lower
    ):
        s += 1.0  # Numbers + question-word starter without calculation → factual

    return s


def _score_sentiment(prompt: str) -> float:
    """Sentiment analysis — positive/negative/neutral classification."""
    lower = prompt.lower()
    s = 0.0

    # Negative guard: suppress sentiment if this is actually a financial/biomedical
    # NER extraction task with ticker symbols, monetary values, or clinical entities.
    has_extract_pattern = re.search(
        r"\b(extract|identify|find|list|tag)\b.{0,60}\b(compan(y|ies)|ticker|stock|"
        r"currenc|monetar|regulator|percentag|diseases|genes|medications|patients|"
        r"titles?|positions?|draft|legislation)\b", lower, re.DOTALL
    )
    if has_extract_pattern:
        return s  # This is an extraction/NER task, not sentiment

    # Explicit sentiment task indicators
    if re.search(
        r"\b(sentiment|positive|negative|neutral|opinion|tone|mood|emotion)\b", lower
    ):
        s += 2.5
    if re.search(
        r"\b(classify|determine|identify|rate|analy[sz]e|judge|assess)\b"
        r".{0,40}\b(sentiment|tone|mood|emotion|opinion|feeling)\b", lower
    ):
        s += 2.0
    if re.search(r"\b(positive or negative|happy or sad|satisfied or dissatisfied)\b", lower):
        s += 2.0
    if re.search(r"\bhow (do|does|would) .{0,20} (feel|think)\b", lower):
        s += 1.5

    # Review-specific patterns
    if re.search(r"\b(review|feedback|rating|stars?)\b", lower) and \
       re.search(r"\b(positive|negative|good|bad|worst|best|terrible|excellent)\b", lower):
        s += 1.5

    return s


def _score_ner(prompt: str) -> float:
    """Named entity recognition — extract people, orgs, locations, etc."""
    lower = prompt.lower()
    s = 0.0

    # Explicit NER instruction
    if re.search(
        r"\b(?:named\s+entit\w*|NER|entity\s+recognition|entity\s+extraction)\b", lower
    ):
        s += 3.0
    if re.search(
        r"\b(extract|identify|find|list|tag|label|pull\s+out|pick\s+out)\b"
        r".{0,80}\b(persons?|people|organi[sz]ations?|locations?|"
        r"entit(y|ies)|companies|cities|countries|diseases|genes|"
        r"teams?|players?|positions?|currencies|patents|inventors|authors|institutions)\b", lower
    ):
        s += 2.5

    # Biomedical/clinical NER
    if re.search(
        r"\b(extract|find|list|identify)\b.*\b(diseases|conditions|genes|proteins|"
        r"medications|drugs|symptoms|diagnoses)\b", lower
    ):
        s += 2.0

    # Disease suffix patterns in extraction context
    disease_suffixes = ["itis", "osis", "oma", "emia", "pathy"]
    has_disease = any(re.search(rf"\\b\\w+{suf}\\b", lower) for suf in disease_suffixes)
    if has_disease and re.search(r"\b(extract|find|list|identify)\b", lower):
        s += 2.0

    return s


def _score_summarization(prompt: str) -> float:
    """Summarization — condense text, extract key points.

    Uses four signal families:
      1. Explicit summarization keywords (e.g. ``summarize``, ``tl;dr``).
      2. Document-structure markers (source attribution, legal/doc headers,
         multi-paragraph layout, bullet lists).
      3. Narrative / prose structure (news datelines, connecting words,
         academic citation patterns).
      4. Length-based implicit signal (generous boost for long prose that is
         not code or math).
    """
    lower = prompt.lower()
    s = 0.0

    # ── 1. Explicit summarization keywords ──────────────────────────────────
    if re.search(
        r"\b(summarize|summary|tl;?dr|tldr|condense|shorten|compress|"
        r"recap|gist|boil down|key points|main idea|overview)\b", lower
    ):
        s += 2.5

    # Single-sentence/word constraints (strong summarization signal)
    if re.search(
        r"\bin (one|two|three|\d+) (sentence|word|bullet|line|paragraph)\b", lower
    ):
        s += 2.0

    # ── 2. Document-structure markers ──────────────────────────────────────

    # Source attribution: "SOURCE N (Name, Year):" or "SOURCE:" at line start
    if re.search(
        r"\b(?:SOURCE|source|Source)\s+\d+\s*\(|^Source\s*:|^SOURCES?:",
        prompt,
    ):
        s += 1.5

    # According to / reported by / published in attribution
    if re.search(
        r"\b(According to|reported by|published in|as reported by|"
        r"as stated by|according to a (?:report|study|article|analysis))\b",
        lower,
    ):
        s += 1.5

    # Document / legal header patterns
    if re.search(
        r"(?:^|\n)\s*(?:LEGAL BRIEF|STATEMENT BY THE|PRESS RELEASE|"
        r"EXECUTIVE SUMMARY|WHITE PAPER|POLICY BRIEF|MEMORANDUM)\b", prompt
    ):
        s += 1.5

    # All-caps headline followed by colon or opening quote
    if re.search(
        r"(?:^|\n)\s*[A-Z][A-Z\s'\"]{10,}\s*:?\s*[\"'\u201C]", prompt
    ):
        s += 1.0

    # Multi-source comparison (two or more SOURCE / STUDY markers)
    source_count = len(re.findall(r"\b(?:SOURCE|STUDY)\s+\d+", prompt))
    if source_count >= 2:
        s += 1.0

    # ── Generic prose-structure patterns ──
    # These patterns (paragraph breaks, lists, narrative connectives) are
    # intentionally broader but gated behind a weak summarization-context
    # check: they only contribute when the prompt already has SOME other
    # summarization-relevant signal.  This prevents scoring pure logic
    # puzzles that happen to have paragraph breaks and numbered choices.

    _has_summary_context = bool(re.search(
        r"\b(summarize|summary|tl;?dr|tldr|condense|shorten|compress|"
        r"recap|gist|boil down|key points|main idea|overview|"
        r"tl|dr|according to|published in|reported by|"
        r"SOURCE\s+\d+|LEGAL BRIEF|STATEMENT BY|PRESS RELEASE|"
        r"EXECUTIVE SUMMARY|WHITE PAPER|POLICY BRIEF|MEMORANDUM|"
        r"HEADLINE|DATELINE|BREAKING|BRIEF|MEMORANDUM|"
        r"presents a|provides a|offers a|delivers a)\b", prompt
    )) or bool(re.search(
        r"\b(Read|Consider|Review|Analyze)\s+(?:the following|"
        r"this (?:text|article|passage|source|document|report))\b", lower
    ))

    # Paragraph breaks (double newlines) indicate long-form prose
    para_breaks = prompt.count("\n\n")
    if para_breaks >= 2 and _has_summary_context:
        s += 1.0
    if para_breaks >= 4 and _has_summary_context:
        s += 0.5

    # Bullet / numbered list patterns
    if re.search(r"(?:^|\n)\s*[-*\u2022]\s+", prompt):
        s += 0.5
    if re.search(r"(?:^|\n)\s*\d+[.)]\s+", prompt) and _has_summary_context:
        s += 0.5

    # ── 3. Narrative / prose structure ─────────────────────────────────────

    # News-style dateline: "On December 5, 2022, ..."
    if re.search(
        r"\bOn\s+(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{1,2},?\s+\d{4}\b", prompt
    ):
        s += 1.0

    # Academic / report citation patterns:
    #   "A new study published in ..." / "The report, titled ..."
    #   "STUDY A (Journal, Year):" / "The document ... highlights"
    if re.search(
        r"\b(?:A new study|The report|STUDY [A-Z]\b|The document|"
        r"The article|This study|Our analysis)\b", prompt
    ) and re.search(
        r"\b(published|titled|presents|highlights|analyzes|details|"
        r"examines|investigates|describes|reviews)\b", lower
    ):
        s += 1.0

    # Report / document introductory clauses:
    #   "X presents a [analysis/overview/summary]" — common in report-style
    #   summarization prompts that lack explicit "summarize" keywords.
    if re.search(
        r"\b(presents|provides|offers|delivers)\s+(a|an|the)\s+"
        r"(analysis|overview|summary|report|study|assessment|"
        r"review|guide|introduction|breakdown)\b", lower
    ):
        s += 1.0

    # Multi-sentence narrative with connecting words (≥3 long sentences)
    sentences = [st.strip() for st in re.split(r"[.!?]+", prompt)
                 if len(st.strip()) > 20]
    if len(sentences) >= 3 and _has_summary_context and re.search(
        r"\b(however|therefore|meanwhile|furthermore|moreover|"
        r"nevertheless|consequently|additionally|in addition|"
        r"as a result|for example|for instance|in particular|"
        r"specifically|notably|importantly)\b", lower
    ):
        s += 0.5

    # "Read the following" / "Consider the following" instructions
    if re.search(
        r"\b(Read|Consider|Review|Analyze)\s+(?:the following|"
        r"this (?:text|article|passage|source|document|report))\b", lower
    ):
        s += 1.5

    # ── 4. Length-based implicit signal ────────────────────────────────────

    words = lower.split()
    word_count = len(words)

    code_score = _score_code_gen(prompt)
    math_score = _score_math(prompt)
    if word_count > 80 and not code_score and not math_score:
        s += 0.5
    if word_count > 150:
        s += 0.5

    return s


def _score_code_gen(prompt: str) -> float:
    """Code generation — write/implement/create programs."""
    lower = prompt.lower()
    s = 0.0

    # Strong code signals
    if _CODE_FENCE_RE.search(prompt):
        # Suppress for SQuAD-style context where "return" appears in prose
        if "context:" in lower:
            # Only score if there's actual code structure, not just the word "return"
            if re.search(r"```|\bdef\s+\w+\s*\(|\bclass\s+\w+", prompt):
                s += 3.0
        else:
            s += 3.0

    # Code verb + target combinations
    trigger = re.search(r"\b(write|implement|create|build|generate|develop|complete)\b", lower)
    target = re.search(
        r"\b(function|method|class|program|script|algorithm|snippet|"
        r"implementation|query|decorator|endpoint|code)\b", lower
    )
    if trigger and target:
        s += 3.0

    # Language-specific hints
    if re.search(r"\bin (python|java|javascript|typescript|rust|go|c\+\+|ruby|swift|kotlin)", lower):
        s += 1.5
    if re.search(r"\b(leetcode|hackerrank|coding challenge|write a function to)\b", lower):
        s += 2.0

    # Data structure / algorithm references
    if re.search(
        r"\b(array|list|tree|graph|hash map|dictionary|stack|queue|heap|"
        r"linked list|binary search|DFS|BFS|dynamic programming|recursion)\b", lower
    ):
        s += 1.5

    # Code-specific keywords (shared with code_debug; suppress when debugging)
    has_debug_signal = bool(re.search(
        r"\b(debug|bug|fix|error|broken|incorrect|wrong|crash|fault|traceback)\b", lower
    ))
    # Also suppress SQuAD-style code matches (e.g. "sample return" in factual context)
    is_squad = "context:" in lower
    if not has_debug_signal and not is_squad:
        if re.search(r"\b(?:import |from \w+ import|def \w+\(|class \w+\s*[:\(]|print\(|return\b)", prompt):
            s += 2.0

    # Negative guard: if there's no code signal, tone it down
    return s


def _score_code_debug(prompt: str) -> float:
    """Code debugging — fix bugs, errors, incorrect output."""
    lower = prompt.lower()
    s = 0.0

    # Traceback / error stack detection (strongest signal)
    has_traceback = bool(re.search(
        r"(Traceback|Error:|Exception:|SyntaxError|IndentationError|NameError|ValueError|TypeError|KeyError|IndexError|AttributeError|ZeroDivisionError|RuntimeError|ImportError|ModuleNotFoundError|FileNotFoundError|RecursionError|StopIteration|OverflowError)",
        prompt
    ))

    # Must have both code AND debugging signals to get score
    has_code = bool(_CODE_FENCE_RE.search(prompt)) or bool(
        re.search(r"\b(def \w+\(|class \w+|import |print\(|return\b)", prompt)
    )
    has_debug = bool(re.search(
        r"\b(debug|bug|fix|error|not working|broken|incorrect|wrong|"
        r"issue|crash|fails?|doesn't work|fault|traceback|exception)\b", lower
    ))

    if has_traceback:
        s += 5.0
    elif has_code and has_debug:
        s += 5.0

    # Specific debugging patterns
    if re.search(
        r"\b(fix\s+(this|the|a|my)\s+(code|bug|function|error|problem)|"
        r"(what|why|spot)\s+(is\s+)?wrong|correct\s+(the\s+)?(code|error)|"
        r"error\s+(in|at|during|when))", lower
    ):
        s += 2.0

    # Error-type mentions
    if re.search(
        r"\b(TypeError|ValueError|KeyError|IndexError|SyntaxError|"
        r"NameError|ImportError|AttributeError|ZeroDivisionError|RuntimeError)\b", prompt
    ):
        s += 2.0

    # Off-by-one / logic bugs
    if re.search(r"\b(off.by.one|null pointer|segfault|memory leak|infinite loop|race condition)", lower):
        s += 2.0

    return s


# ---------------------------------------------------------------------------
# Scorer registry
# ---------------------------------------------------------------------------

SCORERS: Dict[str, Callable] = {
    "math":          _score_math,
    "logic":         _score_logic,
    "factual":       _score_factual,
    "sentiment":     _score_sentiment,
    "ner":           _score_ner,
    "summarization": _score_summarization,
    "code_gen":      _score_code_gen,
    "code_debug":    _score_code_debug,
}

# Priority for tiebreaking
PRIORITY = PRIORITY

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(prompt: str) -> Tuple[str, float, Dict[str, float]]:
    """
    Run the 8-way scorer classifier.

    Returns:
        (category_8way, confidence, {category: score})
    """
    if not prompt or not prompt.strip():
        return "factual", 0.0, {}

    text = prompt.strip()
    lower = text.lower()

    # Compute all scores
    scores: Dict[str, float] = {}
    for cat, scorer_fn in SCORERS.items():
        scores[cat] = scorer_fn(text)

    # Find winner: highest score, ties broken by priority
    best_cat = "factual"
    best_score = -1.0
    best_pri = -1

    for cat, sc in scores.items():
        pri = PRIORITY.get(cat, 0)
        if sc > best_score or (sc == best_score and pri > best_pri):
            best_cat = cat
            best_score = sc
            best_pri = pri


    # ── V4 IMPROVEMENTS: targeted post-processing ──
    lower = prompt.lower()

    # Zero-score guard: if nothing scores, return factual immediately
    if all(v <= 0 for v in scores.values()):
        return "factual", 0.10, scores

    # Math guard: cap math on context passages without math intent
    _hmk = bool(re.search(
        r"(solve|calculate|compute|equation|formula|derivative|integral|"
        r"algebra|geometry|trig|calculus|probability|permutation|"
        r"combination|factorial|matrix|vector)", lower))
    _hmc = bool(re.search(
        r"(liters|gallons|kilograms|meters|kilometers|miles|hours|"
        r"minutes|percent|ratio|mixture|distance|speed|velocity|rate|"
        r"time|work|age|interest|discount|profit|loss|area|volume|"
        r"perimeter)", lower))
    _hcm = bool(re.search(r"^(Context|Passage|Text|Article):", prompt, re.I))

    if scores.get("math", 0) > 0 and not _hmk and not _hmc and _hcm:
        scores["math"] = scores.get("math", 0) * 0.1

    # Logic structural patterns (exclude context passages)
    _logic_boost = 0.0
    if not _hcm:
        if re.search(r"(reservations|appointments|meetings)", lower) and \
           re.search(r"\d+:\d{2}", prompt) and \
           re.search(r"(at \d+|different time|arrang)", lower):
            _logic_boost = max(_logic_boost, 4.0)
        if re.search(r"i am (thinking|looking)", lower) and \
           re.search(r"(digit|number|word|clue|hint|crypt|pattern)", lower):
            _logic_boost = max(_logic_boost, 4.0)
        if len(re.findall(r"[A-Z][a-z]+", prompt)) >= 2 and \
           re.search(r"(different|distinct|each \w+ (has|is|works|owns|"
                     r"lives|sits|drives|likes))", lower):
            _logic_boost = max(_logic_boost, 3.0)
        # NEW: Physics/scientific reasoning boost (why does X happen)
        if re.search(r"\b(why|explain|reason)\b", lower) and \
           re.search(r"\b(refract|prism|spectrum|light|wavelength|lens|"
                     r"mirror|gravity|acceleration|force|velocity|momentum|"
                     r"electric|magnetic|circuit|current|voltage|resistance|"
                     r"converge|gradient|derivative|algorithm|asymptotic|"
                     r"time complexity|big O|O\(n|proof|theorem)\b", lower):
            _logic_boost = max(_logic_boost, 4.0)
        # NEW: "Why does X happen" / "Explain step by step why Y" patterns
        if re.search(r"\b(why (does|is|are|do|would|did)|explain (why|how|step))\b", lower):
            _logic_boost = max(_logic_boost, 2.0)
    if _logic_boost > 0:
        scores["logic"] = scores.get("logic", 0) + _logic_boost

    # Summarization: news/document structure (only when NOT NER task)
    _ner_task = bool(re.search(
        r"(extract (all )?|identify (the )?"
        r"(persons|organizations|locations|entities))", lower))
    
    # NER task guard: if prompt starts with explicit NER instruction, suppress
    # competing categories (factual/math/logic) so NER wins
    if _ner_task and re.search(r'^(Extract|Identify|Find|List|Tag)\s', prompt, re.I):
        for _comp in ('factual', 'math', 'logic', 'sentiment', 'summarization'):
            scores[_comp] = scores.get(_comp, 0) * 0.1
    if re.search(r"^(On|In|At) \w+ \d{1,2},? \d{4}", prompt) and not _ner_task:
        scores["summarization"] = scores.get("summarization", 0) + 2.0
    if re.search(r"(HEADLINE|DATELINE|BREAKING|BRIEF|MEMORANDUM)", prompt) \
       and not _ner_task:
        scores["summarization"] = scores.get("summarization", 0) + 2.0

    # Code gen: typing imports should not trigger math
    if scores.get("code_gen", 0) > 0 and \
       re.search(r"(from \w+ import|def \w+\()", prompt) and \
       scores.get("math", 0) > scores.get("code_gen", 0):
        scores["math"] = scores.get("math", 0) * 0.2

    # Code debug: fix prompts with heavy arithmetic should not trigger math
    if scores.get("code_debug", 0) >= 5.0 and \
       scores.get("math", 0) > 0 and \
       re.search(r"\b(fix|bug|debug|error|traceback|exception)\b", lower) and \
       re.search(r"\b(def |return|import|class |print\()", prompt):
        scores["math"] = scores.get("math", 0) * 0.3

    # Factual boost: "who was" + actor/entertainment/role + no constraint context → factual
    _who_role_score = 0
    if re.search(r'who\s+(was|is|are)\s+(the|a|an)\s+(actor|actress|president|singer|writer|author|character|star|artist|director|producer|host|player|founder|inventor|leader|queen|king|prime minister|governor|mayor|senator|representative|judge|attorney|general|chief|executive|manager|coach|teacher|professor|doctor|nurse|attorney|lawyer|artist|musician|painter|sculptor|architect|engineer|scientist|philosopher|poet|novelist|dramatist|composer|choreographer|designer|photographer|reporter|journalist|anchor|commentator|caster|analyst|expert|specialist)',
        lower
    ) and not re.search(
        r'\b(if|then|conclusion|inference|premise|deduce|syllogism|knight|knave|liar|therefore|hence|which of the following|must be|cannot be|all\s+\w+\s+are)',
        lower
    ):
        _who_role_score = 2.0

    # Entity lookup with "who was" + proper noun (named person) + no constraint → factual boost
    if _who_role_score == 0 and \
       re.search(r'who\s+(was|is|are)\s+(the|a|an)\s+\w+', lower) and \
       re.search(r'\b[A-Z][a-z]+\b\s+\b[A-Z][a-z]+\b', prompt) and \
       not re.search(r'\b(if|then|conclusion|inference|deduce|must be|cannot be)', lower):
        _who_role_score = 1.5

    if _who_role_score > 0:
        scores['factual'] = scores.get('factual', 0) + _who_role_score

    # Re-evaluate winner after adjustments
    sorted_scores = sorted(
        scores.items(),
        key=lambda x: (-x[1], -PRIORITY.get(x[0], 0))
    )
    best_cat = sorted_scores[0][0]
    best_score = sorted_scores[0][1]
    # ── END V4 IMPROVEMENTS ──

    # Map confidence from score
    confidence = _score_to_confidence(best_score)


    return best_cat, confidence, scores


def classify_with_detail(prompt: str) -> dict:
    """
    Full detailed result including all signals.
    """
    best_cat, confidence, scores = classify(prompt)

    return {
        "category": best_cat,
        "category_4way": get_4way(best_cat),
        "category_human": get_human_name(best_cat),
        "confidence": confidence,
        "raw_scores": scores,
        "score_delta": _get_score_delta(scores),
    }


def classify_batch(prompts: List[str]) -> List[dict]:
    """Classify multiple prompts."""
    return [classify_with_detail(p) for p in prompts]


def _score_to_confidence(score: float) -> float:
    """
    Map raw score to 0-1 confidence.
      score >= 3.0  →  0.90  (very confident)
      score >= 1.5  →  0.60  (moderately confident)
      score >= 0.5  →  0.30  (weak signal)
      score <  0.5  →  0.10  (barely above noise == factual default)
    """
    if score >= 3.0:
        return 0.90
    elif score >= 1.5:
        return 0.60
    elif score >= 0.5:
        return 0.30
    return 0.10


def _get_score_delta(scores: Dict[str, float]) -> float:
    """Difference between highest and second-highest score."""
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) < 2:
        return 0.0
    return sorted_scores[0] - sorted_scores[1]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m agent.stage2 <prompt>          # Classify one prompt")
        print("  python -m agent.stage2 --batch <file>    # Classify prompts from JSON array")
        print("  python -m agent.stage2 --validate <file> # Validate against labeled data")
        sys.exit(1)

    arg = sys.argv[1]

    if arg == "--batch" and len(sys.argv) >= 3:
        with open(sys.argv[2]) as f:
            data = json.load(f)
        prompts = [d["prompt"] if isinstance(d, dict) else d for d in data]
        results = classify_batch(prompts)
        print(json.dumps(results, indent=2))

    elif arg == "--validate" and len(sys.argv) >= 3:
        with open(sys.argv[2]) as f:
            items = json.load(f)
        questions = items.get("questions", items) if isinstance(items, dict) else items

        correct = 0
        total = len(questions)
        by_cat = {}

        for q in questions:
            prompt = q.get("prompt", q.get("question", ""))
            true_cat = q.get("category", q.get("label", q.get("label_8way", "unknown")))
            short_true = get_short_name(true_cat)

            predicted, confidence, scores = classify(prompt)

            if predicted == short_true:
                correct += 1

            by_cat.setdefault(short_true, {"correct": 0, "total": 0})
            by_cat[short_true]["total"] += 1
            if predicted == short_true:
                by_cat[short_true]["correct"] += 1

        print(json.dumps({
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total > 0 else 0,
            "per_category": {
                cat: {
                    "correct": v["correct"],
                    "total": v["total"],
                    "accuracy": round(v["correct"] / v["total"], 4) if v["total"] > 0 else 0,
                }
                for cat, v in sorted(by_cat.items())
            },
        }, indent=2))

    else:
        prompt = arg
        result = classify_with_detail(prompt)
        print(json.dumps(result, indent=2))
