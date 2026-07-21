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
_CODE_FENCE_RE = re.compile(r"```|\bdef\s|\breturn\b|\bfunction\b|\bclass\b|\bimport\s")

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
        r"what is the (value|area|sum|product|probability)|"
        r"what\s+(distance|speed|time|volume|area|perimeter|probability|fraction)|"
        r"how\s+(fast|far|long|many|much)|^(?:problem|question)\s*\d*[:.]?\s)\b",
        lower
    ))
    # Competition math: factorial, Asymptotic, geometry diagrams, $...$ LaTeX
    competition = bool(re.search(
        r"\b\d+!|\\circ|\\triangle|\\sqrt|\\frac|\\log|\\sin|\\cos|\\tan|"
        r"\\angle|^\\|\$.*?\$|\\begin\{|\\[a-zA-Z]+\{|\[asy\]|"  # LaTeX environments
        r"\b(integer|integers|positive integer|prime|factor|divisible|digit)\b",
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

    return s


def _score_logic(prompt: str) -> float:
    """Logical reasoning — syllogisms, puzzles, constraints."""
    lower = prompt.lower()
    s = 0.0

    # Strong logic signals
    if re.search(r"\b(knight|knave|lying|liar|syllogism|deduce|infer)", lower):
        s += 3.0
    if re.search(r"\beach\b.*\b(who|which)\b", lower):
        s += 2.5
    if re.search(r"\bdifferent\b.*\b(who|which)\b", lower):
        s += 2.0
    if re.search(r"\bif\b.{0,60}\bthen\b", lower):
        s += 2.0
    if re.search(r"\b(either|neither|exactly one|must be (true|false))", lower):
        s += 2.0
    if re.search(r"\b(logical|conclusion|implies|therefore|hence|iff)", lower):
        s += 1.5
    if re.search(r"\bwhich of the following\b", lower):
        s += 1.5
    if re.search(r"\border|rank|arrangement|seat|adjacent\b", lower):
        s += 1.5

    # ── NEW: Mathematical proof / justification patterns ──
    if re.search(r"\b(prove|proof|justify|justification)\b", lower):
        s += 3.0
    if re.search(r"\b(convergence|converge|divergence|diverge)\b", lower):
        s += 2.5

    # ── NEW: Algorithm / complexity analysis patterns ──
    if re.search(
        r"\b(time\s+complexity|space\s+complexity|big.\s*O|O\s*\(|\bO\b\s*\(n|"
        r"algorithmic|asymptotic|worst.case|best.case|average.case|"
        r"runtime\s+(analysis|complexity))\b", lower
    ):
        s += 3.0
    if re.search(r"\b(analyze|analysis)\b.{0,40}\b(algorithm|complexity|function)\b", lower):
        s += 2.0

    # ── NEW: Physics / scientific reasoning patterns ──
    if re.search(
        r"\b(refraction|refract|prism|spectral|spectrum|wavelength|"
        r"gradient\s+descent|gradient\s+boost|backpropagation)\b", lower
    ):
        s += 2.5

    # Constraint-puzzle patterns
    if re.search(
        r"(?:sit|sits|seated|are|is|placed|arranged)\s+in a row", lower
    ):
        s += 2.5
    if re.search(r"\ball\s+\w+\s+are\s+\w+", lower):
        s += 1.0
    # People-name logic puzzles: "Four friends — X, Y, Z — each have..."
    if re.search(
        r"\b(friends?|colleagues?|neighbors?|people|team|members?|classmates?|"
        r"candidates?|students?)\s.{0,30}(each|all|one|two|three|four|five|six|seven|eight|nine|ten)\b", lower
    ):
        s += 2.0
    # List of names as puzzle setup: "Tom, Oscar, Nina, Maya, and Leo"
    name_list = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?", prompt)
    name_count = sum(1 for n in name_list if len(n) > 2)
    if name_count >= 3 and re.search(
        r"\b(each\s+has|each\s+work|each\s+is|each\s+like|each\s+own|"
        r"different\s+(department|role|job|position|color|item|type|one)|"
        r"who\s+(is|has|works|likes|owns|sits|lives|drives))\b", lower
    ):
        s += 2.5

    return s


def _score_factual(prompt: str) -> float:
    """Factual knowledge / QA — who/what/when/where/why questions."""
    lower = prompt.lower()
    s = 0.0

    # Question-starter patterns
    if re.search(
        r"\b(what is|who (is|was|are|were)|when (did|was|is)|where is|"
        r"why (did|does|is|are)|how (does|do|did|many|much|long|far|old)|"
        r"which of the following is)\b", lower
    ):
        s += 1.5

    # Information-seeking verbs
    if re.search(r"\b(define|explain|describe|tell me about|facts? about|"
                 r"history of|capital of|population of|meaning of)\b", lower):
        s += 1.5

    # SQuAD-style QA format
    if ("question:" in lower or "q:" in lower) and \
       ("context:" in lower or "passage:" in lower or "text:" in lower):
        s += 3.0

    # Simple factual lookup patterns
    if re.search(r"^what\s+is\b", lower):
        s += 0.5

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
    """Summarization — condense text, extract key points."""
    lower = prompt.lower()
    s = 0.0

    if re.search(
        r"\b(summarize|summary|tl;?dr|tldr|condense|shorten|compress|"
        r"recap|gist|boil down|key points|main idea|overview)\b", lower
    ):
        s += 2.5

    # Single-sentence/word constraints
    if re.search(
        r"\bin (one|two|three|\d+) (sentence|word|bullet|line|paragraph)\b", lower
    ):
        s += 2.0

    # Detect long text passed for summarization (implicit)
    words = lower.split()
    if len(words) > 80 and not _score_code_gen(prompt) and not _score_math(prompt):
        s += 0.5
    if len(words) > 150:
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
        if re.search(r"\b(?:import |from \w+ import|def \w+\(|class \w+:?\b|print\(|return\b)", prompt):
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
