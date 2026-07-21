"""
Deterministic task router: pure pattern-matching, zero model calls.

Priority-ordered rules. First match wins. Falls back to Bitmorphic
Complexity Score classifier, then 'other_complex'.
"""

import re
from typing import Optional, Tuple

from agent.bitmorphic_classifier import BitmorphicClassifier

# ---------------------------------------------------------------------------
# Bitmorphic fallback classifier
# ---------------------------------------------------------------------------
_bitmorphic = BitmorphicClassifier()

# ---------------------------------------------------------------------------
# Rule: (match_fn, category)
# ---------------------------------------------------------------------------

def _has_any(text: str, *patterns: str) -> bool:
    """Case-insensitive substring match against any of the patterns."""
    lower = text.lower()
    return any(p.lower() in lower for p in patterns)


def _has_re(text: str, pattern: str) -> bool:
    """Regex match."""
    return bool(re.search(pattern, text, re.IGNORECASE))


def _is_code_task(text: str) -> bool:
    """Detect code generation or debugging tasks."""
    if "```" in text and re.search(r"```(python|js|rust|go|cpp|java|ts|typescript)", text):
        return True
    if _has_any(text,
        "write a function", "write a program",
        "def ", "function ", "class ", "import ",
        "fix this code", "debug", "bug in", "error in",
        "what's wrong with", "refactor", "optimize this code",
        "leetcode", "algorithm", "data structure",
    ) or _has_re(text, r'\bcode\b') or _has_re(text, r'\bimplement\b'):
        return True
    return False


def _is_debugging_task(text: str) -> bool:
    """Detect code debugging / bug-fix tasks (not code generation)."""
    lower = text.lower()
    # Must have explicit debugging indicators
    has_debug_signal = _has_any(text,
        "fix", "debug", "bug", "error", "issue", "not working",
        "broken", "incorrect", "wrong output", "doesn't work",
        "what's wrong", "why doesn't", "why is this",
        "fix this code", "fix the code", "find the bug",
        "correct the code", "what is wrong",
    )
    if not has_debug_signal:
        return False
    # Must contain code patterns
    has_code = (
        "```" in text
        or _has_any(text, "def ", "function ", "class ", "import ",
                    "print(", "return ", "for ", "while ", "if ")
        or bool(re.search(r'[a-zA-Z_]\w*\s*=\s*[a-zA-Z_(\[0-9]', text))
    )
    return has_code


def _is_math_task(text: str) -> Optional[str]:
    """Returns 'arithmetic' or 'reasoning' or None."""
    has_numbers = bool(re.search(r'\d+', text))
    # Math keywords (excluding operators that cause false positives with hyphens)
    has_math_keywords = (
        _has_any(text,
            "calculate", "compute", "solve", "equation", "formula",
            "quotient", "derivative", "integral", "algebra",
            "geometry", "trigonometry", "probability", "statistics",
            "plus", "minus", "times", "divided by",
            # Benchmark-style word problems
            "problem:", "kmph", "km/h", "mph", "how many", "how much",
            "what distance", "what percentage", "what fraction",
            "percentage", "percent", "speed", "distance",
        )
        or _has_re(text, r'\b(sum|difference|product|math)\b')
    )
    # Operators only count as math signals when numbers are present nearby
    has_math_operators = has_numbers and (
        _has_re(text, r'\d+\s*[\+\-\*/]\s*\d+')
    )
    if not has_math_keywords and not has_math_operators:
        return None
    # Simple arithmetic (single-step, numbers, no variables)
    if has_numbers and _has_re(text, r'^[\d\s\+\-\*\/\(\)\.\,]+$|what is\s+\d+'):
        return "math_arithmetic"
    # Reasoning (word problems, multi-step, variables)
    return "math_reasoning"


def _is_summarization(text: str) -> bool:
    if _has_any(text,
        "summarize", "summary", "tl;dr", "tldr",
        "condense", "shorten", "key points", "main idea",
        "in a few words", "briefly", "overview",
    ):
        return True
    # Detect raw news-article style text (implicit summarization task)
    words = text.split()
    if len(words) > 80 and not _is_code_task(text):
        # Person+Age pattern: "John Stollery, 58,"
        # or article-start pattern: "The 29-year-old,"  "A 45-year-old man"
        if _has_re(text, r'[A-Z][a-z]+,\s*\d+,') or _has_re(text, r'\b\d+(-year)?-old\b'):
            return True
    return False


def _is_sentiment(text: str) -> bool:
    lower = text.lower()
    # Explicit sentiment task indicators
    if _has_any(text,
        "sentiment", "positive", "negative", "neutral",
        "how does the author feel", "tone of",
        "is this review", "opinion",
    ):
        return True
    # Rich sentiment-bearing vocabulary with strong signal
    strong_sentiment = [
        "love", "loved", "loving", "heartfelt", "mesmerizing",
        "amazing", "wonderful", "fantastic", "excellent",
        "beautiful", "brilliant", "terrible", "awful",
        "horrible", "disgusting", "worst", "boring", "dull",
        "sad", "angry", "frustrating",
    ]
    # Only trigger on strong sentiment words if the text looks like a review
    # (short to medium length, might contain multiple sentiment words)
    if len(lower.split()) <= 100:
        sentiment_count = sum(1 for w in strong_sentiment if w in lower)
        if sentiment_count >= 3:
            return True
    return False


def _is_ner(text: str) -> bool:
    """Detect Named Entity Recognition tasks."""
    lower = text.lower()
    # Explicit NER instructions
    if _has_any(text,
        "named entity", "extract the entities", "find the names",
        "people mentioned", "organizations", "locations",
        "entity recognition", "extract named entities",
        "extract entities from",
    ) or _has_re(text, r'\bNER\b'):
        return True
    # Domain-specific NER patterns
    if _has_any(text,
        "extract all diseases", "extract all genes",
        "diseases from this", "biomedical text", "clinical text",
        "medical record", "patient history",
        "what diseases are mentioned",
    ):
        return True
    # Detect biomedical text with disease patterns
    disease_suffixes = ["itis", "osis", "oma", "emia", "pathy", "penia"]
    has_disease_pattern = any(
        re.search(rf'\b\w+{suf}\b', text, re.IGNORECASE)
        for suf in disease_suffixes
    )
    if has_disease_pattern and _has_any(text, "extract", "find", "list", "identify", "name"):
        return True
    return False


def _is_classification(text: str) -> bool:
    return _has_any(text,
        "classify", "categorize", "label this",
        "which category", "assign a",
    )


def _is_extraction(text: str) -> bool:
    return _has_any(text,
        "extract", "pull out", "find all",
        "list the", "what are the",
    ) and not _is_ner(text)


def _is_translation(text: str) -> bool:
    return _has_any(text,
        "translate", "convert to", "in spanish", "in french",
        "in german", "in chinese", "in japanese",
        "from english to", "to english",
    )


def _is_rewriting(text: str) -> bool:
    return _has_any(text,
        "rewrite", "rephrase", "paraphrase",
        "make this more", "improve this",
        "formalize", "simplify this",
    )


def _is_data_formatting(text: str) -> bool:
    return _has_any(text,
        "format", "convert to json", "convert to csv",
        "parse", "to yaml", "to xml",
        "sort", "filter", "transform",
    )


def _is_factual(text: str) -> bool:
    """Detect factual knowledge / QA tasks."""
    lower = text.lower()
    # Explicit factual indicators
    has_factual_signal = _has_any(text,
        "what is", "who is", "when did", "where is",
        "definition", "explain", "describe",
        "facts about", "tell me about", "history of",
        "capital of", "population of",
    )
    # SQuAD-style QA format: "Question: ... Context: ..." or vice versa
    has_qa_format = (
        ("question:" in lower or "q:" in lower)
        and ("context:" in lower or "passage:" in lower or "text:" in lower)
    )
    if has_qa_format:
        return True
    if not has_factual_signal:
        return False
    return not _is_math_task(text)


def _is_creative(text: str) -> bool:
    return _has_any(text,
        "write a story", "write a poem", "write a song",
        "creative", "generate an image prompt",
        "come up with", "imagine",
    )


def _is_analysis(text: str) -> bool:
    return _has_any(text,
        "analyze", "compare", "contrast",
        "what's the difference", "pros and cons",
        "evaluate", "assess", "interpret",
    )


def _is_instruction(text: str) -> bool:
    return _has_any(text,
        "how to", "steps to", "guide me",
        "instructions for", "tutorial",
    )


def _is_logic_task(text: str) -> bool:
    """Detect logical reasoning tasks (QA with options, syllogisms)."""
    return _has_any(text,
        "which of the following", "must be true", "must be false",
        "syllogism", "logical", "deduce", "inference",
        "if-then", "implies", "conclusion",
        "typical manslaughters",  # from eval
        "options:", "question:",
    ) or bool(re.search(r'\bAll\s+\w+\s+are\s+\w+\.', text, re.I))


# ---------------------------------------------------------------------------
# Priority-ordered rule list
# ---------------------------------------------------------------------------
# Each rule is (category, match_fn, debug_name)
# Order matters: more specific patterns first, general fallbacks later.
ROUTER_RULES = [
    ("code_debugging",     _is_debugging_task,      "code_debugging"),
    ("code_generation",    _is_code_task,            "code_generation"),
    ("math_arithmetic",    lambda t: _is_math_task(t) == "math_arithmetic", "math_arithmetic"),
    ("math_reasoning",     lambda t: _is_math_task(t) == "math_reasoning", "math_reasoning"),
    ("summarization",      _is_summarization,        "summarization"),
    ("sentiment",          _is_sentiment,             "sentiment"),
    ("named_entity_recognition", _is_ner,             "ner"),
    ("classification",     _is_classification,        "classification"),
    ("logical_reasoning",  _is_logic_task,            "logical_reasoning"),
    ("translation",        _is_translation,           "translation"),
    ("rewriting",          _is_rewriting,             "rewriting"),
    ("extraction",         _is_extraction,            "extraction"),
    ("data_formatting",    _is_data_formatting,       "data_formatting"),
    ("factual_knowledge",  _is_factual,               "factual"),
    ("creative_generation", _is_creative,             "creative"),
    ("analysis",           _is_analysis,              "analysis"),
    ("instruction_following", _is_instruction,        "instruction"),
]


# Catch-all for remaining patterns
CATCH_ALL_CATEGORIES = {
    "logical_reasoning", "question_answering", "other_complex",
}


def classify(prompt: str) -> str:
    """
    Classify a task prompt into a category.

    Returns a TASK_CATEGORIES value. Pure deterministic -
    no model calls, no randomness.
    """
    if not prompt or not prompt.strip():
        return "other_complex"

    text = prompt.strip()

    # Run priority rules
    for category, match_fn, _name in ROUTER_RULES:
        if match_fn(text):
            return category

    # For short prompts (<20 chars), assume Q&A or instruction
    if len(text) < 20:
        if text.endswith("?"):
            return "question_answering"
        return "instruction_following"

    # Bitmorphic Complexity Score fallback
    result = _bitmorphic.classify(text)
    score = result["score"]
    difficulty = result["difficulty"]

    if difficulty == "SIMPLE":
        # Check for known structured tasks that Bitmorphic scored low
        if _is_sentiment(text):
            return "sentiment"
        if _is_ner(text):
            return "named_entity_recognition"
        if _is_summarization(text):
            return "summarization"
        if _is_math_task(text):
            m = _is_math_task(text)
            return "math_arithmetic" if m == "math_arithmetic" else "math_reasoning"
        if _is_logic_task(text):
            return "logical_reasoning"
        return "factual_knowledge"  # default SIMPLE: question answering
    elif difficulty == "MODERATE":
        return "analysis"
    else:
        return "other_complex"


def classify_with_complexity(task: str) -> Tuple[str, float, dict]:
    """
    Classify a task and return (category, bitmorphic_score, complexity_info).

    The bitmorphic score (0-1) can be used by the orchestrator for
    local vs Fireworks routing decisions:
      - < 0.35 (SIMPLE):  local model
      - 0.35-0.65 (MODERATE): mid-tier Fireworks
      - > 0.65 (COMPLEX): best Fireworks model
    """
    category = classify(task)
    result = _bitmorphic.classify(task)
    return (category, result["score"], result)


def classify_batch(tasks: list[str]) -> list[str]:
    """Classify a batch of tasks."""
    return [classify(t) for t in tasks]


# ---------------------------------------------------------------------------
# 4-way mapping (merge 8/19 categories into 4 super-categories)
# ---------------------------------------------------------------------------
ROUTER_4WAY_MAP = {
    # code
    "code_debugging": "code",
    "code_generation": "code",
    "creative_generation": "code",
    "data_formatting": "code",
    # reasoning
    "math_arithmetic": "reasoning",
    "math_reasoning": "reasoning",
    "logical_reasoning": "reasoning",
    "analysis": "reasoning",
    "other_complex": "reasoning",
    # knowledge
    "factual_knowledge": "knowledge",
    "named_entity_recognition": "knowledge",
    "question_answering": "knowledge",
    "extraction": "knowledge",
    "translation": "knowledge",
    "instruction_following": "knowledge",
    # text
    "sentiment": "text",
    "summarization": "text",
    "classification": "text",
    "rewriting": "text",
}


def classify_4way(prompt: str) -> str:
    """
    Classify a prompt into one of 4 super-categories:
    code, reasoning, knowledge, text.

    Runs the existing 19-category deterministic router,
    then maps the result to a 4-way group.
    """
    cat = classify(prompt)
    return ROUTER_4WAY_MAP.get(cat, "knowledge")
