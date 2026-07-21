"""ReadyClassifier — Lightweight bulk classifier for staging.

Pure regex/heuristic classifier that replicates the category_filter
without importing the agent package (avoids the spacy/sympy import chain).

8-way categories: code_debug, code_gen, factual, logic, math, ner, sentiment, summarization
4-way grouping: code, reasoning, knowledge, text
"""
import re
from typing import Optional

# ── Category names ──
CATEGORIES = [
    "code_debug", "code_gen", "factual", "logic",
    "math", "ner", "sentiment", "summarization",
]

CATEGORY_4WAY = {
    "code_debug": "code", "code_gen": "code",
    "math": "reasoning", "logic": "reasoning",
    "factual": "knowledge",
    "sentiment": "text", "ner": "text", "summarization": "text",
}

SCORE_PRIORITY = {
    "code_debug": 8, "code_gen": 7, "math": 6, "logic": 5,
    "sentiment": 4, "ner": 3, "summarization": 2, "factual": 1,
}


def classify(prompt: str) -> tuple[str, str, float, dict]:
    """Classify a prompt into one of 8 categories.

    Returns (category_8way, category_4way, confidence, raw_scores).
    Uses a simplified heuristic scoring approach.
    """
    lower = prompt.lower()

    scores = {}
    scores["code_debug"] = _score_code_debug(lower, prompt)
    scores["code_gen"] = _score_code_gen(lower, prompt)
    scores["math"] = _score_math(lower, prompt)
    scores["logic"] = _score_logic(lower, prompt)
    scores["sentiment"] = _score_sentiment(lower, prompt)
    scores["ner"] = _score_ner(lower, prompt)
    scores["summarization"] = _score_summarization(lower, prompt)
    scores["factual"] = _score_factual(lower, prompt)

    # Pick highest score, break ties by priority
    best_cat = max(scores, key=lambda c: (scores[c], SCORE_PRIORITY.get(c, 0)))
    best_score = scores[best_cat]

    # Map 0-1 range to confidence
    confidence = min(0.95, max(0.1, best_score / 5.0))

    return best_cat, CATEGORY_4WAY.get(best_cat, "knowledge"), confidence, scores


def classify_batch(prompts: list[str]) -> list[dict]:
    """Classify multiple prompts."""
    results = []
    for p in prompts:
        cat, cat4, conf, scores = classify(p)
        sorted_scores = sorted(scores.values(), reverse=True)
        delta = (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else 0.0
        results.append({
            "category": cat,
            "category_4way": cat4,
            "confidence": conf,
            "raw_scores": scores,
            "score_delta": delta,
        })
    return results


# ── Scoring functions (simplified versions) ──

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

def _has_code_fence(text: str) -> bool:
    return bool(re.search(r"```|`[^`]+`|\\bdef\\s|\\breturn\\b|\\bfunction\\b|\\bclass\\b|\\bimport\\s", text))


def _score_math(lower: str, raw: str) -> float:
    s = 0.0
    explicit = bool(re.search(
        r"\b(calculate|compute|solve|equation|formula|algebra|geometry|"
        r"trigonometry|calculus|derivative|integral|probability|statistics|"
        r"what\s+is\s+\d|how many|how much|percent|percentage|quotient|"
        r"remainder|square root|factorial|modulo|triangle|angle|"
        r"perimeter|circumference|diameter|radius|velocity|acceleration|"
        r"derivative|integral|matrix|determinant|logarithm|log\s+base)\b", lower
    ))
    if explicit:
        s += 3.0
    nums = _NUM_RE.findall(raw)
    if len(nums) >= 2:
        s += 1.0
    if re.search(r"\d+\s*[+*/\-^]\s*\d+", raw):
        s += 1.0
    if re.search(r"\b(?:math|arithmetic)\b", lower):
        s += 1.0
    return s


def _score_sentiment(lower: str, raw: str) -> float:
    s = 0.0
    if re.search(r"\b(sentiment|positive|negative|neutral|feeling|opinion|"
                 r"emotion|angry|happy|sad|review|rating|star|"
                 r"terrible|awful|great|amazing|boring|excellent|worst|best)\b", lower):
        s += 2.0
    if re.search(r"\b(is this|is the|how does|what does).*\b(feel|think|sentiment)\b", lower):
        s += 2.0
    return s


def _score_code_gen(lower: str, raw: str) -> float:
    s = 0.0
    if re.search(r"\b(write|generate|create|implement|code|program|"
                 r"function|script|algorithm|sort|search|parse)\b", lower):
        s += 2.0
    if _has_code_fence(raw):
        s += 2.0
    if re.search(r"\bpython\b|write\s+a\s+function|using\s+python", lower):
        s += 1.0
    return s


def _score_code_debug(lower: str, raw: str) -> float:
    s = 0.0
    if re.search(r"\b(debug|bug|fix|error|issue|problem|incorrect|wrong|"
                 r"not working|doesn't work|failed|crash|exception)\b", lower):
        s += 2.0
    if _has_code_fence(raw):
        s += 2.0
    if re.search(r"\b(find|what.*wrong|why|explain.*bug|correct|repair)\b", lower):
        s += 1.0
    return s


def _score_logic(lower: str, raw: str) -> float:
    s = 0.0
    if re.search(r"\b(logic|reason|deduce|inference|premise|conclusion|"
                 r"if.*then|puzzle|riddle|brain.teaser|syllogism|"
                 r"zebra|einstein|who|which.*of.*the)\b", lower):
        s += 2.0
    if re.search(r"\b(all|some|none|every|always|never|must|therefore|"
                 r"implies|follows|contradiction)\b", lower):
        s += 1.0
    return s


def _score_factual(lower: str, raw: str) -> float:
    s = 0.5  # Default fallback score
    if re.search(r"\b(what is|who is|where is|when did|how did|"
                 r"define|explain|describe|what are|tell me about)\b", lower):
        s += 1.5
    if re.search(r"\b(capital|president|country|city|population|"
                 r"located|founded|invented|discovered|author|"
                 r"history|definition|meaning|example)\b", lower):
        s += 1.0
    return s


def _score_ner(lower: str, raw: str) -> float:
    s = 0.0
    if re.search(r"\b(ner|named entity|entity|extract|"
                 r"person|organization|location|date|time|"
                 r"find.*names|identify.*(?:person|place|organization))\b", lower):
        s += 2.0
    if re.search(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}", raw):
        s += 1.0
    return s


def _score_summarization(lower: str, raw: str) -> float:
    s = 0.0
    if re.search(r"\b(summarize|summary|summarise|abstract|"
                 r"tl;dr|in short|briefly|condense|key points|"
                 r"main idea|overview|recap|gist)\b", lower):
        s += 2.0
    if len(raw) > 500:
        s += 1.0
    if re.search(r"\b(source|text|article|passage|paragraph|"
                 r"document|following)\b", lower) and len(raw) > 200:
        s += 1.0
    return s
