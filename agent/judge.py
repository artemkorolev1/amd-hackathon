"""
Multi-judge ensemble for answer quality scoring.

Combines:
  1. Deterministic cascade (4 tiers: quick reject, format val, confidence gating, structure)
  2. Learned judge (RandomForest trained on 732 eval records, 87% test acc)
  3. Consensus voting (agreement across multiple samples)

Produces a unified confidence score (0.0-1.0) per answer.
"""

from __future__ import annotations

import ast
import json
import os
import pickle
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from agent.solvers.verify import (
    _has_hedge, _is_degenerate, _is_too_short, _is_too_long,
    _extract_code, _valid_python, _has_import_statement,
    _split_sentences, _count_bullets, format_and_lint,
)
from scripts.grade_answer import fuzzy_match

# ── Constants ───────────────────────────────────────────────────────────

CATEGORIES = [
    "code_gen", "code_debug", "math", "sentiment",
    "ner", "summarization", "factual", "logic",
]

METRICS_PASS_THRESHOLD = 0.5

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_MODEL_PATH = os.path.join(_PROJECT, "models", "learned_judge.pkl")
_VEC_PATH = os.path.join(_PROJECT, "models", "judge_vectorizer.pkl")

# Classifier confidence penalty brackets
_CLASSIFIER_PENALTY = [
    (0.9, 0.0), (0.6, 0.2), (0.3, 0.4), (0.0, 0.6),
]

# Tier 4 structure format expectations per category
_FORMAT_EXPECTATIONS = {
    "code_gen":       {"label": "code_block",    "penalty": 0.15},
    "code_debug":     {"label": "code_block",    "penalty": 0.10},
    "math":           {"label": "numeric",       "penalty": 0.15},
    "sentiment":      {"label": "short_label",   "penalty": 0.10},
    "ner":            {"label": "entity_format", "penalty": 0.15},
    "summarization":  {"label": "prose",         "penalty": 0.05},
    "factual":        {"label": "short_answer",  "penalty": 0.05},
    "logic":          {"label": "explanatory",   "penalty": 0.05},
}


# ── Result type ─────────────────────────────────────────────────────────


@dataclass
class JudgeResult:
    score: float = 0.0
    passed: bool = False
    reasons: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════
#  Learned judge feature extractors (must be importable for pickle compat)
# ═══════════════════════════════════════════════════════════════════════


class _StructuralFeatureExtractor:
    """Structural feature extractor for the learned judge model."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        features = []
        for rec in X:
            prompt = rec.get("prompt", "")
            answer = rec.get("answer", "")
            expected = rec.get("expected", "")
            f = [
                len(answer), len(prompt), len(expected),
                len(answer.split()), len(prompt.split()), len(expected.split()),
                len(answer) / max(len(expected), 1),
                len(answer.split()) / max(len(expected.split()), 1),
                1.0 if "```" in answer else 0.0,
                answer.count("\n"),
                len(set(re.findall(r"\w+", answer.lower()))),
                1.0 if answer.strip() and answer.strip()[-1] in ".!?" else 0.0,
                len(re.findall(r"\d+(?:\.\d+)?", answer)),
                len(re.findall(r"\d+(?:\.\d+)?", expected)),
                len(re.findall(r"\d+(?:\.\d+)?", answer)) - len(re.findall(r"\d+(?:\.\d+)?", expected)),
                1.0 if re.search(r"\bdef |\bclass |\breturn\b|\bimport ", answer) else 0.0,
                1.0 if re.search(r"\bdef |\bclass ", expected) else 0.0,
            ]
            ans_words = set(re.findall(r"[a-zA-Z]{3,}", answer.lower()))
            exp_words = set(re.findall(r"[a-zA-Z]{3,}", expected.lower()))
            f.append(len(ans_words & exp_words) / len(exp_words) if exp_words else 0.0)
            f.append(1.0 if answer.strip().lower() == expected.strip().lower() else 0.0)
            f.append(1.0 if expected.strip().lower() in answer.strip().lower() else 0.0)
            features.append(f)
        return np.array(features)


class _TextFeatureExtractor:
    """TF-IDF feature extractor for the learned judge."""
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf = TfidfVectorizer(
            max_features=1000, ngram_range=(1, 2),
            stop_words="english", sublinear_tf=True,
        )
    def fit(self, X, y=None):
        texts = [f"{rec.get('category', '')} {rec.get('answer', '')} {str(rec.get('prompt', ''))[:200]}" for rec in X]
        self.tfidf.fit(texts)
        return self
    def transform(self, X):
        texts = [f"{rec.get('category', '')} {rec.get('answer', '')} {str(rec.get('prompt', ''))[:200]}" for rec in X]
        return self.tfidf.transform(texts).toarray()


class LearnedJudge:
    """Wrapper for the trained RandomForest judge model."""

    def __init__(self):
        self.model = None
        self.struct = None
        self.txt = None
        self._load()

    def _load(self):
        if not os.path.isfile(_MODEL_PATH) or not os.path.isfile(_VEC_PATH):
            return
        try:
            with open(_MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            with open(_VEC_PATH, "rb") as f:
                vec = pickle.load(f)
            self.struct = vec["struct"]
            self.txt = vec["txt"]
        except Exception as e:
            print(f"[judge] Failed to load learned model: {e}", file=sys.stderr)

    @property
    def available(self):
        return self.model is not None

    def predict_proba(self, answer: str, prompt: str, expected: str, category: str) -> float:
        if not self.available:
            return 0.5
        rec = {"prompt": prompt, "answer": answer, "expected": expected, "category": category}
        X = np.hstack([self.struct.transform([rec]), self.txt.transform([rec])])
        return float(self.model.predict_proba(X)[0, 1])


# ═══════════════════════════════════════════════════════════════════════
#  Deterministic cascade (4 tiers)
# ═══════════════════════════════════════════════════════════════════════


def _get_classifier_penalty(confidence: float) -> tuple[float, str]:
    for threshold, penalty in _CLASSIFIER_PENALTY:
        if confidence >= threshold:
            if penalty == 0.0:
                return 0.0, "classifier confident"
            return penalty, f"classifier uncertainty (penalty: {penalty:.1f}) [conf={confidence:.2f}]"
    return 0.6, "classifier highly uncertain"


def _tier1_quick_reject(answer: str, category: str) -> Optional[JudgeResult]:
    if not answer or not answer.strip():
        return JudgeResult(0.0, False, ["empty answer"], {"tier1": "empty"})
    if _has_hedge(answer):
        return JudgeResult(0.0, False, ["hedge words detected"], {"tier1": "hedge"})
    if _is_degenerate(answer):
        return JudgeResult(0.0, False, ["degenerate repetition detected"], {"tier1": "degenerate"})
    if _is_too_short(answer):
        return JudgeResult(0.0, False, ["answer too short (< 2 chars)"], {"tier1": "too_short"})
    max_chars = 4000 if category == "summarization" else 8000
    if _is_too_long(answer, max_chars=max_chars):
        return JudgeResult(0.0, False, [f"answer too long (>{max_chars} chars)"], {"tier1": "too_long"})
    return None


def _tier2_format_validation(answer: str, category: str) -> tuple[float, list[str], dict]:
    penalty = 0.0
    reasons: list[str] = []
    details: dict = {}
    text = answer.strip()

    if category in ("code_gen", "code_debug"):
        code = _extract_code(answer) or text
        details["has_code_block"] = code is not None
        if not _valid_python(code):
            penalty += 0.4
            reasons.append("code has invalid Python syntax")
            details["valid_python"] = False
        else:
            details["valid_python"] = True
        if _has_import_statement(code):
            p = 0.3 if category == "code_gen" else 0.15
            penalty += p
            reasons.append(f"{category} answer contains import statement (unsafe)")
            details["has_import"] = True
        if isinstance(code, str) and code.strip():
            is_fragment = not code.strip().startswith(("def ", "class ", "@"))
            lint_result = format_and_lint(code, relaxed=is_fragment)
            lint_errors = lint_result.get("lint_errors", [])
            details["lint_errors_count"] = len(lint_errors)
            if lint_errors:
                lp = min(0.3, len(lint_errors) * 0.1)
                penalty += lp
                reasons.append(f"{len(lint_errors)} lint error(s) (penalty: {lp:.1f})")

    elif category == "math":
        has_number = bool(re.search(r"\d+(?:\.\d+)?", text))
        details["has_number"] = has_number
        if not has_number:
            penalty += 0.35
            reasons.append("math answer contains no number")
        else:
            numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
            words = text.split()
            if numbers and len(words) > 30 and len(words) / len(numbers) > 10:
                penalty += 0.15
                reasons.append("math answer is prose-heavy")
                details["prose_heavy"] = True

    elif category == "sentiment":
        found = [l for l in ("positive", "negative", "neutral", "mixed") if l in text.lower()]
        details["found_labels"] = found
        if not found:
            penalty += 0.4
            reasons.append("sentiment answer missing positive/negative/neutral label")
        else:
            words = re.findall(r"[a-zA-Z]+", text.lower())[:3]
            if not any(w in {"positive", "negative", "neutral", "mixed"} for w in words) and len(text.split()) > 5:
                penalty += 0.1
                reasons.append("sentiment label found but not prominent")

    elif category == "ner":
        ep = re.compile(r"(PERSON|ORG|ORGANIZATION|LOC|LOCATION|DATE|TIME|GPE|EVENT|PRODUCT|NORP|FAC|LAW|LANGUAGE|WORK_OF_ART|MONEY|PERCENT|QUANTITY|CARDINAL)\s*:\s*.+", re.IGNORECASE)
        gp = re.compile(r"[A-Z][A-Z]+(?:\s*:\s*.+)", re.MULTILINE)
        matches = ep.findall(text) or gp.findall(text)
        details["entity_count"] = len(matches)
        if not matches:
            penalty += 0.35
            reasons.append("NER answer does not match TYPE: value entity format")

    elif category == "summarization":
        sentences = _split_sentences(text)
        details["sentence_count"] = len(sentences)
        if len(sentences) < 1:
            penalty += 0.3
            reasons.append("summarization has no complete sentences")
        elif len(sentences) > 15:
            penalty += 0.15
            reasons.append(f"summarization too long ({len(sentences)} sentences)")
        bullets = _count_bullets(text)
        if bullets > len(sentences) * 0.5 and bullets > 3:
            penalty += 0.1
            reasons.append("summarization is mostly bullet points")
        if len(text.split()) < 5:
            penalty += 0.2
            reasons.append("summarization too short (< 5 words)")

    elif category == "factual":
        wc = len(text.split())
        details["word_count"] = wc
        if wc > 120:
            penalty += 0.1
            reasons.append(f"factual answer too wordy ({wc} words)")
        if re.search(r"^here.*(?:list|table|details)", text.lower()):
            penalty += 0.1
            reasons.append("factual answer starts with preamble")

    elif category == "logic":
        indicators = [r"\btherefore\b", r"\bbecause\b", r"\bif\b", r"\bthen\b", r"\bso\b", r"\bimplies\b", r"\bhence\b", r"\bthus\b", r"\bconclusion\b", r"\breason\b", r"\bdeduce\b"]
        has = any(re.search(p, text.lower()) for p in indicators)
        details["has_reasoning_language"] = has
        if not has and len(text.split()) >= 3:
            penalty += 0.1
            reasons.append("logic answer lacks reasoning language")

    return penalty, reasons, details


def _tier3_classifier_gating(classifier_conf: Optional[float]) -> tuple[float, list[str], dict]:
    if classifier_conf is None:
        return 0.0, [], {"classifier_conf": None, "penalty": 0.0}
    penalty, reason = _get_classifier_penalty(classifier_conf)
    if penalty != 0.0:
        return penalty, [reason], {"classifier_conf": classifier_conf, "penalty": penalty}
    return 0.0, [], {"classifier_conf": classifier_conf, "penalty": 0.0}


def _tier4_answer_structure(answer: str, category: str, expected: Optional[str]) -> tuple[float, list[str], dict]:
    penalty = 0.0
    reasons: list[str] = []
    details: dict = {}
    text = answer.strip()
    expect = _FORMAT_EXPECTATIONS.get(category)
    if expect is None:
        return 0.0, [], {"structure_checked": False}

    bp = expect["penalty"]
    label = expect["label"]

    if label == "code_block":
        has_fence = text.startswith("```")
        is_def = bool(re.match(r"^(def |class |@)", text))
        details["has_code_fence"] = has_fence
        details["is_definition"] = is_def
        if category == "code_gen" and not has_fence and not is_def:
            penalty += bp; reasons.append("code_gen answer not in code block format")
        elif category == "code_debug" and not has_fence and not is_def:
            if not re.search(r"(return |print |for |while |if |else|elif)", text):
                penalty += bp * 0.5; reasons.append("code_debug answer not in expected format")

    elif label == "numeric":
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        words = text.split()
        if not numbers:
            penalty += bp; reasons.append("math answer contains no numeric value")
        elif len(words) > 50:
            penalty += bp * 0.5; reasons.append("math answer is prose-heavy rather than compact numeric")
        details["word_count"] = len(words); details["number_count"] = len(numbers)

    elif label == "short_label":
        wc = len(text.split())
        details["word_count"] = wc
        if wc > 10:
            penalty += bp; reasons.append(f"sentiment answer too long ({wc} words)")

    elif label == "entity_format":
        el = re.findall(r"^[ \t]*[A-Z][A-Z_]+\s*:\s*.+", text, re.MULTILINE)
        details["entity_line_count"] = len(el)
        if not el:
            penalty += bp; reasons.append("NER answer lacks TYPE: value entity format")
        else:
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            if lines and len(el) < len(lines) * 0.3:
                penalty += bp * 0.5; reasons.append("NER answer has non-entity content mixed in")

    elif label == "prose":
        wc = len(text.split()); details["word_count"] = wc
        code_indicators = ["def ", "class ", "import ", "```", "return "]
        if any(ind in text for ind in code_indicators):
            penalty += bp * 2; reasons.append("summarization answer contains code fragments");

    elif label == "short_answer":
        wc = len(text.split()); details["word_count"] = wc
        if wc > 50:
            penalty += bp; reasons.append(f"factual answer too verbose ({wc} words)")

    elif label == "explanatory":
        wc = len(text.split())
        rw = [r"\btherefore\b", r"\bbecause\b", r"\bif\b", r"\bthen\b", r"\bso\b", r"\bthus\b", r"\bhence\b"]
        has = any(re.search(p, text.lower()) for p in rw)
        details["has_reasoning"] = has
        if not has and wc > 5:
            penalty += bp; reasons.append("logic answer lacks explanatory reasoning language")

    if expected is not None and expected.strip():
        details["fuzzy_match"] = fuzzy_match(text, expected.strip())

    return penalty, reasons, details


# ── Learned judge singleton ─────────────────────────────────────────────

_LEARNED_JUDGE = None

def _get_learned_judge():
    global _LEARNED_JUDGE
    if _LEARNED_JUDGE is None:
        _LEARNED_JUDGE = LearnedJudge()
    return _LEARNED_JUDGE


# ═══════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════


def judge_answer(
    answer: str,
    category: str = "",
    prompt: str = "",
    expected: Optional[str] = None,
    classifier_conf: Optional[float] = None,
    consensus_samples: Optional[list[str]] = None,
) -> JudgeResult:
    """
    Multi-judge ensemble for answer quality.

    Combines:
      1. Deterministic cascade (4 tiers)
      2. Learned judge (RandomForest, when expected is available)
      3. Consensus voting (when multiple samples are provided)

    Returns JudgeResult with score (0.0-1.0), passed (bool), reasons, and details.
    """
    category = (category or "").lower().strip()
    all_reasons: list[str] = []
    all_details: dict[str, dict] = {}

    # ── Part 1: Deterministic cascade ──
    tier1 = _tier1_quick_reject(answer, category)
    if tier1 is not None:
        return JudgeResult(0.0, False, tier1.reasons, {
            "tier1": tier1.details, "tier2": {"skipped": True},
            "tier3": {"skipped": True}, "tier4": {"skipped": True},
        })

    det_score = 1.0
    p2, r2, d2 = _tier2_format_validation(answer, category)
    det_score -= p2; all_reasons.extend(r2); all_details["tier2"] = {"penalty": p2, **d2}

    p3, r3, d3 = _tier3_classifier_gating(classifier_conf)
    det_score -= p3; all_reasons.extend(r3); all_details["tier3"] = d3

    p4, r4, d4 = _tier4_answer_structure(answer, category, expected)
    det_score -= p4; all_reasons.extend(r4); all_details["tier4"] = d4

    det_score = max(0.0, min(1.0, det_score))

    # Expected-answer boost/penalty
    expected_boost = 0.0
    if expected is not None and expected.strip():
        fm = fuzzy_match(answer.strip(), expected.strip())
        if fm:
            expected_boost = 0.15
            all_reasons.append("answer fuzzy-matches expected -- score boosted")
        elif det_score > 0.7:
            expected_boost = -0.1
            all_reasons.append("expected available but fuzzy_match failed")
        all_details["expected_boost"] = expected_boost

    det_adjusted = max(0.0, min(1.0, det_score + expected_boost))

    # ── Part 2: Learned judge ──
    learned_score = None
    if expected and expected.strip():
        lj = _get_learned_judge()
        if lj.available:
            learned_score = lj.predict_proba(answer, prompt, expected, category)
            all_details["learned_score"] = learned_score

    # ── Part 3: Consensus ──
    cons_score = None
    if consensus_samples and len(consensus_samples) >= 2:
        agreements = 0
        total = 0
        for i, a in enumerate(consensus_samples):
            for j in range(i + 1, len(consensus_samples)):
                if a.strip().lower() == consensus_samples[j].strip().lower():
                    agreements += 1
                total += 1
        cons_score = agreements / max(total, 1) if total > 0 else 0.0
        all_details["consensus_score"] = cons_score
        all_details["consensus_samples"] = len(consensus_samples)

    # ── Combine ──
    # Weights: deterministic=0.4, learned=0.4 (if avail), consensus=0.2 (if avail)
    weights = [0.4]
    scores = [det_adjusted]

    if learned_score is not None:
        weights.append(0.4)
        scores.append(learned_score)

    if cons_score is not None:
        weights.append(0.2)
        scores.append(cons_score)

    tw = sum(weights)
    weights = [w / tw for w in weights]
    final_score = sum(s * w for s, w in zip(scores, weights))

    # Consensus bonuses
    if cons_score is not None:
        if cons_score > 0.8:
            final_score = min(1.0, final_score + 0.1)
            all_reasons.append("consensus: strong agreement")
        elif cons_score < 0.3:
            final_score = max(0.0, final_score - 0.2)
            all_reasons.append("consensus: weak agreement")

    final_score = max(0.0, min(1.0, final_score))
    all_details["final_score"] = final_score

    return JudgeResult(
        score=final_score,
        passed=final_score >= METRICS_PASS_THRESHOLD,
        reasons=all_reasons,
        details=all_details,
    )


def judge_batch(answers: list[dict]) -> list[JudgeResult]:
    results = []
    for item in answers:
        results.append(judge_answer(
            answer=item.get("answer", ""),
            category=item.get("category", ""),
            prompt=item.get("prompt", ""),
            expected=item.get("expected"),
            classifier_conf=item.get("classifier_conf"),
            consensus_samples=item.get("consensus_samples"),
        ))
    return results
