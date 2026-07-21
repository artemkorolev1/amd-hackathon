#!/usr/bin/env python3
"""
agent/solvers/sentiment_hybrid.py — VADER + LLM hybrid sentiment classifier.

Routes simple cases to deterministic VADER (fast, free) and delegates
hard/mixed cases to the LLM with VADER hints.

Typical usage:

    from agent.solvers.sentiment_hybrid import classify_sentiment_hybrid

    def my_llm(system, user):
        # call your model ...
        return raw_text

    result = classify_sentiment_hybrid(
        text="This product is terrible!",
        llm_infer_fn=my_llm,
        system_prompt="Analyze the tone.",
    )
    print(result["label"], result["source"])
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ── Import VADER components from deterministic solver ──────────────────────
from agent.solvers.deterministic import (
    _classify_sentiment_vader,
    _get_vader_analyzer,
    _RE_BACKHANDED,
    _RE_GENERAL_BUT,
    _RE_SARCASM_OH,
    _RE_SARCASM_RHET,
    _RE_SARCASM_YEAH,
    _VADER_POS_THRESH,
)

# ── Import format normalizer ───────────────────────────────────────────────
from agent.solvers.format_normalizer import normalize_sentiment_output

# ── VADER routing thresholds (tunable) ─────────────────────────────────────

# When compound < VADER_THRESHOLD, trust VADER directly (92 % accurate on these)
_DEFAULT_VADER_THRESHOLD = -0.3

# When LLM says positive/neutral but compound < OVERRIDE_THRESHOLD + pattern, override
_DEFAULT_OVERRIDE_THRESHOLD = -0.1

# ============================================================================
# Internal helpers
# ============================================================================


def _check_vader_pattern(text: str, compound: float) -> Optional[str]:
    """
    Check which VADER pattern (if any) would match this text.

    Returns the pattern name (str) or *None*.
    """
    # Pattern A: "Oh [positive]..." — override to NEGATIVE if compound > -0.1
    if _RE_SARCASM_OH.search(text) and compound > -0.1:
        return "sarcasm_oh"
    # Pattern B: "Yeah right" / dismissive agreement — unconditional
    if _RE_SARCASM_YEAH.search(text):
        return "sarcasm_yeah"
    # Pattern C: Rhetorical question with positive surface — override if compound > 0.0
    if _RE_SARCASM_RHET.search(text) and compound > 0.0:
        return "sarcasm_rhet"
    # Backhanded compliment — unconditional
    if _RE_BACKHANDED.search(text):
        return "backhanded"
    # "X but Y" negative bias — override if compound > -0.1
    if _RE_GENERAL_BUT.search(text) and compound > -0.1:
        return "but"
    return None


def _compute_vader_confidence(compound: float, pattern: Optional[str]) -> str:
    """
    Determine VADER confidence level.

    - **high**: strong signal (|compound| > 0.3) *or* pattern matched
    - **medium**: moderate signal (0.05 < |compound| <= 0.3)
    - **low**: near neutral (|compound| <= 0.05)
    """
    if pattern is not None:
        return "high"
    if compound < -0.3 or compound > 0.5:
        return "high"
    if compound <= -0.05 or compound >= _VADER_POS_THRESH:
        return "medium"
    return "low"


# ============================================================================
# Public API
# ============================================================================


def classify_sentiment_vader_only(text: str) -> dict:
    """
    Run VADER and return a full scoring dict.

    Parameters
    ----------
    text : str
        Input text to analyze.

    Returns
    -------
    dict
        - **label** (*str*) — ``"positive"``, ``"negative"``, ``"neutral"``,
          ``"mixed"``, or ``None``.
        - **compound** (*float*) — VADER compound score (-1 to 1).
        - **pos** (*float*) — positive sub-score.
        - **neg** (*float*) — negative sub-score.
        - **confidence** (*str*) — ``"high"``, ``"medium"``, ``"low"``.
        - **vader_pattern** (*str | None*) — which pattern matched, or ``None``.
    """
    analyzer = _get_vader_analyzer()
    if analyzer is None:
        return {
            "label": None,
            "compound": 0.0,
            "pos": 0.0,
            "neg": 0.0,
            "confidence": "low",
            "vader_pattern": None,
        }

    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    pos = scores["pos"]
    neg = scores["neg"]

    # Get standard VADER label (includes pattern overrides inside _classify_sentiment_vader)
    label = _classify_sentiment_vader(text)

    # Check for patterns separately (for routing decisions)
    pattern = _check_vader_pattern(text, compound)

    # Compute confidence
    confidence = _compute_vader_confidence(compound, pattern)

    return {
        "label": label,
        "compound": compound,
        "pos": pos,
        "neg": neg,
        "confidence": confidence,
        "vader_pattern": pattern,
    }


def _build_vader_hint_prompt(
    user_text: str,
    vader_score: dict,
    system_prompt: str | None = None,
) -> list[dict]:
    """Build messages with VADER hint injected."""
    system_msg = (
        system_prompt
        or "Analyze the tone as positive, negative, neutral, or mixed."
    )

    # Add VADER hint
    hint = ""
    compound = vader_score.get("compound", 0.0)

    if compound < -0.1:
        hint = (
            f"\n\n[Note: A preliminary analysis suggests this text may be negative "
            f"(score: {compound:.2f}). Consider sarcasm or indirect language.]"
        )
    elif 0.5 < compound < 0.95:
        hint = (
            f"\n\n[Note: A preliminary analysis suggests a mildly positive tone "
            f"(score: {compound:.2f}). Verify if the sentiment is genuinely positive.]"
        )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_text + hint},
    ]


def classify_sentiment_hybrid(
    text: str,
    llm_infer_fn: Callable[[str, str], str],
    system_prompt: str | None = None,
    vader_threshold: float = _DEFAULT_VADER_THRESHOLD,
    override_threshold: float = _DEFAULT_OVERRIDE_THRESHOLD,
    llm_params: dict | None = None,
) -> dict:
    """
    Hybrid VADER + LLM sentiment classification.

    Flow
    ----
    1. Run VADER scoring.
    2. If **compound < *vader_threshold*** AND confidence is high → use VADER directly.
    3. If VADER has a sarcasm/backhanded pattern match → use VADER (as negative).
    4. Otherwise → run LLM with VADER hint injected.
    5. If LLM says ``"positive"`` / ``"neutral"`` AND
       ``compound < override_threshold`` AND pattern matched → override to VADER.

    Parameters
    ----------
    text : str
        Input text to classify.
    llm_infer_fn : Callable[[str, str], str]
        Function that takes ``(system_prompt, user_prompt)`` and returns the
        **raw** output string from the LLM.
    system_prompt : str | None
        System-level instruction for the LLM.
    vader_threshold : float
        Compound score below which VADER is trusted directly (default: -0.3).
    override_threshold : float
        Compound score below which an LLM positive/neutral verdict is overridden
        when VADER detected a negative pattern (default: -0.1).
    llm_params : dict | None
        Reserved for future extensibility.

    Returns
    -------
    dict
        - **label** (*str*) — ``"positive"``, ``"negative"``, ``"neutral"``,
          ``"mixed"``, or ``"unknown"``.
        - **source** (*str*) — ``"vader"``, ``"llm"``, ``"llm_override"``,
          ``"vader_pattern"``.
        - **compound** (*float*) — VADER compound score.
        - **vader_label** (*str*) — VADER's verdict.
        - **llm_label** (*str | None*) — LLM's verdict (if called).
        - **format_confidence** (*str | None*) — confidence from the format
          normalizer (``"high"``, ``"medium"``, ``"low"``).
    """
    # Step 1: Run VADER scoring
    vader_result = classify_sentiment_vader_only(text)
    vader_label: str | None = vader_result["label"]
    compound: float = vader_result["compound"]
    pattern: str | None = vader_result["vader_pattern"]
    vader_confidence: str = vader_result["confidence"]

    # Step 2: If compound is strongly negative, trust VADER directly
    if compound < vader_threshold and vader_confidence == "high":
        return {
            "label": vader_label if vader_label is not None else "unknown",
            "source": "vader",
            "compound": compound,
            "vader_label": vader_label,
            "llm_label": None,
            "format_confidence": vader_confidence,
        }

    # Step 3: If VADER has a sarcasm/backhanded pattern match → use VADER
    if pattern is not None:
        return {
            "label": "negative",  # All patterns override to negative
            "source": "vader_pattern",
            "compound": compound,
            "vader_label": vader_label,
            "llm_label": None,
            "format_confidence": "high",
        }

    # Step 4: Call LLM with VADER hint
    messages = _build_vader_hint_prompt(text, vader_result, system_prompt)
    system = messages[0]["content"]
    user = messages[1]["content"]

    try:
        raw_output = llm_infer_fn(system, user)
    except Exception as e:
        logger.error(f"LLM inference failed: {e}")
        return {
            "label": vader_label if vader_label is not None else "unknown",
            "source": "vader",
            "compound": compound,
            "vader_label": vader_label,
            "llm_label": None,
            "format_confidence": vader_confidence,
        }

    # Normalize LLM output
    llm_label, format_confidence = normalize_sentiment_output(raw_output)

    # Step 5: Override rules
    # If LLM says positive/neutral and VADER detected negative signal + pattern
    if llm_label in ("positive", "neutral") and compound < override_threshold:
        # Broader pattern check (without compound constraints that _check_vader_pattern uses)
        # Re-check with relaxed conditions for the override rule
        if _check_vader_pattern(text, compound) is not None:
            return {
                "label": "negative",
                "source": "llm_override",
                "compound": compound,
                "vader_label": vader_label,
                "llm_label": llm_label,
                "format_confidence": "high",
            }

    # Return LLM result (fallback to VADER if LLM returned unknown)
    final_label = llm_label if llm_label != "unknown" else (
        vader_label if vader_label is not None else "unknown"
    )

    return {
        "label": final_label,
        "source": "llm",
        "compound": compound,
        "vader_label": vader_label,
        "llm_label": llm_label,
        "format_confidence": format_confidence,
    }
