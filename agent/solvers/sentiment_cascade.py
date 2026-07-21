#!/usr/bin/env python3
"""
agent/solvers/sentiment_cascade.py — Two-level cascading sentiment classifier.

Level 1 (coarse): Uses existing VADER+LLM hybrid classifier to extract
    positive/negative/neutral/mixed/unknown.

Level 2 (fine-grained): Calls the LLM again (model already loaded from L1)
    with the coarse label as context to select the most specific emotion
    from a per-category taxonomy.

Usage:
    from agent.solvers.sentiment_cascade import classify_sentiment_cascade

    def my_llm(system, user):
        # call your model ...
        return raw_text

    result = classify_sentiment_cascade(
        text="This product is terrible!",
        llm_infer_fn=my_llm,
    )
    print(result["coarse_label"], result["fine_emotion"])
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ── Emotion taxonomy ──────────────────────────────────────────────────────────

EMOTION_TAXONOMY: dict[str, list[str]] = {
    "positive": ["joyful", "satisfied", "excited", "grateful", "amused", "hopeful", "proud", "loving"],
    "negative": ["angry", "sad", "frustrated", "disappointed", "fearful", "annoyed", "disgusted", "ashamed"],
    "neutral":  ["factual", "informative", "questioning", "uncertain", "objective", "mixed"],
    "mixed":    ["bittersweet", "conflicted", "ambivalent"],
    "unknown":  ["unknown"],
}

ALL_EMOTIONS: set[str] = {
    e for emotions in EMOTION_TAXONOMY.values() for e in emotions
}

# ── Default prompts ──────────────────────────────────────────────────────────

DEFAULT_COARSE_SYSTEM_PROMPT = (
    "Analyze the tone as positive, negative, neutral, or mixed."
)

DEFAULT_FINE_SYSTEM_PROMPT_TEMPLATE = (
    'The coarse sentiment is "{coarse_label}".\n'
    "Task: Choose the single best emotion from this list.\n"
    "List: {emotion_list}\n"
    "Output one word."
)


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr.append(min(curr[-1] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def extract_fine_emotion(
    text: str,
    coarse_label: str,
    llm_infer_fn: Callable[[str, str], str],
    temperature: float = 0.3,
    max_tokens: int = 32,
) -> tuple[str, str]:
    """
    Extract fine-grained emotion given a coarse sentiment label.

    Parameters
    ----------
    text : str
        Original input text.
    coarse_label : str
        Coarse sentiment label (positive/negative/neutral/mixed/unknown).
    llm_infer_fn : Callable[[str, str], str]
        Function that takes (system_prompt, user_prompt) and returns
        the raw output string from the LLM.
    temperature : float
        LLM temperature for fine-grained call (default: 0.3).
    max_tokens : int
        Max tokens for fine-grained call (default: 32).

    Returns
    -------
    tuple[str, str]
        (emotion, source) where emotion is the selected fine emotion
        and source is "llm_fine" or "unknown" if extraction fails.
    """
    emotion_list = EMOTION_TAXONOMY.get(coarse_label, EMOTION_TAXONOMY["unknown"])
    emotion_list_str = ", ".join(emotion_list)

    # For Level 2, strip any "Classify the sentiment..." prefix from the text
    # since the model doesn't need to see classification instructions again
    clean_text = text
    import re as _re
    # Remove leading "Classify the sentiment..." or "Classify..." instructions
    clean_text = _re.sub(
        r'^(Classify the sentiment\s*(of this review)?\s*:\s*)?'
        r'(Classify the sentiment as POSITIVE or NEGATIVE[^.]*\.\s*)?'
        r'(Classify the sentiment of this review as POSITIVE or NEGATIVE[^.]*\.\s*)?'
        r'\"?',
        '', clean_text, count=1
    ).strip()
    # Remove trailing quotes if any
    clean_text = clean_text.strip('" \'')
    if not clean_text:
        clean_text = text  # fallback to original

    system_prompt = DEFAULT_FINE_SYSTEM_PROMPT_TEMPLATE.format(
        coarse_label=coarse_label,
        emotion_list=emotion_list_str,
    )

    try:
        raw_output = llm_infer_fn(system_prompt, clean_text)
    except Exception as e:
        logger.error(f"Fine-grained LLM inference failed: {e}")
        return "unknown", "unknown"

    if not raw_output or not raw_output.strip():
        return "unknown", "unknown"

    emotion = raw_output.strip().lower().rstrip(".,;:!?").strip()

    # Exact match against valid emotions
    if emotion in ALL_EMOTIONS:
        return emotion, "llm_fine"

    # Partial match: find first known emotion contained in output
    # (model often wraps the word in explanation)
    for valid_emotion in emotion_list:
        if valid_emotion in emotion:
            return valid_emotion, "llm_fine_partial"
    for valid_emotion in ALL_EMOTIONS:
        if valid_emotion in emotion:
            return valid_emotion, "llm_fine_partial"

    # Check if model output contains the coarse label name itself
    # (e.g., model outputs "positive" when asked for fine emotion)
    if coarse_label in emotion:
        # Map coarse label to a default fine emotion
        default_map = {
            "positive": "satisfied",
            "negative": "frustrated",
            "neutral": "factual",
            "mixed": "mixed",
        }
        return default_map.get(coarse_label, "unknown"), "llm_fine_partial"

    # Check individual words in the output for matches
    words = _re.findall(r'[a-zA-Z]+', emotion)
    for word in words:
        if word in ALL_EMOTIONS:
            return word, "llm_fine_partial"

    # Levenshtein fuzzy match (corrects typos)
    best_emotion = None
    best_dist = 3
    for valid_emotion in emotion_list:
        if len(valid_emotion) < 3:
            continue
        dist = _levenshtein_distance(emotion, valid_emotion)
        if dist < best_dist:
            best_dist = dist
            best_emotion = valid_emotion
    if best_emotion:
        return best_emotion, "llm_fine_typo"

    return "unknown", "unknown"


def classify_sentiment_cascade(
    text: str,
    llm_infer_fn: Callable[[str, str], str],
    coarse_classifier_fn: Callable | None = None,
    fine_llm_infer_fn: Callable[[str, str], str] | None = None,
    coarse_temperature: float = 0.0,
    fine_temperature: float = 0.3,
    coarse_system_prompt: str | None = None,
    fine_system_prompt: str | None = None,
) -> dict:
    """
    Two-level cascading sentiment classification.

    Level 1: Get coarse label using hybrid VADER+LLM classifier.
    Level 2: Get fine-grained emotion via LLM with coarse label as context.

    Parameters
    ----------
    text : str
        Input text to classify.
    llm_infer_fn : Callable[[str, str], str]
        Function that takes (system_prompt, user_prompt) and returns
        the raw LLM output string. Used for Level 1 (coarse).
    coarse_classifier_fn : Callable | None
        The coarse classifier function. If None, uses
        classify_sentiment_hybrid from sentiment_hybrid.py.
    fine_llm_infer_fn : Callable[[str, str], str] | None
        Separate LLM function for Level 2. If None, uses llm_infer_fn
        (same model, but temperature is controlled by the closure).
    coarse_temperature : float
        Temperature for coarse LLM call (default: 0.0).
    fine_temperature : float
        Temperature for fine-grained LLM call (default: 0.3).
    coarse_system_prompt : str | None
        System prompt for Level 1. If None, uses default.
    fine_system_prompt : str | None
        System prompt for Level 2. If None, uses template.

    Returns
    -------
    dict with keys:
        coarse_label, coarse_source, coarse_confidence,
        fine_emotion, fine_source, fine_confidence, full_path
    """
    # Lazy import to allow module to be importable without the hybrid dependency
    if coarse_classifier_fn is None:
        try:
            from agent.solvers.sentiment_hybrid import classify_sentiment_hybrid as _hybrid
            coarse_classifier_fn = _hybrid
        except ImportError as e:
            raise ImportError(
                "coarse_classifier_fn is required; could not auto-import "
                f"classify_sentiment_hybrid: {e}"
            )

    # Use separate LLM function for fine if provided
    fine_fn = fine_llm_infer_fn or llm_infer_fn

    # ── Level 1: Coarse classification ──
    coarse_result = coarse_classifier_fn(
        text=text,
        llm_infer_fn=llm_infer_fn,
        system_prompt=coarse_system_prompt or DEFAULT_COARSE_SYSTEM_PROMPT,
        llm_params={
            "temperature": coarse_temperature,
            "top_p": 0.9,
            "top_k": 20,
            "min_p": 0.05,
            "seed": 42,
        },
    )

    coarse_label: str = coarse_result.get("label", "unknown")
    coarse_source: str = coarse_result.get("source", "unknown")
    coarse_confidence: str = coarse_result.get("format_confidence", "medium")

    # ── Level 2: Fine-grained emotion extraction ──
    fine_emotion, fine_source = extract_fine_emotion(
        text=text,
        coarse_label=coarse_label,
        llm_infer_fn=fine_fn,
        temperature=fine_temperature,
    )

    # Confidence heuristic based on extraction clarity
    fine_confidence: str = "high"
    if fine_source == "unknown":
        fine_confidence = "low"
    elif fine_source == "llm_fine_partial":
        fine_confidence = "medium"
    elif fine_source == "llm_fine_typo":
        fine_confidence = "medium"

    return {
        "coarse_label": coarse_label,
        "coarse_source": coarse_source,
        "coarse_confidence": coarse_confidence,
        "fine_emotion": fine_emotion,
        "fine_source": fine_source,
        "fine_confidence": fine_confidence,
        "full_path": [
            {"level": 1, "label": coarse_label, "source": coarse_source},
            {"level": 2, "label": fine_emotion, "source": fine_source},
        ],
    }


# ============================================================================
# Validation via LLM-as-judge
# ============================================================================


def validate_emotion_is_reasonable(
    text: str,
    coarse: str,
    fine: str,
    judge_llm_fn: Callable[[str, str], str] | None = None,
) -> dict:
    """
    Validate whether the fine-grained emotion is reasonable.

    Uses a two-tier approach:
    1. **Rule-based check**: Quick heuristics (emotion matches coarse category,
       emotion appears valid for the taxonomy).
    2. **LLM-as-judge** (if *judge_llm_fn* is provided): Asks a separate LLM
       whether the fine emotion is reasonable.

    Parameters
    ----------
    text : str
        Original input text.
    coarse : str
        Coarse sentiment label.
    fine : str
        Fine-grained emotion to validate.
    judge_llm_fn : Callable | None
        Optional LLM inference function for deeper validation. If None,
        only rule-based validation is used.

    Returns
    -------
    dict
        - **reasonable** (*bool*) — whether the emotion is deemed reasonable.
        - **method** (*str*) — ``"rule"``, ``"llm"``, or ``"none"``.
        - **raw_judge_output** (*str*) — raw judge output (if LLM used).
    """
    if fine == "unknown":
        return {"reasonable": False, "method": "rule", "raw_judge_output": ""}

    # ── Rule-based check ──────────────────────────────────────────────────
    # 1. Fine emotion must belong to the coarse category's taxonomy
    valid_fines = EMOTION_TAXONOMY.get(coarse, [])
    if fine in valid_fines:
        return {"reasonable": True, "method": "rule", "raw_judge_output": ""}

    # 2. Fine emotion must at least be a known emotion somewhere
    if fine in ALL_EMOTIONS:
        return {"reasonable": True, "method": "rule", "raw_judge_output": ""}

    # 3. If we can't determine via rules, try LLM judge if available
    if judge_llm_fn is not None:
        system_prompt = "Strict judge. Reply with exactly one word: YES or NO."
        user_prompt = (
            f"Text: {text}\n"
            f"Fine emotion: {fine}\n"
            f"Coarse category: {coarse}\n"
            f"Question: Is '{fine}' a reasonable emotion for this text?\n"
            f"YES or NO."
        )

        try:
            raw_judge = judge_llm_fn(system_prompt, user_prompt)
        except Exception as e:
            logger.error(f"Judge LLM failed: {e}")
            return {"reasonable": False, "method": "error", "raw_judge_output": f"error: {e}"}

        if not raw_judge:
            return {"reasonable": False, "method": "llm", "raw_judge_output": ""}

        judge_text = raw_judge.strip().upper()
        first_token = judge_text.split()[0] if judge_text.split() else ""
        first_token = first_token.rstrip(".,!?")
        is_reasonable = first_token == "YES"

        return {
            "reasonable": is_reasonable,
            "method": "llm",
            "raw_judge_output": raw_judge.strip(),
        }

    # No judge available and rules can't decide
    return {"reasonable": False, "method": "none", "raw_judge_output": ""}
