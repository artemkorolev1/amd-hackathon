"""
Hierarchical Classifier — secondary classifiers for known 8-way confusions.

Architecture:
  8-way primary (category_filter.py) → classifies into 8 categories
    → if prediction is code_debug/code_gen:
        → ML binary (preferred) + deterministic fallback
    → if prediction is logic/math:
        → deterministic reasoner disambiguates
    → if prediction is factual:
        → factual QA detector checks for SQuAD/context patterns
    → if prediction is logic/math but looks factual:
        → factual detector overrides (bidirectional)

Secondary modules:
  - agent/secondary_code.py       (deterministic + ML wrapper)
  - agent/secondary_reasoning.py  (deterministic logic vs math)
  - agent/secondary_factual.py    (deterministic factual QA detector)
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Secondary module imports
try:
    from agent.secondary_code import resolve_code as secondary_code
    CODE_AVAILABLE = True
except ImportError:
    CODE_AVAILABLE = False
    logger.warning("secondary_code.py not available")

try:
    from agent.secondary_reasoning import resolve_reasoning as secondary_reasoning
    REASONING_AVAILABLE = True
except ImportError:
    REASONING_AVAILABLE = False
    logger.warning("secondary_reasoning.py not available")

try:
    from agent.secondary_factual import resolve_factual as secondary_factual
    FACTUAL_AVAILABLE = True
except ImportError:
    FACTUAL_AVAILABLE = False
    logger.warning("secondary_factual.py not available")


def classify(prompt: str) -> Tuple[str, str, float]:
    """
    Run 8-way primary then pass through secondary classifiers.
    Returns (category, method, confidence).
    """
    from agent.category_filter import classify_with_detail
    
    # Primary 8-way
    result = classify_with_detail(prompt)
    primary_cat = result["category"]
    primary_conf = result["confidence"]
    method = "primary"
    
    # ── Secondary: code debug vs gen ──
    if CODE_AVAILABLE and primary_cat in ("code_debug", "code_gen"):
        corrected = secondary_code(primary_cat, prompt)
        if corrected != primary_cat:
            primary_cat = corrected
            method = "code_secondary"
    
    # ── Secondary: logic vs math ──
    if REASONING_AVAILABLE and primary_cat in ("logic", "math"):
        corrected = secondary_reasoning(primary_cat, prompt)
        if corrected != primary_cat:
            primary_cat = corrected
            method = "reasoning_secondary"
    
    # ── Secondary: factual QA detector (bidirectional) ──
    if FACTUAL_AVAILABLE:
        corrected = secondary_factual(primary_cat, prompt)
        if corrected != primary_cat:
            primary_cat = corrected
            method = "factual_secondary"
    
    return primary_cat, method, primary_conf


def classify_fast(prompt: str) -> str:
    """Convenience wrapper — returns just the category string."""
    cat, _, _ = classify(prompt)
    return cat
