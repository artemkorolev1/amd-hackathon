"""
agent/classifier.py — Full classification pipeline.

Chains 8-way primary scorer → secondary classifiers for known confusions.

Usage:
    from agent.classifier import classify
    category, method, confidence = classify("Fix this buggy code:")
    # → ("code_debug", "primary", 0.6)

    # Named entity recognition / NER solver (separate):
    from agent.classifier import classify_ner
    entities = classify_ner("Extract entities: {@RocketPunch members@} and Trump")
    # → {"entities": ["group: {@RocketPunch members@}", "person: Trump"], ...}
"""

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Load primary 8-way classifier (direct import to avoid broken agent/__init__.py chain) ──
_PRIMARY = None
def _get_primary():
    global _PRIMARY
    if _PRIMARY is None:
        spec = importlib.util.spec_from_file_location(
            "category_filter", os.path.join(_HERE, "category_filter.py")
        )
        _PRIMARY = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_PRIMARY)
    return _PRIMARY

# ── Load secondary classifiers ──
_SECONDARIES = {}
def _get_secondary(name):
    if name not in _SECONDARIES:
        fname = os.path.join(_HERE, f"secondary_{name}.py")
        if os.path.exists(fname):
            spec = importlib.util.spec_from_file_location(f"secondary_{name}", fname)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _SECONDARIES[name] = mod
    return _SECONDARIES.get(name)


def classify(prompt: str):
    """
    Run full classification pipeline.
    
    Returns:
        (category_8way: str, method: str, confidence: float)
        
    method is one of: "primary", "code_secondary", "reasoning_secondary", "factual_secondary"
    """
    primary = _get_primary()
    result = primary.classify_with_detail(prompt)
    category = result["category"]
    confidence = result["confidence"]
    method = "primary"
    
    # ── NEW SECONDARIES (v15+: factual-structure, code-context, NER-tweet) ──
    # These run AFTER the primary but BEFORE the existing secondaries to
    # fix the most common misroutes before they reach broader-scope logic.

    # Math→factual resolver: catch word-problem arithmetic misrouted as factual
    mod = _get_secondary("mathfact")
    if mod and category == "factual":
        corrected = mod.resolve_mathfact(category, prompt)
        if corrected != category:
            category = corrected
            method = "mathfact_secondary"

    # NER→factual: catch biomedical/tweet NER prompts that the primary misses
    mod = _get_secondary("nertweet")
    if mod and category != "ner":
        corrected = mod.resolve_nertweet(category, prompt)
        if corrected != category:
            category = corrected
            method = "nertweet_secondary"

    # factual→summarization / factual→code_gen: catch SQuAD-style QA and MCQs
    mod = _get_secondary("qa")
    if mod:
        corrected = mod.resolve_qa(category, prompt)
        if corrected != category:
            category = corrected
            method = "qa_secondary"

    # code_gen→factual: catch MCQ formatting without code structure
    mod = _get_secondary("codeguard")
    if mod and category in ("code_gen", "code_debug"):
        corrected = mod.resolve_codeguard(category, prompt)
        if corrected != category:
            category = corrected
            method = "codeguard_secondary"

    # ── EXISTING SECONDARIES ──
    mod = _get_secondary("code")
    if mod and category in ("code_debug", "code_gen"):
        corrected = mod.resolve_code(category, prompt)
        if corrected != category:
            category = corrected
            method = "code_secondary"
    
    # ── Secondary: logic vs math ──
    mod = _get_secondary("reasoning")
    if mod and category in ("logic", "math"):
        corrected = mod.resolve_reasoning(category, prompt)
        if corrected != category:
            category = corrected
            method = "reasoning_secondary"
    
    # ── Secondary: factual QA detector ──
    mod = _get_secondary("factual")
    if mod:
        corrected = mod.resolve_factual(category, prompt)
        if corrected != category:
            category = corrected
            method = "factual_secondary"
    
    # ── Secondary: summarization document-structure detector ──
    mod = _get_secondary("summarization")
    if mod:
        corrected = mod.resolve_summarization(category, prompt)
        if corrected != category:
            category = corrected
            method = "summarization_secondary"
    
    return category, method, confidence


def classify_with_detail(prompt: str) -> dict:
    """
    Cascade classify returning the same dict shape as category_filter.classify_with_detail.

    Runs primary 8-way scorer to get raw_scores + score_delta, then chains
    through secondary classifiers for corrections.

    Returns:
        dict with keys: category, category_4way, category_human, confidence,
                        raw_scores, score_delta, method
    """
    # Get primary detail first (for raw_scores)
    primary_mod = _get_primary()
    primary_detail = primary_mod.classify_with_detail(prompt)
    raw_scores = primary_detail.get("raw_scores", {})
    score_delta = primary_detail.get("score_delta", 0.0)

    # Run cascade
    category, method, confidence = classify(prompt)

    return {
        "category": category,
        "category_4way": getattr(_get_primary(), "get_4way", lambda c: "knowledge")(category),
        "category_human": getattr(_get_primary(), "get_human_name", lambda c: c)(category),
        "confidence": confidence,
        "raw_scores": raw_scores,
        "score_delta": score_delta,
        "method": method,
    }


def classify_fast(prompt: str) -> str:
    """Convenience: returns just the category string."""
    return classify(prompt)[0]


def classify_ner(prompt: str, category_hint: str = None) -> dict:
    """
    Run NER-specific solver on a prompt.
    
    Args:
        prompt: The text to extract entities from
        category_hint: Optional — if the prompt's category is known (e.g. from classify())
                       pass it here to confirm it's actually NER
    
    Returns:
        {"entities": [str], "f1": float, "method": str}
    """
    from agent.solvers.prototype_ner_v3 import solve_ner as ner_solver
    from agent.solvers.deterministic import solve_ner as old_ner_solver
    
    result = ner_solver(prompt, "ner")
    if result:
        lines = [l.strip() for l in result.split("\n") if l.strip()]
        return {"entities": lines, "raw": result, "method": "prototype_v3"}
    
    # Fallback to old solver
    result = old_ner_solver(prompt, "ner")
    if result:
        return {"entities": [result], "raw": result, "method": "old_deterministic"}
    
    return {"entities": [], "raw": "", "method": "none"}
