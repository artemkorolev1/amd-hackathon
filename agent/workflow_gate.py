"""
workflow_gate.py — Deterministic pre-check for workflow operators.

Decides which (if any) multi-step workflow to apply based on:
  - Category (from classifier)
  - Prompt features (length, structure, keyword patterns)
  - Complexity score (from complexity_scorer)

This runs AFTER complexity scoring and BEFORE the decision table.

Returns a workflow template name from agent/workflow.py TEMPLATE_REGISTRY,
or 'single_shot' for direct LLM inference.

Usage:
    from agent.workflow_gate import select_workflow

    template, config = select_workflow(prompt, category, complexity=0.5)
    if template != "single_shot":
        steps = TEMPLATE_REGISTRY[template]
        # run workflow...
"""

import re
from typing import Tuple

# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _has_multi_constraint(prompt: str) -> bool:
    """Detect 3+ distinct named entities with constraint language."""
    names = re.findall(r'\b[A-Z][a-z]+\b', prompt)
    constraints = re.findall(
        r'\b(each|different|distinct|if|unless|but|however|either|neither)\b',
        prompt.lower()
    )
    return len(set(names)) >= 3 and len(constraints) >= 2


def _multi_hop_signal_count(prompt: str) -> int:
    """Count multi-hop reasoning indicators (total matches across all patterns)."""
    signals = [
        r'\b(first|second|third|then|next|finally|after that|subsequently)\b',
        r'\b(if.*then|given.*find|assuming|suppose|derive|infer|deduce)\b',
        r'\b(compare|contrast|relationship|connection|difference|similarity)\b',
        r'\b(chain|cascade|sequence|series|multi.step|multi.hop)\b',
        r'\b(step \d|stage \d|phase \d)\b',
    ]
    return sum(len(re.findall(s, prompt, re.I)) for s in signals)


def _count_requirements(prompt: str) -> int:
    """Count explicit requirements in a code spec."""
    bullets = len(re.findall(r'^[-*]\s', prompt, re.M))
    numbers = len(re.findall(r'^\d+\.\s', prompt, re.M))
    musts = len(re.findall(
        r'\b(must|should|need to|required|shall|has to|have to)\b', prompt, re.I
    ))
    return bullets + numbers + musts


def _has_calc_verbs(prompt: str) -> bool:
    """Check if prompt contains calculation verbs (for suppressing factual false-positives)."""
    return bool(re.search(
        r'\b(calculate|compute|solve|equation|formula|derivative|integral|'
        r'sum of|difference of|product of|quotient|remainder|modulo)\b',
        prompt, re.I
    ))


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def should_split_into_steps(prompt: str, category: str, complexity: float = 0.0) -> Tuple[bool, str | None]:
    """
    Legacy alias — returns (should_split: bool, workflow_type: str or None).

    Deprecated; prefer select_workflow().
    """
    template, _config = select_workflow(prompt, category, complexity)
    return (template != "single_shot", template if template != "single_shot" else None)


def select_workflow(prompt: str, category: str, complexity: float = 0.0) -> Tuple[str, dict]:
    """
    Return (workflow_template_name, config_overrides).

    Template names match TEMPLATE_REGISTRY keys in agent/workflow.py:
      'single_shot'  — no workflow, direct LLM inference
      'math_3step'   — plan → solve → compose
      'logic_3step'  — plan → reason → compose
      'ner_2step'    — extract → verify
      'plan_solve'   — ad-hoc plan → solve (for complex code/spec)
      'analyze_answer' — analyze → compose output (for summarization/long texts)
      'verify'       — single-shot + verification step appended

    Config overrides dict may contain:
      add_verify: bool  — append a verification step to the workflow
      format: str       — output format hint (e.g. "bullets", "paragraph")
    """
    cat = category.lower().replace(" ", "_").replace("-", "_").replace("__", "_")

    # ── Math: always benefit from structured reasoning ──
    if cat in ("math", "math_reasoning", "math_arithmetic"):
        return ("math_3step", {"add_verify": complexity > 0.5})

    # ── Logic: always use plan→reason→compose ──
    if cat in ("logic", "logical_reasoning"):
        n_entities = len(set(re.findall(r'\b[A-Z][a-z]+\b', prompt)))
        return ("logic_3step", {"add_verify": n_entities >= 3})

    # ── Code generation: plan-solve for complex specs ──
    if cat in ("code_gen", "code_generation"):
        req_count = _count_requirements(prompt)
        word_count = len(prompt.split())
        if req_count >= 3 or word_count > 300:
            return ("plan_solve", {})
        return ("single_shot", {})

    # ── Code debugging: analyze for multiple errors ──
    if cat in ("code_debug", "code_debugging"):
        error_count = len(re.findall(
            r'\b(error|bug|fix|issue|traceback|not working|broken|fault)\b', prompt, re.I
        ))
        if error_count >= 2:
            return ("analyze_answer", {})
        return ("single_shot", {})

    # ── Summarization: structured output for long/analytical tasks ──
    if cat in ("summarization", "text_summarisation"):
        words = len(prompt.split())
        has_bullets = bool(re.search(
            r'\b(bullet|highlight|key point|list|numbered|outline|tl;dr)\b', prompt, re.I
        ))
        if words > 80 or has_bullets:
            return ("analyze_answer", {"format": "bullets" if has_bullets else "paragraph"})
        return ("single_shot", {})

    # ── Factual QA: verify for multi-hop questions ──
    if cat in ("factual", "factual_knowledge"):
        hop_count = _multi_hop_signal_count(prompt)
        if hop_count >= 3 and not _has_calc_verbs(prompt):
            return ("verify", {})
        return ("single_shot", {})

    # ── NER: always verify (existing workflow) ──
    if cat in ("ner", "named_entity_recognition"):
        return ("ner_2step", {})

    # ── Sentiment: verify for ambiguous/long texts ──
    if cat in ("sentiment", "sentiment_classification"):
        words = len(prompt.split())
        has_mixed = bool(re.search(
            r'\b(but|however|although|despite|nevertheless|on the other hand)\b',
            prompt, re.I
        ))
        if words > 200 or has_mixed:
            return ("analyze_answer", {})
        return ("single_shot", {})

    # ── Default: single shot ──
    return ("single_shot", {})


# ===================================================================
# Shortcut: one-call decision
# ===================================================================

def decide(prompt: str, category: str, complexity: float = 0.0) -> str:
    """
    Convenience: return the workflow template name only (no config dict).

    Example:
        >>> decide("Solve 2x + 5 = 13", "math")
        'math_3step'
        >>> decide("What is the capital of France?", "factual")
        'single_shot'
    """
    template, _config = select_workflow(prompt, category, complexity)
    return template
