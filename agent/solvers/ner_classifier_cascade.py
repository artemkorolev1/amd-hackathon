"""
Binary reject cascade for routing NER prompts to the correct solver tool.

Each level is a high-precision binary classifier that checks if a prompt
matches a specific NER subtype. If yes, it routes to the corresponding
solver; if no, it falls through to the next level.

Cascade structure:
  Level 1: Has {@...@} markers?           → prototype_ner_v3 (handles tweet-style NER)
  Level 2: Has explicit extraction keywords → spaCy NER (general NER extraction)
  Level 3: Otherwise                       → None (LLM handles it)

All classifiers are pure deterministic regex/heuristic — zero model calls.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

from agent.solvers.prototype_ner_v3 import solve_ner as solve_ner_v3

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Level 1: Tweet-style NER with {@...@} markers
# ═══════════════════════════════════════════════════════════════════════════

_TWEET_MARKER_PATTERN = re.compile(r'\{@[^@]+@\}')

# Negative guards — passages that happen to contain {@ but aren't NER tasks
_TWEET_MARKER_NEGATIVE = [
    r"\bexample\s*\{@",
    r"\bformat\s*\{@",
    r"\boutput\s*\{@",
]


def has_tweet_markers(prompt: str) -> Tuple[bool, float]:
    """Detect NER prompts with {@...@} annotated markers (tweet-style).
    
    Returns (True, confidence) if the prompt contains {@...@} markers.
    """
    text = prompt.lower()
    
    # Negative guards
    for neg in _TWEET_MARKER_NEGATIVE:
        if re.search(neg, text):
            return False, 0.0
    
    matches = _TWEET_MARKER_PATTERN.findall(prompt)
    if matches:
        # Confidence based on number of markers found
        confidence = min(0.7 + len(matches) * 0.1, 0.98)
        return True, confidence
    
    return False, 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Level 2: General NER extraction prompts
# ═══════════════════════════════════════════════════════════════════════════

_NER_EXTRACTION_PATTERNS = [
    # Explicit extraction keywords
    r"extract\s+(?:all\s+)?(?:named\s+)?entities?",
    r"extract\s+(?:all\s+)?(?:the\s+)?(?:names?\s+of\s+)?(?:people|persons?|organizations?|locations?|companies?)",
    r"list\s+(?:all\s+)?(?:named\s+)?entities?",
    r"find\s+(?:all\s+)?(?:named\s+)?entities?",
    r"identify\s+(?:all\s+)?(?:named\s+)?entities?",
    r"named\s+entity\s+recognition",
    r"named\s+entities?\s+in\s+(?:the\s+)?(?:following\s+)?text",
    r"what\s+(?:are\s+the|is\s+the)\s+(?:named\s+)?entities?",
    r"types?\s+of\s+entities?\s+(?:in|from|found)",
    r"entities?\s+extraction",
    r"ner\s*(?:task|extraction|recognition)",
]

# Patterns that indicate the prompt is NOT a general NER extraction
_NER_NEGATIVE = [
    r"\b(?:how|what|why)\s+(?:many|much|is|are|do|does|was|were)\b",
    r"\bsolve\b",
    r"\bclassif",
    r"\bsummariz",
    r"\btranslat",
    r"\bcorrect\b",
    r"\brewrite\b",
    r"\bparaphras",
    r"\bcode\b",
    r"\bdebug\b",
    r"\bfix\s+(?:the\s+)?bug\b",
    r"\bfunction\b",
    r"\bpython\b",
    r"\breview\b",
    r"\btruth\b",
    r"\bhouse\b",
    r"\b(?:first|second|third|fourth)\s+house\b",
    r"\bthere\s+are\s+\d+\s+houses?\b",
]


def is_ner_extraction(prompt: str) -> Tuple[bool, float]:
    """Detect general NER extraction prompts.
    
    Returns (True, confidence) if the prompt asks for named entity extraction.
    """
    text = prompt.lower()
    
    # Negative guards
    for neg in _NER_NEGATIVE:
        if re.search(neg, text):
            return False, 0.0
    
    # Check for NER keywords
    matches = 0
    for pat in _NER_EXTRACTION_PATTERNS:
        if re.search(pat, text):
            matches += 1
    
    if matches == 0:
        return False, 0.0
    
    # Check for text to extract from
    has_capitalized = bool(re.findall(r'\b[A-Z][a-z]{2,}\b', prompt))
    
    confidence = min(0.5 + matches * 0.2, 0.9)
    if has_capitalized:
        confidence = min(confidence + 0.15, 0.95)
    
    return confidence >= 0.6, confidence


# Load spaCy once at module level
_SPACY_NLP = None


def _get_spacy():
    """Get or load spaCy model (lazy-loaded)."""
    global _SPACY_NLP
    if _SPACY_NLP is None:
        try:
            import spacy
            _SPACY_NLP = spacy.load("en_core_web_sm")
        except Exception:
            _SPACY_NLP = False  # Sentinel for failed load
    return _SPACY_NLP if _SPACY_NLP else None


def _solve_spacy_ner(prompt: str) -> Optional[str]:
    """Use spaCy NER to extract entities from the text portion of a prompt.
    
    Tries to extract the actual text after the instruction, runs spaCy NER,
    and formats the output as TYPE: entity lines.
    """
    nlp = _get_spacy()
    if nlp is None:
        # Fall back to prototype_ner_v3 if spaCy unavailable
        return solve_ner_v3(prompt, "ner")
    
    # Try to extract the text portion after the instruction
    text_to_analyze = prompt
    
    # Remove common instruction prefixes
    patterns = [
        r"Extract\s+all\s+named\s+entities\s+from\s+the\s+following\s+text\s*:\s*",
        r"Extract\s+entities?\s*:\s*",
        r"List\s+(?:all\s+)?(?:the\s+)?named\s+entities?\s*:\s*",
        r"Identify\s+(?:all\s+)?(?:the\s+)?named\s+entities?\s*:\s*",
        r"Find\s+(?:all\s+)?(?:the\s+)?named\s+entities?\s*:\s*",
        r"Named\s+entity\s+recognition\s*:\s*",
    ]
    for pat in patterns:
        text_to_analyze = re.sub(pat, "", text_to_analyze, flags=re.IGNORECASE).strip()
    
    if not text_to_analyze or len(text_to_analyze) < 5:
        text_to_analyze = prompt
    
    # Run spaCy NER
    doc = nlp(text_to_analyze)
    
    # Map spaCy entity types to expected output format
    type_mapping = {
        "PERSON": "person",
        "NORP": "norp",
        "FAC": "fac",
        "ORG": "org",
        "GPE": "gpe",
        "LOC": "loc",
        "PRODUCT": "product",
        "EVENT": "event",
        "WORK_OF_ART": "creative_work",
        "LAW": "law",
        "LANGUAGE": "language",
        "DATE": "date",
        "TIME": "time",
        "PERCENT": "percent",
        "MONEY": "money",
        "QUANTITY": "quantity",
        "ORDINAL": "ordinal",
        "CARDINAL": "cardinal",
    }
    
    lines = []
    seen = set()
    for ent in doc.ents:
        # Normalize type
        etype = type_mapping.get(ent.label_, ent.label_.lower())
        # Normalize entity text
        entity = ent.text.strip().rstrip(".,!?;:")
        if not entity:
            continue
        
        key = (etype, entity.lower().strip())
        if key in seen:
            continue
        seen.add(key)
        
        lines.append(f"{etype}: {entity}")
    
    if not lines:
        return None
    
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Routing function
# ═══════════════════════════════════════════════════════════════════════════

_ROUTE_NAMES = {
    1: "tweet_ner",
    2: "general_ner",
}


def route_ner(prompt: str) -> Optional[str]:
    """Walk the binary reject cascade for NER and return the first solver's output.
    
    Args:
        prompt: The full prompt text.
    
    Returns:
        Solver output string, or None if no solver matched (LLM fallback).
    """
    # Level 1: Tweet-style NER with {@...@} markers
    matched, conf = has_tweet_markers(prompt)
    if matched:
        logger.debug(f"NER Cascade Level 1 (tweet_ner, conf={conf:.2f})")
        result = solve_ner_v3(prompt, "ner")
        if result is not None:
            return result

    # Level 2: General NER extraction → spaCy
    matched, conf = is_ner_extraction(prompt)
    if matched:
        logger.debug(f"NER Cascade Level 2 (general_ner, conf={conf:.2f})")
        result = _solve_spacy_ner(prompt)
        if result is not None:
            return result

    # Fallback — return None for LLM to handle
    return None
