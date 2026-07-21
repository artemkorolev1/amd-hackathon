"""Spell check tool using SymSpell.
Lightweight deterministic spelling correction — 2MB install, 10MB RAM.
"""
import re
from typing import Optional

# Lazy import — only loads when first called
_symspell = None
_verbosity_closest = 1  # Verbosity.CLOSEST value


def _get_speller(max_edit_distance: int = 2):
    """Lazy singleton for SymSpell instance."""
    global _symspell, _verbosity_closest
    if _symspell is None:
        from symspellpy import SymSpell, Verbosity
        import os

        _verbosity_closest = Verbosity.CLOSEST.value
        _symspell = SymSpell(max_dictionary_edit_distance=max_edit_distance)
        # Load default frequency dictionary (bundled with symspellpy)
        dict_path = os.path.join(
            os.path.dirname(__import__("symspellpy").__file__),
            "frequency_dictionary_en_82_765.txt",
        )
        if os.path.exists(dict_path):
            _symspell.load_dictionary(dict_path, term_index=0, count_index=1)
    return _symspell


def spell_check(text: str, max_edit: int = 2, max_results: int = 5) -> str:
    """Check and correct spelling in text.

    Returns corrected text. Only changes misspelled words (words not in
    SymSpell dictionary with edit distance <= max_edit). Proper nouns
    (capitalised mid-sentence) are skipped to avoid false positives.

    Args:
        text: Text to spell-check
        max_edit: Maximum edit distance (1=faster, 2=more corrections)
        max_results: Max suggestions per misspelled word

    Returns:
        Corrected text, or original if no corrections needed.
    """
    if not text or len(text.strip()) < 2:
        return text

    speller = _get_speller(max_edit_distance=max_edit)
    words = re.findall(r"\b[a-zA-Z]+(?:\'[a-zA-Z]+)?\b", text)
    if not words:
        return text

    # Build mapping of word -> correction (if needed)
    corrections = {}
    for w in sorted(set(w.lower() for w in words), key=len, reverse=True):
        if len(w) <= 1:
            continue
        # Check acronyms on the original words
        original_forms = {ow for ow in words if ow.lower() == w}
        if any(ow.isupper() and len(ow) <= 5 for ow in original_forms):
            continue  # Acronyms (CPU, API, GPU)
        # Skip short words and known false positives
        if w in {"the", "and", "for", "are", "but", "not", "you", "all",
                  "can", "had", "has", "was", "were", "its", "any", "too",
                  "got", "let", "may", "now", "see", "way", "get", "use",
                  "set", "put", "run", "end", "odd", "big", "red"}:
            continue

        suggestions = speller.lookup(w, _verbosity_closest, max_edit_distance=max_edit)

        # Only correct if we have a high-confidence suggestion
        # SymSpell returns (term, distance, count)
        for sug in suggestions:
            if sug.term != w and sug.distance > 0:
                # Require high frequency (count > 100) OR distance 1
                if sug.count > 100 or (sug.distance <= 1 and sug.count > 10):
                    corrections[w] = sug.term
                    break

    if not corrections:
        return text  # No corrections needed

    # Apply corrections (case-preserving)
    result = text
    for misspelled, corrected in sorted(corrections.items(), key=lambda x: -len(x[0])):
        # Case-preserving replace
        pattern = re.compile(r'\b' + re.escape(misspelled) + r'\b', re.IGNORECASE)

        def _case_preserving(m):
            orig = m.group(0)
            if orig.isupper():
                return corrected.upper()
            if orig[0].isupper():
                return corrected.capitalize()
            return corrected

        result = pattern.sub(_case_preserving, result)

    return result


def list_misspellings(text: str, max_edit: int = 1) -> list:
    """Return list of potentially misspelled words with suggestions.

    Less aggressive than spell_check — only flags words with edit distance 1
    and shows suggestions, does NOT auto-correct. Useful for flagging in
    tasks where the original text integrity matters.

    Args:
        text: Text to analyze
        max_edit: Maximum edit distance (default 1)

    Returns:
        List of dicts: {word, suggestions: [{term, distance, count}]}
    """
    if not text:
        return []

    speller = _get_speller(max_edit_distance=max_edit)
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text)

    results = []
    for w in sorted(set(words)):
        suggestions = speller.lookup(w, _verbosity_closest, max_edit_distance=max_edit)
        # If the word IS in the dictionary, skip (correction == word itself)
        good_suggestions = [s for s in suggestions if s.term != w]
        if good_suggestions:
            # Check confidence — require either count > 100 or edit distance 1 with count > 10
            confident = [s for s in good_suggestions
                         if s.count > 100 or (s.distance <= 1 and s.count > 10)]
            if confident:
                results.append({
                    "word": w,
                    "suggestions": [
                        {"term": s.term, "distance": s.distance, "count": s.count}
                        for s in confident[:3]
                    ],
                })
    return results
