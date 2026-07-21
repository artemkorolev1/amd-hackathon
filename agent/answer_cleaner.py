"""Lightweight response cleaning — stub for fireworks.py compatibility."""
import re
import json


def clean_response(text: str) -> str:
    """Strip markdown fences, preamble, and trailing commentary."""
    if not text:
        return ""
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:python|json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    # Strip common preamble
    text = re.sub(r"^(Here'?s?\s+(?:the|a|an)\s+|The\s+(?:answer|result|solution)\s+(?:is|would be)\s*:?\s*)", "", text, flags=re.I)
    # Strip trailing fluff
    text = re.sub(r"\s*(\.\s*)?(Let me know|Hope this helps|I hope|Feel free).*$", "", text, flags=re.I)
    return text.strip()


def extract_json(text: str) -> str:
    """Extract valid JSON from text that may contain markdown fences or prose wrappers.

    Tries: markdown-fenced JSON block -> raw JSON object in text -> first valid JSON parse.
    Returns the JSON string on success, or empty string if no valid JSON found.
    """
    if not text:
        return ""
    text = text.strip()

    # Try to find a fenced JSON block first
    m = re.search(r'```(?:json)?\s*\n?({.*?})\n?```', text, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # Try to find any {...} object in the text
    m = re.search(r'({[^{}]*})', text, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # Try the whole text as JSON
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    return ""
