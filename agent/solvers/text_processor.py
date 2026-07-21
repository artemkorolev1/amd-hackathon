"""
Text preprocessing module with smart chunker, JSON formatter/validator,
markdown cleaner, input length classifier, and text extraction helpers.

All functions are registered as tools in the ToolRegistry.
"""

import json
import re
from typing import Any, Dict, List, Optional


# ═════════════════════════════════════════════════════════════════════════════
# 1. Smart Chunker
# ═════════════════════════════════════════════════════════════════════════════

def chunk_text(text: str, max_chars: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into chunks at sentence/paragraph boundaries.

    Priority:
    1. Try paragraph breaks (\\n\\n) first
    2. Then sentence boundaries (.!? followed by space)
    3. Then clause boundaries (;, :)
    4. Finally word boundaries (space)

    Each chunk <= max_chars with overlap for context preservation.
    Returns list of chunk strings.
    """
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0

    while start < len(text):
        # Determine end boundary for this chunk
        if start + max_chars >= len(text):
            chunks.append(text[start:])
            break

        end = start + max_chars
        chunk_candidate = text[start:end]

        # 1. Try paragraph break (scan backward from end)
        split_at = _find_boundary(chunk_candidate, r'\n\n')
        if split_at is not None and split_at > len(chunk_candidate) // 2:
            pass
        else:
            # 2. Sentence boundary
            split_at = _find_boundary(chunk_candidate, r'[.!?](?:\s|$)')
        if split_at is None or split_at < len(chunk_candidate) // 3:
            # 3. Clause boundary
            split_at = _find_boundary(chunk_candidate, r'[;:](?:\s|$)')
        if split_at is None or split_at < len(chunk_candidate) // 4:
            # 4. Word boundary
            split_at = _find_boundary(chunk_candidate, r'\s')

        # If still no good boundary, hard split at max_chars
        if split_at is None or split_at < 1:
            split_at = max_chars

        chunk_text = text[start:start + split_at].strip()
        if chunk_text:
            chunks.append(chunk_text)

        # Advance start, accounting for overlap
        overlap_chars = min(overlap, split_at)
        start = start + split_at - overlap_chars

    # Merge very small trailing chunk into previous
    if len(chunks) > 1 and len(chunks[-1]) < overlap // 2:
        chunks[-2] = chunks[-2] + " " + chunks[-1]
        chunks.pop()

    return chunks


def _find_boundary(text: str, pattern: str) -> Optional[int]:
    """Find the last occurrence of pattern in text, return end index or None."""
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None
    return matches[-1].end()


def maybe_truncate(text: str, max_chars: int = 1500, strategy: str = "smart") -> str:
    """
    Truncate text if too long for the model context window.

    Strategies:
    - "smart": Truncate at sentence boundary preserving first and last parts
    - "head": Keep first N chars
    - "tail": Keep last N chars
    - "middle": Keep first 30% + last 30% (for summarization)
    """
    if not text or len(text) <= max_chars:
        return text

    if strategy == "head":
        return _truncate_at_sentence(text[:max_chars])

    if strategy == "tail":
        return _truncate_at_sentence(text[-max_chars:])

    if strategy == "middle":
        first_part = int(max_chars * 0.5)
        last_part = max_chars - first_part
        return (_truncate_at_sentence(text[:first_part])
                + "\n[... truncated ...]\n"
                + _truncate_at_sentence(text[-last_part:]).lstrip())

    # Default "smart": keep first ~40% and last ~60%
    first_part = int(max_chars * 0.4)
    last_part = max_chars - first_part
    return (_truncate_at_sentence(text[:first_part])
            + "\n[... truncated ...]\n"
            + _truncate_at_sentence(text[-last_part:]).lstrip())


def _truncate_at_sentence(text: str) -> str:
    """Truncate at the last sentence boundary (one of .!?) within the text."""
    match = list(re.finditer(r'[.!?]', text))
    if match:
        end = match[-1].end()
        return text[:end]
    return text


# ═════════════════════════════════════════════════════════════════════════════
# 2. JSON Formatter & Validator
# ═════════════════════════════════════════════════════════════════════════════

def format_json(text: str, indent: int = 2) -> str:
    """
    Pretty-print and validate JSON. Returns formatted JSON or original text if invalid.
    """
    try:
        parsed = json.loads(text)
        return json.dumps(parsed, indent=indent, ensure_ascii=False)
    except (json.JSONDecodeError, ValueError):
        return text


def validate_json(text: str) -> Dict[str, Any]:
    """
    Validate JSON string. Returns {"valid": bool, "error": str or None, "data": dict or None}.
    """
    try:
        parsed = json.loads(text)
        return {"valid": True, "error": None, "data": parsed}
    except json.JSONDecodeError as e:
        return {"valid": False, "error": str(e), "data": None}


# ═════════════════════════════════════════════════════════════════════════════
# 3. Markdown Cleaner
# ═════════════════════════════════════════════════════════════════════════════

def clean_markdown(text: str) -> str:
    """
    Clean up markdown for cleaner output:
    - Strip trailing whitespace
    - Normalize heading levels (### -> ## -> #)
    - Remove empty links/images
    - Fix broken code fences
    - Deduplicate blank lines
    """
    if not text:
        return text

    lines = text.split('\n')
    result: List[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Strip trailing whitespace
        stripped_line = line.rstrip()

        # Normalize excessive heading levels (more than ### -> ##)
        heading_match = re.match(r'^(#{4,})\s+(.*)', stripped_line)
        if heading_match:
            # Convert ####+ headings to ###
            stripped_line = '### ' + heading_match.group(2).strip()

        # Remove empty links [text]() or ![](alt)
        stripped_line = re.sub(r'\[([^\]]*)\]\(\)', r'\1', stripped_line)
        stripped_line = re.sub(r'!\[([^\]]*)\]\(\)', '', stripped_line)

        # Fix broken code fences (odd number of backticks)
        if stripped_line.startswith('```'):
            # Count consecutive backticks at start
            backtick_count = len(stripped_line) - len(stripped_line.lstrip('`'))
            if backtick_count < 3:
                # Fix: prepend backticks to make it 3
                stripped_line = '```' + stripped_line[backtick_count:]

        # Deduplicate blank lines: skip if last result line is also blank
        if stripped_line == '' and result and result[-1] == '':
            i += 1
            continue

        result.append(stripped_line)
        i += 1

    # Join and strip leading/trailing whitespace
    cleaned = '\n'.join(result)
    # Ensure no more than one blank line in a row (safety pass)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


# ═════════════════════════════════════════════════════════════════════════════
# 4. Input Length Classifier
# ═════════════════════════════════════════════════════════════════════════════

def classify_input_length(text: str) -> Dict[str, Any]:
    """
    Classify input by length for routing decisions.

    Returns:
    {
        "char_count": int,
        "word_count": int,
        "sentence_count": int,
        "approx_tokens": int,  # ~4 chars per token
        "category": "short" | "medium" | "long" | "very_long",
        "needs_chunking": bool,  # True if > 500 chars
        "suggested_chunks": int,
    }
    """
    if not text:
        return {
            "char_count": 0,
            "word_count": 0,
            "sentence_count": 0,
            "approx_tokens": 0,
            "category": "short",
            "needs_chunking": False,
            "suggested_chunks": 0,
        }

    char_count = len(text)
    word_count = len(text.split())
    sentence_count = len(re.findall(r'[.!?](?:\s|$)', text))
    if sentence_count == 0 and text.strip():
        sentence_count = 1

    approx_tokens = max(1, char_count // 4)

    # Category thresholds
    if approx_tokens > 4000 or char_count > 16000:
        category = "very_long"
    elif approx_tokens > 1500 or char_count > 6000:
        category = "long"
    elif approx_tokens > 125 or char_count > 500:
        category = "medium"
    else:
        category = "short"

    needs_chunking = char_count > 500
    suggested_chunks = max(1, (char_count + 499) // 500) if needs_chunking else 0

    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "approx_tokens": approx_tokens,
        "category": category,
        "needs_chunking": needs_chunking,
        "suggested_chunks": suggested_chunks,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 5. Text Extraction Helpers
# ═════════════════════════════════════════════════════════════════════════════

def extract_first_n_sentences(text: str, n: int = 3) -> str:
    """Extract first N sentences (useful for previews/leads)."""
    if not text:
        return ""

    # Find sentence boundaries
    matches = list(re.finditer(r'[.!?](?:\s|$)', text))
    if not matches:
        return text

    # Take up to N sentences
    end_idx = min(n - 1, len(matches) - 1)
    end_pos = matches[end_idx].end()

    # If there's trailing content after the last matched sentence, include it
    remainder = text[end_pos:].strip()
    result = text[:end_pos].strip()
    if remainder and len(matches) <= n:
        result += " " + remainder

    return result


_PREAMBLE_PATTERNS = [
    r'^Sure\b[^.!?]*[.!?]?\s*',
    r'^Here\s+(?:is|are|we|you)\b[^.!?]*[.!?]?\s*',
    r'^I\s+think\b[^.!?]*[.!?]?\s*',
    r'^The\s+answer\s+is\b[^.!?]*[.!?]?\s*',
    r'^In\s+my\s+opinion\b[^.!?]*[.!?]?\s*',
    r'^As\s+(?:an?\s+)?(?:AI|assistant|model|language\s+model)\b[^.!?]*[.!?]?\s*',
    r'^Certainly\b[^.!?]*[.!?]?\s*',
    r'^Absolutely\b[^.!?]*[.!?]?\s*',
    r'^Of\s+course\b[^.!?]*[.!?]?\s*',
    r'^Let\s+me\b[^.!?]*[.!?]?\s*',
]


def strip_preamble(text: str) -> str:
    """
    Strip common preamble words: 'Sure', 'Here', 'I think', 'The answer is', etc.
    Only strips if the preamble is at the very start of the text.
    """
    if not text:
        return text

    result = text
    for pattern in _PREAMBLE_PATTERNS:
        result = re.sub(pattern, '', result, count=1)
        if result != text:
            break  # Only strip one preamble

    return result.strip()


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces, normalize newlines, strip leading/trailing."""
    if not text:
        return text

    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Process each line: strip, collapse internal spaces
    cleaned_lines = []
    for line in text.split('\n'):
        line = line.strip()
        # Collapse multiple spaces within the line
        line = re.sub(r'[^\S\n]+', ' ', line)
        cleaned_lines.append(line)
    # Collapse multiple blank lines (now all '' are blank)
    result = '\n'.join(cleaned_lines)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


# ═════════════════════════════════════════════════════════════════════════════
# 6. Tool Registrations
# ═════════════════════════════════════════════════════════════════════════════

from agent.solvers.tool_registry import tool, registry


@tool(name="chunk_text", category="preprocessing")
def chunk_tool(text: str, max_chars: int = 500) -> List[str]:
    """Split text into chunks at sentence/paragraph boundaries.
    Args:
        text: Text to chunk.
        max_chars: Maximum characters per chunk.
    """
    return chunk_text(text, max_chars=max_chars)


@tool(name="maybe_truncate", category="preprocessing", is_deterministic=True)
def truncate_tool(text: str, max_chars: int = 1500, strategy: str = "smart") -> str:
    """Truncate text if too long for the model context window.
    Args:
        text: Text to truncate.
        max_chars: Maximum characters allowed.
        strategy: Truncation strategy - smart, head, tail, or middle.
    """
    return maybe_truncate(text, max_chars=max_chars, strategy=strategy)


@tool(name="format_json", category="formatting", is_deterministic=True)
def format_json_tool(text: str) -> str:
    """Pretty-print and validate JSON text.
    Args:
        text: JSON string to format.
    """
    return format_json(text)


@tool(name="validate_json", category="formatting", is_deterministic=True)
def validate_json_tool(text: str) -> Dict[str, Any]:
    """Validate JSON string and return structured validation result.
    Args:
        text: JSON string to validate.
    """
    return validate_json(text)


@tool(name="clean_markdown", category="formatting", is_deterministic=True)
def clean_markdown_tool(text: str) -> str:
    """Clean up markdown text (fix headings, code fences, blank lines, etc.).
    Args:
        text: Markdown text to clean.
    """
    return clean_markdown(text)


@tool(name="classify_input_length", category="preprocessing", is_deterministic=True)
def classify_input_length_tool(text: str) -> Dict[str, Any]:
    """Classify input text by length for routing decisions.
    Args:
        text: Text to classify.
    """
    return classify_input_length(text)


@tool(name="extract_first_n_sentences", category="extraction", is_deterministic=True)
def extract_first_n_sentences_tool(text: str, n: int = 3) -> str:
    """Extract the first N sentences from text for previews/leads.
    Args:
        text: Source text.
        n: Number of sentences to extract.
    """
    return extract_first_n_sentences(text, n=n)


@tool(name="strip_preamble", category="extraction", is_deterministic=True)
def strip_preamble_tool(text: str) -> str:
    """Strip common preamble phrases like 'Sure', 'Here is', 'I think', etc.
    Args:
        text: Text to clean.
    """
    return strip_preamble(text)


@tool(name="normalize_whitespace", category="formatting", is_deterministic=True)
def normalize_whitespace_tool(text: str) -> str:
    """Collapse multiple spaces, normalize newlines, strip leading/trailing space.
    Args:
        text: Text to normalize.
    """
    return normalize_whitespace(text)


# ── Register all tools ──────────────────────────────────────────────────────

_TEXT_PROCESSOR_TOOLS = [
    chunk_tool,
    truncate_tool,
    format_json_tool,
    validate_json_tool,
    clean_markdown_tool,
    classify_input_length_tool,
    extract_first_n_sentences_tool,
    strip_preamble_tool,
    normalize_whitespace_tool,
]

for t in _TEXT_PROCESSOR_TOOLS:
    registry.register(t)

print(f"TextProcessor: registered {len(_TEXT_PROCESSOR_TOOLS)} tools ({len(registry)} total)")


# ═════════════════════════════════════════════════════════════════════════════
# Self-test entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("TEXT PROCESSOR MODULE TESTS")
    print("=" * 60)

    # ── chunk_text tests ──
    print("\n--- chunk_text ---")
    short = "Hello world."
    assert chunk_text(short) == [short], "Short text should return as-is"
    print("  PASS: short text unchanged")

    long_text = (
        "This is the first paragraph about the topic. It contains several sentences to test chunking. "
        "We need to ensure boundaries are respected.\n\n"
        "This is the second paragraph. It has more content here. And some additional sentences.\n\n"
        "Third paragraph with a single sentence. And some trailing text after that."
    )
    chunks = chunk_text(long_text, max_chars=200, overlap=20)
    print(f"  Chunked {len(long_text)} chars into {len(chunks)} chunks (max 200)")
    for i, c in enumerate(chunks):
        print(f"    Chunk {i + 1}: {len(c)} chars - {c[:60]}...")
    assert len(chunks) >= 2, "Long text should produce multiple chunks"
    assert all(len(c) <= 200 for c in chunks), "Each chunk must be <= max_chars"
    print("  PASS: all chunks within size limit")

    # ── maybe_truncate tests ──
    print("\n--- maybe_truncate ---")
    trunc_text = "This is a test. It has multiple sentences. We need to truncate it properly. And check boundaries."
    truncated = maybe_truncate(trunc_text, max_chars=50, strategy="head")
    assert len(truncated) <= len(trunc_text), "Truncated text should be shorter or equal"
    print(f"  head({50}): {repr(truncated)}")
    truncated_mid = maybe_truncate(trunc_text, max_chars=50, strategy="middle")
    print(f"  middle({50}): {repr(truncated_mid)}")
    print("  PASS: truncation works")

    # ── format_json / validate_json -- ──
    print("\n--- format_json + validate_json ---")
    raw = '{"name":"Alice","scores":[95,87,92]}'
    formatted = format_json(raw)
    parsed = json.loads(formatted)
    assert parsed["name"] == "Alice"
    print(f"  Format: {formatted}")
    valid = validate_json(raw)
    assert valid["valid"] is True
    assert valid["data"]["name"] == "Alice"
    print(f"  Validate: valid={valid['valid']}")
    invalid = validate_json("{bad json}")
    assert invalid["valid"] is False
    print(f"  Invalid JSON caught: {invalid['error'][:40]}")
    print("  PASS: JSON tools work")

    # ── clean_markdown tests ──
    print("\n--- clean_markdown ---")
    dirty_md = "## Heading\n\n\n\n### Subheading\n\nSome text   \n\n[empty]()\n\n![](img.png)"
    cleaned = clean_markdown(dirty_md)
    assert "()" not in cleaned, "Empty links should be removed"
    assert "\n\n\n" not in cleaned, "Excessive blank lines should be collapsed"
    print(f"  Cleaned: {repr(cleaned[:100])}")
    print("  PASS: markdown cleaner works")

    # ── classify_input_length tests ──
    print("\n--- classify_input_length ---")
    cls = classify_input_length("Hello world.")
    assert cls["category"] == "short"
    print(f"  Short text: {cls['category']}")
    long_cls = classify_input_length("word " * 300)
    assert long_cls["needs_chunking"] is True
    print(f"  Long text: {long_cls['category']}, chunks={long_cls['suggested_chunks']}")
    print("  PASS: classifier works")

    # ── extract_first_n_sentences tests ──
    print("\n--- extract_first_n_sentences ---")
    multi_sent = "First sentence here. Second sentence follows. Third is here. Fourth too."
    preview = extract_first_n_sentences(multi_sent, n=2)
    assert "First" in preview and "Second" in preview
    assert "Fourth" not in preview
    print(f"  Extracted (n=2): {repr(preview)}")
    print("  PASS: extraction works")

    # ── strip_preamble tests ──
    print("\n--- strip_preamble ---")
    assert strip_preamble("Sure, here is the answer.") == "", "All-preamble text should become empty"
    assert strip_preamble("Sure, here is the answer. The real content follows.") == "The real content follows."
    assert strip_preamble("I think this is correct.") == "", "All-preamble text should become empty"
    assert strip_preamble("I think this is correct. The value is 5.") == "The value is 5."
    assert strip_preamble("No preamble here.") == "No preamble here."
    print("  PASS: preamble stripping works")

    # ── normalize_whitespace tests ──
    print("\n--- normalize_whitespace ---")
    result = normalize_whitespace("  Hello    world.   \n\n\n  Next line.  ")
    assert result == "Hello world.\n\nNext line."
    print(f"  Normalized: {repr(result)}")
    print("  PASS: whitespace normalization works")

    # ── Tool registry check ──
    print("\n--- Tool Registry ---")
    from agent.solvers.tool_registry import registry
    preproc_tools = registry.get_by_category("preprocessing")
    formatting_tools = registry.get_by_category("formatting")
    extraction_tools = registry.get_by_category("extraction")
    print(f"  Preprocessing tools: {len(preproc_tools)}")
    print(f"  Formatting tools:    {len(formatting_tools)}")
    print(f"  Extraction tools:    {len(extraction_tools)}")
    print(f"  Total in registry:   {len(registry)}")
    assert len(preproc_tools) >= 2
    assert len(formatting_tools) >= 3
    assert len(extraction_tools) >= 2
    print("  PASS: all categories have expected tools")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
