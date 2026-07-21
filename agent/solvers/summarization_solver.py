#!/usr/bin/env python3
"""
summarization_solver.py — Chunk-and-summarize pipeline for long texts.

Splits long texts into sentence-boundary-aligned chunks (~200 words),
summarizes each chunk, then merges into a final coherent summary.
Falls back to single-pass LLM summarization for short texts.

Usage:
    from agent.solvers.summarization_solver import chunk_text, summarize_workflow

    result = summarize_workflow(
        text="long article here...",
        llm_infer_fn=my_llm_fn,
        system_prompt="Summarize concisely.",
        chunk_prompt="Summarize this section briefly.",
        merge_prompt="Combine these summaries into one coherent summary.",
    )
    print(result["summary"])       # final summary
    print(result["chunks"])        # list of text chunks
    print(result["chunk_summaries"])  # per-chunk summaries
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Optional

# ── Chunking ──────────────────────────────────────────────────────────────────


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex."""
    # Preserve common abbreviations to avoid over-splitting
    text_clean = re.sub(r"\b([A-Z][a-z]?)\.(?=\s+[A-Z])", r"\1<DOT>", text)
    sentences = re.split(r"(?<=[.!?])\s+", text_clean)
    result = []
    for s in sentences:
        s = s.replace("<DOT>", ".").strip()
        if s:
            result.append(s)
    return result


def chunk_text(text: str, max_words: int = 200) -> list[str]:
    """Split text into ~max_words chunks at sentence boundaries.

    Args:
        text: Input text to chunk.
        max_words: Target maximum words per chunk.

    Returns:
        List of text chunks (each is 1+ sentences, roughly max_words).
    """
    if not text or not text.strip():
        return []

    sentences = _split_sentences(text.strip())
    if not sentences:
        return [text.strip()]

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_word_count = 0

    for sent in sentences:
        sent_words = len(sent.split())

        # If adding this sentence would exceed max_words AND we already have content,
        # finalize the current chunk and start a new one
        if current_word_count + sent_words > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_word_count = sent_words
        else:
            current_chunk.append(sent)
            current_word_count += sent_words

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Edge case: if a single sentence exceeds max_words, keep it as its own chunk
    # (we can't split mid-sentence meaningfully without NLP tools)
    return chunks


def _estimate_tokens(text: str) -> int:
    """Rough estimate: ~1.3 tokens per word for English text."""
    return int(len(text.split()) * 1.3)


# ── Summarization workflow ──────────────────────────────────────────────────


# Default prompts
DEFAULT_SYSTEM_PROMPT = "Summarize the following text concisely."
DEFAULT_CHUNK_PROMPT = "Summarize this section in 1-2 sentences. Include key names, numbers, and facts."
DEFAULT_MERGE_PROMPT = (
    "Combine these section summaries into a coherent 2-3 sentence summary. "
    "Preserve the most important facts, names, and numbers. Make it read naturally."
)


def summarize_workflow(
    text: str,
    llm_infer_fn: Callable,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    chunk_prompt: str = DEFAULT_CHUNK_PROMPT,
    merge_prompt: str = DEFAULT_MERGE_PROMPT,
    max_words_per_chunk: int = 200,
    max_tokens_per_call: int = 128,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Multi-step summarization:
    1. Split text into chunks at sentence boundaries
    2. Summarize each chunk
    3. Merge chunk summaries into final summary

    For short texts (< max_words_per_chunk words): single direct LLM call.

    Args:
        text: Input text to summarize.
        llm_infer_fn: Callable(messages) -> str. Messages is a list of dicts
            with 'role' and 'content'.
        system_prompt: System prompt for all LLM calls.
        chunk_prompt: System prompt used for per-chunk summarization.
        merge_prompt: System prompt used for merging chunk summaries.
        max_words_per_chunk: Maximum words per chunk.
        max_tokens_per_call: Max tokens for each LLM response.
        temperature: Temperature for LLM calls.

    Returns:
        dict with keys:
            - summary: final merged summary (str)
            - chunks: list of text chunks
            - chunk_summaries: list of per-chunk summaries
            - method: "chunk_and_merge" or "direct"
            - timing: dict of timing info
            - num_chunks: int
    """
    start_time = time.time()
    result: dict[str, Any] = {
        "summary": "",
        "chunks": [],
        "chunk_summaries": [],
        "method": "direct",
        "timing": {},
        "num_chunks": 0,
    }

    word_count = len(text.split())

    # ── Short text: direct LLM call ──────────────────────────────────────
    if word_count < max_words_per_chunk:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
        t0 = time.time()
        summary = llm_infer_fn(messages, max_tokens=max_tokens_per_call, temperature=temperature)
        result["summary"] = (summary or "").strip()
        result["method"] = "direct"
        result["timing"]["total"] = time.time() - t0
        result["num_chunks"] = 1
        return result

    # ── Long text: chunk-and-merge ───────────────────────────────────────
    result["method"] = "chunk_and_merge"

    # Step 1: Chunk
    t0 = time.time()
    chunks = chunk_text(text, max_words=max_words_per_chunk)
    result["chunks"] = chunks
    result["num_chunks"] = len(chunks)
    result["timing"]["chunking"] = time.time() - t0

    # Step 2: Summarize each chunk
    chunk_summaries: list[str] = []
    timing_chunks: list[float] = []
    for i, chunk in enumerate(chunks):
        messages = [
            {"role": "system", "content": chunk_prompt},
            {"role": "user", "content": chunk},
        ]
        t1 = time.time()
        summary = llm_infer_fn(messages, max_tokens=max_tokens_per_call, temperature=temperature)
        elapsed = time.time() - t1
        timing_chunks.append(elapsed)
        chunk_summary = (summary or "").strip()
        chunk_summaries.append(chunk_summary)

    result["chunk_summaries"] = chunk_summaries
    result["timing"]["per_chunk"] = timing_chunks
    result["timing"]["chunk_summarization"] = sum(timing_chunks)

    # Step 3: Merge chunk summaries
    if len(chunk_summaries) == 1:
        # Only one chunk — use its summary directly
        result["summary"] = chunk_summaries[0]
    else:
        # Build merge input
        merge_input_lines = []
        for i, cs in enumerate(chunk_summaries, 1):
            merge_input_lines.append(f"Section {i}: {cs}")
        merge_input = "\n\n".join(merge_input_lines)

        messages = [
            {"role": "system", "content": merge_prompt},
            {"role": "user", "content": merge_input},
        ]
        t2 = time.time()
        final_summary = llm_infer_fn(messages, max_tokens=max_tokens_per_call + 32, temperature=temperature)
        result["summary"] = (final_summary or "").strip()
        result["timing"]["merge"] = time.time() - t2

    result["timing"]["total"] = time.time() - start_time
    return result


# ── Convenience: build an llm_infer_fn from a loaded Llama model ────────────


def build_llm_infer(llm_model) -> Callable:
    """Wrap a llama_cpp.Llama instance into a simple callable.

    Returns a function that accepts (messages, max_tokens, temperature) and returns
    the response text.
    """

    def infer_fn(messages: list[dict], max_tokens: int = 128, temperature: float = 0.0) -> str:
        try:
            kwargs = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "top_k": 40,
                "min_p": 0.0,
                "repeat_penalty": 1.0,
            }
            resp = llm_model.create_chat_completion(**kwargs)
            return resp["choices"][0]["message"]["content"] or ""
        except Exception as e:
            return f""

    return infer_fn
