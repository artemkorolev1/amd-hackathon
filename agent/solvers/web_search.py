"""Web search tool for live factual fallback.
Uses duckduckgo_search (2MB install, network calls only).

Provides two strategies:
1. search_web() — general web search, returns top snippet results
2. search_factual() — optimised for answering factual questions
"""
import json
import re
from typing import Optional


# Lazy import — duckduckgo_search only imports when called
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from duckduckgo_search import DDGS
        _engine = DDGS()
    return _engine


def search_web(query: str, max_results: int = 5, timeout: int = 10) -> str:
    """Search the web via DuckDuckGo. Returns snippet-style results.

    Args:
        query: Search query
        max_results: Max results to return (1-10)
        timeout: Network timeout seconds

    Returns:
        Formatted result strings, or error message if search fails.
    """
    if not query or len(query.strip()) < 3:
        return "Query too short"

    try:
        ddgs = _get_engine()
        results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"[Web search error: {e}]"

    if not results:
        return "No results found"

    output = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "").strip()
        body = r.get("body", "").strip()
        href = r.get("href", "")
        if title or body:
            snippet = f"{i}. {title}\n   {body[:300]}"
            if href:
                snippet += f"\n   {href}"
            output.append(snippet)

    return "\n\n".join(output[:max_results]) if output else "No results found"


def search_factual(question: str, max_results: int = 3, timeout: int = 10) -> str:
    """Answer a factual question by searching the web.

    Optimised for factoid QA: appends "site:en.wikipedia.org" to improve
    result quality for known topics, then falls back to general search.

    Args:
        question: Factual question (e.g. "What is the capital of France?")
        max_results: Max results to return
        timeout: Network timeout seconds

    Returns:
        Answer text with sources, or error/empty message.
    """
    if not question or len(question.strip()) < 5:
        return "Question too short"

    try:
        ddgs = _get_engine()
        results = []

        # Strategy 1: Try with Wikipedia source restriction first
        wiki_query = f"{question} site:en.wikipedia.org"
        try:
            wiki_results = list(ddgs.text(wiki_query, max_results=2))
            if wiki_results and any(
                "wikipedia" in r.get("href", "") or len(r.get("body", "")) > 50
                for r in wiki_results
            ):
                # If at least one result looks substantive, use it
                results.extend(wiki_results)
        except Exception:
            pass

        # Strategy 2: General search (always, for breadth)
        try:
            general = list(ddgs.text(question, max_results=max_results))
            # De-duplicate by URL
            seen_urls = {r.get("href", "") for r in results}
            for r in general:
                if r.get("href", "") not in seen_urls:
                    results.append(r)
        except Exception:
            pass

        if not results:
            return "No factual results found"

        # Build answer
        output = []
        for i, r in enumerate(results[:max_results], 1):
            title = r.get("title", "").strip()
            body = r.get("body", "").strip()
            href = r.get("href", "")

            # Skip very short bodies
            if body and len(body) < 20:
                continue

            entry = f"[{i}] {title}"
            if body:
                entry += f"\n    {body[:500]}"
            if href and "wikipedia" in href:
                entry += f"\n    (Wikipedia)"
            output.append(entry)

        return "\n\n".join(output[:max_results]) if output else "No factual results found"

    except Exception as e:
        return f"[Web search error: {e}]"
