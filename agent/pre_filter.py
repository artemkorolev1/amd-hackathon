"""
Pre-Filter: Deterministic pre-filter pipeline.

Filters run in strict priority order:

  Tier 0 — Immediate bypass (direct answer or deterministic solver)
  Tier 1 — Route to category (skip classifier, go to complexity)
  Tier 2 — Feature flags only (enrich downstream, never bypass)

Main entry: pre_filter(prompt: str) -> PreFilterResult
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PreFilterResult:
    """Result of Pre-Filter."""

    # What to do next: "bypass", "route_to_stage3", "continue"
    action: str = "continue"

    # Direct answer (for Tier 0 bypass)
    direct_answer: Optional[str] = None

    # Category hint (for Tier 1 routing to complexity)
    category: Optional[str] = None

    # Feature flags for downstream stages
    flags: dict[str, bool] = field(default_factory=dict)


# ============================================================
# TIER 0 — Immediate bypass (direct answer or deterministic solver)
# ============================================================

# 0A: Greetings / trivial acknowledgements
RE_GREETING = re.compile(r"^(hi|hello|hey|thanks|thank you|ok|okay)[\s!.]*$", re.I)

# 0B: Pure backtick fence — prompt IS code, no surrounding prose
RE_PURE_FENCE = re.compile(r"^```[a-z]*\n[\s\S]*?\n```\s*$")

# 0C: Pure arithmetic query — "what is 12 + 5" or "12 + 5 = ?"
RE_PURE_ARITH = re.compile(
    r"^what is\s+(\d+)\s*([+\-*/])\s*(\d+)\s*\??$", re.I
)

# 0D: Single-line function/class/method definition
RE_SINGLE_DEF = re.compile(
    r"^(def|function|class)\s+\w+\s*\(.*\).*$", re.MULTILINE
)


# ============================================================
# TIER 1 — Route to category (skip classifier, go to complexity)
# ============================================================

# 1A: Fenced code block with language tag
RE_CODE_FENCE = re.compile(r"```[a-z]+\n[\s\S]*?\n```", re.MULTILINE)

# 1B: Implicit code — def/function/class/import at line start
RE_CODE_HEADER = re.compile(r"^\s*(def|function|class|import)\b", re.MULTILINE)

# 1C: Arithmetic expression (only claims on short prompts)
RE_ARITH_EXPR = re.compile(r"\d+\s*[\+\-\*/\^\%]\s*\d+")

# 1D: Explicit summarization instruction at start
RE_SUMMARIZE = re.compile(
    r"^(summarize|summary|tl;dr|tldr|condense|key points)[:\s]", re.I
)


# ============================================================
# PRIORITY RESOLUTION
# ============================================================

# For Tier 1 conflicts — most specific pattern wins
TIER1_PRIORITY = ["fenced_code", "implicit_code", "arithmetic", "summarization"]


def _is_debug_prose(prose: str) -> bool:
    """Check prose outside code fences for debugging signals."""
    return bool(re.search(
        r"\b(debug|bug|fix|error|broken|wrong|issue|not working|incorrect)\b",
        prose, re.I,
    ))


def _refine_code_vs_debug(prompt: str) -> str:
    """Decide between code_generation and code_debugging."""
    prose = RE_CODE_FENCE.sub("", prompt) if RE_CODE_FENCE.search(prompt) else prompt
    if _is_debug_prose(prose):
        return "code_debugging"
    return "code_generation"


def _is_all_code(prompt: str) -> bool:
    """Check if prompt is predominantly code (little prose)."""
    lines = prompt.strip().split("\n")
    code_lines = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if re.search(
            r"\b(def |function |class |import |from \w+ import|return |print\()",
            stripped,
        ):
            code_lines += 1
    total_non_empty = sum(1 for l in lines if l.strip())
    if total_non_empty == 0:
        return False
    return code_lines / total_non_empty > 0.5


def pre_filter(prompt: str) -> PreFilterResult:
    """
    Run Pre-Filter in priority order.
    
    Returns a PreFilterResult directing what should happen next.
    """
    text = prompt.strip()
    if not text:
        return PreFilterResult(action="bypass", direct_answer="")

    # ── Tier 0: Immediate bypass ────────────────────────────

    # 0A: Greeting
    if RE_GREETING.match(text):
        return PreFilterResult(
            action="bypass",
            direct_answer="Hello! How can I help you today?",
        )

    # 0B: Pure backtick fence (entire prompt is a code fence)
    if RE_PURE_FENCE.match(text):
        return PreFilterResult(
            action="route_to_stage3",
            category=_refine_code_vs_debug(text),
            flags={},
        )

    # 0C: Pure arithmetic query
    m = RE_PURE_ARITH.match(text)
    if m:
        return PreFilterResult(
            action="bypass",
            direct_answer=None,  # route to deterministic solver
            category="math_arithmetic",
            flags={},
        )

    # 0D: Single-line function/class definition (predominantly code)
    if RE_SINGLE_DEF.match(text) and _is_all_code(text):
        return PreFilterResult(
            action="route_to_stage3",
            category="code_generation",
            flags={},
        )

    # ── Tier 1: Route to category (skip classifier) ────────

    tier1_matches: list[str] = []

    # 1A: Fenced code block with language tag
    if RE_CODE_FENCE.search(text):
        tier1_matches.append("fenced_code")

    # 1B: Implicit code (def/function/class/import at line start)
    if RE_CODE_HEADER.search(text) and _is_all_code(text):
        tier1_matches.append("implicit_code")

    # 1C: Arithmetic expression + short prompt
    if RE_ARITH_EXPR.search(text) and len(text.split()) < 15:
        fences = RE_CODE_FENCE.findall(text)
        if not fences or not any(RE_ARITH_EXPR.search(f) for f in fences):
            tier1_matches.append("arithmetic")

    # 1D: Explicit summarization instruction
    if RE_SUMMARIZE.match(text):
        tier1_matches.append("summarization")

    # Resolve Tier 1 conflicts
    if tier1_matches:
        chosen: Optional[str] = None
        for priority in TIER1_PRIORITY:
            if priority in tier1_matches:
                chosen = priority
                break
        if not chosen:
            chosen = tier1_matches[0]

        category_map = {
            "fenced_code": _refine_code_vs_debug(text),
            "implicit_code": "code_generation",
            "arithmetic": "math_arithmetic",
            "summarization": "summarization",
        }

        return PreFilterResult(
            action="route_to_stage3",
            category=category_map.get(chosen, chosen),
            flags={},
        )

    # ── Nothing matched ─────────────────────────────────────
    return PreFilterResult(action="continue", category=None, flags={})
