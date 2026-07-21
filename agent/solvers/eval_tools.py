#!/usr/bin/env python3
"""
Tool Evaluation Harness — runs all registered deterministic tools against
labeled evaluation datasets and produces an accuracy report.

Usage:
    python agent/solvers/eval_tools.py                          # default: validation sets
    python agent/solvers/eval_tools.py --all                    # all available eval sets
    python agent/solvers/eval_tools.py --category math          # single category
    python agent/solvers/eval_tools.py --tool math_solve        # single tool
    python agent/solvers/eval_tools.py --report report.json     # save report
    python agent/solvers/eval_tools.py --filters                # use deterministic pre-filters
"""
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Category → Tool name mapping ──────────────────────────────────────────
# Each category maps to one or more tools that should attempt it.
CATEGORY_TOOLS = {
    "sentiment":      ["sentiment_analysis"],
    "summarization":  ["summarize"],
    "math":           ["math_solve"],
    "ner":            ["ner_extract"],
    "factual":        ["factual_qa", "spell_check", "search_factual"],
    "logic":          ["solve_logical_reasoning", "solve_logic_puzzle", "solve_syllogism",
                       "solve_truth_teller_liar", "solve_number_sequence"],
    "code_debug":     ["code_debug"],
    "code_gen":       ["code_gen_templates"],
}

# Answer-producing tools (their output can be compared to expected_answer)
ANSWER_TOOLS = {
    "sentiment_analysis", "summarize", "math_solve", "ner_extract",
    "factual_qa",
    "solve_logic_puzzle", "solve_syllogism", "solve_truth_teller_liar",
    "solve_number_sequence", "solve_logical_reasoning",
    "code_debug", "code_gen_templates",
}

# Tools that can be sanity-checked but not accuracy-compared
SANITY_TOOLS = {
    "spell_check", "list_misspellings",
    "search_web", "search_factual",
    "format_python", "execute_code_safe",
    "format_csv", "text_stats", "reverse_text", "top_words",
    "to_leetspeak", "is_palindrome", "days_until_april_fools",
    "weather_hot_take", "to_emoji", "flip_coin",
}

# ── Eval datasets ─────────────────────────────────────────────────────────
EVAL_SETS = [
    ("data/eval/validation-v1.json",      "validation-v1",     200, None),
    ("data/eval/validation-v2.json",      "validation-v2",     200, None),
    ("data/eval/validation-v3.json",      "validation-v3",     200, None),
    ("data/eval/primary/eval_60_medium_hard.json", "eval_60_mh", 60, None),
    ("data/eval/primary/eval_hard_218.json",       "eval_hard", 218, None),
    ("data/eval/math_combined_80.json",   "math_combined",     80, "math_solve"),
    ("data/eval/factual_combined_80.json","factual_combined",   80, None),
    ("data/eval/tests/gsm8k_100.json",    "gsm8k_100",         100, "math_solve"),
    ("data/eval/tests/sst2_100.json",     "sst2_100",          100, "sentiment_analysis"),
]


# ═════════════════════════════════════════════════════════════════════════════
# Matching
# ═════════════════════════════════════════════════════════════════════════════

def _normalize(text: str) -> str:
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(text.lower().split())


def _match(output: str, expected: str) -> bool:
    if not output or not expected:
        return False
    o_norm = _normalize(output)
    e_norm = _normalize(expected)
    if not o_norm or not e_norm:
        return False
    if o_norm == e_norm:
        return True
    if e_norm in o_norm or o_norm in e_norm:
        return True
    # Number comparison
    o_nums = re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', o_norm)
    e_nums = re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', e_norm)
    if o_nums and e_nums:
        if len(o_nums) == 1 and len(e_nums) == 1:
            try:
                o_f = float(o_nums[0].split('/')[0]) / float(o_nums[0].split('/')[1]) if '/' in o_nums[0] else float(o_nums[0])
                e_f = float(e_nums[0].split('/')[0]) / float(e_nums[0].split('/')[1]) if '/' in e_nums[0] else float(e_nums[0])
                if abs(o_f - e_f) < 0.01:
                    return True
            except (ValueError, ZeroDivisionError):
                pass
        if o_nums == e_nums:
            return True
    # Token overlap (fuzzy)
    o_tokens = set(o_norm.split())
    e_tokens = set(e_norm.split())
    if o_tokens and e_tokens:
        overlap = len(o_tokens & e_tokens)
        smaller = min(len(o_tokens), len(e_tokens))
        if smaller > 0 and overlap / smaller > 0.6:
            o_content = {t for t in o_tokens if len(t) > 3}
            e_content = {t for t in e_tokens if len(t) > 3}
            if len(e_content) > 0:
                content_overlap = len(o_content & e_content) / len(e_content)
                if content_overlap > 0.5:
                    return True
    return False


# ═════════════════════════════════════════════════════════════════════════════
# Tool runner
# ═════════════════════════════════════════════════════════════════════════════

_TOOL_CACHE = {}


def _get_tool(name: str):
    if name not in _TOOL_CACHE:
        from agent.solvers.tool_registry import registry
        tool = registry.get(name)
        if tool:
            _TOOL_CACHE[name] = tool
    return _TOOL_CACHE.get(name)


# Determine which parameter name each tool expects
_TOOL_PARAMS = {
    # Answer tools
    "sentiment_analysis": "text",
    "summarize": "text",
    "math_solve": "expression",
    "ner_extract": "text",
    "factual_qa": "question",
    "solve_logic_puzzle": "prompt",
    "solve_syllogism": "premises",
    "solve_truth_teller_liar": "prompt",
    "solve_number_sequence": "prompt",
    "solve_logical_reasoning": "prompt",
    # Code tools
    "format_python": "code",
    "execute_code_safe": "code",
    "code_debug": "task",
    "code_gen_templates": "task",
    # Spell tools
    "spell_check": "text",
    "list_misspellings": "text",
    # Search tools
    "search_web": "query",
    "search_factual": "question",
    # Fun tools
    "format_csv": "text",
    "text_stats": "text",
    "reverse_text": "text",
    "top_words": "text",
    "to_leetspeak": "text",
    "is_palindrome": "text",
    "flip_coin": None,
    "days_until_april_fools": None,
    "weather_hot_take": "text",
    "to_emoji": "text",
}


def run_tool_on_prompt(tool_name: str, prompt: str) -> Optional[str]:
    """Run a single tool on a prompt. Returns output string or None on error."""
    try:
        tool = _get_tool(tool_name)
        if not tool:
            return None
        param = _TOOL_PARAMS.get(tool_name, "text")
        if param is None:
            # Tool takes no arguments
            result = tool()
        else:
            result = tool(**{param: prompt})

        if isinstance(result, dict):
            if result.get("status") == "success":
                data = result["data"]
                return str(data) if data is not None else None
            elif result.get("status") == "fallback":
                return result.get("data")
            return None
        return str(result) if result else None
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════════════
# Dataset loader
# ═════════════════════════════════════════════════════════════════════════════

def load_eval_items(path: str, limit: int = 0) -> list:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    items = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for key in ("items", "questions", "data", "examples"):
            if key in data and isinstance(data[key], list):
                items = data[key]
                break
    normalized = []
    for item in items[:limit] if limit else items:
        cat = item.get("category") or item.get("category_label", "")
        prompt = item.get("prompt") or item.get("text", "")
        expected = item.get("expected_answer") or item.get("answer", "")
        if prompt and expected and cat:
            normalized.append({
                "category": cat,
                "prompt": prompt,
                "expected": expected,
                "source": path,
            })
    return normalized


# ═════════════════════════════════════════════════════════════════════════════
# Pre-filters
# ═════════════════════════════════════════════════════════════════════════════

_FILTERS_LOADED = False


def _load_filters():
    global _FILTERS_LOADED
    if not _FILTERS_LOADED:
        from agent.solvers import deterministic_filters as df
        globals()['_FILTERS'] = df
        _FILTERS_LOADED = True


def apply_prefilter(category: str, prompt: str) -> Optional[list]:
    """Apply deterministic pre-filter for this category.

    Returns:
        List of tool names to try, or None if the pre-filter isn't available.
        Empty list means no tool can handle this prompt.
    """
    _load_filters()
    df = globals().get('_FILTERS')
    if df is None:
        return None

    if category == "math":
        result = df.can_solve_math(prompt)
        if result == 'direct':
            return ['math_solve']
        return []  # word_problem or skip — no tool for these yet

    elif category == "logic":
        tool = df.detect_logic_type(prompt)
        if tool:
            return [tool]
        return []

    elif category == "ner":
        if df.can_ner(prompt):
            return ['ner_extract']
        return []

    elif category == "summarization":
        if df.can_summarize_extractive(prompt):
            return ['summarize']
        return []

    elif category == "sentiment":
        if df.can_solve_sentiment(prompt):
            return ['sentiment_analysis']
        return []

    return None  # No pre-filter for this category


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation runner
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_tools(
    category_filter: str = "",
    tool_filter: str = "",
    max_items: int = 200,
    use_filters: bool = False,
) -> dict:
    """Run tool evaluation across all datasets.

    Args:
        use_filters: If True, apply deterministic pre-filters before each tool.
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "categories": {},
        "tools": {},
        "filters": {"accepted": 0, "rejected": 0, "no_filter": 0},
        "summary": {
            "total_items": 0,
            "total_attempts": 0,
            "total_matches": 0,
            "total_errors": 0,
            "accuracy": 0.0,
        },
    }

    for eval_path, label, limit, tool_override in EVAL_SETS:
        if category_filter and category_filter not in label:
            continue

        items = load_eval_items(eval_path, limit=limit)
        if not items:
            continue

        print(f"\n  📁 {label} ({eval_path}) — {len(items)} items")

        for item in items:
            cat = item["category"]
            if category_filter and cat != category_filter:
                continue

            prompt = item["prompt"]
            expected = item["expected"]

            # Get candidate tools for this category
            tool_names = CATEGORY_TOOLS.get(cat, [])

            # Apply tool_override (for dataset-specific overrides)
            if tool_override:
                tool_names = [t for t in tool_names if tool_override in t]

            if tool_filter:
                tool_names = [t for t in tool_names if tool_filter in t]

            if not tool_names:
                continue

            # Apply pre-filters if enabled
            if use_filters:
                filtered = apply_prefilter(cat, prompt)
                if filtered is not None:
                    if not filtered:
                        # Pre-filter says no tool can handle this
                        report["filters"]["rejected"] += 1
                        continue
                    report["filters"]["accepted"] += 1
                    # Only try the tools the filter recommends
                    tool_names = filtered
                else:
                    report["filters"]["no_filter"] += 1

            # Initialize per-category tracking
            if cat not in report["categories"]:
                report["categories"][cat] = {"attempts": 0, "matches": 0, "errors": 0, "items": 0}

            report["summary"]["total_items"] += 1
            report["categories"][cat]["items"] += 1

            # Try each tool in order
            matched = False
            for tool_name in tool_names:
                report["summary"]["total_attempts"] += 1
                report["categories"][cat]["attempts"] += 1

                start = time.time()
                output = run_tool_on_prompt(tool_name, prompt)
                elapsed = time.time() - start

                if tool_name not in report["tools"]:
                    report["tools"][tool_name] = {"attempts": 0, "matches": 0, "errors": 0}

                if output is None:
                    report["summary"]["total_errors"] += 1
                    report["categories"][cat]["errors"] += 1
                    report["tools"][tool_name]["errors"] += 1
                    continue

                report["tools"][tool_name]["attempts"] += 1

                if _match(output, expected):
                    matched = True
                    report["summary"]["total_matches"] += 1
                    report["categories"][cat]["matches"] += 1
                    report["tools"][tool_name]["matches"] += 1

                # If we matched, don't try more tools for this item
                if matched:
                    break

    # Compute summary
    total = report["summary"]["total_attempts"]
    matches = report["summary"]["total_matches"]
    report["summary"]["accuracy"] = round(matches / total * 100, 1) if total > 0 else 0.0

    for cat, stats in report["categories"].items():
        stats["accuracy"] = round(stats["matches"] / stats["attempts"] * 100, 1) if stats["attempts"] > 0 else 0.0
        stats["coverage"] = round(stats["matches"] / stats["items"] * 100, 1) if stats["items"] > 0 else 0.0

    for tool_name, stats in report["tools"].items():
        stats["accuracy"] = round(stats["matches"] / stats["attempts"] * 100, 1) if stats["attempts"] > 0 else 0.0

    return report


# ═════════════════════════════════════════════════════════════════════════════
# Report printer
# ═════════════════════════════════════════════════════════════════════════════

def print_report(report: dict):
    print("\n" + "=" * 68)
    print("  TOOL EVALUATION REPORT")
    print("=" * 68)
    print(f"  Timestamp: {report['timestamp']}")
    print(f"  Total items evaluated: {report['summary']['total_items']}")
    print(f"  Total tool attempts:   {report['summary']['total_attempts']}")
    print(f"  Total matches:         {report['summary']['total_matches']}")
    print(f"  Total errors:          {report['summary']['total_errors']}")
    print(f"  Overall accuracy:      {report['summary']['accuracy']}%")
    print()

    if report["filters"]["accepted"] > 0 or report["filters"]["rejected"] > 0:
        print(f"  ── Pre-filter Stats ──")
        print(f"  Accepted: {report['filters']['accepted']}")
        print(f"  Rejected: {report['filters']['rejected']}")
        print(f"  No filter: {report['filters']['no_filter']}")
        print(f"  Pass rate: {report['filters']['accepted'] / (report['filters']['accepted'] + report['filters']['rejected'] + 1) * 100:.0f}%")
        print()

    print("  ── Per Category ──")
    print(f"  {'Category':20s} {'Items':>5s} {'Attempts':>8s} {'Matches':>8s} {'Accuracy':>8s} {'Coverage':>8s}")
    print(f"  {'-'*20} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for cat, stats in sorted(report["categories"].items()):
        print(f"  {cat:20s} {stats['items']:>5d} {stats['attempts']:>8d} {stats['matches']:>8d} {stats['accuracy']:>7.1f}% {stats['coverage']:>7.1f}%")
    print()

    print("  ── Per Tool ──")
    print(f"  {'Tool':27s} {'Attempts':>8s} {'Matches':>8s} {'Accuracy':>8s}")
    print(f"  {'-'*27} {'-'*8} {'-'*8} {'-'*8}")
    for tool_name, stats in sorted(report["tools"].items()):
        acc_str = f"{stats['accuracy']:>7.1f}%" if stats['attempts'] > 0 else "   N/A  "
        print(f"  {tool_name:27s} {stats['attempts']:>8d} {stats['matches']:>8d} {acc_str}")
    print("=" * 68)


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    sys.path.insert(0, project_root)

    parser = argparse.ArgumentParser(description="Tool evaluation harness")
    parser.add_argument("--category", default="", help="Filter by category")
    parser.add_argument("--tool", default="", help="Filter by tool name")
    parser.add_argument("--max-items", type=int, default=200, help="Max items per set")
    parser.add_argument("--report", default="", help="Save report to JSON file")
    parser.add_argument("--filters", action="store_true", help="Use deterministic pre-filters")
    args = parser.parse_args()

    print(f"🔧 Tool Evaluation Harness")
    print(f"   Filter: category={args.category or 'all'}, tool={args.tool or 'all'}")
    print(f"   Pre-filters: {'ON' if args.filters else 'OFF'}")
    print(f"   Eval sets: {len(EVAL_SETS)}")

    report = evaluate_tools(
        category_filter=args.category,
        tool_filter=args.tool,
        max_items=args.max_items,
        use_filters=args.filters,
    )

    print_report(report)

    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {args.report}")
