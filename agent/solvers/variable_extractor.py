"""
variable_extractor.py — Pre-filter that extracts structured variables from
math word problems before sending to ToRA.

Architecture:
  1. LLM extracts all numeric values + relationships from the word problem
     as named variables in a structured dict
  2. Extractors validate: all variables present, no contradictions
  3. A clean computation prompt is built from the variables
  4. ToRA receives the cleaned prompt instead of raw text

This decouples "understanding the problem" (extraction) from
"computing the answer" (ToRA code generation), so each step is simpler.
"""

import json
import logging
import re
from typing import Any, Callable, Optional

logger = logging.getLogger("variable_extractor")


def _first_key_wins(pairs):
    """JSON object_pairs_hook that keeps the first occurrence of each key."""
    seen = set()
    result = {}
    for key, val in pairs:
        if key not in seen:
            seen.add(key)
            result[key] = val
    return result


# ── Extraction prompt ──

EXTRACT_SYSTEM_PROMPT = """\
You extract numeric variables and relationships from math word problems.

Output ONLY a JSON object with:
1. All numeric values as named variables
2. A "steps" array showing the computation plan (each step is one arithmetic operation)
3. A "final_answer" field set to the expression that computes the final answer

Rules:
- Use descriptive variable names (no abbreviations)
- Include EVERY numeric value from the problem as a separate variable
- Do NOT compute intermediate values in the variable definitions (use raw numbers)
- Do NOT use duplicate variable names
- Steps reference variables, not raw numbers
- Each step is one arithmetic operation: "var_name = expression"
- The final step must compute the answer as "answer = <expression>"

Examples:

Problem: Janet has 5 apples. She buys 3 more. Then she gives 2 to Tom.
{
  "variables": {
    "initial_apples": 5,
    "bought_apples": 3,
    "given_apples": 2
  },
  "steps": [
    "total_after_buy = initial_apples + bought_apples",
    "answer = total_after_buy - given_apples"
  ],
  "final_answer": "answer"
}

Problem: Kylar wants to buy 16 glasses. One glass costs $5, every second glass costs 60% of the price.
{
  "variables": {
    "total_glasses": 16,
    "full_price": 5,
    "discount_ratio": 0.6
  },
  "steps": [
    "full_price_count = total_glasses / 2",
    "discounted_count = total_glasses / 2",
    "discounted_price = full_price * discount_ratio",
    "total_full = full_price_count * full_price",
    "total_discounted = discounted_count * discounted_price",
    "answer = total_full + total_discounted"
  ],
  "final_answer": "answer"
}
"""

EXTRACT_USER_PROMPT = """\
Extract variables and steps from this problem. Output ONLY a JSON object.

Problem: {problem}
"""


def extract_structured(problem: str, infer_fn: Callable, max_tokens: int = 512) -> Optional[dict]:
    """Extract structured variables from a word problem.

    Args:
        problem: The raw math word problem text.
        infer_fn: Pipeline._infer-compatible callable.
        max_tokens: Max tokens for LLM response.

    Returns:
        dict with 'variables', 'steps', 'final_answer' keys, or None.
    """
    try:
        raw = infer_fn(
            [
                {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": EXTRACT_USER_PROMPT.format(problem=problem)},
            ],
            max_tok=max_tokens,
            stop_seq=[],
            timeout=30.0,
            category="math",
        )
    except Exception as e:
        logger.warning("Extraction inference failed: %s", e)
        return None

    if not raw:
        return None

    # Try to parse JSON from the output
    # The model might wrap in ```json or have extra text
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL)
    json_str = json_match.group(1) if json_match else raw

    # Find the outermost { }
    brace_start = json_str.find("{")
    brace_end = json_str.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        json_str = json_str[brace_start:brace_end + 1]

    try:
        # Handle duplicate keys: keep the first occurrence (literal values)
        # The model sometimes outputs computed expressions AND their values
        # as duplicate keys. First occurrence is the expression, second is value.
        # We use a custom parser that keeps only the first occurrence.
        if brace_start >= 0 and brace_end > brace_start:
            json_str = json_str[brace_start:brace_end + 1]
        # Use object_pairs_hook to handle dupes: keep first
        data = json.loads(json_str, object_pairs_hook=_first_key_wins)
    except json.JSONDecodeError as e:
        logger.debug("JSON parse failed: %s. Raw: %s", e, raw[:300])
        return None

    # Validate required keys
    if not isinstance(data.get("variables"), dict) or not data["variables"]:
        logger.debug("Extraction missing variables")
        return None
    if not isinstance(data.get("steps"), list) or not data["steps"]:
        logger.debug("Extraction missing steps")
        return None
    if not data.get("final_answer"):
        logger.debug("Extraction missing final_answer")
        return None

    return data


def build_clean_prompt(extracted: dict) -> str:
    """Build a clean computation prompt from extracted variables.

    The output is a structured description ToRA can directly work with.
    """
    vars = extracted["variables"]
    steps = extracted["steps"]

    var_lines = "\n".join(f"- {k} = {v}" for k, v in vars.items())
    step_lines = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))

    return f"""\
Variables:
{var_lines}

Computation plan:
{step_lines}

Write Python code that implements this plan. Use the variable names directly.
The last expression must compute 'answer'."""


def solve_with_extraction(
    problem: str,
    llm: Any,
    infer_fn: Callable,
    max_tokens: int = 512,
) -> Optional[str]:
    """Solve a math problem using variable extraction + ToRA.

    1. Extract structured variables from the word problem
    2. Build a clean computation prompt
    3. Run ToRA on the clean prompt

    Args:
        problem: Raw math word problem text.
        llm: Llama model instance.
        infer_fn: Pipeline._infer-compatible callable.
        max_tokens: Max tokens for ToRA code generation.

    Returns:
        Answer string, or None on failure.
    """
    # Step 1: Extract structured variables
    extracted = extract_structured(problem, infer_fn, max_tokens=max_tokens)
    if not extracted:
        logger.info("Extraction failed, falling back to raw prompt")
        return None  # Caller should fall back

    logger.debug(
        "Extracted %d variables, %d steps, answer=%s",
        len(extracted["variables"]),
        len(extracted["steps"]),
        extracted.get("final_answer", "?"),
    )

    # Step 2: Build clean prompt
    clean_prompt = build_clean_prompt(extracted)

    # Step 3: Run ToRA on the clean prompt
    from agent.solvers.tora_solver import solve_with_tora

    try:
        answer = solve_with_tora(
            clean_prompt,
            llm,
            infer_fn,
            max_tokens=max_tokens,
            timeout=10,
        )
    except Exception as e:
        logger.warning("ToRA on clean prompt failed: %s", e)
        return None

    return answer


__all__ = ["extract_structured", "build_clean_prompt", "solve_with_extraction"]
