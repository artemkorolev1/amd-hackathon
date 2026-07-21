"""
iterative_tora_solver.py — Step-by-step iterative ToRA solver.

Unlike the single-shot ToRA (which generates all code at once),
this solver decomposes the problem into sub-steps and solves each one
independently, passing intermediate results forward as context.

Architecture:
  1. DECOMPOSE: LLM analyzes the problem → numbered sub-goal plan
  2. SOLVE per step: For each sub-goal, LLM generates Python code → executes → result
  3. AGGREGATE: Combine all intermediate results into final answer

This avoids the 1.5B model's weakness: holding multi-step logic in a
single generation. Each step is simpler and isolated.
"""

import ast
import logging
import re
import subprocess
import sys
from typing import Any, Callable, Optional

logger = logging.getLogger("iterative_tora")

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

DECOMPOSE_SYSTEM_PROMPT = """\
You are a math problem decomposer. Your job is to break a math word problem
into a sequence of simple computation steps.

For each step, specify:
- What to compute (in plain English)
- What Python expression to use

Rules:
1. Each step must compute exactly ONE intermediate value.
2. Steps are numbered sequentially.
3. The LAST step must compute the final answer.
4. Do NOT solve the problem — just list the steps.

Output format:
Step 1: <description> → <python expression>
Step 2: <description> → <python expression>
...
"""

DECOMPOSE_USER_PROMPT = """\
Break this problem into computation steps:

Problem: {problem}

Output each step on a separate line in the format:
Step N: <what to compute> → <python expression>
"""

STEP_SOLVE_SYSTEM_PROMPT = """\
You are a math step solver. Given a sub-problem, previous results,
and how many total steps remain, write Python code to compute ONE step.

Rules:
1. Write a single Python expression or short code block.
2. Only use standard library (no imports besides math if needed).
3. Print the result at the end.
4. Put code inside a single ```python ... ``` block.
"""

STEP_SOLVE_USER_PROMPT = """\
Problem: {problem}

Current step ({current_step}/{total_steps}):
{step_description}

Previous results: {previous_results}

Compute this step. Output Python code in a ```python ... ``` block.
"""

AGGREGATE_PROMPT = """\
Problem: {problem}

All intermediate results:
{results}

Combine these to produce the final answer.
Output only the final numeric answer.
"""

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_PYTHON_BLOCK_RE = re.compile(
    r"```python\s*\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)

_STEP_LINE_RE = re.compile(
    r"Step\s+(\d+)\s*:\s*(.+?)\s*(?:→|->)\s*(.+)",
    re.IGNORECASE,
)

# Alternative: "Step N: description\n\nPython expression: expr" format
# Must split on step boundaries first, then parse each block
_STEP_HEADER_RE = re.compile(
    r"Step\s+(\d+)\s*:\s*(.+?)(?=\s*Step\s+\d+\s*:|\s*$)",
    re.IGNORECASE | re.DOTALL,
)

_STEP_EXPR_RE = re.compile(
    r"Python\s+expression\s*:\s*(.+?)(?:\n|$)",
    re.IGNORECASE,
)


def parse_decomposition(raw: str) -> list[dict]:
    """Parse LLM decomposition output into step list."""
    steps = []

    # Try primary format: "Step 1: desc → expr"
    for line in raw.split("\n"):
        m = _STEP_LINE_RE.match(line.strip())
        if m:
            steps.append({
                "num": int(m.group(1)),
                "desc": m.group(2).strip(),
                "expr": m.group(3).strip(),
            })

    if steps:
        return steps

    # Try alternative format: "Step 1: desc\n\nPython expression: expr"
    matches = list(_STEP_HEADER_RE.finditer(raw))
    for m in matches:
        step_num = int(m.group(1))
        desc_text = m.group(2).strip()
        # Find the Python expression line within this block
        expr_m = _STEP_EXPR_RE.search(desc_text)
        if expr_m:
            steps.append({
                "num": step_num,
                "desc": desc_text[:100],
                "expr": expr_m.group(1).strip(),
            })
        elif desc_text:
            steps.append({
                "num": step_num,
                "desc": desc_text[:100],
                "expr": "",
            })

    return steps


def extract_python_code(raw: str) -> Optional[str]:
    """Extract Python code from ```python ... ``` block."""
    m = _PYTHON_BLOCK_RE.search(raw)
    if m:
        return m.group(1).strip()
    # Fallback: lines with = or print
    lines = []
    for line in raw.split("\n"):
        s = line.strip()
        if re.match(r"^[a-zA-Z_]\w*\s*=", s) or s.startswith("print("):
            lines.append(s)
    return "\n".join(lines) if lines else None


def execute_code(code: str, timeout: int = 10) -> Optional[str]:
    """Execute Python code in a sandboxed subprocess."""
    try:
        ast.parse(code)
    except SyntaxError as e:
        logger.warning("Step code syntax error: %s", e)
        return None
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={"PYTHONIOENCODING": "utf-8"},
        )
    except subprocess.TimeoutExpired:
        logger.warning("Step code timed out")
        return None
    if proc.returncode != 0:
        logger.warning("Step code error (code=%d): %s", proc.returncode, proc.stderr.strip()[:200])
        return None
    return proc.stdout.strip()


def extract_answer(text: str) -> Optional[str]:
    """Extract the last meaningful number or value from text."""
    text = text.strip()
    if not text:
        return None
    # Take last non-empty line
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return None
    last = lines[-1]
    # Handle "5 + 3 - 2 = 6" pattern — extract the result after =
    eq_m = re.search(r"=\s*(-?\d+(?:\.\d+)?)", last)
    if eq_m:
        return eq_m.group(1).strip()
    return last


# ---------------------------------------------------------------------------
# Iterative ToRA solver
# ---------------------------------------------------------------------------

def solve_iterative_tora(
    problem: str,
    llm: Any,
    infer_fn: Callable,
    max_tokens: int = 256,
    timeout: int = 10,
) -> Optional[str]:
    """Solve a math word problem using step-by-step iterative ToRA.

    1. Decompose the problem into sub-steps (1 LLM call)
    2. Solve each step independently with code execution (N LLM calls)
    3. Combine results into final answer

    Args:
        problem: The math word problem text.
        llm: Llama model instance.
        infer_fn: Pipeline._infer-compatible callable.
        max_tokens: Max tokens per LLM call.
        timeout: Max seconds for code execution.

    Returns:
        Final answer string, or None on failure.
    """
    # ---- STEP 1: Decompose ----
    logger.info("Iterative ToRA: decomposing problem...")
    try:
        decomp_raw = infer_fn(
            [
                {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
                {"role": "user", "content": DECOMPOSE_USER_PROMPT.format(problem=problem)},
            ],
            max_tok=max_tokens,
            stop_seq=[],
            timeout=30.0,
            category="math",
        )
    except Exception as e:
        logger.warning("Decomposition failed: %s", e)
        return None

    if not decomp_raw:
        logger.debug("Decomposition: empty output")
        return None

    steps = parse_decomposition(decomp_raw)
    if not steps:
        logger.debug("Decomposition: no steps parsed from:\n%s", decomp_raw[:500])
        # Fall back to single-shot ToRA
        return None

    logger.info("Iterative ToRA: decomposed into %d steps", len(steps))

    # ---- STEP 2: Solve each step ----
    previous_results = {}
    for i, step in enumerate(steps):
        step_num = step["num"]
        step_desc = step["desc"]
        step_expr = step["expr"]
        prev_str = "; ".join(f"step{k}={v}" for k, v in previous_results.items()) or "none yet"

        try:
            solve_raw = infer_fn(
                [
                    {"role": "system", "content": STEP_SOLVE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": STEP_SOLVE_USER_PROMPT.format(
                            problem=problem,
                            current_step=step_num,
                            total_steps=len(steps),
                            step_description=step_desc,
                            previous_results=prev_str,
                        ),
                    },
                ],
                max_tok=max_tokens,
                stop_seq=[],
                timeout=30.0,
                category="math",
            )
        except Exception as e:
            logger.warning("Step %d: LLM call failed: %s", step_num, e)
            continue

        if not solve_raw:
            logger.debug("Step %d: empty LLM output", step_num)
            continue

        # Extract and execute code
        code = extract_python_code(solve_raw)
        if not code:
            logger.debug("Step %d: no code block found", step_num)
            continue

        result = execute_code(code, timeout=timeout)
        if result is None:
            logger.debug("Step %d: code execution failed", step_num)
            continue

        # Store result for next steps
        result_val = extract_answer(result)
        if result_val:
            previous_results[step_num] = result_val
            logger.info("Step %d: %s → %s", step_num, step_desc[:40], result_val)
        else:
            previous_results[step_num] = result

    # ---- STEP 3: Aggregate ----
    if not previous_results:
        logger.debug("Iterative ToRA: no steps produced results")
        return None

    results_str = "\n".join(
        f"Step {k}: {v}" for k, v in sorted(previous_results.items())
    )

    try:
        agg_raw = infer_fn(
            [
                {"role": "system", "content": "You are a math answer aggregator."},
                {
                    "role": "user",
                    "content": AGGREGATE_PROMPT.format(
                        problem=problem,
                        results=results_str,
                    ),
                },
            ],
            max_tok=128,
            stop_seq=[],
            timeout=15.0,
            category="math",
        )
    except Exception as e:
        logger.warning("Aggregation failed: %s", e)
        # Use last step result as fallback
        last_key = max(previous_results.keys())
        return previous_results[last_key]

    final = extract_answer(agg_raw) if agg_raw else None
    if final:
        return final
    # Fallback: last step
    last_key = max(previous_results.keys())
    return previous_results[last_key]


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

__all__ = ["solve_iterative_tora"]
