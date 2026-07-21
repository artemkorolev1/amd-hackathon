"""
stepwise_math_solver.py — Deterministic step-by-step math solver.

Instead of generating free-form Python code (which the 1.5B model often
mangles), the LLM outputs simple arithmetic expressions line by line.

Format:
  Step 1: <arithmetic expression>  →  <result>
  Step 2: <arithmetic expression>  →  <result>
  ...
  Answer: <final value>

A deterministic parser evaluates each expression and carries forward
any named variables. This leverages the LLM's strength (understanding
what to compute) while eliminating its weakness (generating buggy
Python with import errors, wrong variable names, etc.).

The parser supports:
  - Basic arithmetic: +, -, *, /, //, %
  - Parentheses
  - Named variables from previous steps
  - The special variable `answer` for the final result
"""

import ast
import logging
import operator
import re
from typing import Any, Callable, Optional

logger = logging.getLogger("stepwise_math")

# ── Supported operators ──
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

def _safe_eval(expr: str, variables: dict) -> float:
    """Evaluate a safe arithmetic expression with named variables."""
    tree = ast.parse(expr.strip(), mode="eval")
    
    def _eval_node(node):
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant: {node.value}")
        elif isinstance(node, ast.Name):
            if node.id in variables:
                return variables[node.id]
            # Try parsing as number
            try:
                return float(node.id)
            except ValueError:
                raise NameError(f"Unknown variable: {node.id}")
        elif isinstance(node, ast.BinOp):
            op_fn = _OPERATORS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            return op_fn(left, right)
        elif isinstance(node, ast.UnaryOp):
            op_fn = _OPERATORS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            return op_fn(_eval_node(node.operand))
        else:
            raise ValueError(f"Unsupported syntax: {type(node).__name__}")
    
    return _eval_node(tree)

# ── Prompt templates ──

DECOMPOSE_PROMPT = """\
Break this math problem into simple arithmetic steps.

Each step must be a pure arithmetic expression using ONLY numbers and +, -, *, /, //.
No words, no units, no dollar signs, no explanations.

Format:
Step 1: <arithmetic expression>
Step 2: <arithmetic expression>
...
answer: <final expression>

Example:
Problem: Janet has 5 apples. She buys 3 more. Then she gives 2 to Tom.

Step 1: 5 + 3
Step 2: 8 - 2
answer: 6

Another example:
Problem: A robe takes 2 bolts of blue fiber and half that much white fiber.

Step 1: 2 / 2
Step 2: 2 + 1
answer: 3
"""

DECOMPOSE_USER = """\
Problem: {problem}

Write the arithmetic steps using ONLY numbers and +, -, *, /, //.
One step per line. Each step must be a valid arithmetic expression.
End with 'answer:' followed by the final result."""

# ── Parser ──

_STEP_LINE_RE = re.compile(
    r"Step\s+(\d+)\s*:\s*(.+)",
    re.IGNORECASE,
)

_ANSWER_LINE_RE = re.compile(
    r"(?:answer|final)\s*[:=]\s*(.+)",
    re.IGNORECASE,
)

def parse_step_output(raw: str) -> Optional[str]:
    """Parse the LLM's step-by-step output and compute the final answer.

    Returns the final answer string, or None on failure.
    """
    variables = {}
    final_answer = None

    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Check for answer/final line first
        am = _ANSWER_LINE_RE.match(line)
        if am:
            try:
                final_answer = str(_safe_eval(am.group(1), variables))
            except Exception as e:
                logger.debug("Answer eval failed: %s", e)
            continue

        # Check for step line
        sm = _STEP_LINE_RE.match(line)
        if sm:
            step_num = int(sm.group(1))
            expr = sm.group(2).strip()
            try:
                result = _safe_eval(expr, variables)
                var_name = f"Step{step_num}"
                variables[var_name] = result
                logger.debug("Step %d: %s = %s", step_num, expr, result)
            except Exception as e:
                logger.debug("Step %d eval failed: %s -> %s", step_num, expr, e)
            continue

        # Bare expression line (no Step prefix)
        try:
            result = _safe_eval(line, variables)
            # Treat as an intermediate step
            variables.setdefault("_last", result)
            logger.debug("Bare expr: %s = %s", line, result)
        except Exception:
            pass

    if final_answer is not None:
        return final_answer

    # Fallback: last numbered step
    if variables:
        last_key = sorted(k for k in variables if k.startswith("Step"))
        if last_key:
            return str(variables[last_key[-1]])

    return None


def solve_stepwise(
    problem: str,
    llm: Any,
    infer_fn: Callable,
    max_tokens: int = 512,
) -> Optional[str]:
    """Solve a math problem using stepwise arithmetic decomposition.

    1. LLM decomposes the problem into arithmetic steps
    2. Deterministic parser evaluates each step
    3. Returns final answer

    Args:
        problem: The math word problem text.
        llm: Llama model instance (unused, kept for API compat).
        infer_fn: Pipeline._infer-compatible callable.
        max_tokens: Max tokens for LLM response.

    Returns:
        Final answer string, or None.
    """
    try:
        raw = infer_fn(
            [
                {"role": "system", "content": DECOMPOSE_PROMPT},
                {"role": "user", "content": DECOMPOSE_USER.format(problem=problem)},
            ],
            max_tok=max_tokens,
            stop_seq=[],
            timeout=30.0,
            category="math",
        )
    except Exception as e:
        logger.warning("Stepwise: inference failed: %s", e)
        return None

    if not raw:
        return None

    logger.debug("Stepwise raw output:\\n%s", raw[:500])
    return parse_step_output(raw)


__all__ = ["solve_stepwise", "parse_step_output"]
