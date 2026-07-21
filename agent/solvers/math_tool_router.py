"""
agent/solvers/math_tool_router.py — Routes each math problem type to the
best solver tool.

Architecture:
    classify_math(prompt) → problem_type
    route_math(problem_type) → (tool_name, solver_function)
    solve_math_routed(task, category) → Optional[str]

The router maps problem types to solver tools:

    simple_arithmetic      → solve_arithmetic()        (deterministic.py)
    narrative_entity       → _solve_narrative_math()   (deterministic.py)
    narrative_remaining    → _solve_narrative_math()   (deterministic.py)
    comparison_ratio       → _solve_narrative_math()   (deterministic.py)
    rate_per_unit          → solve_arithmetic()        (deterministic.py)
    multiplication_chain   → solve_arithmetic()        (deterministic.py)
    fraction_percentage    → solve_arithmetic()        (deterministic.py)
    speed_distance         → solve_arithmetic()        (has speed/distance patterns)
    unit_conversion        → _solve_narrative_math()   (deterministic.py)
    age_ratio              → python_executor           (tools.py) — for solving age equations
    money_shopping         → _solve_narrative_math()   (deterministic.py)
    profit_cost            → _solve_narrative_math()   (deterministic.py)
    multi_step_complex     → WorkflowEngine (MATH_3STEP) or pipeline fallback
    unknown                → None (let the model handle it)
"""

import logging
from typing import Callable, Optional, Tuple

from agent.solvers.deterministic import (
    solve_arithmetic,
    _solve_narrative_math,
)
from agent.solvers.math_classifier import classify_math

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Routing table: problem_type → (tool_name, solver_function)
# ---------------------------------------------------------------------------

# The primary solvers we route to
SOLVER_MAP: dict[str, Tuple[str, Callable]] = {
    "simple_arithmetic":       ("solve_arithmetic", solve_arithmetic),
    "narrative_entity":        ("_solve_narrative_math", _solve_narrative_math),
    "narrative_remaining":     ("_solve_narrative_math", _solve_narrative_math),
    "comparison_ratio":        ("_solve_narrative_math", _solve_narrative_math),
    "rate_per_unit":           ("solve_arithmetic", solve_arithmetic),
    "multiplication_chain":    ("solve_arithmetic", solve_arithmetic),
    "fraction_percentage":     ("solve_arithmetic", solve_arithmetic),
    "speed_distance":          ("solve_arithmetic", solve_arithmetic),
    "unit_conversion":         ("_solve_narrative_math", _solve_narrative_math),
    "age_ratio":               ("_solve_narrative_math", _solve_narrative_math),
    "money_shopping":          ("_solve_narrative_math", _solve_narrative_math),
    "profit_cost":             ("_solve_narrative_math", _solve_narrative_math),
    "multi_step_complex":      ("solve_arithmetic", solve_arithmetic),  # try arithmetic first
}


def route_math(prompt: str, category_hint: str = None) -> Tuple[str, Optional[Callable]]:
    """Route a math problem to the best solver tool.

    Args:
        prompt: The math problem text.
        category_hint: Optional 8-way category hint (e.g., "math_arithmetic").

    Returns:
        (tool_name, solver_function) where solver_function accepts
        (task, category) or (task) and returns Optional[str].
    """
    problem_type = classify_math(prompt)
    tool_name, solver_fn = SOLVER_MAP.get(problem_type, ("unknown", None))
    return tool_name, solver_fn


def solve_math_routed(task: str, category: str = "math") -> Optional[str]:
    """Solve a math problem using the classifier → router → solver pipeline.

    Steps:
        1. Classify the problem into a type via classify_math()
        2. Look up the best solver tool
        3. Call the solver with (task, category) or (task)
        4. Return the answer string, or None if unsolvable

    Args:
        task: The math problem text.
        category: The 8-way category (default: "math").

    Returns:
        Answer string, or None if no solver could handle it.
    """
    problem_type = classify_math(task)
    entry = SOLVER_MAP.get(problem_type)
    if entry is None:
        logger.debug("solve_math_routed: type=%s no solver found", problem_type)
        return None

    tool_name, solver_fn = entry

    logger.debug(
        "solve_math_routed: type=%s tool=%s task=%.60s...",
        problem_type, tool_name, task,
    )

    # Try the primary solver
    result: Optional[str] = None
    try:
        # Some solvers take (task, category), others take just (task)
        if tool_name == "_solve_narrative_math":
            result = solver_fn(task)
        else:
            result = solver_fn(task, category)
    except Exception as e:
        logger.debug("Primary solver %s failed: %s", tool_name, e)

    if result is not None:
        return result

    # Fallback: if narrative solver failed, try arithmetic
    if tool_name == "_solve_narrative_math":
        try:
            result = solve_arithmetic(task, category)
        except Exception as e:
            logger.debug("Fallback arithmetic also failed: %s", e)

    # Fallback: if arithmetic solver failed, try narrative
    if tool_name == "solve_arithmetic":
        try:
            result = _solve_narrative_math(task)
        except Exception as e:
            logger.debug("Fallback narrative also failed: %s", e)

    return result


# ---------------------------------------------------------------------------
# Convenience: direct access to the classifier if needed downstream
# ---------------------------------------------------------------------------
__all__ = [
    "classify_math",
    "route_math",
    "solve_math_routed",
]
