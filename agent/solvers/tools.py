"""
Deterministic tools for task solving.

These cost zero tokens and run locally. The local model (Qwythos-9B)
has native function calling and can invoke these tools automatically
via its chat template's <tool_call> format.

Tools available:
  - python_executor: run Python code for math verification, data processing
  - calculator: evaluate arithmetic expressions safely
  - json_validator: validate structured output format
  - code_syntax_check: basic syntax checking for common languages
"""

import ast
import math
import json
import os
import re
import subprocess
import sys
import traceback
from typing import Any, Dict, Optional

import sympy as sp


# ---------------------------------------------------------------------------
# Safe calculator (pure Python eval with restricted globals)
# ---------------------------------------------------------------------------

_CALC_SAFE_GLOBALS = {
    "abs": abs, "round": round, "int": int, "float": float,
    "min": min, "max": max, "sum": sum, "len": len,
    "pi": math.pi, "e": math.e, "inf": math.inf,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
    "exp": math.exp, "pow": pow, "floor": math.floor, "ceil": math.ceil,
    "factorial": math.factorial,
}


def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.

    Accepts Python math syntax. Returns result as string.
    """
    try:
        # Remove whitespace and validate characters
        cleaned = expression.strip()
        if not cleaned:
            return "Error: empty expression"

        # Only allow safe characters
        if not re.match(r'^[\d\s\+\-\*\/\(\)\.,\%\^\w\[\]]+$', cleaned):
            return "Error: invalid characters in expression"

        result = eval(cleaned, {"__builtins__": {}}, _CALC_SAFE_GLOBALS)
        if isinstance(result, float):
            # Format: trim trailing zeros
            return f"{result:.10f}".rstrip("0").rstrip(".")
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# SymPy equation solver (symbolic math validation)
# ---------------------------------------------------------------------------


def sympy_solve(expr_str: str) -> Optional[str]:
    """
    Solve a mathematical expression using SymPy.
    
    Handles:
    - Basic arithmetic: 2 + 3 * 4
    - Equations: x + 5 = 10 (solves for x)
    - Symbolic: sqrt(144), sin(pi/2)
    - Multi-variable: x**2 + y**2 = 25, y = 3 (solves for x)
    
    Returns the result as a string, or None on failure.
    """
    # Use SymPy's implicit multiplication parser
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication,
        function_exponentiation,
    )
    _transformations = (
        standard_transformations
        + (implicit_multiplication, function_exponentiation)
    )

    # Pre-strip command words that would be misinterpreted as variables
    command_words = r'\b(solve|find|calculate|compute|evaluate|simplify|determine|what|is|the|for)\b'
    cleaned = re.sub(command_words, '', expr_str, flags=re.I).strip()
    cleaned = re.sub(r'^[;:!?\s]+', '', cleaned).strip()
    cleaned = re.sub(r'[;:!?]+$', '', cleaned).strip()
    
    # Preprocess log subscripts: log₂(x) → log(x, 2)
    cleaned = cleaned.translate(str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789'))
    cleaned = re.sub(r'log_?(\d+)\s*\(([^)]+)\)', r'log(\2, \1)', cleaned)

    def _parse(s: str):
        """Try to parse with implicit multiplication support."""
        try:
            return parse_expr(s, local_dict={}, transformations=_transformations)
        except Exception:
            try:
                return sp.sympify(s, dict())
            except Exception:
                return None

    # Try parsing as a regular expression first
    try:
        expr = _parse(cleaned)
    except Exception:
        expr = None

    if expr is not None:
        try:
            # If it's already a number (no variables), evaluate it
            if expr.is_Number or expr.is_constant():
                val = sp.N(expr)
                # Format nicely: integer if whole number
                if val.is_Float:
                    fval = float(val)
                    if abs(fval - round(fval)) < 1e-12:
                        return str(int(round(fval)))
                    return f"{fval:.10f}".rstrip("0").rstrip(".")
                return str(val)
            return None  # Has variables, can't simplify without equation context
        except Exception:
            return None
    
    # Try equation with "=" sign
    if "=" in cleaned:
        try:
            left, right = cleaned.split("=", 1)
            left_expr = _parse(left.strip())
            right_expr = _parse(right.strip())
            if left_expr is None or right_expr is None:
                return None
            equation = sp.Eq(left_expr, right_expr)
            solution = sp.solve(equation)
            if solution:
                s = str(solution)
                return s
            return None
        except Exception:
            pass
    
    return None


# ---------------------------------------------------------------------------
# Python executor (runs user-supplied Python code in a subprocess)
# ---------------------------------------------------------------------------

def python_executor(code: str, timeout: int = 5) -> Dict[str, Any]:
    """
    Run Python code in an isolated subprocess.

    Returns dict with stdout, stderr, returncode.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        return {
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "Timeout", "returncode": -1}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": -1}


# ---------------------------------------------------------------------------
# JSON validator
# ---------------------------------------------------------------------------

def json_validator(text: str) -> Dict[str, Any]:
    """Try to parse text as JSON. Returns parsed result or error."""
    try:
        parsed = json.loads(text)
        return {"valid": True, "data": parsed}
    except json.JSONDecodeError as e:
        return {"valid": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Code syntax checker (ast-based for Python)
# ---------------------------------------------------------------------------

def code_syntax_check(code: str, language: str = "python") -> Dict[str, Any]:
    """
    Basic syntax check for code snippets.

    For Python: uses ast.parse.
    For others: returns "check not implemented".
    """
    if language == "python":
        try:
            ast.parse(code)
            return {"valid": True, "error": None}
        except SyntaxError as e:
            return {"valid": False, "error": str(e)}
    return {"valid": True, "note": f"Syntax check not implemented for {language}"}


# ---------------------------------------------------------------------------
# Function registry for the local model's function calling
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "calculator": {
        "description": "Evaluate mathematical expressions. Use for arithmetic, trigonometry, etc.",
        "fn": lambda expression: {"result": calculator(expression)},
        "schema": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate (e.g., 'sin(pi/7) * cos(pi/11)')",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    "sympy_solve": {
        "description": "Solve mathematical expressions using SymPy symbolic math. Handles equations, symbolic simplification, trigonometric functions, and exact rational results.",
        "fn": lambda expression: {"result": sympy_solve(expression) or "Error: could not solve"},
        "schema": {
            "name": "sympy_solve",
            "description": "Solve a mathematical expression using SymPy symbolic math",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression or equation to solve (e.g., 'x + 5 = 10', 'sin(pi/3)', 'sqrt(144)')",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    "python_executor": {
        "description": "Run Python code for verification, data processing, or computation.",
        "fn": lambda code: python_executor(code),
        "schema": {
            "name": "python_executor",
            "description": "Execute Python code in an isolated environment",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    }
                },
                "required": ["code"],
            },
        },
    },
    "json_validator": {
        "description": "Validate whether a string is valid JSON.",
        "fn": lambda text: json_validator(text),
        "schema": {
            "name": "json_validator",
            "description": "Validate JSON format",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to validate as JSON"}
                },
                "required": ["text"],
            },
        },
    },
}


def get_tool_schemas() -> list:
    """Return OpenAI-compatible tool schemas for the local model."""
    return [{"type": "function", "function": info["schema"]} for info in TOOL_REGISTRY.values()]


def execute_tool(name: str, args: dict) -> Any:
    """Execute a tool by name with the given arguments."""
    tool = TOOL_REGISTRY.get(name)
    if not tool:
        return {"error": f"Unknown tool: {name}"}
    try:
        return tool["fn"](**args)
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}
