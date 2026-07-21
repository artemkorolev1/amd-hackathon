"""
Safe Python code execution using RestrictedPython + subprocess guard.
"""
import sys
import ast
import traceback
from typing import Optional, Any, Dict, List
from io import StringIO
from RestrictedPython import compile_restricted, safe_builtins, limited_builtins, utility_builtins
from RestrictedPython.Guards import (
    guarded_iter_unpack_sequence,
    guarded_unpack_sequence,
    safer_getattr,
    guarded_setattr,
    full_write_guard,
)
from RestrictedPython.PrintCollector import PrintCollector


# ── Banned AST patterns ──

_BANNED_AST_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
)

_BANNED_FUNCTIONS = {
    "exec", "eval", "compile", "__import__", "open", "input",
    "breakpoint", "exit", "quit", "help",
}

_BANNED_ATTRIBUTES = {
    "__subclasses__", "__bases__", "__globals__", "__code__",
    "__closure__", "__func__", "__self__", "__builtins__",
    "__import__", "__class__", "__mro__", "__dict__",
}


def check_ast_safe(tree: ast.AST) -> Optional[str]:
    """
    AST-level safety check. Returns error message if dangerous code detected.
    """
    for node in ast.walk(tree):
        if isinstance(node, _BANNED_AST_NODES):
            return f"Import statements are not allowed (found: {ast.dump(node)[:60]})"

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in _BANNED_FUNCTIONS:
                    return f"Use of '{node.func.id}()' is not allowed"
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in _BANNED_ATTRIBUTES:
                    return f"Access to '{node.func.attr}' is restricted"
                if node.func.attr.startswith("__") and node.func.attr.endswith("__"):
                    return f"Access to dunder method '{node.func.attr}' is restricted"

        if isinstance(node, ast.Attribute):
            if node.attr in _BANNED_ATTRIBUTES:
                return f"Access to '{node.attr}' is restricted"
            if node.attr.startswith("__") and node.attr.endswith("__"):
                return f"Access to dunder attribute '{node.attr}' is restricted"

    return None


def _get_stdout(local_vars: dict, global_vars: dict) -> str:
    """Extract captured stdout from RestrictedPython's PrintCollector."""
    # Check local_vars first (where exec stores it), fall back to global_vars
    printer = local_vars.get("_print") or global_vars.get("_print")
    if printer is not None and hasattr(printer, "__call__"):
        try:
            return str(printer())
        except Exception:
            pass
    return ""


def execute_code_safe(
    code: str,
    timeout: int = 10,
    safe_modules: Optional[List[str]] = None,
) -> dict:
    """
    Execute Python code safely with RestrictedPython and AST checks.

    1. AST check for dangerous patterns
    2. RestrictedPython compile
    3. Execute with limited builtins
    4. Timeout via signal

    Args:
        code: Python code to execute
        timeout: Max execution time in seconds
        safe_modules: List of module names to expose (math, json, random, etc.)

    Returns:
        dict with keys: stdout, stderr, success, result
    """
    # ── Step 1: Parse and AST-check ──
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {
            "stdout": "",
            "stderr": f"SyntaxError: {e}",
            "success": False,
            "result": None,
        }

    ast_error = check_ast_safe(tree)
    if ast_error:
        return {
            "stdout": "",
            "stderr": ast_error,
            "success": False,
            "result": None,
        }

    # ── Step 2: Compile with RestrictedPython ──
    try:
        byte_code = compile_restricted(tree, filename="<safe_code>", mode="exec")
    except SyntaxError as e:
        return {
            "stdout": "",
            "stderr": f"RestrictedPython compile error: {e}",
            "success": False,
            "result": None,
        }

    # ── Step 3: Build restricted execution environment ──
    local_globals: Dict[str, Any] = {
        "__builtins__": {
            **safe_builtins,
            **limited_builtins,
            **utility_builtins,
        },
        "__name__": "__restricted__",
        # Guards that RestrictedPython injects into compiled code
        "_getattr_": safer_getattr,
        "_setattr_": guarded_setattr,
        "_write_": full_write_guard,
        "_unpack_sequence_": guarded_unpack_sequence,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        # Print collector
        "_print_": PrintCollector,
    }

    # Add safe modules if requested
    if safe_modules:
        for mod_name in safe_modules:
            try:
                local_globals[mod_name] = __import__(mod_name)
            except ImportError:
                pass

    # ── Step 4: Execute with timeout ──
    import signal

    class TimeoutError(Exception):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Code execution timed out after {timeout}s")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    local_dict: Dict[str, Any] = {}
    stderr_capture = StringIO()
    old_stderr = sys.stderr
    sys.stderr = stderr_capture

    try:
        exec(byte_code, local_globals, local_dict)
        signal.alarm(0)

        stdout = _get_stdout(local_dict, local_globals)
        result = local_dict.get("result", None)
        stderr = stderr_capture.getvalue()

        return {
            "stdout": stdout,
            "stderr": stderr,
            "success": True,
            "result": result,
        }

    except TimeoutError as e:
        signal.alarm(0)
        stdout = _get_stdout(local_dict, local_globals)
        return {
            "stdout": stdout,
            "stderr": str(e),
            "success": False,
            "result": None,
        }

    except Exception as e:
        signal.alarm(0)
        tb = traceback.format_exc()
        stdout = _get_stdout(local_dict, local_globals)
        stderr = f"{type(e).__name__}: {e}\n{tb}" if tb else str(e)
        stderr += "\n" + stderr_capture.getvalue()

        return {
            "stdout": stdout,
            "stderr": stderr,
            "success": False,
            "result": None,
        }

    finally:
        sys.stderr = old_stderr
        signal.signal(signal.SIGALRM, old_handler)
