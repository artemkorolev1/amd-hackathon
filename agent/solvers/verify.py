"""
Quality gates for solver outputs.

Runs after any solver returns an answer. If verification fails,
the answer is discarded and the task falls through to Fireworks.

Checks:
- hedge detection (degenerate / non-answers)
- code parse validation (for code outputs)
- code formatting/lint validation via black + ruff
- constraint enforcement (sentence count, word limit for summaries)
- empty / null check
"""

import ast
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional

# ── Degenerate output markers ──────────────────────────────────────────

_DEGENERATE_PATTERNS = [
    r"\bi don'?t know\b",
    r"\bi do not know\b",
    r"\bi cannot\b",
    r"\bi can'?t\b",
    r"\bas an ai\b",
    r"\bunable to\b",
    r"\bno information\b",
    r"\bis not provided\b",
    r"\bdoes not contain\b",
    r"\bcannot answer\b",
    r"\bcannot provide\b",
    r"\bnot enough information\b",
    r"\binsufficient\b",
    r"\bsorry\b",
    r"\bthe text does not\b",
]


def _has_hedge(text: str) -> bool:
    """Check if output contains degenerate / non-answer markers."""
    low = text.lower()
    for pat in _DEGENERATE_PATTERNS:
        if re.search(pat, low):
            return True
    return False


def _is_degenerate(text: str) -> bool:
    """Check for heavy repetition indicating failed generation.
    Only flags when there are multiple words AND one dominates.
    Short answers (numerical, single-word) are never degenerate."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if not words or len(words) < 3:
        return False  # Very short answers (numbers, single words) are valid
    most = max((words.count(w) for w in set(words)), default=0)
    return most > len(words) * 0.5


def _is_too_short(text: str) -> bool:
    """Return True if output is suspiciously short."""
    return len(text.strip()) < 2


def _is_too_long(text: str, max_chars: int = 8000) -> bool:
    """Return True if exceeds expected output length."""
    return len(text) > max_chars


# ── Code-specific checks ───────────────────────────────────────────────

def _extract_code(text: str) -> Optional[str]:
    """Extract code from markdown fences or standalone."""
    if text.startswith("```python"):
        code = text.split("```python", 1)[1]
        if "```" in code:
            code = code.split("```", 1)[0]
        return code.strip()
    if text.startswith("```"):
        code = text.split("```", 1)[1]
        if "```" in code:
            code = code.split("```", 1)[0]
        return code.strip()
    # Check if the whole output looks like python code
    if re.search(r"^(def |class |import |from |return |print\()", text):
        return text.strip()
    return None


def _normalize_code_fragment(code: str) -> str:
    """
    Normalize a code fragment (possibly a function body) into valid Python
    so it can be parsed/formatted/linted.

    If the code is already a valid module-level construct (def, class, import,
    assignment, expression), return as-is. Otherwise, wrap indented fragments
    in a synthetic function so black and ruff can process them.
    """
    # Try direct parse first
    try:
        ast.parse(code)
        return code  # Already valid
    except SyntaxError:
        pass

    # Check if code starts with function-level constructs that suggest
    # it's a function body fragment (indented lines)
    lines = code.splitlines()
    if not lines:
        return code

    # Check first non-empty line
    first_line = next((l for l in lines if l.strip()), "")
    first_indent = len(first_line) - len(first_line.lstrip())

    if first_indent > 0 or not code.strip().startswith(("def ", "class ", "@", "import ", "from ", "#", "\"\"\"")):
        # This is likely a function body fragment. Wrap in a function.
        wrapped_lines = ["def _wrapper():"]
        for line in lines:
            if line.strip():
                wrapped_lines.append("    " + line)
            else:
                wrapped_lines.append("")
        wrapped = "\n".join(wrapped_lines)
        try:
            ast.parse(wrapped)
            return wrapped
        except SyntaxError:
            pass

    # Try wrapping with consistent 4-space indent
    try:
        dedented_lines = []
        for line in lines:
            stripped = line.lstrip()
            if stripped:
                dedented_lines.append(stripped)
            else:
                dedented_lines.append("")
        wrapped = "def _wrapper():\n    " + "\n    ".join(dedented_lines)
        ast.parse(wrapped)
        return wrapped
    except SyntaxError:
        pass

    return code  # Give up, return original


def _valid_python(code: str) -> bool:
    """Verify code is syntactically valid Python.
    Handles both full modules and partial code fragments (function bodies).
    """
    # Try direct parse first
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        pass

    # Try normalized (wrapped) version
    normalized = _normalize_code_fragment(code)
    if normalized != code:
        try:
            ast.parse(normalized)
            return True
        except SyntaxError:
            pass

    return False


def _has_import_statement(code: str) -> bool:
    """Warning: code uses import (possible side effect)."""
    return bool(re.search(r"^\s*(import|from)\s+", code, re.MULTILINE))


# ── Black + Ruff code validation ──────────────────────────────────────


def _can_format_or_lint() -> bool:
    """Check if black and ruff are available on PATH."""
    for cmd in ("black", "ruff"):
        try:
            subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    return True


def format_and_lint(code: str, relaxed: bool = False) -> dict:
    """
    Run black (formatter) + ruff (linter) on a code string.

    Handles both full Python modules and code fragments (function bodies)
    by auto-wrapping fragments into synthetic function stubs.

    Args:
        code: Python code string to validate.
        relaxed: If True, use relaxed ruff config (ignore style warnings
                 and 'return outside function' which is common for fragments).

    Returns:
        {
            "formatted": str or None (formatted code, or None if error/unparseable),
            "lint_errors": list of str (ruff violations),
            "syntax_ok": bool,
            "ast_ok": bool,
            "error": str or None (fatal error message),
        }
    """
    result = {
        "formatted": None,
        "lint_errors": [],
        "syntax_ok": False,
        "ast_ok": False,
        "error": None,
    }

    if not code or not code.strip():
        result["error"] = "empty code"
        return result

    # Check if black/ruff are available
    if not _can_format_or_lint():
        result["error"] = "black/ruff not installed"
        return result

    # Normalize fragment to valid Python
    normalized = _normalize_code_fragment(code)

    # Verify syntax of normalized code
    try:
        ast.parse(normalized)
        result["ast_ok"] = True
        result["syntax_ok"] = True
    except SyntaxError as e:
        result["syntax_ok"] = False
        result["lint_errors"].append(f"SyntaxError: {e}")
        result["error"] = f"syntax error: {e}"
        return result  # Can't format/lint broken syntax

    # Write to temp file
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(normalized)
            tmp_path = f.name

        # Run black (format in-place)
        fmt_result = subprocess.run(
            ["black", "--quiet", tmp_path],
            capture_output=True, text=True, timeout=15
        )
        if fmt_result.returncode == 0:
            with open(tmp_path) as f:
                formatted = f.read()
            # If we wrapped the code, unwrap the result
            if normalized != code:
                result["formatted"] = _unwrap_wrapper(formatted)
            else:
                result["formatted"] = formatted
        else:
            # Black couldn't format
            result["formatted"] = code  # Return original
            if fmt_result.stderr.strip():
                result["lint_errors"].append(f"black: {fmt_result.stderr.strip()}")

        # Run ruff (lint only)
        ruff_args = ["ruff", "check", "--quiet"]
        if relaxed:
            # Relaxed mode: ignore pycodestyle (E, W) and
            # 'return outside function' (F706) which is common for fragments
            ruff_args.extend(["--ignore", "E,W,F706"])
        else:
            # Strict mode: still ignore F706 since fragments are common
            ruff_args.extend(["--ignore", "F706"])

        lint_result = subprocess.run(
            ruff_args + [tmp_path],
            capture_output=True, text=True, timeout=15
        )
        if lint_result.returncode != 0:
            raw = (lint_result.stdout + "\n" + lint_result.stderr).strip()
            if raw:
                result["lint_errors"] = [
                    line for line in raw.split('\n') if line.strip()
                ]

    except FileNotFoundError as e:
        result["error"] = f"tool not found: {e}"
    except subprocess.TimeoutExpired:
        result["error"] = "black or ruff timed out"
    except Exception as e:
        result["error"] = str(e)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return result


def _unwrap_wrapper(wrapped: str) -> str:
    """
    Remove the synthetic function wrapper added by _normalize_code_fragment.
    Takes the body of `def _wrapper():` and un-indents it by one level.
    """
    lines = wrapped.splitlines()
    # Skip the 'def _wrapper():' line
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("def _wrapper"):
            body_start = i + 1
            break

    body_lines = lines[body_start:]

    # Remove the outer 4-space indent
    result = []
    for line in body_lines:
        if line.startswith("    "):
            result.append(line[4:])
        elif line.strip() == "":
            result.append("")
        else:
            result.append(line)

    return "\n".join(result).strip()


# ── Constraint checks ──────────────────────────────────────────────────

_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "inc", "ltd", "co", "e.g", "i.e", "u.s", "a.m", "p.m",
})


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences: list[str] = []
    for part in parts:
        if sentences:
            last_word = re.split(r"\s+", sentences[-1].rstrip())[-1].rstrip(".").lower()
            if last_word in _ABBREVIATIONS:
                sentences[-1] += " " + part
                continue
        sentences.append(part.strip())
    return sentences


def _count_bullets(text: str) -> int:
    return len([ln for ln in text.splitlines() if re.match(r"\s*(?:[-*•]|\d+[.)])\s+", ln)])


# ── Public API ─────────────────────────────────────────────────────────

@dataclass
class VerifyResult:
    passed: bool
    reason: str = ""
    details: dict = field(default_factory=dict)


def verify(answer: str, category: str = "", task: str = "") -> VerifyResult:
    """
    Run quality gates on a solver output.

    Returns (passed: bool, reason: str). When passed=False, the answer
    should be discarded and the task falls through to Fireworks.
    """
    if not answer or not answer.strip():
        return VerifyResult(False, "empty answer")

    # Hedge / degenerate check (all categories)
    if _has_hedge(answer):
        return VerifyResult(False, "hedge words detected (I don't know, sorry, etc.)")
    if _is_degenerate(answer):
        return VerifyResult(False, "degenerate output (heavy repetition)")
    if _is_too_short(answer):
        return VerifyResult(False, "output too short")

    # Category-specific checks
    if category == "code_gen" or category == "code_debug":
        code = _extract_code(answer)
        if code:
            if not _valid_python(code):
                return VerifyResult(False, "invalid Python syntax")
            # Run black + ruff validation
            # For code_debug, fragments (return/expr/for/if stmts) are common
            # answers — use relaxed linting to avoid false positives from
            # undefined names and return-outside-function
            is_fragment = not code.strip().startswith(("def ", "class ", "@"))
            lint_result = format_and_lint(code, relaxed=is_fragment)
            if lint_result.get("error"):
                # Only reject if it's a real error (not a tool issue)
                if lint_result["error"] != "black/ruff not installed":
                    return VerifyResult(False, f"code validation error: {lint_result['error']}")
            if lint_result["lint_errors"]:
                # For strict code_gen, lint errors are rejections
                # For code_debug fragments, only reject if there are > 5 lint issues
                if category == "code_gen":
                    return VerifyResult(
                        False,
                        f"lint errors: {len(lint_result['lint_errors'])} issues",
                        details={"lint_errors": lint_result["lint_errors"]}
                    )
                elif not is_fragment and len(lint_result["lint_errors"]) > 0:
                    # code_debug full function with lint issues
                    return VerifyResult(
                        False,
                        f"lint errors: {len(lint_result['lint_errors'])} issues",
                        details={"lint_errors": lint_result["lint_errors"]}
                    )
        else:
            return VerifyResult(False, "no code block found")
        if _has_import_statement(code or ""):
            return VerifyResult(False, "unsafe import statement in generated code")

    if category == "summarization":
        if _is_too_long(answer, max_chars=4000):
            return VerifyResult(False, "summary too long")

    return VerifyResult(True, "")


def verify_strict(answer: str, category: str = "",
                  expected_sentences: int = 0,
                  max_words: int = 0,
                  expected_bullets: int = 0) -> VerifyResult:
    """
    Stricter verification with explicit constraint checks.
    Used by the validation harness.
    """
    result = verify(answer, category)
    if not result.passed:
        return result

    text = answer.strip()

    # Sentence count
    if expected_sentences > 0:
        n = len(_split_sentences(text))
        if n != expected_sentences:
            return VerifyResult(False, f"expected {expected_sentences} sentences, got {n}")

    # Word limit
    if max_words > 0:
        n = len(re.findall(r"\S+", text))
        if n > max_words:
            return VerifyResult(False, f"exceeded {max_words} word limit, got {n}")

    # Bullet count
    if expected_bullets > 0:
        n = _count_bullets(text)
        if n != expected_bullets:
            return VerifyResult(False, f"expected {expected_bullets} bullets, got {n}")

    return VerifyResult(True, "passed all checks")
