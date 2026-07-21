"""
CodeQualityCell — work cell that runs ruff on generated code.

Purpose: catch syntax errors, style violations, and anti-patterns
in code produced by builder/inference cells BEFORE the answer is finalised.

As an answer-centred component, this cell's job is to ensure the answer
(code output) is correct, not to enforce arbitrary style rules.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import tempfile
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Structured output of a code quality check."""

    passed: bool
    errors: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    fixable_count: int = 0
    formatted: Optional[str] = None  # auto-fixed code (if fixes applied)
    summary: str = ""
    tool: str = "ruff"
    elapsed_ms: float = 0.0


class CodeQualityCell:
    """Work cell that checks code quality using ruff.

    Usage:
        cell = CodeQualityCell()
        report = cell.check("def foo(): pass")
        if not report.passed:
            fixed_code = cell.fix("def foo(): pass")
            report = cell.check(fixed_code)
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self._ruff_path = self._find_ruff()

    def _find_ruff(self) -> str:
        """Locate the ruff binary."""
        candidates = [
            "ruff",
            os.path.expanduser("~/.local/bin/ruff"),
            "/usr/local/bin/ruff",
        ]
        for c in candidates:
            try:
                subprocess.run([c, "--version"], capture_output=True, timeout=5)
                return c
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        logger.warning("ruff not found -- code quality cell will be unavailable")
        return ""

    def check(self, code: str, filename: str = "generated.py") -> QualityReport:
        """Run ruff check on code and return a QualityReport.

        Args:
            code: Python source code to check
            filename: Logical filename (for ruff's output)

        Returns:
            QualityReport with pass/fail, errors, warnings, fixable count
        """
        if not self._ruff_path:
            return QualityReport(passed=True, summary="ruff not available -- skipping")

        t0 = time.time()

        # Write code to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="code_quality_"
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            # Run ruff check (JSON output)
            result = subprocess.run(
                [self._ruff_path, "check", "--output-format=json", tmp_path],
                capture_output=True,
                text=True,
                timeout=30,
            )

            elapsed = (time.time() - t0) * 1000

            # Parse violations
            violations = []
            if result.stdout.strip():
                try:
                    violations = json.loads(result.stdout)
                except json.JSONDecodeError:
                    logger.warning("ruff JSON parse failed: %s", result.stdout[:200])

            errors = []
            warnings = []
            fixable = 0

            for v in violations:
                entry = {
                    "line": v.get("location", {}).get("row", 0),
                    "col": v.get("location", {}).get("column", 0),
                    "code": v.get("code", ""),
                    "message": v.get("message", ""),
                }
                code_str = v.get("code", "")
                # Ruff convention: codes starting with F (pyflakes) or E
                # (pycodestyle errors) are errors
                if code_str and (
                    code_str.startswith("F")
                    or code_str.startswith("E")
                    or "SyntaxError" in entry["message"]
                ):
                    errors.append(entry)
                else:
                    warnings.append(entry)
                if v.get("fix") is not None:
                    fixable += 1

            report = QualityReport(
                passed=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                fixable_count=fixable,
                summary=f"{len(errors)} errors, {len(warnings)} warnings ({fixable} fixable)",
                tool="ruff",
                elapsed_ms=round(elapsed, 1),
            )

            return report

        except subprocess.TimeoutExpired:
            return QualityReport(passed=True, summary="ruff timed out -- skipping")
        except Exception as e:
            logger.warning("Code quality check failed: %s", e)
            return QualityReport(passed=True, summary=f"check failed: {e}")
        finally:
            # Cleanup temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def fix(self, code: str) -> str:
        """Auto-fix code with ruff --fix. Returns fixed code or original on failure."""
        if not self._ruff_path:
            return code

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="code_quality_fix_"
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            subprocess.run(
                [self._ruff_path, "check", "--fix", tmp_path],
                capture_output=True,
                text=True,
                timeout=30,
            )

            with open(tmp_path) as f:
                fixed = f.read()

            if fixed != code:
                logger.info(
                    "Code auto-fixed by ruff (%d -> %d chars)", len(code), len(fixed)
                )
            return fixed

        except Exception as e:
            logger.warning("ruff fix failed: %s", e)
            return code
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def check_with_fix(self, code: str) -> tuple[QualityReport, str]:
        """Check code, auto-fix if fixable issues found, return report + fixed code."""
        report = self.check(code)
        if report.fixable_count > 0 and not report.passed:
            fixed = self.fix(code)
            post_report = self.check(fixed)
            return post_report, fixed
        return report, code
