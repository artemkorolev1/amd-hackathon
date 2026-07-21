"""Tests for CodeQualityCell."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.cells.code_quality import CodeQualityCell


def test_clean_code():
    cell = CodeQualityCell()
    report = cell.check("x = 42\nprint(x)\n")
    if report.summary == "ruff not available -- skipping":
        print("Skipping: ruff not available")
        return
    assert report.passed, f"Clean code should pass: {report.summary}"
    print(f"Clean code: {report.summary} ({report.elapsed_ms:.0f}ms)")


def test_bad_code():
    cell = CodeQualityCell()
    report = cell.check("import os\nimport os\nx = undefined_var\n")
    if report.summary == "ruff not available -- skipping":
        print("Skipping: ruff not available")
        return
    assert not report.passed, "Bad code should fail"
    assert len(report.errors) > 0, "Should have errors"
    print(f"Bad code detected: {report.summary}")


def test_auto_fix():
    cell = CodeQualityCell()
    # Duplicate import is a fixable ruff violation (F811)
    bad = "import os\nimport os\nx = 1\n"
    report, fixed = cell.check_with_fix(bad)
    if report.summary == "ruff not available -- skipping":
        print("Skipping: ruff not available")
        return
    # Should have removed the duplicate import
    assert fixed == "x = 1\n", f"Fix didn't work: {fixed!r}"
    print(f"Auto-fix: {bad!r} -> {fixed!r}")


def test_artifact_output():
    cell = CodeQualityCell()
    code = "def foo():\n    pass\n"
    report = cell.check(code)
    if report.summary == "ruff not available -- skipping":
        print("Skipping: ruff not available")
        return
    # Turn into artifact
    from agent.cell import Artifact

    artifact = Artifact(
        source="code_quality_cell",
        content=code,
        metadata={
            "passed": report.passed,
            "errors": len(report.errors),
            "warnings": len(report.warnings),
            "summary": report.summary,
            "elapsed_ms": report.elapsed_ms,
        },
    )
    assert artifact.source == "code_quality_cell"
    assert not artifact.metadata["errors"]
    print(f"Artifact OK: {artifact.metadata['summary']}")


if __name__ == "__main__":
    test_clean_code()
    test_bad_code()
    test_auto_fix()
    test_artifact_output()
    print("\nAll CodeQualityCell tests passed!")
