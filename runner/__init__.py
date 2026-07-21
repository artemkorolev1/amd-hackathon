"""Runner — parallel orchestration, evaluation, and deployment wrapper.

Exports the public API for local evaluation runs:
  BatchRunner   — parallel task execution via multiprocessing
  run_parallel  — convenience wrapper
  evaluate_tasks — grade pipeline output against ground truth
  build_report   — aggregate results into structured report
  write_xlsx     — write report as Excel workbook
  build_image    — Docker build automation
  push_image     — Docker push automation
  verify_image   — Docker image import sanity check

All of these wrap the untouchable agent.Pipeline for local evaluation.
"""

from .batch_runner import BatchRunner, run_parallel
from .deploy import build_image, push_image, verify_image
from .evaluate import evaluate_tasks, build_report, write_xlsx
from .instrumented_evaluate import (
    evaluate_tasks_detailed,
    build_detailed_report,
    build_html_report,
    grade_results_detailed,
)
from .regression import compute_deltas, compare_results

__all__ = [
    "BatchRunner",
    "run_parallel",
    "evaluate_tasks",
    "build_report",
    "write_xlsx",
    "evaluate_tasks_detailed",
    "build_detailed_report",
    "build_html_report",
    "grade_results_detailed",
    "compute_deltas",
    "compare_results",
    "build_image",
    "push_image",
    "verify_image",
]
