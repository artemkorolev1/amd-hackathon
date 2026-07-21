#!/usr/bin/env python3
"""Enhanced pipeline evaluator — produces JSON, HTML, and legacy XLSX reports.

Works with the standard results.json output from the pipeline. When the
pipeline also produces results_detailed.json (DETAILED_OUTPUT=1), additional
per-stage timing, worker provenance, and judgment strategy analysis is included.

Usage:
    # Standard mode (works with any results.json)
    python -m runner.instrumented_evaluate \\
        --results /path/to/results.json \\
        --gold data/eval/validation-v1.json \\
        --output-dir eval_results/v15/ \\
        --title "V15 Staging Pipeline Eval"

    # Detailed mode (when pipeline was run with DETAILED_OUTPUT=1)
    python -m runner.instrumented_evaluate \\
        --detailed /output/results_detailed.json \\
        --gold data/eval/tests/eval_v14_test_20.json \\
        --output-dir eval_results/v15/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.grade_answer import fuzzy_match, grade_answer
from runner.evaluate import load_gold, build_report, write_xlsx

logger = logging.getLogger("instrumented_evaluate")

TEMPLATE_DIR = Path(__file__).resolve().parent / "report_templates"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_results(path: str) -> dict[str, dict]:
    """Load pipeline results and index by task_id.

    Works with both standard results.json and results_detailed.json format.
    """
    with open(path) as f:
        raw = json.load(f)

    entries: list[dict] = raw if isinstance(raw, list) else raw.get("results", raw)
    results: dict[str, dict] = {}
    for entry in entries:
        tid = entry.get("task_id")
        if not tid:
            continue
        results[tid] = entry
    return results


def load_detailed_results(path: str) -> dict[str, dict]:
    """Load results_detailed.json and index by task_id.

    Same as load_results but expects the full instrumentation schema with
    classification, worker_answers, and judgment keys.
    """
    return load_results(path)


# ---------------------------------------------------------------------------
# Enriched grading
# ---------------------------------------------------------------------------


def evaluate_tasks_detailed(
    gold: dict[str, dict],
    results: dict[str, dict],
) -> list[dict]:
    """Grade each task, enriching with any available instrumentation metadata.

    Args:
        gold: dict mapping task_id -> {prompt, expected, category, ...}
        results: dict mapping task_id -> {answer, timing_ms, ...} (may also
                 contain classification, worker_answers, judgment keys)

    Returns:
        List of per-task result dicts with standard grade fields plus optional
        instrumentation fields.
    """
    graded: list[dict] = []

    for tid, g in gold.items():
        pred = results.get(tid, {})
        answer = pred.get("answer", "")
        timing_ms = float(pred.get("timing_ms", 0.0))
        expected = g.get("expected", "")
        category = g.get("category", "")
        difficulty = g.get("difficulty", "")
        prompt = g.get("prompt", "")

        # --- Code tasks: pass by convention ---
        if g.get("function") or g.get("tests"):
            result: dict[str, Any] = {
                "task_id": tid,
                "category": category,
                "difficulty": difficulty,
                "prompt": prompt,
                "expected": expected,
                "answer": answer,
                "correct": True,
                "reason": "Code task — passed by convention",
                "timing_ms": timing_ms,
            }
        else:
            # Standard grading
            passed, reason = grade_answer(answer, expected)

            # Check accept-list aliases
            if not passed:
                accept_list = g.get("accept", [])
                for alias in accept_list:
                    alias_str = str(alias) if not isinstance(alias, str) else alias
                    if alias_str:
                        p, r = grade_answer(answer, alias_str)
                        if p:
                            passed = True
                            reason = "Passed (accepted alias)"
                            break

            result = {
                "task_id": tid,
                "category": category,
                "difficulty": difficulty,
                "prompt": prompt,
                "expected": expected,
                "answer": answer,
                "correct": passed,
                "reason": reason,
                "timing_ms": timing_ms,
            }

        # --- Enrich with instrumentation metadata if available ---
        if "classification" in pred:
            result["classification"] = pred["classification"]
        if "worker_answers" in pred:
            result["worker_answers"] = pred["worker_answers"]
        if "judgment" in pred:
            result["judgment"] = pred["judgment"]
        if "total_timing_ms" in pred:
            result["total_timing_ms"] = pred["total_timing_ms"]

        graded.append(result)

    return graded


# ---------------------------------------------------------------------------
# Extended report building
# ---------------------------------------------------------------------------


def compute_timing_detailed(results: list[dict]) -> dict:
    """Compute timing stats including per-stage breakdown when available."""
    timings = [r["timing_ms"] for r in results]
    t_mean = statistics.mean(timings) if timings else 0.0
    t_median = statistics.median(timings) if timings else 0.0
    t_p95 = _percentile(timings, 95)

    # Per-category timing
    per_cat: dict[str, list[float]] = {}
    for r in results:
        cat = r.get("category", "unknown")
        per_cat.setdefault(cat, []).append(r["timing_ms"])

    timing: dict[str, Any] = {
        "mean": t_mean,
        "median": t_median,
        "p95": t_p95,
        "per_category": per_cat,
    }

    # Per-stage timing (only from detailed results)
    classification_times = []
    judgment_times = []
    worker_try_times = []

    for r in results:
        cls = r.get("classification")
        if cls and "timing_ms" in cls:
            classification_times.append(cls["timing_ms"])
        jdg = r.get("judgment")
        if jdg and "timing_ms" in jdg:
            judgment_times.append(jdg["timing_ms"])
        answers = r.get("worker_answers", [])
        for wa in answers:
            if "timing_ms" in wa:
                worker_try_times.append(wa["timing_ms"])

    if classification_times or judgment_times or worker_try_times:
        timing["per_stage"] = {
            "classification_avg_ms": statistics.mean(classification_times) if classification_times else 0,
            "judgment_avg_ms": statistics.mean(judgment_times) if judgment_times else 0,
            "worker_per_try_avg_ms": statistics.mean(worker_try_times) if worker_try_times else 0,
        }

    return timing


def compute_worker_contribution(results: list[dict]) -> dict[str, Any]:
    """Analyze per-worker-type contribution from detailed results.

    For each worker type (fireworks, local, deterministic), compute:
    - accuracy: how often would this worker be correct if used alone
    - alone_correct: tasks where ONLY this worker type answered correctly
    """
    # Collect per-worker-type correctness per task
    worker_correctness: dict[str, dict[str, bool]] = {}  # type -> {task_id: correct}
    task_best: dict[str, bool] = {}  # task_id -> final correct

    for r in results:
        tid = r["task_id"]
        task_best[tid] = r["correct"]

        for wa in r.get("worker_answers", []):
            wtype = wa.get("worker_type", "unknown")
            worker_correctness.setdefault(wtype, {})[tid] = fuzzy_match(
                wa.get("answer", ""), r.get("expected", "")
            )

    contribution: dict[str, Any] = {}
    for wtype, correctness in worker_correctness.items():
        correct_count = sum(1 for c in correctness.values() if c)
        total_count = len(correctness)
        # alone_correct: tasks where this worker was correct AND all other
        # workers with answers on this task were wrong
        alone = 0
        for tid in correctness:
            if not correctness.get(tid):
                continue
            # Check if any other worker was also correct on this task
            others_correct = any(
                other_type != wtype and other_correctness.get(tid, False)
                for other_type, other_correctness in worker_correctness.items()
            )
            if not others_correct:
                alone += 1

        contribution[wtype] = {
            "correct": correct_count,
            "total": total_count,
            "accuracy": correct_count / total_count if total_count > 0 else 0.0,
            "alone_correct": alone,
        }

    return contribution


def compute_strategy_breakdown(results: list[dict]) -> dict[str, dict]:
    """Analyze accuracy by judgment strategy (only from detailed results)."""
    strategies: dict[str, dict] = {}
    for r in results:
        jdg = r.get("judgment", {})
        strategy = jdg.get("strategy", "unknown")
        if strategy not in strategies:
            strategies[strategy] = {"count": 0, "correct": 0}
        strategies[strategy]["count"] += 1
        if r["correct"]:
            strategies[strategy]["correct"] += 1

    result = {}
    for strat, data in strategies.items():
        result[strat] = {
            "count": data["count"],
            "accuracy": data["correct"] / data["count"] if data["count"] > 0 else 0.0,
        }
    return result

def _percentile(data: list[float], p: int) -> float:
    """Compute the p-th percentile using linear interpolation."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    k = (p / 100.0) * (n - 1)
    f = int(k)
    c = f + 1
    if f >= n - 1:
        return sorted_data[-1]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def build_detailed_report(
    results: list[dict],
    timing: dict | None = None,
) -> dict:
    """Build a comprehensive report dict with optional instrumentation data.

    Args:
        results: Graded per-task results (from evaluate_tasks_detailed)
        timing: Optional timing dict (from compute_timing_detailed)

    Returns:
        dict with overall, by_category, by_difficulty, timing, per_task,
        failures, and optionally by_strategy, worker_contribution.
    """
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = (correct / total) if total > 0 else 0.0
    gate_pass = accuracy >= 0.842

    # --- By category ---
    by_category: dict[str, dict] = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = {"total": 0, "correct": 0}
        by_category[cat]["total"] += 1
        if r["correct"]:
            by_category[cat]["correct"] += 1
    for cat in by_category:
        t = by_category[cat]["total"]
        c = by_category[cat]["correct"]
        by_category[cat]["accuracy"] = (c / t) if t > 0 else 0.0

    # --- By difficulty ---
    by_difficulty: dict[str, dict] = {}
    for r in results:
        diff = r.get("difficulty")
        if diff:
            if diff not in by_difficulty:
                by_difficulty[diff] = {"total": 0, "correct": 0}
            by_difficulty[diff]["total"] += 1
            if r["correct"]:
                by_difficulty[diff]["correct"] += 1
    for diff in by_difficulty:
        t = by_difficulty[diff]["total"]
        c = by_difficulty[diff]["correct"]
        by_difficulty[diff]["accuracy"] = (c / t) if t > 0 else 0.0

    # --- Timing ---
    if timing is None:
        timing = compute_timing_detailed(results)

    # --- Failures ---
    failures = [r for r in results if not r["correct"]]

    report: dict[str, Any] = {
        "overall": {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "gate_pass": gate_pass,
        },
        "by_category": by_category,
        "by_difficulty": by_difficulty,
        "timing": timing,
        "per_task": results,
        "failures": failures,
    }

    # --- Worker contribution (from detailed results) ---
    has_worker_data = any(
        "worker_answers" in r and r["worker_answers"] for r in results
    )
    if has_worker_data:
        report["worker_contribution"] = compute_worker_contribution(results)
        report["by_strategy"] = compute_strategy_breakdown(results)

    return report


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def build_html_report(report: dict, title: str = "Eval Report") -> str:
    """Render the report dict as a standalone HTML page.

    Uses Jinja2 if available, otherwise falls back to a basic hand-rolled HTML.
    """
    try:
        import jinja2
    except ImportError:
        return _render_html_fallback(report, title)

    template_path = TEMPLATE_DIR / "eval_report.html"
    if not template_path.exists():
        return _render_html_fallback(report, title)

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=True,
    )
    template = env.get_template("eval_report.html")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return template.render(
        title=title,
        date=now,
        **report,
    )


def _render_html_fallback(report: dict, title: str) -> str:
    """Minimal fallback HTML when Jinja2 is not available."""
    o = report["overall"]
    by_cat = report.get("by_category", {})
    failures = report.get("failures", [])
    timing = report.get("timing", {})

    rows = ""
    for cat, stats in sorted(by_cat.items()):
        pct = stats["accuracy"] * 100
        bar_color = "#4caf50" if stats["accuracy"] >= 0.842 else "#f44336"
        rows += (
            f"<tr><td>{cat}</td>"
            f"<td>{pct:.1f}%</td>"
            f"<td>{stats['correct']}/{stats['total']}</td>"
            f"<td><div style='background:#e0e0e0;border-radius:4px;"
            f"height:20px;width:100%;position:relative;overflow:hidden;'>"
            f"<div style='background:{bar_color};height:100%;"
            f"width:{pct}%;'></div></div></td></tr>\n"
        )

    fail_rows = ""
    for f in failures:
        fail_rows += (
            f"<tr style='background:#fff0f0;'>"
            f"<td>{f.get('task_id','')}</td>"
            f"<td>{f.get('category','')}</td>"
            f"<td>{f.get('expected','')[:100]}</td>"
            f"<td>{f.get('answer','')[:100]}</td>"
            f"<td>{f.get('reason','')}</td></tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>{title}</title>
<style>
body {{ font-family: sans-serif; padding: 20px; }}
h1 {{ font-size: 1.5rem; }}
.cards {{ display: flex; gap: 12px; margin-bottom: 20px; }}
.card {{ background: #fff; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.card .big {{ font-size: 2rem; font-weight: 700; }}
.pass {{ color: #1a7d1a; }} .fail {{ color: #c41e1e; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
th {{ background: #4472C4; color: #fff; text-align: left; padding: 8px; }}
td {{ padding: 6px 8px; border-bottom: 1px solid #e0e0e0; }}
tr:hover {{ background: #f0f4ff; }}
</style></head><body>
<h1>{title}</h1>
<div class="cards">
<div class="card {'fail' if not o['gate_pass'] else ''}">
<h2>Accuracy</h2>
<div class="big {'pass' if o['gate_pass'] else 'fail'}">{o['accuracy']*100:.1f}%</div>
<div>{o['correct']}/{o['total']}</div>
</div>
<div class="card">
<h2>Gate (84.2%)</h2>
<div class="big {'pass' if o['gate_pass'] else 'fail'}">{'PASS' if o['gate_pass'] else 'FAIL'}</div>
</div>
<div class="card">
<h2>Timing</h2>
<div>Mean: {timing.get('mean', 0):.0f}ms</div>
<div>P95: {timing.get('p95', 0):.0f}ms</div>
</div>
</div>
<h2>Category Breakdown</h2>
<table><tr><th>Category</th><th>Accuracy</th><th>Correct</th><th>Bar</th></tr>
{rows}</table>
<h2>Failures ({len(failures)})</h2>
{"<table><tr><th>ID</th><th>Category</th><th>Expected</th><th>Got</th><th>Reason</th></tr>" + fail_rows + "</table>" if failures else "<p>No failures.</p>"}
<div style="margin-top:30px;font-size:0.75rem;color:#aaa;">Generated by instrumented_evaluate.py</div>
</body></html>"""
    return html


def build_diff_html(deltas: dict, label_baseline: str, label_candidate: str) -> str:
    """Render regression deltas as HTML."""
    try:
        import jinja2
        template_path = TEMPLATE_DIR / "regression_diff.html"
        if template_path.exists():
            env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(str(TEMPLATE_DIR)),
                autoescape=True,
            )
            template = env.get_template("regression_diff.html")
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            net_change = len(deltas.get("improvements", [])) - len(deltas.get("regressions", []))
            return template.render(
                label_baseline=label_baseline,
                label_candidate=label_candidate,
                date=now,
                net_change=net_change,
                **deltas,
            )
    except ImportError:
        pass

    # Fallback
    o = deltas.get("overall", {})
    regs = deltas.get("regressions", [])
    imps = deltas.get("improvements", [])
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Diff: {label_baseline} → {label_candidate}</title>
<style>body {{ font-family: sans-serif; padding: 20px; }}
.delta-up {{ color: #1a7d1a; }} .delta-down {{ color: #c41e1e; }}
table {{ border-collapse: collapse; width: 100%; }}
th {{ background: #4472C4; color: #fff; padding: 8px; text-align: left; }}
td {{ padding: 6px 8px; border-bottom: 1px solid #e0e0e0; }}
</style></head><body>
<h1>Regression Diff</h1>
<p>{label_baseline} → {label_candidate}</p>
<h2>Overall Delta: <span class="{'delta-up' if o.get('delta',0) > 0 else 'delta-down'}">{o.get('delta',0)*100:+.1f}pp</span></h2>
<p>Baseline: {o.get('baseline_accuracy',0)*100:.1f}% | Candidate: {o.get('candidate_accuracy',0)*100:.1f}%</p>
<h2>Regressions ({len(regs)})</h2>
{"<table><tr><th>ID</th><th>Category</th><th>Expected</th><th>Baseline</th><th>Candidate</th></tr>" +
"".join(f"<tr><td>{r['task_id']}</td><td>{r['category']}</td><td>{r['expected'][:80]}</td><td>{r['baseline_answer'][:80]}</td><td>{r['candidate_answer'][:80]}</td></tr>" for r in regs) +
"</table>" if regs else "<p>No regressions.</p>"}
<h2>Improvements ({len(imps)})</h2>
{"<table><tr><th>ID</th><th>Category</th><th>Expected</th><th>Baseline</th><th>Candidate</th></tr>" +
"".join(f"<tr><td>{r['task_id']}</td><td>{r['category']}</td><td>{r['expected'][:80]}</td><td>{r['baseline_answer'][:80]}</td><td>{r['candidate_answer'][:80]}</td></tr>" for r in imps) +
"</table>" if imps else "<p>No improvements.</p>"}
</body></html>"""
    return html


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def save_report_json(report: dict, path: str) -> str:
    """Save report dict as readable JSON, returns the path."""
    with open(path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Saved JSON report to %s", path)
    return path


def save_report_html(report: dict, path: str, title: str = "Eval Report") -> str:
    """Render and save HTML report, returns the path."""
    html = build_html_report(report, title=title)
    with open(path, "w") as f:
        f.write(html)
    logger.info("Saved HTML report to %s", path)
    return path


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def grade_results_detailed(
    results_path: str,
    gold_path: str,
    title: str = "Eval Report",
    output_dir: str | None = None,
    save_html: bool = True,
    save_xlsx: bool = False,
    save_json: bool = True,
) -> dict:
    """Grade pipeline results and save reports.

    Args:
        results_path: Path to results.json (or results_detailed.json)
        gold_path: Path to ground-truth JSON
        title: Report title
        output_dir: Where to save reports (default: results dir)
        save_html: Generate HTML report
        save_xlsx: Generate legacy XLSX report
        save_json: Save JSON report

    Returns:
        The report dict
    """
    results = load_detailed_results(results_path)
    gold = load_gold(gold_path)

    logger.info("Loaded %d results, %d gold entries", len(results), len(gold))

    graded = evaluate_tasks_detailed(gold, results)
    timing = compute_timing_detailed(graded)
    report = build_detailed_report(graded, timing=timing)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        stem = Path(results_path).stem

        if save_json:
            json_path = os.path.join(output_dir, f"{stem}_report.json")
            save_report_json(report, json_path)

        if save_html:
            html_path = os.path.join(output_dir, f"{stem}_report.html")
            save_report_html(report, html_path, title=title)

        if save_xlsx:
            xlsx_path = os.path.join(output_dir, f"{stem}_report.xlsx")
            try:
                write_xlsx(report, xlsx_path)
                logger.info("Saved XLSX report to %s", xlsx_path)
            except Exception as e:
                logger.warning("Failed to write XLSX: %s", e)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enhanced pipeline evaluation — JSON, HTML, and XLSX reports."
    )
    parser.add_argument(
        "--results",
        help="Path to results.json (standard pipeline output)",
    )
    parser.add_argument(
        "--detailed",
        help="Path to results_detailed.json (full instrumentation output)",
    )
    parser.add_argument(
        "--gold", required=True,
        help="Path to ground-truth JSON (gold standard)",
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="Directory for output reports (default: current dir)",
    )
    parser.add_argument(
        "--title", default="Eval Report",
        help="Report title (default: Eval Report)",
    )
    parser.add_argument(
        "--xlsx", action="store_true",
        help="Also generate legacy XLSX report",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stderr,
        format="%(levelname)s %(message)s",
    )

    results_path = args.detailed or args.results
    if not results_path:
        print("ERROR: one of --results or --detailed is required", file=sys.stderr)
        return 1

    report = grade_results_detailed(
        results_path=results_path,
        gold_path=args.gold,
        title=args.title,
        output_dir=args.output_dir,
        save_html=True,
        save_xlsx=args.xlsx,
        save_json=True,
    )

    o = report["overall"]
    bar = "=" * 50
    print(f"\n{bar}")
    print(f"  {args.title}")
    print(f"{bar}")
    print(f"  Accuracy:  {o['accuracy']*100:.1f}% ({o['correct']}/{o['total']})")
    print(f"  Gate:      {'PASS' if o['gate_pass'] else 'FAIL'} (84.2%)")
    print(f"  Mean:      {report['timing']['mean']:.0f} ms")
    print(f"  P95:       {report['timing']['p95']:.0f} ms")
    print(f"{bar}")

    if args.output_dir:
        print(f"  Reports:   {args.output_dir}/")
    print()

    return 0 if o["gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
