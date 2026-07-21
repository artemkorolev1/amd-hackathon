# Comprehensive Evaluation System Improvement Plan

> **Project:** AMD ACT II Hackathon — Track 1 Token Efficient Routing Agent  
> **Root:** `/home/artem/dev/amd-hackathon/`  
> **Date:** 2026-07-13  
> **Context:** Parallel staging pipeline (`staging/`) with 4-worker majority-vote architecture  
> **Grader constraints:** 2 vCPU, 4 GB RAM, 600s deadline, no GPU, optional Fireworks API  
> **Golden Rule:** The `agent/` directory is **untouchable**. All new code lives alongside or outside it.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Analysis](#2-current-state-analysis)
3. [Architecture Overview](#3-architecture-overview)
4. [File-by-File Implementation](#4-file-by-file-implementation)
5. [Data Model & Output Formats](#5-data-model--output-formats)
6. [Test Data Requirements](#6-test-data-requirements)
7. [Running the Evaluation](#7-running-the-evaluation)
8. [Interpretation Guide](#8-interpretation-guide)
9. [Scenarios & Usage Patterns](#9-scenarios--usage-patterns)
10. [Integration with Grader Constraints](#10-integration-with-grader-constraints)
11. [Appendix: Report Schema](#11-appendix-report-schema)

---

## 1. Executive Summary

The current evaluation system (`runner/evaluate.py`) grades pipeline output against ground truth using fuzzy match and produces XLSX reports. It is a **flat, post-hoc grader** — it sees only final answers and a single `timing_ms` per task. It has no visibility into the staging pipeline's internal structure: classification phase, worker-specific answers, per-try timing, judgment strategy, or vote distribution.

This plan extends the evaluation system to **instrument the full pipeline end-to-end**. It adds:

- **Rich output mode** — the pipeline writes a detailed, per-task instrumentation JSON alongside the grader-contract output
- **Per-stage timing** — classification, per-worker per-try, judgment
- **Worker provenance tracking** — which worker produced which answer, with model type and parameters
- **Ablation config** — env-var-driven toggles for worker types, voting, temperature settings
- **Regression comparison** — diff reports between two evaluation runs
- **Multi-format reports** — JSON, HTML (with embedded diagnostic plots), and the existing XLSX
- **Graceful degradation tests** — deliberately induced failures to verify pipeline resilience

### New files (all in `runner/` or `scripts/`):

| File | Purpose |
|------|---------|
| `runner/instrumented_evaluate.py` | Enhanced evaluator: loads rich pipeline output, produces JSON+HTML+XLSX reports |
| `runner/regression.py` | Compare two evaluation runs, report accuracy deltas |
| `runner/ablation_runner.py` | Run catalog of ablation experiments with config diff report |
| `runner/degradation_test.py` | Graceful degradation test suite |
| `scripts/grade_answer.py` | Shared grading logic (extracted from `scripts/evaluate.py` for reuse) |
| `runner/report_templates/eval_report.html` | Jinja2 HTML report template |
| `runner/report_templates/regression_diff.html` | Jinja2 regression diff template |

### Modified files:

| File | Change |
|------|--------|
| `staging/entrypoint.py` | Add `DETAILED_OUTPUT=1` mode that writes `results_detailed.json` with full instrumentation |
| `staging/ready_judge.py` | Expose detailed judgment metadata (group sizes, vote distribution) |
| `staging/ready_pool.py` | Expose per-worker timing breakdown |
| `staging/ready_config.py` | Add ablation toggle env vars |
| `staging/workers/*.py` | Include model info + temperature in answer metadata |

---

## 2. Current State Analysis

### 2.1 What exists now

**`runner/evaluate.py`** — Grade-only post-processor:
- `load_gold()` — reads ground truth in two formats (heldout dict-style, plain answer-style)
- `load_predictions()` — reads pipeline output as `[{"task_id", "answer", "timing_ms"}]`
- `evaluate_tasks()` — grades each task using `scripts/evaluate.fuzzy_match` cascade
- `build_report()` — aggregates into overall, per-category, per-difficulty, timing stats
- `write_xlsx()` — three-sheet Excel workbook (Summary, Details, Failures)
- `grade_results()` — convenience wrapper

**`scripts/evaluate.py`** — Official grading functions (both used by runner and by grader harness):
- `fuzzy_match(answer, expected)` — 4-strategy cascade
- `grade_answer(answer, expected)` — wraps fuzzy_match with diagnostics
- `run_agent()` / `load_tasks()` / `load_ground_truth()` — legacy CLI harness

**`staging/entrypoint.py`** — Pipeline output (currently `results.json`):
```json
[{"task_id": "fact-1", "answer": "Canberra"}]
```
All internal metadata (`_judgment`, per-worker answers, timing breakdowns) is **stripped** before writing. This satisfies the grader contract but makes detailed evaluation impossible.

**`staging/ready_judge.py`** — Judge already tracks per-task:
- `strategy` (majority_3plus, majority_2plus, escalate_fireworks, fallback_best, etc.)
- `votes_cast`, `largest_group`, `num_groups`
- But this metadata only exists in memory and is stripped from final output.

### 2.2 Gaps

| Gap | Impact |
|-----|--------|
| No per-stage timing | Cannot identify bottlenecks (classification vs worker inference vs judgment) |
| No worker provenance | Cannot attribute which worker type contributed which answer |
| No vote distribution | Cannot analyze majority quality (weak vs strong consensus) |
| No ablation support | Requires manual env-var changes and separate runs |
| No regression comparison | No automated way to compare two builds |
| No HTML/plots | Reports are XLSX-only, requiring manual analysis |
| No degradation tests | Pipeline resilience is untested |
| Single timing_ms per task | Cannot separate model inference time from overhead |
| Per-try metadata lost | Temperature, model, worker type not recorded per-try |

---

## 3. Architecture Overview

### 3.1 Data Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                        staging/entrypoint.py                         │
│                                                                      │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │ Read     │───→│ Bulk Classify│───→│ ReadyQueue   │               │
│  │ tasks    │    │ (timing_t1)  │    │              │               │
│  └──────────┘    └──────────────┘    └──────┬───────┘               │
│                                             │                        │
│                                    ┌────────▼────────┐               │
│                                    │   ReadyPool     │               │
│                                    │  dispatch_loop  │               │
│                                    │                 │               │
│                                    │  ┌─ fw_worker   │── try(t=0.1) │
│                                    │  │              │── try(t=0.3) │
│                                    │  │              │── ...        │
│                                    │  ├─ loc_worker  │── try(t=0.1) │
│                                    │  ├─ det_worker_0│── try(same)  │
│                                    │  └─ det_worker_1│── try(same)  │
│                                    └────────┬────────┘               │
│                                             │                        │
│                                    ┌────────▼────────┐               │
│                                    │   ReadyJudge    │               │
│                                    │  judge_all()    │               │
│                                    └────────┬────────┘               │
│                                             │                        │
│                    ┌────────────────────────┼──────────────┐         │
│                    ▼                        ▼              ▼         │
│           ┌──────────────┐       ┌──────────────────┐ ┌──────────┐  │
│           │ results.json │       │results_detailed  │ │ timing   │  │
│           │ (contract)   │       │.json (full inst- │ │ .json    │  │
│           └──────────────┘       │ rumentation)     │ └──────────┘  │
│                                  └──────────────────┘               │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     runner/instrumented_evaluate.py                   │
│                                                                      │
│  ┌────────────┐    ┌──────────────────┐    ┌──────────────────────┐  │
│  │ Load gold  │───→│ Grade each task  │───→│ Build report dict    │  │
│  │ + detailed │    │ (reuse fuzzy_    │    │ (overall, cat, diff, │  │
│  │ results    │    │  match cascade)  │    │  per-worker, timing) │  │
│  └────────────┘    └──────────────────┘    └───────┬──────────────┘  │
│                                                     │                 │
│                          ┌──────────────────────────┼──────────┐     │
│                          ▼                          ▼          ▼     │
│                   ┌────────────┐            ┌──────────┐  ┌──────┐  │
│                   │ report.json │            │report.html│  │report│  │
│                   │(machine    │            │(rich HTML │  │.xlsx │  │
│                   │ readable)  │            │ + plots)  │  │(legacy)│ │
│                   └────────────┘            └──────────┘  └──────┘  │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     runner/regression.py                              │
│                                                                      │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │ Load baseline│───→│ Compute per-task  │───→│ Generate diff    │   │
│  │ report.json  │    │ accuracy delta    │    │ HTML/JSON        │   │
│  └──────────────┘    └──────────────────┘    └──────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 Instrumentation Contract

The pipeline entrypoint gets a new output mode. When `DETAILED_OUTPUT=1`:

**`/output/results_detailed.json`** — Full instrumentation (never sent to grader):
```json
[
  {
    "task_id": "fact-1",
    "answer": "Canberra",
    "category": "factual",
    "category_4way": "knowledge",
    "classification": {
      "category": "factual",
      "confidence": 0.85,
      "score_delta": 0.42,
      "raw_scores": {"factual": 2.5, "math": 0.5, ...},
      "timing_ms": 1.2
    },
    "worker_answers": [
      {
        "worker_id": "fw_worker_0",
        "worker_type": "fireworks",
        "model_id": "accounts/fireworks/models/deepseek-v4-flash",
        "try_index": 0,
        "temperature": 0.1,
        "answer": "Canberra",
        "timing_ms": 1234.5
      },
      {
        "worker_id": "loc_worker_0",
        "worker_type": "local",
        "model_id": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "try_index": 0,
        "temperature": 0.1,
        "answer": "Canberra",
        "timing_ms": 3456.7
      },
      {
        "worker_id": "det_worker_0",
        "worker_type": "deterministic",
        "model_id": "factual_qa_solver",
        "try_index": 0,
        "temperature": null,
        "answer": "Canberra",
        "timing_ms": 0.5
      }
      // ... up to 5 answers total
    ],
    "judgment": {
      "strategy": "majority_3plus",
      "votes_cast": 5,
      "largest_group": 3,
      "num_groups": 2,
      "group_sizes": {"Canberra": 3, "Sydney": 2},
      "winner_answer": "Canberra",
      "timing_ms": 0.3
    },
    "total_timing_ms": 4693.0,
    "judged_by": "ready_judge"
  }
]
```

**`/output/timing.json`** — Aggregate timing summary:
```json
{
  "classification_s": 0.5,
  "worker_setup_s": 12.3,
  "dispatch_loop_s": 45.2,
  "judgment_s": 0.8,
  "total_s": 58.8,
  "per_worker_type": {
    "fireworks": {"total_s": 25.1, "per_try_avg_ms": 1250},
    "local": {"total_s": 320.0, "per_try_avg_ms": 3200},
    "deterministic": {"total_s": 1.2, "per_try_avg_ms": 0.5}
  }
}
```

### 3.3 Ablation Configuration

Environment variables that toggle pipeline features for evaluation:

| Env Var | Values | Default | Effect |
|---------|--------|---------|--------|
| `ABLATION_DISABLE_FIREWORKS` | 0/1 | 0 | Skip FW workers entirely |
| `ABLATION_DISABLE_LOCAL` | 0/1 | 0 | Skip local workers entirely |
| `ABLATION_DISABLE_DETERMINISTIC` | 0/1 | 0 | Skip deterministic workers |
| `ABLATION_DISABLE_VOTING` | 0/1 | 0 | Pick first answer instead of majority |
| `ABLATION_VOTES` | 1-5 | 5 | Override `STAGING_JUDGMENT_VOTES` |
| `ABLATION_TEMPERATURE_SWEEP` | csv | 0.1,0.3,0.5,0.7,0.9 | Override temperature settings |
| `ABLATION_TIEBREAKER` | str | fw_priority | Override tiebreaker strategy |
| `ABLATION_SINGLE_WORKER` | str | "" | If set, only run this worker type |
| `ABLATION_FORCE_CRASH` | 0/1 | 0 | Simulate worker crash (for degradation test) |
| `ABLATION_EMPTY_MODEL_PATH` | 0/1 | 0 | Set empty model path for local worker |

---

## 4. File-by-File Implementation

### 4.1 `scripts/grade_answer.py` (NEW — 120 lines)

**Purpose:** Extract the grading logic (`fuzzy_match`, `grade_answer`) from `scripts/evaluate.py` into a reusable module that both the legacy CLI and the new evaluator can import.

```python
"""scripts/grade_answer.py — Reusable answer grading logic.

Extracted from scripts/evaluate.py for shared use by runner/evaluate.py
and runner/instrumented_evaluate.py.
"""

import re
from typing import List, Tuple

def extract_numbers(s: str) -> List[float]:
    """Extract all decimal numbers from a string."""
    ...

def tokenize(s: str) -> set:
    """Lowercase, split on non-alphanumeric, return non-empty tokens."""
    ...

def fuzzy_match(answer: str, expected: str) -> bool:
    """4-strategy cascade: exact → substring → numeric 1% → token overlap."""
    ...

def grade_answer(answer: str, expected: str) -> Tuple[bool, str]:
    """Grade a single answer. Returns (passed, reason_string)."""
    ...
```

**Implementation notes:**
- Pure copy of `fuzzy_match` and `grade_answer` from `scripts/evaluate.py`
- No argparse, no CLI, no file I/O — just pure functions
- Imported by both old and new evaluators
- Keep both copies in sync (or have `scripts/evaluate.py` re-import from here)

### 4.2 `staging/entrypoint.py` — Modify (add ~40 lines)

**Changes:**

1. **New env var `DETAILED_OUTPUT`** — when set to `1`, write additional output files.

2. **Wrap classification with timing:**
   ```python
   t_classify = time.monotonic()
   classified = classify_batch(prompts)
   t_classify_elapsed = time.monotonic() - t_classify
   timing_data["classification_s"] = t_classify_elapsed
   ```

3. **Attach classification metadata to each task:**
   ```python
   ready_task.classification_meta = {
       "raw_scores": cls.get("raw_scores", {}),
       "confidence": cls.get("confidence", 0.5),
       "score_delta": cls.get("score_delta", 0.0),
       "timing_ms": t_classify_elapsed * 1000 / len(prompts),
   }
   ```

4. **After judge_all(), build detailed output before stripping:**
   ```python
   if os.environ.get("DETAILED_OUTPUT") == "1":
       detailed_output = _build_detailed_output(ready_tasks, judge, timing_data)
       _write_output(detailed_output, "results_detailed.json")
       _write_output(timing_data, "timing.json")
   ```

5. **Add `_build_detailed_output()` function:**
   - Iterates over final_results
   - Looks up per-task answers from judge's `_task_answers` 
   - Attaches classification meta from ReadyTask objects
   - Attaches judgment metadata from `_judged` dict
   - Computes per-try timing breakdown

See Section 5 for the full output schema.

### 4.3 `staging/ready_judge.py` — Modify (add ~20 lines)

**Changes:**

1. **Expose judgment metadata with group details:**
   In `judge()` method, add to the `meta` dict:
   ```python
   meta["group_sizes"] = {answer: len(indices) for answer, indices in groups.items()}
   meta["vote_distribution"] = [
       {"worker_id": a["worker_id"], "answer": a["answer"]}
       for a in answers
   ]
   ```

2. **Expose `get_answer_details(task_id)` method:**
   ```python
   def get_answer_details(self, task_id: str) -> list[dict]:
       """Return raw answers for a task (with worker_id, timing, etc.)."""
       return self._task_answers.get(task_id, [])
   ```

### 4.4 `staging/ready_pool.py` — Modify (add ~15 lines)

**Changes:**

1. **Track per-worker-type timing in `dispatch_loop`:**
   ```python
   # At the end of a successful answer collection:
   self._worker_timing[worker_type] = self._worker_timing.get(worker_type, []) + [elapsed]
   ```

2. **Expose `get_timing_summary()` property:**
   ```python
   @property
   def timing_summary(self) -> dict:
       """Return per-worker-type aggregate timing."""
       return {
           wtype: {
               "total_s": sum(times) / 1000,
               "count": len(times),
               "avg_ms": statistics.mean(times) if times else 0,
               "p95_ms": sorted(times)[int(len(times)*0.95)] if times else 0,
           }
           for wtype, times in self._worker_timing.items()
       }
   ```

### 4.5 `staging/ready_config.py` — Modify (add ~20 lines)

**Changes:**

1. **Add ablation env vars to `from_env()`:**
   ```python
   self.ablation_disable_fireworks = int(os.environ.get("ABLATION_DISABLE_FIREWORKS", "0"))
   self.ablation_disable_local = int(os.environ.get("ABLATION_DISABLE_LOCAL", "0"))
   self.ablation_disable_deterministic = int(os.environ.get("ABLATION_DISABLE_DETERMINISTIC", "0"))
   self.ablation_disable_voting = int(os.environ.get("ABLATION_DISABLE_VOTING", "0"))
   self.ablation_votes = int(os.environ.get("ABLATION_VOTES", "0"))
   self.ablation_temperature_sweep = os.environ.get("ABLATION_TEMPERATURE_SWEEP", "")
   self.ablation_tiebreaker = os.environ.get("ABLATION_TIEBREAKER", "")
   self.ablation_single_worker = os.environ.get("ABLATION_SINGLE_WORKER", "")
   self.ablation_force_crash = int(os.environ.get("ABLATION_FORCE_CRASH", "0"))
   self.ablation_empty_model_path = int(os.environ.get("ABLATION_EMPTY_MODEL_PATH", "0"))
   ```

2. **Worker count override from ablation:**
   - If `ABLATION_SINGLE_WORKER=fireworks`, set `det_workers=0`, `loc_workers=0`
   - If `ABLATION_DISABLE_FIREWORKS=1`, set `fw_workers=0`

3. **Temperature sweep override:**
   ```python
   if self.ablation_temperature_sweep:
       temps = [float(t.strip()) for t in self.ablation_temperature_sweep.split(",")]
       # Pass to workers via config; workers read from config instead of hardcoded list
   ```

### 4.6 `staging/workers/loc_worker.py` — Modify (add ~5 lines)

**Changes:**
- Include model info and temperature in each answer dict:
  ```python
  answers.append({
      "worker_id": self.worker_id,
      "worker_type": "local",
      "model_id": self.config.loc_model_path.split("/")[-1],
      "try_index": i,
      "temperature": temperature,
      "task_id": task.task_id,
      "answer": answer,
      "timing_ms": elapsed,
  })
  ```

### 4.7 `staging/workers/fw_worker.py` — Modify (add ~5 lines)

**Changes:** Same pattern as loc_worker:
```python
answers.append({
    "worker_id": self.worker_id,
    "worker_type": "fireworks",
    "model_id": self._model_id,
    "try_index": i,
    "temperature": temperature,
    "task_id": task.task_id,
    "answer": answer,
    "timing_ms": elapsed,
})
```

### 4.8 `staging/workers/det_worker.py` — Modify (add ~5 lines)

**Changes:**
```python
answers.append({
    "worker_id": self.worker_id,
    "worker_type": "deterministic",
    "model_id": task.category,  # which solver was used
    "try_index": i,
    "temperature": None,
    "task_id": task.task_id,
    "answer": answer,
    "timing_ms": elapsed,
})
```

### 4.9 `runner/instrumented_evaluate.py` (NEW — 500 lines)

**Purpose:** Enhanced evaluator that consumes detailed pipeline output and produces rich multi-format reports.

```python
"""runner/instrumented_evaluate.py — Enhanced pipeline evaluator.

Consumes results_detailed.json (with full instrumentation) and ground truth,
produces JSON report, HTML report (with diagnostic plots), and legacy XLSX.

Usage:
    python -m runner.instrumented_evaluate \
        --detailed /output/results_detailed.json \
        --gold data/eval/tests/eval_v14_test_20.json \
        --output-dir /home/artem/dev/amd-hackathon/eval_results/v15/ \
        --title "V15 Staging Pipeline Eval"
"""

import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

from scripts.grade_answer import fuzzy_match, grade_answer
from runner.evaluate import load_gold, build_report, write_xlsx
```

**Key functions:**

#### `load_detailed_results(path: str) -> dict`
Load `results_detailed.json` and index by `task_id`.

#### `evaluate_detailed(gold: dict, results: dict) -> list[dict]`
Grades each task using `grade_answer`, but also extracts:
- Per-stage timing (classification, per-try, judgment)
- Per-worker-type contribution (did FW answer correctly? did Local?)
- Judgment strategy used
- Vote distribution

Returns enriched per-task dicts.

#### `build_detailed_report(results: list[dict], timing: dict) -> dict`
Extended report dict with:

```python
{
    "overall": {...},  # same as before
    "by_category": {...},
    "by_difficulty": {...},
    "timing": {
        "mean": ...,
        "median": ...,
        "p95": ...,
        "per_category": {...},
        "per_stage": {
            "classification_avg_ms": 1.2,
            "worker_per_try_avg_ms": 2450,
            "judgment_avg_ms": 0.3,
        },
        "per_worker_type": {
            "fireworks": {"avg_ms": 1234, "total_s": 25.1, "tasks_done": 20},
            "local": {"avg_ms": 3456, "total_s": 320.0, "tasks_done": 20},
            "deterministic": {"avg_ms": 0.5, "total_s": 1.2, "tasks_done": 20},
        }
    },
    "by_strategy": {
        "majority_3plus": {"count": 15, "accuracy": 0.93},
        "escalate_fireworks": {"count": 3, "accuracy": 0.67},
        "fallback_best": {"count": 2, "accuracy": 0.50},
    },
    "worker_contribution": {
        "fireworks": {"correct": 12, "total": 20, "accuracy": 0.60, "alone_correct": 2},
        "local": {"correct": 15, "total": 20, "accuracy": 0.75, "alone_correct": 1},
        "deterministic": {"correct": 10, "total": 18, "accuracy": 0.56, "alone_correct": 0},
    },
    "confusion_matrix": {
        "majority_3plus": "High confidence — 3+ answers agree",
        "escalate_fireworks": "Weak consensus — escalated to FW",
        ...
    },
    "per_task": [...],  # enriched
    "failures": [...],
}
```

**Worker contribution analysis:**
- `alone_correct` = tasks where ONLY this worker type answered correctly (others wrong)
- Per-worker accuracy = how often would this worker type be correct if used alone
- `ensemble_accuracy` = actual accuracy with majority vote

#### `build_html_report(report: dict, template_path: str) -> str`
Renders a Jinja2 HTML template with:
- Summary stats cards (accuracy, gate, timing)
- Category breakdown bar chart (inline SVG using matplotlib or plotly, or pure HTML/CSS bar)
- Judgment strategy pie chart
- Per-worker-type accuracy comparison
- Per-stage timing waterfall
- Diff-colored per-task table (green=correct, red=fail)
- Failure analysis section

**If matplotlib is not available:** fall back to pure HTML/CSS bar charts (no hard dep).

#### `generate_diagnostic_plots(report: dict, output_dir: str) -> list[str]`
Generates PNG plots (if matplotlib available):
1. `accuracy_by_category.png` — grouped bar chart
2. `accuracy_by_strategy.png` — judgment strategy effectiveness
3. `worker_contribution.png` — per-worker accuracy comparison
4. `timing_waterfall.png` — per-stage timing breakdown
5. `confusion_heatmap.png` — category × difficulty accuracy heatmap

Returns list of paths for embedding in HTML.

### 4.10 `runner/regression.py` (NEW — 200 lines)

**Purpose:** Compare two evaluation runs and report accuracy deltas.

```python
"""runner/regression.py — Compare two evaluation runs.

Usage:
    python -m runner.regression \
        --baseline /path/to/baseline_report.json \
        --candidate /path/to/candidate_report.json \
        --output-dir /path/to/output/ \
        --label-baseline "V15" \
        --label-candidate "V16"
```

**Key functions:**

#### `load_report(path: str) -> dict`
Loads a saved `report.json` from instrumented_evaluate output.

#### `compute_deltas(baseline: dict, candidate: dict) -> dict`
Per-task comparison: which tasks changed from correct→incorrect or vice versa.
Aggregate deltas: overall accuracy delta, per-category delta, per-difficulty delta.

Returns:
```python
{
    "overall": {
        "baseline_accuracy": 0.842,
        "candidate_accuracy": 0.875,
        "delta": +0.033,
        "gate_baseline": True,
        "gate_candidate": True,
    },
    "by_category": {
        "math": {"baseline": 0.80, "candidate": 0.90, "delta": +0.10},
        ...
    },
    "regressions": [
        {"task_id": "math-5", "category": "math", "baseline_correct": True, "candidate_correct": False,
         "baseline_answer": "42", "candidate_answer": "43", "expected": "42"}
    ],
    "improvements": [...],  # opposite of regressions
    "unchanged": [...],  # correct in both or incorrect in both
}
```

#### `build_diff_html(deltas: dict, output_path: str)`
Generates an HTML diff report with:
- Overall delta with color-coded direction (green=up, red=down)
- Per-category delta table
- Regression list (most important — tasks that got worse)
- Improvement list
- Summary statistics (net change count)

### 4.11 `runner/ablation_runner.py` (NEW — 300 lines)

**Purpose:** Run a catalog of ablation experiments and produce a comparative report.

```python
"""runner/ablation_runner.py — Run ablation experiments on the staging pipeline.

Usage:
    python -m runner.ablation_runner \
        --tasks data/eval/tests/eval_v14_test_20.json \
        --gold data/eval/tests/eval_v14_test_20.json \
        --output-dir eval_results/ablation/ \
        --pipeline-cmd "python -m staging.entrypoint"
```

**Ablation experiments (defined as catalog):**

| # | Name | Env Overrides | Purpose |
|---|------|---------------|---------|
| 0 | `baseline` | (none) | Default 4-worker config |
| 1 | `no_fireworks` | `ABLATION_DISABLE_FIREWORKS=1` | Measure FW contribution |
| 2 | `no_local` | `ABLATION_DISABLE_LOCAL=1` | Measure local model contribution |
| 3 | `no_deterministic` | `ABLATION_DISABLE_DETERMINISTIC=1` | Measure deterministic contribution |
| 4 | `fw_only` | `ABLATION_SINGLE_WORKER=fireworks,ABLATION_DISABLE_LOCAL=1,ABLATION_DISABLE_DETERMINISTIC=1` | FW alone |
| 5 | `local_only` | `ABLATION_SINGLE_WORKER=local,ABLATION_DISABLE_FIREWORKS=1,ABLATION_DISABLE_DETERMINISTIC=1` | Local alone |
| 6 | `det_only` | `ABLATION_SINGLE_WORKER=deterministic,ABLATION_DISABLE_FIREWORKS=1,ABLATION_DISABLE_LOCAL=1` | Deterministic alone |
| 7 | `no_voting` | `ABLATION_DISABLE_VOTING=1` | First-answer-wins (no majority) |
| 8 | `cold_temp` | `ABLATION_TEMPERATURE_SWEEP=0.1,0.1,0.1,0.1,0.1` | No temperature diversity |
| 9 | `hot_temp` | `ABLATION_TEMPERATURE_SWEEP=0.9,0.9,0.9,0.9,0.9` | Max temperature diversity |
| 10 | `3_votes` | `ABLATION_VOTES=3` | Fewer votes per task |
| 11 | `1_vote` | `ABLATION_VOTES=1` | Single try, no voting |
| 12 | `tiebreaker_det` | `ABLATION_TIEBREAKER=deterministic` | Deterministic priority tiebreaker |

**Implementation:**

```python
ABLATION_CATALOG = [
    {"name": "baseline", "env": {}},
    {"name": "no_fireworks", "env": {"ABLATION_DISABLE_FIREWORKS": "1"}},
    # ... etc
]

def run_ablation(experiment: dict, tasks_path: str, gold_path: str,
                 pipeline_cmd: list[str], output_dir: str) -> dict:
    """Run one ablation experiment:
    1. Set env vars from experiment['env']
    2. Run pipeline via subprocess
    3. Evaluate results via instrumented_evaluate
    4. Return report dict
    """

def run_catalog(tasks_path: str, gold_path: str, pipeline_cmd: list[str],
                output_dir: str) -> list[dict]:
    """Run all experiments in catalog, collect reports."""

def build_ablation_report(results: list[dict], output_dir: str):
    """Generate HTML comparison of all ablation results.
    
    Shows:
    - Accuracy per experiment (sorted bar chart)
    - Accuracy delta from baseline per experiment
    - Worker contribution table
    - Timing impact per experiment
    """
```

### 4.12 `runner/degradation_test.py` (NEW — 250 lines)

**Purpose:** Test pipeline resilience by deliberately inducing failures.

```python
"""runner/degradation_test.py — Graceful degradation test suite.

Usage:
    python -m runner.degradation_test \
        --tasks data/eval/tests/eval_v14_timeout_stress_19.json \
        --gold data/eval/tests/eval_v14_timeout_stress_19.json \
        --pipeline-cmd "python -m staging.entrypoint" \
        --output-dir eval_results/degradation/
```

**Test scenarios:**

| # | Name | Setup | Expected Behavior |
|---|------|-------|-------------------|
| 1 | `worker_crash` | Inject worker that segfaults (simulate OOM) | Other workers continue, task handled by remaining workers, output has all tasks |
| 2 | `empty_model_path` | `ABLATION_EMPTY_MODEL_PATH=1` | LocWorker fails init, FW+Det handle all tasks, output complete |
| 3 | `no_api_key` | Unset `FIREWORKS_API_KEY` | FW workers return empty, Loc+Det handle all tasks, output complete |
| 4 | `tight_deadline` | `DEADLINE_S=60` | Pipeline adapts, falls back to reduced votes, produces partial output |
| 5 | `very_tight_deadline` | `DEADLINE_S=15` | Pipeline produces whatever it can in time |
| 6 | `empty_tasks` | Empty task list | Pipeline writes empty output gracefully |
| 7 | `malformed_task` | Task with missing `prompt` field | Pipeline handles gracefully, skips or uses fallback |
| 8 | `all_workers_fail` | `ABLATION_DISABLE_FIREWORKS=1 ABLATION_EMPTY_MODEL_PATH=1` (FW+Local both gone) | Det workers handle what they can, remaining tasks get empty answers |

**Each test:**
1. Sets up environment
2. Runs pipeline with a timeout wrapper
3. Checks exit code
4. Checks output file exists
5. Checks number of results matches expected
6. Grades results (within limits — degraded runs may have lower accuracy)
7. Reports pass/fail with diagnostics

**Output:** JSON + HTML report:
```json
{
    "summary": {"passed": 6, "failed": 2, "total": 8},
    "tests": [
        {
            "name": "worker_crash",
            "passed": true,
            "expected_tasks": 19,
            "actual_tasks": 19,
            "accuracy": 0.74,
            "timing_s": 45.2,
            "notes": "One FW worker crashed, others handled tasks. All 19 judged."
        },
        ...
    ]
}
```

### 4.13 `runner/report_templates/eval_report.html` (NEW — ~200 lines)

Jinja2 HTML template with embedded CSS styling:

```html
<!DOCTYPE html>
<html>
<head>
  <title>Eval Report: {{ title }}</title>
  <style>
    /* Print-friendly, responsive */
    .summary-card { ... }
    .pass { background: #d4edda; }
    .fail { background: #f8d7da; }
    table.per-task { ... }
    .bar-chart { ... }
  </style>
</head>
<body>
  <h1>{{ title }}</h1>
  <div class="meta">Date: {{ date }} | Tasks: {{ overall.total }}</div>
  
  <div class="summary-cards">
    <div class="summary-card {% if overall.gate_pass %}pass{% else %}fail{% endif %}">
      <h2>Accuracy</h2>
      <span class="big-number">{{ "%.1f"|format(overall.accuracy*100) }}%</span>
      <span class="gate">84.2% Gate: {{ "PASS" if overall.gate_pass else "FAIL" }}</span>
    </div>
    <div class="summary-card">
      <h2>Timing</h2>
      <span>Mean: {{ "%.0f"|format(timing.mean) }}ms</span>
      <span>P95: {{ "%.0f"|format(timing.p95) }}ms</span>
    </div>
  </div>
  
  <h2>Category Breakdown</h2>
  <table>
    <tr><th>Category</th><th>Accuracy</th><th>Correct/Total</th></tr>
    {% for cat, stats in by_category.items() %}
    <tr class="{{ 'pass' if stats.accuracy >= 0.842 else 'fail' }}">
      <td>{{ cat }}</td>
      <td>{{ "%.1f"|format(stats.accuracy*100) }}%</td>
      <td>{{ stats.correct }}/{{ stats.total }}</td>
    </tr>
    {% endfor %}
  </table>
  
  {% if plots %}
  <h2>Diagnostic Plots</h2>
  {% for plot_path in plots %}
  <img src="{{ plot_path }}" alt="Diagnostic plot" style="max-width: 100%;">
  {% endfor %}
  {% endif %}
  
  <!-- Worker contribution, strategy breakdown, per-task table, failures... -->
  <!-- ... continued in full template file ... -->
</body>
</html>
```

### 4.14 `runner/report_templates/regression_diff.html` (NEW — ~100 lines)

HTML diff report template (similar to eval_report.html but focused on deltas).

### 4.15 `runner/__init__.py` — Add exports

```python
from runner.instrumented_evaluate import (
    evaluate_detailed, build_detailed_report, build_html_report,
    grade_results_detailed,
)
from runner.regression import compute_deltas, build_diff_html
from runner.ablation_runner import run_catalog, build_ablation_report
from runner.degradation_test import run_degradation_tests
```

---

## 5. Data Model & Output Formats

### 5.1 results_detailed.json (per-task, full instrumentation)

Location: `/output/results_detailed.json`

```json
{
  "meta": {
    "pipeline_version": "staging-v15",
    "timestamp": "2026-07-13T12:00:00Z",
    "config": {
      "fw_workers": 1,
      "loc_workers": 1,
      "det_workers": 2,
      "judgment_votes": 5,
      "deadline_s": 600,
      "ablation": {}
    },
    "timing_summary": {
      "classification_s": 0.5,
      "dispatch_loop_s": 45.2,
      "judgment_s": 0.8,
      "total_s": 56.5,
      "per_worker_type": {...}
    }
  },
  "results": [ ... ]  // array of per-task dicts (see Section 3.2)
}
```

### 5.2 report.json (graded, machine-readable)

Location: `{output_dir}/report.json`

```json
{
  "overall": {
    "total": 20,
    "correct": 17,
    "accuracy": 0.85,
    "gate_pass": true
  },
  "by_category": {...},
  "by_difficulty": {...},
  "by_strategy": {
    "majority_3plus": {"count": 14, "accuracy": 0.93},
    "escalate_fireworks": {"count": 4, "accuracy": 0.75},
    "fallback_best": {"count": 2, "accuracy": 0.50}
  },
  "worker_contribution": {
    "fireworks": {"correct": 12, "total": 20, "accuracy": 0.60, "alone_correct": 1},
    "local": {"correct": 15, "total": 20, "accuracy": 0.75, "alone_correct": 2},
    "deterministic": {"correct": 13, "total": 18, "accuracy": 0.72, "alone_correct": 0}
  },
  "timing": {
    "mean": 4560,
    "median": 4200,
    "p95": 8900,
    "per_stage": {...},
    "per_worker_type": {...}
  },
  "per_task": [...],
  "failures": [...]
}
```

### 5.3 regression_diff.json (machine-readable)

```json
{
  "meta": {
    "baseline_label": "V15",
    "candidate_label": "V16",
    "timestamp": "2026-07-13T14:00:00Z"
  },
  "overall": {
    "baseline_accuracy": 0.842,
    "candidate_accuracy": 0.875,
    "delta": 0.033
  },
  "by_category": {...},
  "regressions": [...],
  "improvements": [...]
}
```

---

## 6. Test Data Requirements

### 6.1 Existing eval sets (compatible, no changes needed)

| File | Tasks | Categories | Best for |
|------|-------|------------|----------|
| `data/eval/tests/eval_v14_test_20.json` | 20 | All 8 | Quick smoke test |
| `data/eval/tests/eval_v14_remaining_20.json` | 20 | All 8 | Secondary test |
| `data/eval/tests/eval_v14_timeout_stress_19.json` | 19 | Mixed | Degradation tests |
| `data/eval/primary/eval_60_medium_hard.json` | 60 | 7 (no NER) | Full eval |
| `data/eval/tests/fireworks_eval_20.json` | 20 | All 8 | FW-only test |

### 6.2 New: ablation-specific test set

Create `data/eval/tests/ablation_20.json` — 20 tasks balanced across categories:
```json
{
  "meta": {"total": 20, "note": "Balanced ablation test set — 2-3 per category"},
  "questions": [
    {
      "task_id": "math-1",
      "category": "math",
      "difficulty": "simple",
      "prompt": "...",
      "expected_answer": "..."
    },
    // ... 2-3 per each of the 8 categories
    // Mix of easy/hard within categories
    // At least one task per category that deterministic solvers can handle
    // At least one task per category that requires LLM
  ]
}
```

### 6.3 New: regression test set

Create `data/eval/tests/regression_40.json` — 40 tasks with known stable ground truth:
- Subset of the ablation set (20) + additional 20
- Covers all 8 categories with at least 4 tasks each
- Mix of difficulties (simple, medium, hard)
- Stable answers that are unlikely to change between pipeline versions

### 6.4 New: degradation test fixtures

- `data/eval/tests/empty_tasks.json` — `{"questions": []}`
- `data/eval/tests/malformed_task.json` — task with missing `prompt` field
- `data/eval/tests/single_task.json` — single well-formed task for quick tests

### 6.5 Ground truth format

Both existing formats are compatible. The evaluator's `load_gold()` already handles:

1. **Heldout format** (gold dict with `answer`, `accept`, `keywords`, etc.):
   ```json
   {"task_id": "fact-1", "category": "factual", "prompt": "...",
    "gold": {"answer": "Canberra", "accept": ["..."]}}
   ```

2. **Plain answer format** (flat `expected_answer`):
   ```json
   {"task_id": "fact-1", "category": "factual", "prompt": "...",
    "expected_answer": "Canberra", "difficulty": "simple"}
   ```

---

## 7. Running the Evaluation

### 7.1 Quick smoke test (single command)

```bash
# 1. Run pipeline with detailed output
DETAILED_OUTPUT=1 \
  python -m staging.entrypoint data/eval/tests/eval_v14_test_20.json

# 2. Evaluate against ground truth
python -m runner.instrumented_evaluate \
  --detailed /output/results_detailed.json \
  --gold data/eval/tests/eval_v14_test_20.json \
  --output-dir eval_results/quick_test/ \
  --title "Quick Smoke Test"

# 3. Open report
# open eval_results/quick_test/report.html
```

### 7.2 Full evaluation

```bash
# Run with all instrumentation
DETAILED_OUTPUT=1 \
  STAGING_JUDGMENT_VOTES=5 \
  python -m staging.entrypoint data/eval/primary/eval_60_medium_hard.json

python -m runner.instrumented_evaluate \
  --detailed /output/results_detailed.json \
  --gold data/eval/primary/eval_60_medium_hard.json \
  --output-dir eval_results/full_eval/ \
  --title "V15 Full Eval (60 tasks)" \
  --plots  # generate diagnostic plots (requires matplotlib)
```

### 7.3 Regression test

```bash
# After making changes to the pipeline:
# 1. Run baseline
DETAILED_OUTPUT=1 python -m staging.entrypoint data/eval/tests/regression_40.json

# Save baseline report
cp /output/results_detailed.json eval_results/baseline/results_detailed.json

# 2. Make changes, rebuild, then run candidate
DETAILED_OUTPUT=1 python -m staging.entrypoint data/eval/tests/regression_40.json

# Save candidate report
cp /output/results_detailed.json eval_results/candidate/results_detailed.json

# 3. Compare
python -m runner.regression \
  --baseline-dir eval_results/baseline/ \
  --candidate-dir eval_results/candidate/ \
  --output-dir eval_results/regression_diff/ \
  --label-baseline "V15" \
  --label-candidate "V16"
```

### 7.4 Ablation catalog

```bash
# Run all 13 ablation experiments automatically
python -m runner.ablation_runner \
  --tasks data/eval/tests/ablation_20.json \
  --gold data/eval/tests/ablation_20.json \
  --output-dir eval_results/ablation/ \
  --pipeline-cmd "python -m staging.entrypoint"
```

### 7.5 Degradation tests

```bash
# Run all 8 degradation tests
python -m runner.degradation_test \
  --tasks data/eval/tests/eval_v14_timeout_stress_19.json \
  --gold data/eval/tests/eval_v14_timeout_stress_19.json \
  --output-dir eval_results/degradation/ \
  --pipeline-cmd "python -m staging.entrypoint"
```

### 7.6 Local GPU workflow (development)

```bash
# On development machine (has GPU, no time limit):
python -m runner.ablation_runner \
  --tasks data/eval/tests/ablation_20.json \
  --gold data/eval/tests/ablation_20.json \
  --output-dir eval_results/ablation_gpu/ \
  --pipeline-cmd "python -m staging.entrypoint" \
  --workers 4  # More workers on GPU machine

# Analyze results
open eval_results/ablation_gpu/ablation_report.html
```

### 7.7 Grader-constrained workflow (2 vCPU, 4GB RAM, 600s)

```bash
# Within the container:
# 1. Only run the pipeline (no evaluation — too slow)
python -m staging.entrypoint /input/tasks.json

# 2. Copy /output/results_detailed.json and /output/results.json out
#    (these are small files, typically <1MB)

# On a separate analysis machine:
python -m runner.instrumented_evaluate \
  --detailed /path/to/copied/results_detailed.json \
  --gold /path/to/copied/gold.json \
  --output-dir eval_results/container_run/
```

---

## 8. Interpretation Guide

### 8.1 Reading the HTML Report

**Summary Cards:**
- **Accuracy** — overall percentage; green if ≥ 84.2%, red if below
- **Gate** — PASS/FAIL for the 84.2% submission gate
- **Timing** — mean and P95 per-task timing

**Category Breakdown Table:**
- Per-category accuracy, color-coded
- Identifies weak categories (e.g., logic at 60% vs math at 90%)

**Worker Contribution:**
- `alone_correct` column shows tasks where only that worker type got it right
- High `alone_correct` for a worker type = critical worker for edge cases
- If a worker type has low `alone_correct`, it may not be adding value

**Strategy Breakdown:**
- `majority_3plus` = strong consensus (≥3/5 agree) — high accuracy expected
- `escalate_fireworks` = weak consensus → FW API used — medium accuracy expected
- `fallback_best` = no consensus — low accuracy expected
- If `escalate_fireworks` has high accuracy, the escalation path is working
- If `fallback_best` has many tasks, voting is not producing consensus

**Timing Waterfall:**
- Classification phase should be <1s for all tasks
- Worker per-try timing varies by worker type (det < 1ms, local ~3s, FW ~1.2s)
- Judgment phase should be negligible (<10ms)
- Bottleneck is almost always the local model (3s/try × 5 tries = 15s/task)

### 8.2 Reading Regression Diffs

**Overall Delta:**
- Positive = candidate is better (accuracy improved)
- Negative = candidate regressed
- Green/red coloring for direction

**Regression List (MOST IMPORTANT):**
- Tasks that were correct in baseline but wrong in candidate
- Investigate these first — they represent real regressions
- Common causes: classifier change routing to wrong worker, prompt change, model change

**Improvement List:**
- Tasks that were wrong in baseline but correct in candidate
- Useful for understanding what changed

**Per-Category Deltas:**
- Some categories may improve while others regress
- Trade-off analysis (e.g., better math, worse logic)

### 8.3 Reading Ablation Reports

**Baseline comparison:**
- Each experiment's accuracy is compared to baseline
- Sorting by impact shows the most important features

**Key questions answered:**
- How much does FW contribute? → `no_fireworks` accuracy drop
- How much does local contribute? → `no_local` accuracy drop
- How much does voting help? → `no_voting` accuracy drop
- Is temperature diversity important? → `cold_temp` accuracy change
- Are 5 votes needed or would 3 suffice? → `3_votes` accuracy

**Worker contribution isolation:**
- `fw_only` accuracy = best possible with just FW
- `local_only` accuracy = best possible with just local
- `det_only` accuracy = deterministic baseline
- Ensemble accuracy should exceed all individual worker accuracies (if voting is working)

### 8.4 Reading Degradation Test Reports

**Pass/Fail per scenario:**
- Each test checks specific pipeline resilience properties
- A test passes if the pipeline produces output (even degraded) without crashing

**Expected degradation patterns:**
- Worker crash test: accuracy may be slightly lower but all tasks should be graded
- No API key test: accuracy may drop slightly (no FW escalation) but tasks complete
- Tight deadline test: fewer tasks completed, lower overall accuracy but no crash

**Red flags:**
- Any test that produces 0 results or crashes the process
- Pipeline that hangs (no output at all within timeout)
- Pipeline that crashes the entire container

---

## 9. Scenarios & Usage Patterns

### 9.1 Iterative development (quick feedback loop)

```bash
while true; do
  # Make changes to prompts/solvers

  # Quick smoke test
  DETAILED_OUTPUT=1 python -m staging.entrypoint data/eval/tests/eval_v14_test_20.json
  python -m runner.instrumented_evaluate \
    --detailed /output/results_detailed.json \
    --gold data/eval/tests/eval_v14_test_20.json \
    --output-dir eval_results/iteration/

  # Check if accuracy improved
  cat eval_results/iteration/report.json | python -c "
    import json, sys
    r = json.load(sys.stdin)
    print(f\"Accuracy: {r['overall']['accuracy']:.1%}, Gate: {'PASS' if r['overall']['gate_pass'] else 'FAIL'}\")
    for cat, s in r['by_category'].items():
        print(f\"  {cat}: {s['accuracy']:.1%}\")
  "
done
```

### 9.2 Pre-submission validation

```bash
# Run full evaluation
DETAILED_OUTPUT=1 python -m staging.entrypoint data/eval/primary/eval_60_medium_hard.json

# Grade and check gate
report=$(python -m runner.instrumented_evaluate \
  --detailed /output/results_detailed.json \
  --gold data/eval/primary/eval_60_medium_hard.json \
  --output-dir eval_results/pre_submit/)

python -c "
import json
r = json.load(open('eval_results/pre_submit/report.json'))
if r['overall']['gate_pass']:
    print(f\"✅ GATE PASSED: {r['overall']['accuracy']:.1%}\")
else:
    print(f\"❌ GATE FAILED: {r['overall']['accuracy']:.1%} (need 84.2%)\")
    for f in r['failures'][:5]:
        print(f\"  - {f['task_id']}: {f['reason']}\")
"
```

### 9.3 Worker comparison (which worker type to optimize)

```bash
# Run ablation with single-worker experiments
python -m runner.ablation_runner \
  --tasks data/eval/tests/ablation_20.json \
  --gold data/eval/tests/ablation_20.json \
  --output-dir eval_results/worker_compare/ \
  --experiments baseline,fw_only,local_only,det_only

# Read the report to see per-worker standalone accuracy
```

### 9.4 Debugging a specific task failure

```bash
# Find detailed per-try answers for a failing task
DETAILED_OUTPUT=1 python -m staging.entrypoint data/eval/tests/eval_v14_test_20.json

python -c "
import json
detailed = json.load(open('/output/results_detailed.json'))
results = detailed.get('results', detailed)  # handle meta wrapper
for r in results:
    if r['task_id'] == 'math-5':
        print('Task:', r['task_id'])
        print('Category:', r.get('category'))
        print('Classification:', r.get('classification'))
        print('Judgment strategy:', r.get('judgment', {}).get('strategy'))
        print('Winner:', r.get('answer'))
        print('All answers:')
        for wa in r.get('worker_answers', []):
            print(f'  {wa[\"worker_id\"]} ({wa[\"worker_type\"]}, T={wa[\"temperature\"]}): {wa[\"answer\"][:80]} ({wa[\"timing_ms\"]}ms)')
"
```

---

## 10. Integration with Grader Constraints

### 10.1 Size of instrumentation output

The detailed JSON adds metadata per task. Estimated sizes:

| Field | Estimated bytes | Notes |
|-------|----------------|-------|
| Basic output (contract) | ~200 B/task | task_id + answer |
| Classification meta | ~200 B/task | scores, confidence |
| Per worker answer (×5) | ~400 B/task | worker_id, type, model, temp, answer text |
| Judgment meta | ~150 B/task | strategy, group sizes |
| Timing summary | ~200 B total | per-stage + per-worker-type |
| **Total detailed** | ~1 KB/task | |
| 60 tasks | ~60 KB | Trivially small |
| 300 tasks | ~300 KB | Still small |

**No performance concern.** Writing extra JSON metadata is negligible compared to model inference time.

### 10.2 Dual output strategy

The pipeline always writes:
- `/output/results.json` — grader contract (required, always written, no changes)
- `/output/results_detailed.json` — only if `DETAILED_OUTPUT=1` (optional, for evaluation)

This means:
- **Grader mode:** `DETAILED_OUTPUT=0` (default) — only writes contract file. Zero overhead.
- **Eval mode:** `DETAILED_OUTPUT=1` — writes both files. Adds ~1ms of serialization cost.

### 10.3 CPU/memory budget for evaluation

The evaluation scripts (`runner/instrumented_evaluate.py`, `runner/regression.py`) are designed to run **outside the container** on a machine with more resources (GPU dev box, your laptop). They consume the detailed JSON files which are small (tens of KB).

If evaluation must run inside the constrained container:
- Use `--no-plots` flag to skip matplotlib (saves memory)
- `runner/instrumented_evaluate.py` uses ~50 MB RAM (Python + JSON parsing)
- Report writing is I/O bound, not CPU bound
- HTML generation is fast (<1s for 60 tasks)

### 10.4 Fallback if pipeline crashes mid-eval

```python
# In instrumented_evaluate:
if not os.path.exists(detailed_path):
    # Fall back to standard results.json
    results_path = detailed_path.replace("results_detailed.json", "results.json")
    if os.path.exists(results_path):
        report = grade_results(results_path, gold_path, xlsx_path)
        report["meta"]["note"] = "Detailed instrumentation not available — used flat results"
        return report
```

---

## 11. Appendix: Report Schema

### 11.1 report.json (complete schema)

```json
{
  "meta": {
    "title": "string",
    "timestamp": "ISO8601",
    "pipeline_version": "string",
    "config": "dict",
    "gold_path": "string",
    "detailed_path": "string",
    "note": "string | null"
  },
  "overall": {
    "total": "int",
    "correct": "int",
    "accuracy": "float (0-1)",
    "gate_pass": "bool"
  },
  "by_category": {
    "<category>": {
      "total": "int",
      "correct": "int",
      "accuracy": "float"
    }
  },
  "by_difficulty": {  // if present in gold
    "<difficulty>": {
      "total": "int",
      "correct": "int",
      "accuracy": "float"
    }
  },
  "by_strategy": {
    "<strategy_name>": {
      "count": "int",
      "accuracy": "float"
    }
  },
  "worker_contribution": {
    "<worker_type>": {
      "correct": "int",
      "total": "int",
      "accuracy": "float",
      "alone_correct": "int",
      "tasks_with_answer": "int"
    }
  },
  "timing": {
    "mean": "float (ms)",
    "median": "float (ms)",
    "p95": "float (ms)",
    "per_stage": {
      "classification_avg_ms": "float",
      "worker_per_try_avg_ms": "float",
      "judgment_avg_ms": "float"
    },
    "per_worker_type": {
      "<type>": {
        "avg_ms": "float",
        "total_s": "float",
        "tasks_done": "int"
      }
    }
  },
  "per_task": [
    {
      "task_id": "string",
      "category": "string",
      "difficulty": "string | null",
      "prompt": "string",
      "expected": "string",
      "answer": "string",
      "correct": "bool",
      "reason": "string",
      "timing_ms": "float",
      "strategy": "string",
      "votes_cast": "int",
      "largest_group": "int",
      "num_groups": "int",
      "worker_count": "int",
      "worker_types": ["string"],
      "classification_confidence": "float",
      "classification_delta": "float"
    }
  ],
  "failures": [  // subset of per_task where correct=false
    { ... }
  ]
}
```

### 11.2 regression_diff.json (complete schema)

```json
{
  "meta": {
    "baseline_label": "string",
    "candidate_label": "string",
    "baseline_timestamp": "ISO8601",
    "candidate_timestamp": "ISO8601",
    "baseline_path": "string",
    "candidate_path": "string"
  },
  "overall": {
    "baseline_accuracy": "float",
    "candidate_accuracy": "float",
    "delta": "float",
    "baseline_gate": "bool",
    "candidate_gate": "bool"
  },
  "by_category": {
    "<category>": {
      "baseline": {"total": "int", "correct": "int", "accuracy": "float"},
      "candidate": {"total": "int", "correct": "int", "accuracy": "float"},
      "delta": "float"
    }
  },
  "regressions": [
    {
      "task_id": "string",
      "category": "string",
      "difficulty": "string | null",
      "baseline_correct": true,
      "candidate_correct": false,
      "baseline_answer": "string",
      "candidate_answer": "string",
      "expected": "string"
    }
  ],
  "improvements": [ /* same structure as regressions but reversed */ ],
  "unchanged_correct": ["task_id"],
  "unchanged_incorrect": ["task_id"],
  "summary": {
    "regression_count": "int",
    "improvement_count": "int",
    "unchanged_correct_count": "int",
    "unchanged_incorrect_count": "int",
    "net_change": "int (improvements - regressions)"
  }
}
```

---

## Implementation Order

### Phase 1 — Core instrumentation (highest priority, ~1 day)

1. Create `scripts/grade_answer.py` — extract grading functions
2. Modify `staging/workers/*.py` — add worker metadata to answer dicts
3. Modify `staging/ready_judge.py` — expose group sizes and vote distribution
4. Modify `staging/entrypoint.py` — add `DETAILED_OUTPUT=1` mode
5. Create `runner/instrumented_evaluate.py` — core grading + JSON report

### Phase 2 — Rich reporting (~1 day)

6. Create `runner/report_templates/eval_report.html` — HTML template
7. Add `--plots` support to instrumented_evaluate (diagnostic charts)
8. Test with existing eval sets (20-task, 60-task)
9. Create `data/eval/tests/ablation_20.json` — balanced test set

### Phase 3 — Regression & ablation (~1 day)

10. Create `runner/regression.py` — comparison analysis
11. Create `runner/report_templates/regression_diff.html`
12. Create `runner/ablation_runner.py` — experiment catalog runner
13. Test ablation with all 13 experiments

### Phase 4 — Degradation testing (~0.5 day)

14. Create `runner/degradation_test.py`
15. Test all 8 degradation scenarios
16. Create test fixtures (empty, malformed tasks)

### Phase 5 — Polish & documentation (~0.5 day)

17. Update `runner/__init__.py` with exports
18. Add CLI `--help` documentation to all new scripts
19. Verify dual-output strategy works with grader contract
20. Document in `CONTEXT.md` and `README.md`
