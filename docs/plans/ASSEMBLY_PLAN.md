# ASSEMBLY PLAN: Pipeline Wrapper Layer

> **Project:** AMD ACT II Hackathon — Track 1 Token Efficient Routing Agent
> **Root:** `/home/artem/dev/amd-hackathon/`
> **Latest Container:** `ghcr.io/artemkorolev1/amd-hackathon-submit:v15`
> **Date:** 2026-07-13
> **Golden Rule:** The Pipeline class (`agent/pipeline.py`) and everything it imports is **untouchable**. All wrapper code lives in `runner/` and never modifies `agent/`.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Parallelization Design](#2-parallelization-design)
3. [Evaluation / Judgment Design](#3-evaluation--judgment-design)
4. [Container Deployment Design](#4-container-deployment-design)
5. [File-by-File Implementation Plan](#5-file-by-file-implementation-plan)
6. [Testing Strategy](#6-testing-strategy)

---

## 1. Architecture Overview

### Layer Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                      User / Grader                           │
├──────────────────────────────────────────────────────────────┤
│  runner/deploy.py          Build, tag, push, verify         │
│  runner/evaluate.py        Grade results vs ground truth    │
│  runner/batch_runner.py    Parallel orchestration            │
├──────────────────────────────────────────────────────────────┤
│  agent/pipeline.py         ← UNTOUCHABLE — imports & uses   │
│  agent/__init__.py          "from agent import Pipeline"     │
│  harness.py                 Thin CLI (also untouchable)     │
└──────────────────────────────────────────────────────────────┘

 All wrapper code imports from agent/* but never touches those files.
```

### Communication Pattern

```
tasks.json ──→ runner/batch_runner.py ──→ [worker Pipeline instances] ──→ results.json
                      │                                                    │
                      └──→ runner/evaluate.py ──→ fuzzy_match cascade ──→ report.xlsx
                      │
               runner/deploy.py ──→ docker buildx ──→ push to GHCR
```

### Key Constraints

| Constraint | Value | Implication |
|---|---|---|
| CPU | 2 vCPU | ThreadPoolExecutor may be GIL-bound for CPU inference |
| RAM | 4 GB | ~1.1 GB per GGUF worker process → at most 2-3 concurrent workers |
| Deadline | 600 s | Must finish all tasks within this |
| Per-task | 30 s max | Must timeout individual tasks, not just whole batch |
| Pipeline API | Frozen | Cannot change imports, signatures, or behavior |

---

## 2. Parallelization Design

### 2.1 Worker Model: Multiprocessing (ProcessPoolExecutor)

**Decision: Use `multiprocessing.ProcessPoolExecutor` (not threads).**

Rationale:
- `llama-cpp-python` is a C extension with a Python wrapper. The inference call (`self._llm.create_chat_completion()`) releases the GIL *during* compute, but model loading, tokenization, and response parsing all re-acquire it.
- With threads, all workers compete for the GIL during non-inference sections, causing thrashing.
- With processes, each worker gets its own GIL and its own model copy — true parallelism.
- The downside (higher memory per worker) is manageable with careful budgeting.

### 2.2 Memory Budget

Worker memory breakdown (per-process):

| Component | Estimated RAM |
|---|---|
| Python runtime | ~150 MB |
| numpy/scipy (already imported) | ~100 MB |
| GGUF file (memory-mapped, not fully loaded) | ~200 MB RSS |
| KV cache + compute buffers | ~300-500 MB |
| **Per-worker total** | **~750-950 MB** |

Budget plan:
- **2 concurrent workers**: ~1.6 - 1.9 GB total → leaves ~2 GB for OS + shared libraries ✅
- **3 concurrent workers**: ~2.4 - 2.8 GB total → tighter but feasible with a 900MB GGUF ✅
- **4 concurrent workers**: ~3.2 - 3.8 GB total → OOM risk ❌

**Recommended default: 2 concurrent workers.** A config knob (`MAX_WORKERS`) allows 3 if the model is small or memory proves sufficient.

### 2.3 Per-Task Timeout Strategy

Each worker runs tasks sequentially (one at a time). The batch_runner imposes:

1. **Individual task timeout**: `pipeline.cfg.inference_timeout_s` (default 28s) plus a hard per-task wall-clock limit of 30s enforced via `multiprocessing.Process.join(timeout=30)`.
2. **Worker-level timeout**: If a worker hangs on a task (model OOM, infinite loop), the process pool can kill it and spawn a replacement.
3. **Global deadline**: The wrapper tracks `time.monotonic()` against the 600s wall clock. When time is low, remaining tasks get the shortest possible pipeline path.

**Timeout enforcement cascade:**

```python
# Pseudo-code for per-worker task handling
def _worker_entry(cfg, task_chunk, worker_id):
    pipe = Pipeline(cfg)
    results = []
    for task in task_chunk:
        deadline = time.monotonic() + 30.0
        result = {"task_id": task["task_id"], "answer": ""}
        try:
            answer = pipe.process(task["prompt"])
            result["answer"] = answer
        except Exception:
            result["answer"] = ""
        results.append(result)
    pipe.close()
    return results
```

### 2.4 Graceful Degradation & Crash Handling

| Failure Mode | Response |
|---|---|
| Worker process crashes (OOM/SIGKILL) | `ProcessPoolExecutor` detects broken worker → spawns replacement, logs error, reassigns unprocessed tasks |
| Individual task throws exception | Caught per-task, empty answer returned, continues |
| Worker exceeds 30s per task | `future.result(timeout=30)` raises `TimeoutError` → worker is terminated, new worker spawned, remaining tasks redistributed |
| Deadline approaching (≤60s remaining) | Remaining tasks run sequentially in the main process (avoids spawn overhead, best-effort) |
| No model file found | Pipeline handles this gracefully → local LLM skipped, deterministic solvers + Fireworks fallback only |

### 2.5 Work Distribution

```
Input: 19 tasks, 2 workers
Worker 0: tasks [0..8]   (9 tasks)
Worker 1: tasks [9..18]  (10 tasks)

Each worker creates one Pipeline instance, processes its chunk sequentially.
```

For uneven batch sizes or worker failures, a dynamic work-stealing approach is used: start with a chunk size of `len(tasks) // n_workers`, then after workers finish, check if any tasks are still pending and redistribute.

### 2.6 Result Collection

Results must preserve input order. Workers return lists in the same order as their chunk. The batch_runner concatenates chunks in the correct sequence.

```python
# Parallel execution structure
def run_parallel(tasks, config, n_workers=2):
    chunks = [tasks[i::n_workers] for i in range(n_workers)]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_worker_entry, config, chunk, i)
                   for i, chunk in enumerate(chunks)]
        all_results = []
        for future in as_completed(futures):
            all_results.extend(future.result())
    # Re-sort by original index
    all_results.sort(key=lambda r: r.get("_idx", 0))
    return all_results
```

---

## 3. Evaluation / Judgment Design

### 3.1 Architecture

```
runner/evaluate.py
  ├── load_gold(path)          — load ground-truth JSON (eval set)
  ├── load_predictions(path)   — load batch_runner results (results.json)
  ├── grade_tasks(gold, preds) — grade each task via fuzzy_match cascade
  ├── build_report(grades)     — per-category, per-difficulty, per-task-timing
  └── write_xlsx(report)       — structured Excel with multiple sheets
```

### 3.2 Grading Pipeline

Each task is graded using the official `fuzzy_match(expected, actual)` cascade from `scripts/evaluate.py`:

1. **Exact** (case-insensitive)
2. **Substring** (expected in actual, or short actual in expected)
3. **Numeric 1% tolerance** (pairwise or single-number)
4. **Token overlap** (≥50% for short answers, ≥30% for long)

The wrapper imports `from scripts.evaluate import fuzzy_match` directly — reusing the exact same function the grader uses guarantees consistency.

### 3.3 Per-Category Accuracy Calculation

Categories come from the ground-truth JSON's `"category"` field (e.g., `"sentiment_classification"`, `"code_debugging"`, `"math_reasoning"`). The report computes:

- **Per-category accuracy**: `correct / total` for each category
- **Per-difficulty accuracy**: if `"difficulty"` field present (`simple`, `medium`, `hard`)
- **Overall accuracy**: `total_correct / total_tasks`
- **84.2% gate pass/fail**: threshold check

### 3.4 Excel Report Format

The output XLSX has three sheets:

**Sheet 1: Summary**
```
Run Summary
├── Run timestamp
├── Pipeline version
├── Total tasks graded
├── Overall accuracy (correct/total, %)
├── 84.2% gate: PASS/FAIL
└── Per-category breakdown table
    ├── Category | Total | Correct | Accuracy %
    └── ...
```

**Sheet 2: Per-Task Details**
```
Task ID | Category | Difficulty | Prompt (truncated) | Pipeline Answer | Expected Answer | Correct | Timing (ms)
```

**Sheet 3: Errors & Failures**
```
Task ID | Category | Answer Snippet | Expected Snippet | Failure Reason
```
Only tasks that didn't pass `fuzzy_match`, with the diagnostic reason from `grade_answer()`.

### 3.5 Timing Capture

The batch_runner records:
- Per-task total wall time (ms)
- Pipeline-level timing is already logged by the Pipeline (though we don't modify it)
- Worker startup time (model loading)
- "Per-stage" timing is captured via a lightweight timing wrapper around `pipe.process()` — we record elapsed time before and after the call; we do NOT instrument inside the pipeline.

---

## 4. Container Deployment Design

### 4.1 Build Automation

A single `Makefile` target (or `runner/deploy.py` script) handles the full pipeline:

```
deploy.py: build → tag → push → verify
```

**Build command (automated):**
```bash
docker buildx build \
  --platform linux/amd64 \
  -t ghcr.io/artemkorolev1/amd-hackathon-submit:v<NEXT> \
  --load \
  /home/artem/dev/amd-hackathon/
```

### 4.2 Versioning Strategy

| Component | Convention |
|---|---|
| Docker tag | `v<N>` where N = last v-number + 1 |
| Auto-increment | Read CONTEXT.md submission log, find latest `v<N>`, increment |
| `latest` tag | Also tag with `:latest` after push (optional, configurable) |
| Git | `git tag v<N>` before build to freeze the source |

### 4.3 Push & Verification

**Push:**
```bash
docker push ghcr.io/artemkorolev1/amd-hackathon-submit:v<N>
```

**Verification steps:**
1. Check image exists: `docker image inspect <image>`
2. Check platform: `docker inspect <image> --format '{{.Os}}/{{.Architecture}}'` → `linux/amd64`
3. Check entrypoint: `docker inspect <image> --format '{{json .Config.Entrypoint}}'`
4. Quick import sanity: `docker run --rm --entrypoint python3 <image> -c "from agent import Pipeline; print('OK')"`
5. Smoke test with grader constraints: `--cpus=2 --memory=4g --memory-swap=4g`

### 4.4 Documentation in CONTEXT.md

The deploy script appends an entry to both:
- The `### Submission Log` table (version, tag, build timestamp, status="pushed")
- The `### Version Details` section

### 4.5 Pre-Build Checks

Before building, the script verifies:
- [ ] Git working tree clean (`git status --short` empty)
- [ ] No broken symlinks (Docker COPY doesn't follow external symlinks)
- [ ] Model file exists and is a real GGUF (not a broken symlink)
- [ ] All expected files present (harness.py, agent/, Dockerfile, requirements.txt)

---

## 5. File-by-File Implementation Plan

### 5.1 Directory Layout

```
/home/artem/dev/amd-hackathon/
├── runner/                          # NEW — wrapper layer
│   ├── __init__.py                  # Package init, exports
│   ├── batch_runner.py              # Parallel task distribution
│   ├── evaluate.py                  # Grading + report generation
│   └── deploy.py                    # Docker build, push, verify
├── Makefile                         # [MODIFIED] Add build/push/test targets
├── requirements.txt                 # [MODIFIED] Add openpyxl (for XLSX output)
└── docs/plans/ASSEMBLY_PLAN.md      # [NEW] This document
```

### 5.2 File: `runner/__init__.py`

```python
"""Runner — parallel orchestration, evaluation, and deployment wrapper.

This module wraps the untouchable agent.Pipeline with:
- batch_runner : parallel task distribution via multiprocessing
- evaluate     : grading against ground truth with fuzzy_match cascade
- deploy       : Docker build/tag/push automation
"""

from .batch_runner import BatchRunner, run_parallel
from .evaluate import evaluate_tasks, build_report, write_xlsx
from .deploy import build_image, push_image, verify_image

__all__ = [
    "BatchRunner", "run_parallel",
    "evaluate_tasks", "build_report", "write_xlsx",
    "build_image", "push_image", "verify_image",
]
```

### 5.3 File: `runner/batch_runner.py`

**Purpose:** Distribute tasks across parallel Pipeline instances, collect results in order, enforce deadlines.

**Key classes/functions:**

```python
class BatchRunner:
    """Orchestrates parallel task execution across multiple Pipeline instances."""

    def __init__(self, config: Optional[PipelineConfig] = None,
                 n_workers: int = 2, max_ram_gb: float = 4.0):
        # n_workers auto-capped based on memory budget
        # config defaults to env-var-based PipelineConfig()

    def run(self, tasks: list[dict], deadline_s: float = 600.0) -> list[dict]:
        """
        - Split tasks into n_workers chunks
        - Spawn ProcessPoolExecutor
        - Each worker creates its own Pipeline
        - Per-task timeout enforced via future.result(timeout=30)
        - Global deadline enforced
        - Collect and re-sort results
        - On worker crash: log, continue with remaining workers
        """
```

**Worker entrypoint (module-level for pickling):**
```python
def _worker_process(cfg_dict: dict, task_chunk: list[dict],
                    worker_id: int) -> list[dict]:
    """Runs in a child process. Loads Pipeline, processes chunk."""
    from agent import Pipeline, PipelineConfig
    cfg = PipelineConfig(**cfg_dict)
    pipe = Pipeline(cfg)
    results = []
    for task in task_chunk:
        tid = task.get("task_id", f"w{worker_id}_idx_{len(results)}")
        prompt = task.get("prompt", task.get("question", ""))
        t0 = time.monotonic()
        try:
            answer = pipe.process(prompt)
        except Exception:
            answer = ""
        elapsed_ms = (time.monotonic() - t0) * 1000
        results.append({"task_id": tid, "answer": answer,
                        "timing_ms": elapsed_ms, "worker": worker_id})
    pipe.close()
    return results
```

**Memory-aware worker count:**
```python
def _compute_workers(max_ram_gb: float, model_gb: float = 1.1) -> int:
    """Estimate max workers that fit in RAM budget.
    
    Overhead per worker: 0.4 GB (Python + libs) + model_gb (GGUF).
    Buffer: 0.5 GB for OS.
    """
    overhead = 0.4 + model_gb
    available = max_ram_gb - 0.5
    return max(1, min(4, int(available / overhead)))
```

### 5.4 File: `runner/evaluate.py`

**Purpose:** Grade pipeline output against ground truth, produce structured report and XLSX.

**Key functions:**

```python
def load_gold(path: str) -> dict[str, dict]:
    """
    Load ground-truth JSON.
    Format: array of {task_id, category, prompt, gold: {answer: str, ...}}
    Returns dict mapping task_id → {prompt, expected, category, difficulty}
    """

def load_predictions(path: str) -> dict[str, dict]:
    """
    Load pipeline results (output of BatchRunner.run).
    Format: array of {task_id, answer, timing_ms, ...}
    Returns dict mapping task_id → {answer, timing_ms}
    """

def evaluate_tasks(gold: dict, preds: dict) -> list[dict]:
    """
    Grade every task using fuzzy_match from scripts/evaluate.
    Returns list of dicts:
      {task_id, category, difficulty, prompt, expected, answer,
       correct: bool, reason: str, timing_ms: float}
    """

def build_report(results: list[dict]) -> dict:
    """
    Aggregate results into structured report:
      {overall: {total, correct, accuracy, gate_pass},
       by_category: {cat: {total, correct, accuracy}},
       by_difficulty: {diff: {total, correct, accuracy}},
       per_task: [...full results...],
       failures: [...only incorrect...],
       timing: {mean, median, p95, per_category_timing}}
    """

def write_xlsx(report: dict, output_path: str):
    """
    Write three-sheet Excel workbook:
      Sheet "Summary"     → overall stats, category breakdown, timing
      Sheet "Details"     → per-task rows (all fields)
      Sheet "Failures"    → only incorrect tasks with diagnostic
    Uses openpyxl with formatting (bold headers, colored pass/fail).
    """

def grade_results(results_json: str, gold_json: str,
                  output_xlsx: str, verbose: bool = False) -> dict:
    """
    Convenience wrapper: load → evaluate → report → xlsx.
    Returns the report dict.
    """
```

**Grading logic detail:**
```python
from scripts.evaluate import fuzzy_match, grade_answer

def _grade_one(answer: str, expected: str) -> tuple[bool, str]:
    """Wrapper around scripts.evaluate.grade_answer for consistency."""
    return grade_answer(answer, expected)
```

### 5.5 File: `runner/deploy.py`

**Purpose:** One-command Docker build, push, verify, and document.

**Key functions:**

```python
DEFAULT_IMAGE = "ghcr.io/artemkorolev1/amd-hackathon-submit"
DOCKERFILE = "/home/artem/dev/amd-hackathon/Dockerfile"
BUILD_CTX = "/home/artem/dev/amd-hackathon"
CONTEXT_MD = "/home/artem/dev/amd-hackathon/CONTEXT.md"


def get_next_version() -> str:
    """Read CONTEXT.md submission log, find max v<N>, return v<N+1>."""

def pre_build_checks() -> list[str]:
    """
    Returns list of issues found (empty = all clear).
    Checks:
    - git status clean
    - Dockerfile exists
    - harness.py exists
    - agent/__init__.py exists
    - models/*.gguf is real file (not symlink)
    """

def build_image(tag: str, platform: str = "linux/amd64",
                no_cache: bool = False) -> bool:
    """docker buildx build --platform <platform> -t <tag> --load ."""

def push_image(tag: str) -> bool:
    """docker push <tag>"""

def verify_image(tag: str) -> dict:
    """
    Run verification checks on the image.
    Returns dict of {check_name: passed_bool, detail: str}
    Checks:
    - image exists locally
    - platform is linux/amd64
    - entrypoint is correct
    - imports work (from agent import Pipeline)
    - smoke test with 2 tasks under --cpus=2 --memory=4g
    """

def update_context_md(tag: str, build_ok: bool, push_ok: bool,
                      verify_results: dict):
    """Append entry to CONTEXT.md submission log."""

def deploy(tag: str = None, push: bool = False, verify: bool = True,
           update_context: bool = True) -> int:
    """
    Full pipeline: pre-check → build → (optional push) → verify → document.
    Returns 0 on success, 1 on failure.
    """
```

**Makefile additions** (modify `/home/artem/dev/amd-hackathon/Makefile`):

```makefile
.PHONY: build push verify deploy

# Build Docker image (auto-versioned)
build:
	python -m runner.deploy --build-only

# Push to GHCR
push:
	python -m runner.deploy --push

# Full deploy: build + push + verify + document
deploy:
	python -m runner.deploy --push --verify

# Run evaluation on results
evaluate:
	python -m runner.evaluate --results eval_results/results.json \
		--gold input/dev_40.json --output eval_results/report.xlsx

# Run batch (parallel)
run:
	python -m runner.batch_runner --input input/tasks.json \
		--output results.json --workers 2
```

### 5.6 File modifications: `requirements.txt`

Add `openpyxl` for Excel report generation:

```
numpy>=1.26.0
scipy>=1.13.0
llama-cpp-python>=0.3.1
openpyxl>=3.1.0
```

---

## 6. Testing Strategy

### 6.1 Unit Testing (independent components)

| Component | Test | How |
|---|---|---|
| `batch_runner._compute_workers()` | Memory budget math | Assert correct worker count for various RAM/model sizes |
| `evaluate.fuzzy_match` reuse | Match official grader | Run `fuzzy_match("Paris", "paris")` → True, etc. |
| `evaluate.build_report()` | Aggregation math | Feed known-perfect / half-perfect results → check accuracy |
| `evaluate.write_xlsx()` | File output | Write to temp path, verify 3 sheets exist |
| `deploy.get_next_version()` | Version parsing | Mock CONTEXT.md, assert vN+1 returned |

### 6.2 Integration Testing

| Scenario | Steps | Expected |
|---|---|---|
| **Batch run with 2 workers** | `runner.batch_runner.run(tasks=[...], n_workers=2)` | All tasks processed, results in order, no crashes |
| **Worker crash recovery** | One worker raises intentional exception | Other worker completes, results returned for successful chunk |
| **Global deadline enforcement** | Set `deadline_s=5` on 10 tasks | `deadline_s` is respected, partial results returned |
| **Evaluate against gold** | `evaluate.grade_results(results, gold, output)` | XLSX created, accuracy matches manual count |
| **Deploy pre-flight checks** | `deploy.pre_build_checks()` with dirty git | Reports dirty working tree |

### 6.3 Docker/Container Testing

| Scenario | Command | Expected |
|---|---|---|
| **Build succeeds** | `make build` | Tag created, exit 0 |
| **Platform correct** | `docker inspect <tag> --format '{{.Os}}/{{.Architecture}}'` | `linux/amd64` |
| **Smoke test (no API)** | `docker run --rm --cpus=2 --memory=4g <tag> ...` | Exit 0, results in /output |
| **Smoke test (mock API)** | Same with `FIREWORKS_API_KEY=mock` | Graceful fallback, no crash |
| **Import check** | `docker run --rm --entrypoint python3 <tag> -c "from agent import Pipeline"` | OK |

### 6.4 Testing Order (build recommendation)

1. **Phase 1** — Test `evaluate.py` independently (no parallelization needed)
   - Use existing eval_results/run_001_*.xlsx and grade_results.py as reference
   - Verify `fuzzy_match` output matches official `scripts/evaluate.py`
   - Verify XLSX produces correct breakdown

2. **Phase 2** — Test `batch_runner.py` with `n_workers=1` (sequential fallback)
   - Verify results identical to `Pipeline.process_batch()`
   - Then test with `n_workers=2`, compare timing improvement

3. **Phase 3** — Test `deploy.py` end-to-end
   - Build a new image
   - Run smoke tests from existing `SUBMISSION_CONTAINER_PLAN.md` (Modes A-D)
   - Push and verify public pull

4. **Phase 4** — Full integration test
   - Run BatchRunner on a full eval set
   - Grade with evaluate.py
   - Build container from same code
   - Verify container results match local results

---

## Appendix A: Edge Cases & Failure Modes

| Edge Case | Handling |
|---|---|
| Empty tasks list | Return empty results immediately |
| Single task | Run in main process (no spawn overhead) |
| All workers crash | Fall back to sequential in main process |
| Model file missing | Pipeline logs warning, runs without local LLM |
| No ground truth for some tasks | Skip those in evaluation, log warning |
| Docker build fails | Deploy script exits with descriptive error |
| Docker push fails | Retry once, then exit with error |
| XLSX write fails mid-file | Write to `.tmp` then `os.replace()` (atomic) |
| Pipeline version mismatch | runner captures `Pipeline.__module__` in metadata |

## Appendix B: Per-Stage Timing (Without Touching Pipeline)

Since we cannot instrument inside `Pipeline.process()`, we capture timing at two points:

1. **Task-level wall clock**: `t0 = time.monotonic()` before `pipe.process()`, `t1` after. This gives total processing time for each task.
2. **Pipeline-level granularity**: The existing Pipeline already includes internal timing in its logging (e.g., per-stage ms). We capture the WARNING-level log output via `logging` and parse relevant timestamps, OR we accept that task-level wall time is sufficient for the evaluation report.

For the XLSX report, the `"Timing (ms)"` column represents total elapsed per-task. If per-stage timing is critical, a future enhancement could wrap `pipe.process` with a context manager that patches `logging.getLogger("pipeline")` to intercept timing log lines.

---

## Appendix C: `runner/` CLI Entry Points

Each module in `runner/` exposes a `__main__` block for standalone use:

```bash
# Parallel batch run
python -m runner.batch_runner \
    --input input/tasks.json \
    --output results.json \
    --workers 2 \
    --deadline 600

# Evaluate results
python -m runner.evaluate \
    --results results.json \
    --gold input/dev_40.json \
    --output eval_results/report.xlsx \
    --verbose

# Deploy container
python -m runner.deploy \
    --push \
    --verify \
    --update-context
```
