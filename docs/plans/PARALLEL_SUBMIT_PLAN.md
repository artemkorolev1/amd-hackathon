# Parallel Submission Architecture Plan

> **Project:** AMD ACT II Hackathon — Track 1 Token Efficient Routing Agent
> **Root:** `/home/artem/dev/amd-hackathon/`
> **Current Container:** `ghcr.io/artemkorolev1/amd-hackathon-submit:v15`
> **Date:** 2026-07-13
> **Golden Rule:** The `agent/` directory (pipeline, classifiers, solvers) is **untouchable**. Everything new lives in `staging/` and never modifies `agent/`.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Naming Convention: Staged/Prepared Code](#2-naming-convention-stagedprepared-code)
3. [Component Descriptions](#3-component-descriptions)
4. [Queue and Worker Model](#4-queue-and-worker-model)
5. [ML Classifier Bulk Routing → Task Distribution](#5-ml-classifier-bulk-routing--task-distribution)
6. [5× Processing + Judgment Mechanism](#6-5-processing--judgment-mechanism)
7. [File-by-File Implementation Plan](#7-file-by-file-implementation-plan)
8. [Integration with Grader Contract](#8-integration-with-grader-contract)
9. [How It Builds Around (Not On Top Of) the Current Pipeline](#9-how-it-builds-around-not-on-top-of-the-current-pipeline)
10. [Testing Strategy](#10-testing-strategy)

---

## 1. Architecture Overview

### Text Diagram

```
┌──────────┐     ┌──────────────────────────────────────────────────────────────┐
│ /input/  │────▶│                  staging/entrypoint.py                       │
│tasks.json│     │  (NEW container entrypoint — does NOT import Pipeline)       │
└──────────┘     │                                                              │
                 │  1. READ tasks from /input/tasks.json                        │
                 │  2. BULK CLASSIFY all tasks via agent.category_filter        │
                 │     (fast, pure regex/heuristic — no model load)            │
                 │  3. ENQUEUE tasks into ReadyQueue by category                │
                 │  4. DISPATCH to LLM workers based on type + availability     │
                 │  5. COLLECT up to 5 answers per task                         │
                 │  6. JUDGE: majority vote + consistency check                │
                 │  7. WRITE /output/results.json                              │
                 └──────────────────────────────────────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────────┐
          ▼                       ▼                           ▼
┌──────────────────┐   ┌──────────────────┐       ┌──────────────────────┐
│  Worker Pool A   │   │  Worker Pool B   │       │   Worker Pool C      │
│  Fireworks LLMs  │   │  Local GGUF LLM  │       │  Deterministic       │
│  (API-based)     │   │  (llama-cpp)     │       │  Solvers (fast)      │
│                  │   │                  │       │                      │
│  ┌─ fw_worker_1  │   │  ┌─ loc_worker_1 │       │  ┌─ det_worker_1    │
│  └─ fw_worker_2  │   │  └─ loc_worker_2 │       │  └─ det_worker_2    │
│                  │   │                  │       │                      │
│  Each task →     │   │  Each task →     │       │  Each task →        │
│  1 response      │   │  1 response      │       │  1 response         │
└──────────────────┘   └──────────────────┘       └──────────────────────┘
         │                      │                            │
         └──────────────┬───────┴────────────────────────────┘
                        ▼
               ┌──────────────────┐
               │   ReadyJudge     │
               │  (voting module) │
               │                  │
               │  For each task:  │
               │  collect up to 5 │
               │  answers, apply  │
               │  majority vote   │
               │  → final answer  │
               └──────────────────┘
                        │
                        ▼
               ┌──────────────────┐
               │  /output/        │
               │  results.json    │
               └──────────────────┘
```

### Comparison: Current vs New

| Aspect | Current (`harness.py`) | New (`staging/entrypoint.py`) |
|--------|----------------------|------------------------------|
| Classification | Per-task, inline in `Pipeline.process()` | Bulk classification upfront (all tasks at once via `classify_batch()`) |
| Worker model | Single Pipeline instance, sequential | Multiple worker pools by type, parallel dispatch |
| Redundancy | Single answer per task | Up to 5 answers → majority vote |
| Model usage | One local model + Fireworks fallback | Configurable per-category model routing |
| Entrypoint | `harness.py` (uses `Pipeline`) | `staging/entrypoint.py` (uses `agent.category_filter` + worker processes) |
| Code location | `agent/`, `harness.py` | `staging/` (new directory, no changes to existing files) |

---

## 2. Naming Convention: Staged/Prepared Code

All new code lives in a **`staging/`** directory at the project root. The name implies "prepared but not yet live" — code that's staged for a parallel submission path distinct from the main pipeline.

### Directory and Component Naming

| Component | Name | Rationale |
|-----------|------|-----------|
| Module | `staging/` | "Staging" = prepared, ready to deploy, not yet activated |
| Entrypoint | `staging/entrypoint.py` | The container entrypoint for the new architecture |
| Task queue | `staging/ready_queue.py` | `ReadyQueue` — tasks are "ready" for dispatch |
| Worker manager | `staging/ready_pool.py` | `ReadyPool` — workers are pooled and "ready" |
| Worker base | `staging/ready_worker.py` | `ReadyWorker` — base class for all worker types |
| Fireworks worker | `staging/workers/fw_worker.py` | `FwWorker` — Fireworks API worker |
| Local worker | `staging/workers/loc_worker.py` | `LocWorker` — local GGUF model worker |
| Deterministic worker | `staging/workers/det_worker.py` | `DetWorker` — deterministic solver worker |
| Judgment module | `staging/ready_judge.py` | `ReadyJudge` — votes and judges |
| Config | `staging/ready_config.py` | `ReadyConfig` — configuration for the staging system |
| Dockerfile | `Dockerfile.staging` | Separate Dockerfile for the staging submission |
| Plan | `docs/plans/PARALLEL_SUBMIT_PLAN.md` | This document |

### Prefix/Suffix Convention

- **`Ready`** prefix for classes (e.g., `ReadyQueue`, `ReadyPool`, `ReadyWorker`, `ReadyJudge`)
- **`staging_`** prefix for env vars (e.g., `STAGING_WORKERS`, `STAGING_JUDGMENT_VOTES`)
- **`staging/`** directory for all new code

This makes it clear at a glance which code is part of the new architecture vs. the existing pipeline.

---

## 3. Component Descriptions

### 3.1 `staging/ready_config.py` — Configuration

Loads configuration from environment variables with sensible defaults. Controls worker counts, model assignments, judgment rules, and timeout budgets.

**Key settings:**
- `STAGING_FW_WORKERS` — number of Fireworks API worker processes (default: 2)
- `STAGING_LOC_WORKERS` — number of local model worker processes (default: 1)
- `STAGING_DET_WORKERS` — number of deterministic solver workers (default: 1)
- `STAGING_JUDGMENT_VOTES` — how many answers to collect per task before judging (default: 5)
- `STAGING_VOTE_MIN_AGREEMENT` — minimum agreement fraction for majority vote (default: 0.5)
- `STAGING_WORKER_TIMEOUT` — per-worker task timeout in seconds (default: 30.0)
- `STAGING_DEADLINE_S` — overall deadline (default: inherits from `DEADLINE_S` or 600)
- `STAGING_CATEGORY_WORKER_MAP` — which worker types handle which categories
- `STAGING_FALLBACK_STRATEGY` — what to do when max votes not reached (default: "best_available")

### 3.2 `staging/ready_queue.py` — Task Queue

A priority-aware task queue that holds classified tasks ready for worker dispatch.

**Data structure:**
```python
class ReadyQueue:
    """Multi-category task queue. Tasks are enqueued after bulk classification."""
    
    def __init__(self):
        self._queues: dict[str, queue.Queue] = {}  # category → Queue of ReadyTask
    
    def enqueue_batch(self, classified_tasks: list[ReadyTask]):
        """Add classified tasks to the appropriate category queues."""
    
    def dequeue(self, category: str) -> Optional[ReadyTask]:
        """Get next task for a given category (blocking with timeout)."""
    
    def dequeue_any(self, preferred_categories: list[str]) -> Optional[ReadyTask]:
        """Get next task from any preferred category, or fallback to any category."""
    
    def peek_all(self) -> list[ReadyTask]:
        """View all pending tasks (for deadline-aware decisions)."""
    
    def task_counts_by_category(self) -> dict[str, int]:
        """Return remaining task count per category."""
    
    @property
    def empty(self) -> bool:
        """True when all queues are drained."""
```

**Task object:**
```python
@dataclass
class ReadyTask:
    task_id: str
    prompt: str
    category: str
    category_4way: str
    raw_scores: dict[str, float]
    confidence: float
    score_delta: float
    answers: list[dict]  # populated as workers respond: {worker_id, answer, timing_ms}
    status: str = "pending"  # pending → in_progress → judged
```

### 3.3 `staging/ready_pool.py` — Worker Pool Manager

Manages the lifecycle of all worker processes, distributing tasks based on type and availability.

```python
class ReadyPool:
    """Orchestrates multiple worker pools by type (Fireworks, local, deterministic)."""
    
    def __init__(self, config: ReadyConfig):
        self._fw_pool: list[ReadyWorker] = []
        self._loc_pool: list[ReadyWorker] = []
        self._det_pool: list[ReadyWorker] = []
        self._all_workers: dict[str, ReadyWorker] = {}  # worker_id → worker
    
    def start(self):
        """Spawn all worker processes."""
    
    def assign(self, task: ReadyTask) -> list[str]:
        """Determine which workers should process this task based on category + availability.
        Returns list of worker_ids assigned."""
    
    def collect(self, timeout: float) -> list[ReadyTask]:
        """Collect completed task results from workers."""
    
    def shutdown(self):
        """Gracefully terminate all workers."""
    
    def worker_availability(self) -> dict[str, list[str]]:
        """Return {worker_type: [available_worker_ids]}."""
    
    @property
    def available_count(self) -> int:
        """Total number of workers currently available."""
```

**Worker assignment algorithm:**
1. Look up which worker types handle the task's category (from `STAGING_CATEGORY_WORKER_MAP`)
2. For each eligible type, find available (idle) workers
3. If multiple workers of same type available, round-robin assign
4. If insufficient workers of primary type, fall back to next eligible type
5. If no workers available, task goes to a wait queue

### 3.4 `staging/ready_worker.py` — Worker Base Class

Abstract base for all worker types. Runs in its own process.

```python
class ReadyWorker(ABC):
    """Base class for all worker types. Each worker is a separate process."""
    
    def __init__(self, worker_id: str, worker_type: str, config: ReadyConfig):
        self.worker_id = worker_id
        self.worker_type = worker_type
        self.config = config
    
    @abstractmethod
    def initialize(self):
        """Load model, configure API client, etc."""
    
    @abstractmethod
    def process(self, task: ReadyTask) -> dict:
        """Process a single task and return {'worker_id', 'answer', 'timing_ms'}."""
    
    @abstractmethod
    def shutdown(self):
        """Release resources."""
```

### 3.5 `staging/workers/fw_worker.py` — Fireworks Worker

Calls Fireworks API. Fast, stateless, no model loading overhead.

- Reads `FIREWORKS_API_KEY` from environment
- Uses `agent.solvers.fireworks.FireworksSolver` and `agent.solvers.fw_router.route`
- Handles rate limiting, retries (exponential backoff on 429/500-504)
- Reports completion time for availability tracking

### 3.6 `staging/workers/loc_worker.py` — Local Worker

Loads the local GGUF model via `llama-cpp-python`. Each local worker is a separate process with its own model instance.

- Reads `MODEL_PATH`, `N_GPU_LAYERS`, `N_THREADS`, `N_CTX` from environment
- Uses `agent.pipeline.PipelineConfig` for config (but not the Pipeline class itself)
- Uses `agent.solvers.local_vote.solve_with_consensus` or direct llama_cpp calls
- Memory-intensive — limited to 1-2 concurrent instances (4 GB RAM budget)

### 3.7 `staging/workers/det_worker.py` — Deterministic Worker

Runs deterministic solvers for tasks that don't need LLM inference.

- Uses the same `agent.solvers.deterministic.*` functions as the current pipeline
- Categories: math_arithmetic, sentiment, summarization, factual, code_debugging
- Near-instant execution, acts as a fast-path for suitable tasks

### 3.8 `staging/ready_judge.py` — Judgment / Voting Module

Collects multiple answers per task and applies judgment logic to select the best final answer.

```python
class ReadyJudge:
    """Applies voting/judgment to multi-answer tasks."""
    
    def __init__(self, config: ReadyConfig):
        self.votes_required = config.judgment_votes
        self.min_agreement = config.vote_min_agreement
    
    def add_answer(self, task_id: str, answer: dict):
        """Record one worker's answer for a task."""
    
    def ready_to_judge(self, task_id: str) -> bool:
        """Check if enough votes have been collected."""
    
    def judge(self, task_id: str) -> tuple[str, dict]:
        """Apply majority vote / consistency check. Returns (final_answer, metadata)."""
    
    def judge_all(self) -> list[dict]:
        """Judge all completed tasks and return final results list."""
```

**Judgment strategy (tiered):**

| Tier | Condition | Method |
|------|-----------|--------|
| 1 | ≥3 answers agree | Majority vote — pick the most common answer |
| 2 | 2 answers agree, others disagree | Majority vote with confidence penalty |
| 3 | All answers different | Pick the highest-confidence answer (based on worker priority: Fireworks > Local > Deterministic) |
| 4 | Some workers timed out | Use whatever answers are available, fall back to best single |
| 5 | All workers failed | Return empty string |

**Consistency verification:**
- For each judged answer, run `verify()` from `agent.solvers.verify` if available
- If the majority answer fails verification, try the next most common answer
- If all fail verification, return the highest-confidence answer anyway

---

## 4. Queue and Worker Model

### 4.1 Worker Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      ReadyPool Manager                          │
│  (main process — orchestrates dispatch & collection)            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────┐  ┌──────────────────┐  ┌───────────┐  │
│   │  Fireworks Pool     │  │  Local Pool      │  │  Det Pool │  │
│   │  (2 workers)        │  │  (1-2 workers)   │  │  (1 worker)│  │
│   │                     │  │                  │  │           │  │
│   │  ┌───┐  ┌───┐      │  │  ┌───┐  ┌───┐   │  │  ┌───┐   │  │
│   │  │FW1│  │FW2│      │  │  │L1 │  │L2 │   │  │  │D1 │   │  │
│   │  └───┘  └───┘      │  │  └───┘  └───┘   │  │  └───┘   │  │
│   └─────────────────────┘  └──────────────────┘  └───────────┘  │
│                                                                  │
│  Communication: multiprocessing.Queue (tasks → workers)          │
│                 multiprocessing.Queue (workers → results)        │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Lifecycle of a Task

```
1. Bulk classify ──→ 2. Enqueue ──→ 3. Dispatch ──→ 4. Process ──→ 5. Collect ──→ 6. Judge
                       ReadyQueue        │              │              │
                                         │   ┌──────────┘              │
                                         │   ▼                         ▼
                                         │  fw_worker_1 ──→ answer_1   │
                                         │  fw_worker_2 ──→ answer_2   │
                                         │  loc_worker_1 ─→ answer_3   │
                                         │  loc_worker_2 ─→ answer_4   │
                                         │  det_worker_1 ─→ answer_5   │
                                         │                              │
                                         └──────────────────────────────┘
```

**Detailed flow:**

1. **Bulk classify** — All tasks classified at once via `agent.category_filter.classify_batch()` (pure regex, sub-millisecond per task, no model load)
2. **Enqueue** — Each task wrapped as `ReadyTask` with its category and scores, pushed into `ReadyQueue`
3. **Dispatch** — `ReadyPool` continuously pulls tasks from the queue, assigning each to available workers based on category eligibility. One task may be dispatched to multiple workers simultaneously.
4. **Process** — Each worker receives the task, processes it, returns `{worker_id, task_id, answer, timing_ms}`
5. **Collect** — Results land in a shared results queue. The `ReadyJudge` tracks how many answers per task have been collected.
6. **Judge** — When a task has enough votes (default: 5), judgment runs. If the deadline is approaching, judgment triggers early with whatever votes are available.

### 4.3 Worker Assignment Matrix

Which worker types handle which categories:

| Category | Fireworks | Local | Deterministic | Notes |
|----------|-----------|-------|---------------|-------|
| `math` | ✅ | ✅ | ✅ (arithmetic only) | Deterministic for simple arithmetic; FW/Local for complex math |
| `logic` | ✅ | ✅ | ❌ | Logic puzzles need LLM reasoning |
| `factual` | ✅ | ✅ | ✅ (simple QA only) | Deterministic for factual_qa patterns |
| `sentiment` | ✅ | ✅ | ✅ | Deterministic & LLM both work |
| `ner` | ✅ | ✅ | ✅ | Deterministic handles simple NER |
| `summarization` | ✅ | ✅ | ✅ | Deterministic for simple summarization |
| `code_gen` | ✅ | ✅ | ❌ | Code generation always needs LLM |
| `code_debug` | ✅ | ✅ | ✅ | Deterministic for trivial fixes |

**Priority for each category (which worker types to try first):**

| Category | Primary | Secondary | Tertiary |
|----------|---------|-----------|----------|
| `math` | Deterministic | Local | Fireworks |
| `logic` | Fireworks | Local | — |
| `factual` | Deterministic | Fireworks | Local |
| `sentiment` | Deterministic | Local | Fireworks |
| `ner` | Deterministic | Fireworks | Local |
| `summarization` | Deterministic | Fireworks | Local |
| `code_gen` | Fireworks | Local | — |
| `code_debug` | Fireworks | Local | Deterministic |

### 4.4 Concurrency Model

- **ProcessPoolExecutor** for each worker type pool (separate pools per type)
- **multiprocessing.Queue** for task dispatch and result collection
- **multiprocessing.Manager.dict** for shared state (availability tracking, result counts)
- **No threading** — processes avoid Python GIL issues

### 4.5 Timeout and Deadline Management

```
┌──────────────────────────────────────────────────────────┐
│                   Deadline Enforcement                    │
│                                                          │
│  Global deadline: DEADLINE_S (default 600s)              │
│                                                          │
│  Phase 1 — Bulk classify + queue setup: ≤ 5s            │
│  Phase 2 — Worker dispatch + collection: ≤ deadline - 10s│
│  Phase 3 — Judgment + output write: ≤ 5s                │
│                                                          │
│  Per-worker task timeout: 30s (STAGING_WORKER_TIMEOUT)  │
│  Per-task total wall time: 150s (5 votes × 30s)         │
│                                                          │
│  Deadline adaptation:                                    │
│  ├── >120s remaining: full 5-vote judgment              │
│  ├── 60-120s remaining: reduce to 3-vote judgment       │
│  └── <60s remaining: 1-vote (send to primary worker)   │
└──────────────────────────────────────────────────────────┘
```

---

## 5. ML Classifier Bulk Routing → Task Distribution

### 5.1 Bulk Classification

The current pipeline classifies one task at a time inside `Pipeline.process()`. The new architecture classifies **all tasks upfront** before any worker starts.

```python
from agent.category_filter import classify_batch

# Phase 1: Bulk classify ALL tasks (fast — pure regex, no model)
classified = classify_batch([task["prompt"] for task in tasks])
# Returns list of dicts: {category, category_4way, confidence, raw_scores, score_delta}

# Phase 2: Wrap into ReadyTask objects and enqueue
ready_tasks = []
for task, cls_result in zip(tasks, classified):
    ready_tasks.append(ReadyTask(
        task_id=task.get("task_id", f"task_{i}"),
        prompt=task.get("prompt", ""),
        category=cls_result["category"],
        category_4way=cls_result["category_4way"],
        raw_scores=cls_result["raw_scores"],
        confidence=cls_result["confidence"],
        score_delta=cls_result["score_delta"],
        answers=[],
        status="pending",
    ))

queue = ReadyQueue()
queue.enqueue_batch(ready_tasks)
```

**Why bulk classify vs per-task:**
- Avoids redundant function calls and regex compilation
- Allows workload-aware task dispatch (e.g., send all code tasks to FW workers first)
- Enables early detection of category distribution for worker allocation planning
- No model loading needed — pure regex/heuristic, <1ms per task

### 5.2 Category-Based Dispatch Logic

The dispatch algorithm assigns tasks to specific worker types and specific workers:

```python
def _select_workers_for_task(task: ReadyTask,
                              availability: dict[str, list[str]],
                              priority_map: dict[str, list[str]]) -> list[str]:
    """
    Select workers for a task based on its category and worker availability.
    
    Returns a list of worker_ids to dispatch the task to.
    Number of workers selected is ≤ STAGING_JUDGMENT_VOTES.
    """
    preferred_types = priority_map.get(task.category, ["fireworks"])
    selected_workers = []
    
    for wtype in preferred_types:
        available = availability.get(wtype, [])
        # Take available workers of this type, up to remaining slots needed
        needed = config.judgment_votes - len(selected_workers)
        selected_workers.extend(available[:needed])
        
        if len(selected_workers) >= config.judgment_votes:
            break
    
    return selected_workers[:config.judgment_votes]
```

### 5.3 Availability Tracking

```python
# Worker availability state (shared via Manager.dict)
worker_status = {
    "fw_worker_1": {"type": "fireworks", "busy": False, "current_task": None},
    "fw_worker_2": {"type": "fireworks", "busy": False, "current_task": None},
    "loc_worker_1": {"type": "local", "busy": True, "current_task": "task_03"},
    ...
}
```

- Workers report `busy=False` after completing a task
- Dispatcher checks availability before assigning
- If a worker crashes, it's removed from the pool and (optionally) replaced

---

## 6. 5× Processing + Judgment Mechanism

### 6.1 Goal: Each Task Processed 5 Times

Each task is dispatched to up to 5 different workers (1 per worker process). Workers can be from different pools (Fireworks, Local, Deterministic) or the same pool (multiple Fireworks workers with potentially different models).

**How 5 distinct answers are guaranteed:**
1. Dispatcher assigns the task to ≤5 workers simultaneously
2. Workers process independently (different processes, potentially different models)
3. Results are collected with task_id as correlation key
4. If fewer than 5 results arrive within timeout, judge uses whatever is available

### 6.2 Judgment Algorithm

```
For each task with N answers collected (N ≤ 5):

1. GROUP identical/similar answers using fuzzy_match cascade
   - Exact match → same group
   - Normalized match (lowercase, stripped) → same group
   - Numeric tolerance match (1% for math) → same group
   - Token overlap ≥50% → same group

2. FIND the largest group
   - If largest_group_size >= 3 → MAJORITY VOTE WINNER
   - If largest_group_size == 2 and N >= 4 → MAJORITY VOTE WINNER
   - If largest_group_size == 2 and N == 2 → AMBIGUOUS, use tiebreaker
   - If all groups size 1 → ALL DIFFERENT, use tiebreaker

3. VERIFY the winner
   - Run verify(winner, category) from agent.solvers.verify
   - If verification passes → final answer
   - If verification fails → try second-largest group
   - If no group passes → fall back to primary worker answer

4. TIEBREAKER (when no clear majority)
   - Option A: Trust Fireworks worker (API-based, typically stronger model)
   - Option B: Use the answer from the worker with lowest timing_ms (fastest = simplest = likely correct)
   - Option C: Fall back to deterministic solver if applicable
   - Configurable via STAGING_TIEBREAKER_STRATEGY
```

### 6.3 Configurable Worker Diversity

Each worker pool can be configured with different models:

```yaml
# Example config (via env vars)
STAGING_FW_MODELS=accounts/fireworks/models/deepseek-v4-flash,\
                  accounts/fireworks/models/llama-3.1-nemotron-70b-instruct
STAGING_LOC_MODELS=models/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

This means each Fireworks worker could use a different model, increasing answer diversity.

### 6.4 Impact of Deadline on Judgment Quality

| Remaining Time | Votes Collected | Judgment Strategy |
|----------------|-----------------|-------------------|
| >120s | 5 | Full majority vote |
| 60-120s | 3-5 | Majority vote, reduced confidence threshold (0.4) |
| 30-60s | 2-3 | Best of available, skip verification |
| <30s | 1 | Single best worker, skip judgment entirely |

---

## 7. File-by-File Implementation Plan

### 7.1 Directory Layout

```
/home/artem/dev/amd-hackathon/
├── staging/                          # NEW — parallel submission architecture
│   ├── __init__.py                   # Package init
│   ├── entrypoint.py                 # NEW container entrypoint (replaces harness.py)
│   ├── ready_config.py               # Configuration for staging system
│   ├── ready_queue.py                # Task queue (ReadyTask, ReadyQueue)
│   ├── ready_pool.py                 # Worker pool manager (ReadyPool)
│   ├── ready_worker.py               # Worker base class (ReadyWorker)
│   ├── ready_judge.py                # Judgment/voting module (ReadyJudge)
│   └── workers/                      # Worker implementations
│       ├── __init__.py
│       ├── fw_worker.py              # Fireworks API worker
│       ├── loc_worker.py             # Local GGUF model worker
│       └── det_worker.py             # Deterministic solver worker
├── Dockerfile.staging                # NEW — separate Dockerfile for staging submission
├── docs/plans/PARALLEL_SUBMIT_PLAN.md # NEW — this document
```

### 7.2 File: `staging/__init__.py`

```python
"""Staging — parallel submission architecture for AMD ACT II Track 1.

This module builds AROUND the existing agent pipeline (agent/pipeline.py)
without touching any files in agent/.

Components:
- ready_config  : Configuration from env vars
- ready_queue   : Multi-category task queue
- ready_pool    : Worker pool manager (Fireworks, Local, Deterministic)
- ready_worker  : Worker base class
- ready_judge   : Voting/judgment module
- workers/      : Worker implementations
- entrypoint    : Container entrypoint (replaces harness.py for staging)
"""

from .ready_config import ReadyConfig
from .ready_queue import ReadyQueue, ReadyTask
from .ready_pool import ReadyPool
from .ready_judge import ReadyJudge

__all__ = [
    "ReadyConfig",
    "ReadyQueue", "ReadyTask",
    "ReadyPool",
    "ReadyJudge",
]
```

### 7.3 File: `staging/ready_config.py`

**Purpose:** Central configuration loaded from environment variables.

**Key config fields:**
- `fw_workers: int` — default 2
- `loc_workers: int` — default 1 (memory-constrained)
- `det_workers: int` — default 1
- `judgment_votes: int` — default 5
- `vote_min_agreement: float` — default 0.5
- `worker_timeout_s: float` — default 30.0
- `deadline_s: float` — default from `DEADLINE_S` or 600
- `fw_api_key: str` — from `FIREWORKS_API_KEY`
- `fw_models: list[str]` — from `STAGING_FW_MODELS`
- `loc_model_path: str` — from `MODEL_PATH`
- `tiebreaker_strategy: str` — from `STAGING_TIEBREAKER` (default: "fw_priority")
- `category_priority: dict` — built-in mapping of category → worker type priority
- `fallback_strategy: str` — from `STAGING_FALLBACK` (default: "best_available")

### 7.4 File: `staging/ready_queue.py`

**Purpose:** Multi-category task queue populated after bulk classification.

**Key elements:**
- `@dataclass ReadyTask` — task wrapper with classification results, status, and answer collection
- `class ReadyQueue` — dict of `category → multiprocessing.Queue` with dispatch methods
- `enqueue_batch()` — bulk enqueue from classified task list
- `dequeue_by_category()` — get task for specific category (non-blocking)
- `dequeue_any()` — get task for any of preferred categories
- Thread‑/process‑safe (uses `multiprocessing.Queue` internally)

### 7.5 File: `staging/ready_worker.py`

**Purpose:** Abstract base for worker implementations.

**Key elements:**
- `class ReadyWorker(ABC)` with abstract methods `initialize()`, `process(task)`, `shutdown()`
- Worker communication via shared `multiprocessing.Queue` (one per pool)
- Status reporting via `Manager.dict`
- Built‑in timeout enforcement per task

### 7.6 File: `staging/ready_pool.py`

**Purpose:** Manages all worker pools and orchestrates dispatch.

**Key elements:**
- `class ReadyPool` — starts/ stops all workers, tracks availability
- `start()` — spawns worker processes for each pool
- `dispatch_loop()` — main loop: pull from queue → assign to workers → collect results
- `assign(task)` — determines which workers handle the task based on category priority + availability
- `collect()` — gather completed results from worker output queues
- `shutdown()` — graceful termination with timeout

### 7.7 File: `staging/read_judge.py`

**Purpose:** Collects multiple answers per task, applies voting/ judgment, produces final output.

**Key elements:**
- `class ReadyJudge` — collects answers, decides when to judge, applies tiered strategy
- `add_answer(task_id, worker_answer)` — records a worker's answer
- `ready_to_judge(task_id)` — checks if enough votes are in
- `judge(task_id)` — applies majority vote with fuzzy_match grouping + tiebreaker
- `judge_all(pending_tasks)` — batch judge all completed tasks
- Uses `scripts.evaluate.fuzzy_match` and `agent.solvers.verify` for consistency

### 7.8 File: `staging/workers/fw_worker.py`

**Purpose:** Fireworks API worker — fast, stateless, handles rate limiting.

```python
class FwWorker(ReadyWorker):
    def initialize(self):
        from agent.solvers.fireworks import FireworksSolver
        self._solver = FireworksSolver()
        # Select which FW model this worker uses (from config)
        self._model_id = self.config.fw_models[self.worker_index]
    
    def process(self, task: ReadyTask) -> dict:
        from agent.solvers.fw_router import route
        cfg = route(task.category, task.prompt, 0.5)
        t0 = time.monotonic()
        answer = self._solver.solve(
            self._model_id, cfg.system_prompt, task.prompt,
            max_tokens=cfg.max_tokens, temperature=cfg.temperature,
            prefill=cfg.prefill, task_type=cfg.task_type,
            timeout=self.config.worker_timeout_s,
        )
        elapsed = (time.monotonic() - t0) * 1000
        return {"worker_id": self.worker_id, "task_id": task.task_id,
                "answer": answer, "timing_ms": elapsed}
```

### 7.9 File: `staging/workers/loc_worker.py`

**Purpose:** Local GGUF model worker — memory-intensive but offline-capable.

```python
class LocWorker(ReadyWorker):
    def initialize(self):
        # Load llama-cpp-python model (separate process, own memory)
        from llama_cpp import Llama
        self._llm = Llama(
            model_path=self.config.loc_model_path,
            n_ctx=2048, n_gpu_layers=0, n_threads=2,
            flash_attn=True, verbose=False,
        )
    
    def process(self, task: ReadyTask) -> dict:
        # Build system prompt (same logic as pipeline but copied here)
        sys_prompt = self._get_system_prompt(task.category)
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": task.prompt}]
        t0 = time.monotonic()
        # ... inference ...
        answer = result["choices"][0]["message"]["content"]
        elapsed = (time.monotonic() - t0) * 1000
        return {"worker_id": self.worker_id, "task_id": task.task_id,
                "answer": answer, "timing_ms": elapsed}
    
    def _get_system_prompt(self, category: str) -> str:
        """Replicate the pipeline's prompt construction (without importing Pipeline)."""
        # Minimal copy of the prompt logic to avoid touching agent/pipeline.py
        # Can import from agent.dynamic_prompts if needed
```

**Important design note:** The local worker does NOT import `Pipeline`. It imports only what it needs from `agent.solvers.*` and `agent.dynamic_prompts`. This keeps the staging code truly separate from the pipeline.

### 7.10 File: `staging/workers/det_worker.py`

**Purpose:** Fast deterministic solver path for categories where LLM is overkill.

```python
class DetWorker(ReadyWorker):
    def initialize(self):
        # Import deterministic solvers
        from agent.solvers.deterministic import (
            solve_arithmetic, solve_factual_qa, solve_sentiment,
            solve_summarization, solve_code_debugging,
        )
        self._solvers = {
            "math": solve_arithmetic,
            "factual": solve_factual_qa,
            "sentiment": solve_sentiment,
            "summarization": solve_summarization,
            "code_debug": solve_code_debugging,
        }
    
    def process(self, task: ReadyTask) -> dict:
        solver = self._solvers.get(task.category)
        if not solver:
            return {"worker_id": self.worker_id, "task_id": task.task_id,
                    "answer": "", "timing_ms": 0}
        t0 = time.monotonic()
        answer = solver(task.prompt, task.category) or ""
        elapsed = (time.monotonic() - t0) * 1000
        return {"worker_id": self.worker_id, "task_id": task.task_id,
                "answer": answer, "timing_ms": elapsed}
```

### 7.11 File: `staging/entrypoint.py`

**Purpose:** The NEW container entrypoint. Reads tasks, runs the staging architecture, writes results.

```python
#!/usr/bin/env python3
"""staging/entrypoint.py — Parallel submission container entrypoint.

Reads tasks from /input/tasks.json, runs bulk classification, distributed
worker processing with voting/judgment, writes /output/results.json.

Does NOT import agent.Pipeline — imports only specific agent modules
(category_filter, solvers) needed by workers.
"""

import json, logging, os, sys, time
from staging import ReadyConfig, ReadyQueue, ReadyPool, ReadyJudge
from agent.category_filter import classify_batch

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger("staging-entrypoint")

def _read_tasks() -> list[dict]:
    """Read tasks from /input/tasks.json (or fallbacks)."""
    # Same logic as harness.py

def _write_output(results: list[dict]) -> None:
    """Atomic write to /output/results.json."""
    # Same pattern as harness.py

def main() -> None:
    # 1. Load config
    config = ReadyConfig()
    
    # 2. Read tasks
    tasks = _read_tasks()
    if not tasks:
        _write_output([])
        return
    
    deadline = time.monotonic() + config.deadline_s
    
    # 3. BULK CLASSIFY all tasks
    logger.info("Bulk classifying %d tasks...", len(tasks))
    prompts = [t.get("prompt", t.get("question", "")) for t in tasks]
    classified = classify_batch(prompts)
    
    # 4. Build task queue
    queue = ReadyQueue()
    for task, cls in zip(tasks, classified):
        queue.enqueue(ReadyTask(
            task_id=task.get("task_id", "..."),
            prompt=task.get("prompt", ""),
            category=cls["category"],
            ...
        ))
    
    # 5. Start worker pool
    pool = ReadyPool(config)
    pool.start()
    
    # 6. Dispatch + collect loop
    judge = ReadyJudge(config)
    pool.dispatch_loop(queue, judge, deadline)
    
    # 7. Judge remaining tasks
    final_results = judge.judge_all()
    
    # 8. Write output
    _write_output(final_results)
    pool.shutdown()

if __name__ == "__main__":
    main()
```

### 7.12 File: `Dockerfile.staging`

A new Dockerfile for the staging submission container. Copies the `staging/` module instead of `harness.py`, uses `staging/entrypoint.py` as ENTRYPOINT.

```dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /

# Runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps (same as current Dockerfile + no new deps)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && apt-get purge -y --auto-remove build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Copy agent module (needed for classifier + solvers)
COPY agent/ /agent/

# Copy staging module (NEW parallel architecture)
COPY staging/ /staging/

# Copy model
COPY models/ /models/

# Grader I/O paths
RUN mkdir -p /input /output

ENV PYTHONPATH=/ \
    MODEL_PATH=/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    N_GPU_LAYERS=0 \
    N_THREADS=2 \
    N_CTX=2048

ENTRYPOINT ["python3", "-u", "-m", "staging.entrypoint"]
```

---

## 8. Integration with Grader Contract

### 8.1 Input Contract

Same as current: grader mounts `/input/tasks.json:ro`.

Format:
```json
[
  {"task_id": "t1", "prompt": "What is the capital of France?"},
  {"task_id": "t2", "prompt": "Write a Python function to check if a string is a palindrome."}
]
```

Handles both object arrays and string-only arrays, plus `TASK_COUNT` env var.

### 8.2 Output Contract

Same as current: write `/output/results.json` with atomic write pattern.

Format:
```json
[
  {"task_id": "t1", "answer": "Paris"},
  {"task_id": "t2", "answer": "def is_palindrome(s): ..."}
]
```

**Additional output metadata (optional, not required by grader but useful for debugging):**
- `staging/` can log detailed judgment info to stderr (each task's vote counts, agreement score)
- Worker timing can be logged to stderr for performance analysis
- The results.json contract is strictly preserved — extra fields are allowed but `task_id` and `answer` are guaranteed

### 8.3 Environment Variables

| Variable | Required? | In Dockerfile? | In code? | Purpose |
|----------|-----------|----------------|----------|---------|
| `FIREWORKS_API_KEY` | Yes (if using API) | NO | Fallback `""` | Fireworks access |
| `ALLOWED_MODELS` | Yes (grader sets) | NO | Parsed from env | Model whitelist |
| `DEADLINE_S` | Yes (grader sets) | NO | Default 600 | Total runtime budget |
| `TASK_COUNT` | No | NO | Parsed from env | Limit tasks |
| `MODEL_PATH` | If local model | YES | Default path | GGUF location |
| `N_GPU_LAYERS` | If local model | YES | 0 | GPU offloading |
| `N_THREADS` | If local model | YES | 2 | CPU threads |
| `N_CTX` | If local model | YES | 2048 | Context window |
| `STAGING_FW_WORKERS` | No | NO | 2 | Fireworks worker count |
| `STAGING_LOC_WORKERS` | No | NO | 1 | Local worker count |
| `STAGING_DET_WORKERS` | No | NO | 1 | Deterministic worker count |
| `STAGING_JUDGMENT_VOTES` | No | NO | 5 | Votes per task |
| `STAGING_FW_MODELS` | No | NO | Fireworks default | Comma-separated model IDs |
| `STAGING_TIEBREAKER` | No | NO | `fw_priority` | Tiebreak strategy |
| `STAGING_FALLBACK` | No | NO | `best_available` | Fallback strategy |

### 8.4 Runtime Constraints

All grader constraints are treated the same as the current harness:

| Constraint | Value | How Staging Handles It |
|------------|-------|------------------------|
| CPU | 2 vCPU | `n_threads=2` per local worker; worker count capped by memory |
| RAM | 4 GB | Memory budget formula → cap at 1 local + 2 FW + 1 det workers |
| Deadline | 600s | Global deadline with phased adaptation |
| Per-task | 30s | Per-worker timeout at 30s |
| Startup | <60s | Bulk classify is instant; model loading is the bottleneck (~10s) |
| Architecture | linux/amd64 | Dockerfile.staging uses `--platform linux/amd64` |

---

## 9. How It Builds Around (Not On Top Of) the Current Pipeline

### 9.1 No Changes to `agent/`

The `agent/` directory is **completely untouched**. No files in `agent/` are modified, renamed, or deleted. The staging architecture:

- **Imports from** `agent.category_filter` for bulk classification
- **Imports from** `agent.solvers.fireworks`, `agent.solvers.fw_router`, `agent.solvers.deterministic`, `agent.solvers.verify` for worker implementations
- **Imports from** `agent.dynamic_prompts` for system prompt building (if needed in local worker)
- **Does NOT import** `agent.Pipeline` — the `Pipeline` class is entirely bypassed

### 9.2 Isolation from `harness.py`

- `harness.py` remains untouched and continues to work as the entrypoint for the current submission container
- `staging/entrypoint.py` is a completely separate entrypoint
- Different Dockerfiles (`Dockerfile` vs `Dockerfile.staging`) produce different container images

### 9.3 No Changes to `runner/`

- `runner/batch_runner.py` — untouched (staging has its own worker management)
- `runner/evaluate.py` — can still be used to grade staging output (same output format)
- `runner/deploy.py` — could be extended to also build `Dockerfile.staging`, but no existing code changes needed

### 9.4 Shared Import Paths (Read-Only)

Both the current pipeline and the staging architecture share these same imports:

| Import | Used By Current | Used By Staging |
|--------|----------------|-----------------|
| `agent.category_filter` | ✅ Pipeline.process() | ✅ entrypoint (bulk classify) |
| `agent.solvers.fireworks` | ✅ Pipeline | ✅ FwWorker |
| `agent.solvers.deterministic` | ✅ Pipeline | ✅ DetWorker |
| `agent.solvers.verify` | ✅ Pipeline | ✅ ReadyJudge |
| `agent.dynamic_prompts` | ✅ Pipeline | ✅ LocWorker (if needed) |

The staging code reads these as libraries — it never modifies them.

### 9.5 Coexistence

Both architectures can coexist in the same repository:
- `python3 -u harness.py` → runs current sequential pipeline
- `python3 -u -m staging.entrypoint` → runs new parallel submission architecture
- `Dockerfile` → builds the current container
- `Dockerfile.staging` → builds the new container

---

## 10. Testing Strategy

### 10.1 Unit Tests (Per Component)

| Component | What to Test | How |
|-----------|-------------|-----|
| `ReadyQueue` | Enqueue/dequeue by category, empty detection, concurrent access | Unit tests with `multiprocessing.Queue` |
| `ReadyPool` | Worker assignment by category, availability tracking, crash recovery | Mock workers, verify dispatch logic |
| `ReadyJudge` | Majority vote, tiebreaker, fuzzy grouping, verification pass/fail | Synthetic answers with known correct outcomes |
| `FwWorker` | API call, response parsing, rate limiting, timeout | Mock HTTP, verify behavior |
| `LocWorker` | Model loading, prompt building, answer extraction | Mock llama_cpp, test prompt templates |
| `DetWorker` | Category-to-solver routing, empty results | Test with known task categories |

### 10.2 Integration Tests

| Test | Scenario | Verification |
|------|----------|-------------|
| Bulk classify → queue | Run `classify_batch()` on 19 diverse tasks → verify all classified correctly | ✓ category confidence > 0 |
| 5-vote judgment | Process 3 tasks with 5 votes each → verify 3 final answers | ✓ output has 3 entries |
| Tiebreaker | 5 votes, 2-2-1 split → verify tiebreaker picks correctly | ✓ tiebreak produces deterministic result |
| Deadline adaptation | Set short deadline → verify fewer votes collected but output still valid | ✓ output produced before deadline |
| Worker crash | Kill one worker mid-run → verify remaining workers handle all tasks | ✓ all tasks complete |

### 10.3 Container-Level Tests (Same as Current Pattern)

Use the same smoke test modes from `SUBMISSION_CONTAINER_PLAN.md`:

| Mode | Purpose | Config |
|------|---------|--------|
| **Mode A** | No API, grader constraints | `--cpus=2 --memory=4g`, verify startup + output format |
| **Mode B** | With grader env vars + mock API key | Test graceful fallback with invalid key |
| **Mode C** | Full pipeline coverage | 10 diverse tasks across all categories |
| **Mode D** | Real Fireworks API key | End-to-end with actual API calls |

### 10.4 Comparison Testing

Run both the current pipeline and the staging architecture on the same evaluation set and compare:

```bash
# Run current pipeline
python3 -u harness.py eval_data.json -o output_current.json

# Run staging architecture
python3 -u -m staging.entrypoint eval_data.json -o output_staging.json

# Compare results
python3 -m runner.evaluate output_current.json eval_data.json report_current.xlsx
python3 -m runner.evaluate output_staging.json eval_data.json report_staging.xlsx
```

**Success criteria:**
- Staging accuracy ≥ current pipeline accuracy (benefit of majority voting)
- Staging handles deadline more gracefully (parallel processing is faster)
- No crashes or exceptions in staging runs

### 10.5 Edge Cases

| Edge Case | Expected Behavior |
|-----------|------------------|
| All tasks same category | Workers specialized for that category handle all; queue doesn't block |
| Empty task list | Write empty output, exit cleanly |
| All workers busy | Tasks wait in queue; dispatch resumes when workers free |
| Worker returns empty answer | Judged as normal; adds to pool for vote counting |
| Deadline reached mid-dispatch | Judge whatever votes collected, write output, shutdown |
| No Fireworks API key | FW workers start but return empty; Local + Det handle all |
| Model file missing | Local workers return empty; remaining workers handle all |
| Memory pressure | Worker process OOM-killed; pool detects crash, logs, continues |

---

## Implementation Order

| Step | Files | Effort | Depends On |
|------|-------|--------|------------|
| 1 | `staging/__init__.py`, `staging/ready_config.py` | Small | — |
| 2 | `staging/ready_queue.py` (ReadyTask, ReadyQueue) | Medium | Step 1 |
| 3 | `staging/ready_worker.py` (base class) | Small | Step 1 |
| 4 | `staging/workers/det_worker.py` | Small | Step 3 |
| 5 | `staging/workers/fw_worker.py` | Medium | Step 3 |
| 6 | `staging/workers/loc_worker.py` | Medium | Step 3 |
| 7 | `staging/ready_judge.py` | Medium | Step 2 |
| 8 | `staging/ready_pool.py` | Large | Steps 3-7 |
| 9 | `staging/entrypoint.py` | Medium | Steps 2, 7, 8 |
| 10 | `Dockerfile.staging` | Small | Step 9 |
| 11 | Tests (unit + integration + container) | Medium | Steps 1-10 |

**Total estimated effort:** ~3-4 focused sessions (depending on familiarity with the codebase).

---

*End of plan document.*
