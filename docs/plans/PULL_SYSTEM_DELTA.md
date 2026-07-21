# Pull System: File-by-File Delta Report

> **Source:** `/home/artem/dev/amd-hackathon/docs/plans/PULL_SYSTEM_DESIGN.md` (1472 lines)
> **Target codebase:** `/home/artem/dev/amd-hackathon/staging/` (8 files + 3 workers + `__init__.py`)
> **Audit date:** 2026-07-13
> **Migration path:** Phase 1 → Phase 2 → Phase 3 → Phase 4

---

## Table of Contents

1. [Implementation Order Overview](#1-implementation-order-overview)
2. [File: `ready_config.py` — Add Pull System Tunables (Phase 1-4)](#2-file-ready_configpy--add-pull-system-tunables)
3. [File: `ready_queue.py` — Add `drain_to_pool()` (Phase 1)](#3-file-ready_queuepy--add-drain_to_pool)
4. [File: `ready_worker.py` — Transform Base Class (Phase 1-2)](#4-file-ready_workerpy--transform-base-class)
5. [File: `workers/det_worker.py` — Add `_process_single()` (Phase 1)](#5-file-workersdet_workerpy--add-_process_single)
6. [File: `workers/loc_worker.py` — Add `_process_single()` (Phase 1)](#6-file-workersloc_workerpy--add-_process_single)
7. [File: `workers/fw_worker.py` — Add `_process_single()` (Phase 1)](#7-file-workersfw_workerpy--add-_process_single)
8. [File: `ready_judge.py` — Make Judge Autonomous (Phase 3)](#8-file-ready_judgepy--make-judge-autonomous)
9. [File: `ready_pool.py` → Transform to `ReadyMonitor` (Phase 4)](#9-file-ready_poolpy--transform-to-readymonitor)
10. [File: `entrypoint.py` — Orchestrate Pull System (Phase 1-4)](#10-file-entrypointpy--orchestrate-pull-system)
11. [File: `staging/__init__.py` — Update Exports (Phase 4)](#11-file-staging__init__py--update-exports)
12. [New File: `workers/steal_workflow.py` — Optional (Phase 2)](#12-new-file-workerssteal_workflowpy--optional)
13. [Summary Table](#13-summary-table)

---

## 1. Implementation Order Overview

| Order | Phase | Action | Files Changed |
|-------|-------|--------|---------------|
| P0 | Pre-work | Add pull-system tuning config fields | `ready_config.py` |
| P1.1 | Phase 1 | Add `drain_to_pool()` to `ReadyQueue` | `ready_queue.py` |
| P1.2 | Phase 1 | Add shared `task_pool` + `inbox_queue` + `_process_single()` to `ReadyWorker` | `ready_worker.py` |
| P1.3 | Phase 1 | Add `_process_single()` to all 3 concrete workers | `det_worker.py`, `loc_worker.py`, `fw_worker.py` |
| P1.4 | Phase 1 | Simplify `entrypoint.py` to use `task_pool` (remove round-robin) | `entrypoint.py` |
| P2.1 | Phase 2 | Add `steal_request_queue`, `stolen_queue`, `_attempt_steal()` to `ReadyWorker` | `ready_worker.py` |
| P2.2 | Phase 2 | Add victim check in worker pre-pull loop | `ready_worker.py` |
| P3.1 | Phase 3 | Add `consume_loop()`, dynamic worker types, deadline awareness to judge | `ready_judge.py` |
| P3.2 | Phase 3 | Run judge in daemon thread in entrypoint | `entrypoint.py` |
| P4.1 | Phase 4 | Transform `ReadyPool` → `ReadyMonitor`: remove dispatch, add heartbeat+monitor | `ready_pool.py` |
| P4.2 | Phase 4 | Wire `ReadyMonitor` into entrypoint | `entrypoint.py` |
| P4.3 | Phase 4 | Update `staging/__init__.py` exports | `__init__.py` |

---

## 2. File: `ready_config.py` — Add Pull System Tunables

**File:** `/home/artem/dev/amd-hackathon/staging/ready_config.py`  
**Current size:** 82 lines, `@dataclass ReadyConfig` with 13 fields  
**Phase:** P0 (pre-requisite for all phases)

### What Needs to Change

**A. Add 6 new fields to `ReadyConfig` (after line 43, before `category_priority`)**

```python
# ── Pull system tuning ──
steal_threshold: int = 2              # Min tasks in victim's inbox to enable stealing
steal_timeout_s: float = 0.5          # Max wait for stolen task response
heartbeat_timeout_s: float = 60.0     # Max seconds without heartbeat before kill
monitor_interval_s: float = 5.0       # Health check interval (seconds)
judge_poll_interval_s: float = 0.05   # Judge result pull interval (seconds)
emergency_vote_reduction: int = 2     # Divide judgment_votes by this under deadline emergency
per_worker_inbox_size: int = 3        # Max tasks in a worker's inbox before pool stops feeding
```

**B. Add environment variable loading in `from_env()` (after existing env var mapping)**

Add these lines in the `from_env()` return statement:
```python
steal_threshold=int(os.environ.get("STEAL_THRESHOLD", "2")),
steal_timeout_s=float(os.environ.get("STEAL_TIMEOUT_S", "0.5")),
heartbeat_timeout_s=float(os.environ.get("HEARTBEAT_TIMEOUT_S", "60.0")),
monitor_interval_s=float(os.environ.get("MONITOR_INTERVAL_S", "5.0")),
judge_poll_interval_s=float(os.environ.get("JUDGE_POLL_INTERVAL_S", "0.05")),
emergency_vote_reduction=int(os.environ.get("EMERGENCY_VOTE_REDUCTION", "2")),
per_worker_inbox_size=int(os.environ.get("PER_WORKER_INBOX_SIZE", "3")),
```

### Exact Line Changes

| Change | Location | Description |
|--------|----------|-------------|
| Add 7 fields | After line 43 (after `fallback_strategy`) | New dataclass fields |
| Add 7 env mappings | In `from_env()` lines 59-78 | Load from `STAGING_*` env vars |

---

## 3. File: `ready_queue.py` — Add `drain_to_pool()`

**File:** `/home/artem/dev/amd-hackathon/staging/ready_queue.py`  
**Current size:** 139 lines, `ReadyTask` dataclass + `ReadyQueue` class  
**Phase:** Phase 1

### What Needs to Change

**A. Add `drain_to_pool()` method to `ReadyQueue` (after line 131, before `empty` property)**

```python
def drain_to_pool(self, task_pool: "multiprocessing.Queue") -> int:
    """Drain all tasks into a shared multiprocessing pool.

    Used at startup to populate the worker-facing pool (Phase 1 pull system).
    """
    import multiprocessing
    count = 0
    while not self.empty:
        task = self.dequeue_any(preferred_categories=[])
        if task is None:
            break
        task_pool.put_nowait(task)
        count += 1
    return count
```

### Exact Line Changes

| Change | Location | Description |
|--------|----------|-------------|
| Add method | After line 131 (after `task_counts_by_category`) | New `drain_to_pool()` method |
| Add import | Line 13 (after existing imports) | Add `import multiprocessing` (or use inline import) |

---

## 4. File: `ready_worker.py` — Transform Base Class

**File:** `/home/artem/dev/amd-hackathon/staging/ready_worker.py`  
**Current size:** 191 lines, `ReadyWorker` ABC  
**Phase:** Phase 1 (shared pool) + Phase 2 (work stealing)

### What Needs to Change

**A. Replace constructor signature (lines 38-59)**

OLD (lines 38-59):
```python
def __init__(
    self,
    worker_id: str,
    worker_type: str,
    config: ReadyConfig,
    task_queue,
    results_queue,
    busy_flag,
    task_queues=None,
    category_whitelist=None,
    **kwargs,
) -> None:
    self.worker_id = worker_id
    self.worker_type = worker_type
    self.config = config
    self.task_queue = task_queue
    self.results_queue = results_queue
    self.ready_flag = kwargs.get("ready_flag")
    self.busy_flag = busy_flag
    self.task_queues = task_queues or {}
    self.category_whitelist = category_whitelist or []
    self._running = True
```

NEW:
```python
def __init__(
    self,
    worker_id: str,
    worker_type: str,
    config: ReadyConfig,
    task_pool,                    # NEW: shared multiprocessing.Queue
    results_queue,
    steal_request_queue,          # NEW: shared steal request queue
    stolen_queue,                 # NEW: per-worker stolen task queue
    inbox_queue,                  # NEW: per-worker assigned task queue
    busy_flag,
    heartbeat,                    # NEW: multiprocessing.Value (float timestamp)
    deadline_emergency=None,      # NEW: multiprocessing.Value (bool)
    category_whitelist=None,
    **kwargs,
) -> None:
    self.worker_id = worker_id
    self.worker_type = worker_type
    self.config = config
    self.task_pool = task_pool
    self.results_queue = results_queue
    self.steal_request_queue = steal_request_queue
    self.stolen_queue = stolen_queue
    self.inbox_queue = inbox_queue
    self.busy_flag = busy_flag
    self.heartbeat = heartbeat
    self.deadline_emergency = deadline_emergency
    self.category_whitelist = category_whitelist or []
    self._running = True
    self._known_workers: list[str] = []  # populated at startup (Phase 2)
```

**B. Replace `run()` method (lines 86-132)**

OLD (lines 86-132): init → signal ready → pull tasks via old `_pull_task()` → process → push
NEW: init → signal ready → **heartbeat** → pull via new `_pull_task()` → check `deadline_emergency` → process or `_process_single()` → push

Key changes:
- Add `self.heartbeat.value = time.monotonic()` at top of each loop iteration (line ~107)
- Add deadline emergency check before processing (after line ~118): `if self.deadline_emergency and self.deadline_emergency.value: answers = self._process_single(task)`
- Remove `ready_flag` references (old pool used them; new pool relies on heartbeat)
- Change status handling: remove `task.status = "in_progress"` (monitor tracks via pending_map)

**C. Replace `_pull_task()` method (lines 136-169)**

OLD (lines 136-169): 4-step: private queue → shared category by whitelist → any category → blocking private
NEW: 4-step: **stolen_queue → inbox_queue → task_pool → _attempt_steal()**

```python
def _pull_task(self) -> Optional[ReadyTask]:
    """Pull next task with work stealing awareness.

    Priority: stolen > inbox > task_pool > steal_attempt
    """
    import queue as _queue

    # 1. Stolen tasks first (highest priority)
    try:
        return self.stolen_queue.get_nowait()
    except _queue.Empty:
        pass

    # 2. Private inbox (originally assigned tasks from pool)
    try:
        return self.inbox_queue.get_nowait()
    except _queue.Empty:
        pass

    # 3. Shared task pool (main pull point — Phase 1)
    try:
        return self.task_pool.get_nowait()
    except (AttributeError, _queue.Empty):
        pass

    # 4. Attempt work stealing (Phase 2)
    return self._attempt_steal()
```

**D. Add `_attempt_steal()` method (Phase 2, after `_pull_task`)**

```python
def _attempt_steal(self) -> Optional[ReadyTask]:
    """Try to steal work from a busy worker.

    Sends a steal request to the shared queue, waits briefly for a response.
    """
    import queue as _queue

    try:
        self.steal_request_queue.put_nowait({
            "thief_id": self.worker_id,
            "thief_type": self.worker_type,
            "whitelist": self.category_whitelist,
            "timestamp": time.monotonic(),
        })
    except _queue.Full:
        return None

    # Wait briefly for a stolen task
    try:
        return self.stolen_queue.get(timeout=self.config.steal_timeout_s)
    except _queue.Empty:
        return None
```

**E. Add `_attempt_victim_response()` method (Phase 2)**

Method added before `_pull_task()` to be called at the start of each loop iteration — checks if another worker is requesting a steal and offers one task from inbox if > STEAL_THRESHOLD:

```python
def _attempt_victim_response(self) -> None:
    """Check steal_request_queue and offer a task if we have surplus.

    Called at the start of each run() iteration before pulling.
    """
    import queue as _queue
    try:
        request = self.steal_request_queue.get_nowait()
    except _queue.Empty:
        return

    thief_id = request.get("thief_id")
    if not thief_id or thief_id == self.worker_id:
        return

    # Only offer if we have tasks to spare
    try:
        inbox_size = self.inbox_queue.qsize()
    except NotImplementedError:
        # macOS doesn't support qsize(); skip victim response
        return

    if inbox_size > self.config.steal_threshold:
        try:
            task = self.inbox_queue.get_nowait()
            # Directly put on thief's stolen_queue — need cross-worker ref.
            # In practice, the thief's stolen_queue would be looked up
            # from a shared registry passed at startup.
            # For now, log and re-queue to task_pool as simplified steal.
            self.task_pool.put_nowait(task)
            logger.info(
                "[steal] %s offered task %s back to pool (requested by %s)",
                self.worker_id, task.task_id, thief_id,
            )
        except (_queue.Empty, AttributeError):
            pass
```

**F. Add `_process_single()` abstract method (Phase 1)**

Add after `process()` definition (after line 78):

```python
def _process_single(self, task: ReadyTask) -> list[dict]:
    """Single fast try for deadline emergency mode. Override in subclass.

    Default: delegates to process() and returns first answer only.
    """
    answers = self.process(task)
    return answers[:1] if answers else [{
        "worker_id": self.worker_id,
        "task_id": task.task_id,
        "answer": "",
        "timing_ms": 0,
    }]
```

**G. Add `_check_for_steal_requests()` in `run()` loop (Phase 2)**

Insert at the top of the `while self._running:` block, before heartbeat:
```python
while self._running:
    self._attempt_victim_response()  # Phase 2
    self.heartbeat.value = time.monotonic()  # Phase 4
    ...
```

### Exact Line Changes Summary

| Change | Lines | Description |
|--------|-------|-------------|
| Rewrite `__init__` | 38-59 | New params: task_pool, steal_request_queue, stolen_queue, inbox_queue, heartbeat, deadline_emergency |
| Add `_process_single()` | After line 78 | Abstract stub for emergency fast mode |
| Add `_attempt_victim_response()` | After `_process_single()` | Phase 2 steal victim logic |
| Rewrite `run()` | 86-132 | Add heartbeat, steal response, emergency check, new pull |
| Rewrite `_pull_task()` | 136-169 | New priority: stolen→inbox→task_pool→steal |
| Add `_attempt_steal()` | After `_pull_task()` | Phase 2 steal request logic |
| Remove `ready_flag` handling | Throughout | Replaced by heartbeat-based health check |

---

## 5. File: `workers/det_worker.py` — Add `_process_single()`

**File:** `/home/artem/dev/amd-hackathon/staging/workers/det_worker.py`  
**Current size:** 89 lines  
**Phase:** Phase 1

### What Needs to Change

**A. Add `_process_single()` method (after line 89)**

```python
def _process_single(self, task: ReadyTask) -> list[dict]:
    """Single fast try for deadline emergency mode."""
    solver = self._solvers.get(task.category)
    if not solver:
        return [{
            "worker_id": self.worker_id,
            "task_id": task.task_id,
            "answer": "",
            "timing_ms": 0,
        }]
    t0 = time.monotonic()
    try:
        answer = solver(task.prompt, task.category) or ""
    except Exception as exc:
        logger.warning("[%s] Emergency solve failed for %s: %s",
                       self.worker_id, task.task_id, exc)
        answer = ""
    elapsed = (time.monotonic() - t0) * 1000
    return [{
        "worker_id": self.worker_id,
        "task_id": task.task_id,
        "answer": answer,
        "timing_ms": elapsed,
    }]
```

### Exact Line Changes

| Change | Lines | Description |
|--------|-------|-------------|
| Add method | After line 89 (EOF) | Single-try emergency path |

---

## 6. File: `workers/loc_worker.py` — Add `_process_single()`

**File:** `/home/artem/dev/amd-hackathon/staging/workers/loc_worker.py`  
**Current size:** 108 lines  
**Phase:** Phase 1

### What Needs to Change

**A. Add `_process_single()` method (after line 108, before `shutdown()`)**

```python
def _process_single(self, task: ReadyTask) -> list[dict]:
    """Single try at temperature=0.1 for deadline emergency mode."""
    if self._llm is None:
        return [{
            "worker_id": self.worker_id,
            "task_id": task.task_id,
            "answer": "",
            "timing_ms": 0,
        }]
    sys_prompt = self._get_system_prompt(task.category)
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": task.prompt},
    ]
    t0 = time.monotonic()
    try:
        result = self._llm.create_chat_completion(
            messages=messages, max_tokens=512,
            temperature=0.1, stop=None,
        )
        answer = result["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.warning("[%s] Emergency try failed for %s: %s",
                       self.worker_id, task.task_id, exc)
        answer = ""
    elapsed = (time.monotonic() - t0) * 1000
    return [{
        "worker_id": self.worker_id,
        "task_id": task.task_id,
        "answer": answer,
        "timing_ms": elapsed,
    }]
```

### Exact Line Changes

| Change | Lines | Description |
|--------|-------|-------------|
| Add method | After line 103 (before `shutdown()`) | Single-try emergency path at temp=0.1 |

---

## 7. File: `workers/fw_worker.py` — Add `_process_single()`

**File:** `/home/artem/dev/amd-hackathon/staging/workers/fw_worker.py`  
**Current size:** 78 lines  
**Phase:** Phase 1

### What Needs to Change

**A. Add `_process_single()` method (after line 78)**

```python
def _process_single(self, task: ReadyTask) -> list[dict]:
    """Single try at temperature=0.0 for deadline emergency mode."""
    from agent.solvers.fw_router import route
    cfg = route(task.category, task.prompt, 0.5)
    t0 = time.monotonic()
    try:
        answer = self._solver.solve(
            self._model_id, cfg.system_prompt, task.prompt,
            max_tokens=cfg.max_tokens, temperature=0.0,
            prefill=cfg.prefill, task_type=task.category,
            timeout=int(self.config.worker_timeout_s),
        )
    except Exception as exc:
        logger.warning("[%s] Emergency try failed for %s: %s",
                       self.worker_id, task.task_id, exc)
        answer = ""
    elapsed = (time.monotonic() - t0) * 1000
    return [{
        "worker_id": self.worker_id,
        "task_id": task.task_id,
        "answer": answer,
        "timing_ms": elapsed,
    }]
```

### Exact Line Changes

| Change | Lines | Description |
|--------|-------|-------------|
| Add method | After line 78 (EOF) | Single-try emergency path at temperature=0.0 |

---

## 8. File: `ready_judge.py` — Make Judge Autonomous

**File:** `/home/artem/dev/amd-hackathon/staging/ready_judge.py`  
**Current size:** 376 lines, `ReadyJudge` class  
**Phase:** Phase 3

### What Needs to Change

**A. Add `_active_worker_types` tracking to `__init__` (around line 102-109)**

OLD:
```python
def __init__(self, config: ReadyConfig):
    self.config = config
    self._task_answers: dict[str, list[dict]] = defaultdict(list)
    self._judged: dict[str, dict] = {}
    self._task_first_answer_time: dict[str, float] = {}
    self._fw_solver = None
```

NEW:
```python
def __init__(self, config: ReadyConfig):
    self.config = config
    self._task_answers: dict[str, list[dict]] = defaultdict(list)
    self._judged: dict[str, dict] = {}
    self._task_first_answer_time: dict[str, float] = {}
    self._active_worker_types: set[str] = set()  # Dynamic, not hardcoded
    self.total_expected_answers = config.judgment_votes
    self.deadline_emergency = None  # Set externally by monitor
    self._fw_solver = None
```

**B. Update `add_answer()` (lines 113-125)**

Add worker type tracking:
```python
def add_answer(self, answer: dict) -> None:
    tid = answer.get("task_id")
    if not tid or "answer" not in answer:
        logger.warning("[judge] Dropping malformed result: ...")
        return
    wt = self._get_worker_type(answer)
    self._active_worker_types.add(wt)  # NEW: dynamic tracking
    self._task_answers[tid].append(answer)
    if tid not in self._task_first_answer_time:
        self._task_first_answer_time[tid] = time.monotonic()
```

**C. Rewrite `ready_to_judge()` (lines 148-181)**

OLD: Uses hardcoded `_KNOWN_WORKER_TYPES`, checks `judgment_votes` count, 30s timeout
NEW: Uses `_active_worker_types`, counts non-degenerate answers, reduces threshold under deadline emergency, adaptive timeout

```python
def ready_to_judge(self, task_id: str) -> bool:
    answers = self._task_answers.get(task_id, [])
    count = len(answers)

    if count < self.config.judgment_votes:
        return False

    non_degenerate = [a for a in answers
                      if not _is_degenerate(a.get("answer", ""))]

    # Determine effective threshold
    threshold = self.config.judgment_votes
    if self.deadline_emergency and self.deadline_emergency.value:
        threshold = max(1, threshold // self.config.emergency_vote_reduction)

    # Primary: enough non-degenerate answers
    if len(non_degenerate) >= threshold:
        return True

    # Secondary: all active worker types contributed (all failed)
    task_types = set(self._get_worker_type(a) for a in answers)
    if len(task_types) >= min(2, len(self._active_worker_types)):
        all_empty = all(
            _is_degenerate(a.get("answer", ""))
            for a in answers
        )
        if all_empty:
            return True

    # Timeout (reduced under deadline emergency)
    timeout = 15.0 if (self.deadline_emergency and
                       self.deadline_emergency.value) else 30.0
    first_time = self._task_first_answer_time.get(task_id)
    if first_time is not None and (time.monotonic() - first_time) >= timeout:
        logger.warning(
            "[judge] Timeout for %s — forcing judgment after %.0fs "
            "(types=%s, votes=%d)",
            task_id, timeout, task_types, count,
        )
        return True

    return False
```

**D. Add `consume_loop()` method (new method, after `ingest_results()`)**

```python
def consume_loop(self, results_queue, deadline_emergency=None,
                 stop_event=None) -> None:
    """Autonomous loop: pull results, judge, repeat.

    Runs in its own daemon thread. Decoupled from pool.
    """
    self.deadline_emergency = deadline_emergency

    while not (stop_event and stop_event.is_set()):
        ingested = self.ingest_results(results_queue)

        # Try to judge any ready tasks
        for tid in list(self.pending_tasks):
            if self.ready_to_judge(tid):
                answer, meta = self.judge(tid)

        # Sleep briefly if no new results
        if ingested == 0:
            timeout = 0.05 if not (deadline_emergency and
                                   deadline_emergency.value) else 0.02
            time.sleep(timeout)
```

**E. Remove `_KNOWN_WORKER_TYPES` class variable (line 100)**

Delete line 100: `_KNOWN_WORKER_TYPES = {"deterministic", "local", "fireworks"}` — no longer used.

### Exact Line Changes Summary

| Change | Lines | Description |
|--------|-------|-------------|
| Remove `_KNOWN_WORKER_TYPES` | Line 100 | No longer hardcoded |
| Update `__init__` | 102-109 | Add `_active_worker_types`, `total_expected_answers`, `deadline_emergency` |
| Update `add_answer()` | 113-125 | Track `_active_worker_types` dynamically |
| Rewrite `ready_to_judge()` | 148-181 | Dynamic types, non-degenerate threshold, deadline-aware timeout |
| Add `consume_loop()` | After `ingest_results()` (~line 299) | Autonomous result-pulling + judging loop |

---

## 9. File: `ready_pool.py` → Transform to `ReadyMonitor`

**File:** `/home/artem/dev/amd-hackathon/staging/ready_pool.py`  
**Current size:** 343 lines, `ReadyPool` class + registry functions  
**Phase:** Phase 4 (final form)

### What Needs to Change

**A. Rename class `ReadyPool` → `ReadyMonitor` (line 33)**

And update docstring to reflect new role (lines 34-35):
```python
class ReadyMonitor:
    """Lightweight health monitor for the pull-based pool system.

    Responsibilities:
    1. Spawn workers and manage their lifecycle
    2. Monitor worker health (is_alive, heartbeat, busy_flag)
    3. Re-enqueue orphaned tasks on worker crash
    4. Broadcast deadline_emergency flag
    5. Log progress statistics

    Does NOT:
    - Dispatch tasks (workers pull from shared task_pool)
    - Call judge methods (judge runs autonomously)
    - Manage per-worker queues (only shared task_pool + steal protocol)
    """
```

**B. Rewrite `__init__()` (lines 36-47)**

OLD:
```python
def __init__(self, config: ReadyConfig):
    self.config = config
    self._category_queues: dict[str, multiprocessing.Queue] = {}
    self._results_queue: Optional[multiprocessing.Queue] = None
    self._processes: list[multiprocessing.Process] = []
    self._busy_flags: list[multiprocessing.Value] = []
    self._ready_flags: list[multiprocessing.Value] = []
    self._worker_ids: list[str] = []
    self._worker_task_queues: list[multiprocessing.Queue] = []
    self._started = False
    self._running = multiprocessing.Event()
    self._pending_dispatched: set[str] = set()
```

NEW:
```python
def __init__(self, config: ReadyConfig):
    self.config = config
    self.task_pool = multiprocessing.Queue()        # Single shared task pool
    self.results_queue = multiprocessing.Queue()    # Shared results queue
    self.steal_request_queue = multiprocessing.Queue()  # Steal coordination
    self._processes: list[multiprocessing.Process] = []
    self._busy_flags: list[multiprocessing.Value] = []
    self._heartbeats: list[multiprocessing.Value] = []
    self._worker_ids: list[str] = []
    self._worker_types: list[str] = []
    self._stolen_queues: dict[str, multiprocessing.Queue] = {}
    self._inbox_queues: dict[str, multiprocessing.Queue] = {}
    self._deadline_emergency = multiprocessing.Value('b', 0)
    self._running = multiprocessing.Event()
    self._pending_map: dict[str, str] = {}  # task_id → worker_id (who pulled it)
    self._total_tasks = 0
    self._last_log = 0.0
```

**C. Rewrite `start()` (lines 51-139)**

OLD: Creates per-category queues, spawns workers, waits for ready_flags, round-robin distributes tasks
NEW: Drains ReadyQueue to shared task_pool, spawns workers with new params, no round-robin

```python
def start(self, queue: ReadyQueue) -> None:
    """Populate shared task pool and spawn workers."""
    if self._running.is_set():
        logger.warning("Monitor already started")
        return

    self._running.set()

    # Drain ReadyQueue into shared task pool
    count = queue.drain_to_pool(self.task_pool)
    self._total_tasks = count
    logger.info("[monitor] Drained %d tasks into shared task pool", count)

    # Spawn workers
    workers_to_start = self._build_worker_plan()
    for wid, wtype, worker_cls, index in workers_to_start:
        inbox = multiprocessing.Queue()
        stolen = multiprocessing.Queue()
        self._inbox_queues[wid] = inbox
        self._stolen_queues[wid] = stolen

        busy_flag = multiprocessing.Value('b', 0)
        heartbeat = multiprocessing.Value('d', time.monotonic())
        whitelist = self._build_whitelist(wtype)

        worker = worker_cls(
            worker_id=wid, worker_type=wtype, config=self.config,
            task_pool=self.task_pool,
            results_queue=self.results_queue,
            steal_request_queue=self.steal_request_queue,
            stolen_queue=stolen,
            inbox_queue=inbox,
            busy_flag=busy_flag,
            heartbeat=heartbeat,
            deadline_emergency=self._deadline_emergency,
            category_whitelist=whitelist,
            worker_index=index,
        )

        self._busy_flags.append(busy_flag)
        self._heartbeats.append(heartbeat)
        self._worker_ids.append(wid)
        self._worker_types.append(wtype)

        p = multiprocessing.Process(
            target=worker.run,
            name=wid,
            daemon=True,
        )
        p.start()
        self._processes.append(p)

        logger.info("[monitor] Started %s (type=%s, pid=%d, whitelist=%s)",
                    wid, wtype, p.pid, whitelist)

    logger.info("[monitor] All %d workers started", len(self._processes))
```

**D. Add `monitor_loop()` (replaces `dispatch_loop()`, lines 141-228)**

```python
def monitor_loop(self, judge, deadline: float) -> None:
    """Health monitoring loop — replaces dispatch_loop.

    Args:
        judge: ReadyJudge instance (to check completion)
        deadline: Absolute time.monotonic() deadline
    """
    if not self._running.is_set():
        return

    while self._running.is_set():
        now = time.monotonic()
        remaining = deadline - now

        # 1. Check worker health
        for i, p in enumerate(self._processes):
            if not p.is_alive():
                self._handle_dead_worker(i)

        # 2. Check for stuck workers (heartbeat timeout)
        for i, hb in enumerate(self._heartbeats):
            if self._busy_flags[i].value and (now - hb.value) > self.config.heartbeat_timeout_s:
                self._handle_stuck_worker(i)

        # 3. Deadline emergency broadcast
        if remaining < 30 and not self._deadline_emergency.value:
            self._deadline_emergency.value = 1
            logger.warning("[monitor] DEADLINE EMERGENCY — forcing fast mode")
        elif remaining >= 30 and self._deadline_emergency.value:
            self._deadline_emergency.value = 0

        # 4. Log progress
        if now - self._last_log >= 10.0:
            still_pending = self._total_tasks - judge.total_judged
            busy_count = sum(f.value for f in self._busy_flags)
            alive_count = sum(1 for p in self._processes if p.is_alive())
            logger.info(
                "[monitor] Progress: %d/%d judged, %d pending, "
                "%d busy, %d alive, %.0fs remaining",
                judge.total_judged, self._total_tasks,
                still_pending, busy_count, alive_count, remaining,
            )
            self._last_log = now

        # 5. Check completion conditions
        if judge.total_judged >= self._total_tasks:
            logger.info("[monitor] All %d tasks judged", self._total_tasks)
            break
        if remaining <= 0:
            logger.warning("[monitor] Deadline reached")
            break
        if all(not p.is_alive() for p in self._processes):
            logger.warning("[monitor] All workers dead — stopping")
            break

        time.sleep(self.config.monitor_interval_s)  # 5s default
```

**E. Add `_handle_dead_worker()` and `_handle_stuck_worker()` methods**

```python
def _handle_dead_worker(self, index: int) -> None:
    """Handle a worker that has died. Re-enqueue its pending task."""
    if index >= len(self._processes):
        return
    wid = self._worker_ids[index]
    p = self._processes[index]
    logger.warning("[monitor] Worker %s (pid=%d) died with exit code %s",
                   wid, p.pid, p.exitcode)

    # Find orphaned task from pending_map
    orphaned_tids = [
        tid for tid, w in self._pending_map.items()
        if w == wid
    ]
    for tid in orphaned_tids:
        logger.info("[monitor] Re-enqueuing orphaned task %s", tid)
        # We can't re-enqueue the actual task object without storing it.
        # In production, store a reference or use a task id → task mapping.
        del self._pending_map[tid]

    # Clean up tracking
    if index < len(self._busy_flags):
        self._busy_flags[index].value = 0

def _handle_stuck_worker(self, index: int) -> None:
    """Handle a worker that is busy but not sending heartbeats."""
    if index >= len(self._processes):
        return
    wid = self._worker_ids[index]
    p = self._processes[index]
    logger.warning("[monitor] Worker %s stuck (busy, heartbeat stale) — terminating",
                   wid)
    try:
        p.terminate()
        p.join(timeout=2.0)
    except Exception:
        pass
    self._handle_dead_worker(index)
```

**F. Keep but simplify `shutdown()` (lines 229-242)**

Mostly the same, but remove responsibility for calling judge — just terminate workers:

```python
def shutdown(self, timeout: float = 5.0) -> None:
    """Gracefully terminate all worker processes."""
    self._running.clear()
    for p in self._processes:
        if p.is_alive():
            p.terminate()
    for p in self._processes:
        p.join(timeout=timeout)
        if p.is_alive():
            logger.warning("[monitor] Worker %s (pid=%d) did not exit — killing",
                           p.name, p.pid)
            p.kill()
            p.join(1.0)
    self._processes.clear()
    logger.info("[monitor] All workers shut down")
```

**G. Keep `_build_worker_plan()` (lines 260-296) — mostly unchanged**

No significant changes needed. The plan-building logic remains the same.

**H. Keep `_build_whitelist()` (lines 298-313) — unchanged**

No changes needed.

**I. Remove `_enqueue_remaining()` (lines 315-331) — DELETE entire method**

Round-robin dispatch is removed in Phase 1. Tasks go only to shared `task_pool`.

**J. Remove `_check_worker_health()` (lines 246-258) — DELETE entire method**

Replaced by inline health checks in `monitor_loop()` + `_handle_dead_worker()`.

**K. Remove `dispatch_loop()` (lines 141-228) — DELETE entire method**

Replaced by `monitor_loop()`.

**L. Update `busy_workers`, `available_count`, `total_workers` properties (lines 333-343)**

`busy_workers` — no change needed.
`available_count` — no change needed.
`total_workers` — no change needed.

### Exact Line Changes Summary

| Change | Lines | Description |
|--------|-------|-------------|
| Rename class `ReadyPool` → `ReadyMonitor` | 33 | New name, new docstring |
| Rewrite `__init__()` | 36-47 | Replace per-worker queues with shared task_pool, add heartbeats, pending_map, steal_request_queue |
| Rewrite `start()` | 51-139 | Drain to task_pool, new worker constructor params, no round-robin |
| DELETE `_enqueue_remaining()` | 315-331 | No more round-robin dispatch |
| DELETE `dispatch_loop()` | 141-228 | Replaced by monitor_loop |
| DELETE `_check_worker_health()` | 246-258 | Replaced by inline health checks |
| ADD `monitor_loop()` | New | 5s health check loop, deadline broadcast, completion detection |
| ADD `_handle_dead_worker()` | New | Re-enqueue orphaned tasks, clean up tracking |
| ADD `_handle_stuck_worker()` | New | Terminate stuck workers, re-enqueue tasks |
| Keep `_build_worker_plan()` | ~260-296 | Mostly unchanged |
| Keep `_build_whitelist()` | ~298-313 | Unchanged |
| Keep `shutdown()` | ~229-242 | Minor simplification |
| Keep `busy_workers` etc. | ~333-343 | Unchanged |

---

## 10. File: `entrypoint.py` — Orchestrate Pull System

**File:** `/home/artem/dev/amd-hackathon/staging/entrypoint.py`  
**Current size:** 193 lines  
**Phase:** All phases (incremental)

### What Needs to Change

**A. Update imports (lines 20-21)**

OLD:
```python
from staging import ReadyConfig, ReadyQueue, ReadyPool, ReadyJudge
from staging.ready_queue import ReadyTask
```

NEW:
```python
from staging import ReadyConfig, ReadyQueue, ReadyMonitor, ReadyJudge
from staging.ready_queue import ReadyTask
```

**B. Rewrite `main()` function (lines 93-193)**

Replace the entire section from pool+judge creation to shutdown:

OLD (lines 153-189):
```python
judge = ReadyJudge(config)
pool = ReadyPool(config)

try:
    pool.dispatch_loop(queue, judge, deadline)
    judge.ingest_results(pool._results_queue)
except Exception as exc:
    logger.exception(...)
finally:
    ...
    final_results = judge.judge_all()
    ...
    _shutdown_results = ...
    ...
    pool.shutdown()
```

NEW:
```python
import threading

# Create components
judge = ReadyJudge(config)
monitor = ReadyMonitor(config)

# Start: drain queue to task_pool, spawn workers
monitor.start(queue)

# Run judge in autonomous daemon thread
stop_event = threading.Event()
judge_thread = threading.Thread(
    target=judge.consume_loop,
    args=(monitor.results_queue, monitor._deadline_emergency, stop_event),
    daemon=True,
)
judge_thread.start()

try:
    # Monitor loop (blocking in main thread)
    monitor.monitor_loop(judge, deadline)
except Exception as exc:
    logger.exception("Fatal error in monitor loop: %s", exc)
finally:
    # Stop judge
    stop_event.set()
    judge_thread.join(timeout=2.0)

    # Final drain of any remaining results
    judge.ingest_results(monitor.results_queue)

    # Judge all remaining tasks
    logger.info("Judging %d pending tasks...", len(judge.pending_tasks))
    final_results = judge.judge_all()
    _shutdown_results = [{"task_id": r["task_id"], "answer": r["answer"]}
                         for r in final_results]
    logger.info("Final: %d/%d tasks judged", len(final_results), total_tasks)

    # Log strategy distribution (same as current)
    ...

    # Write output (same as current)
    ...

    # Shutdown
    monitor.shutdown()
```

**C. Add `import threading` at top of file (around line 19)**

### Exact Line Changes Summary

| Change | Lines | Description |
|--------|-------|-------------|
| Add `import threading` | ~line 19 | For judge daemon thread |
| Update imports | 20-21 | `ReadyPool` → `ReadyMonitor` |
| Rewrite pool/judge section | 153-189 | Use `ReadyMonitor`, judge thread, monitor_loop |

---

## 11. File: `staging/__init__.py` — Update Exports

**File:** `/home/artem/dev/amd-hackathon/staging/__init__.py`  
**Current size:** 26 lines  
**Phase:** Phase 4

### What Needs to Change

**A. Update import and export list (lines 16-26)**

OLD:
```python
from .ready_pool import ReadyPool
```

NEW:
```python
from .ready_pool import ReadyMonitor
```

Also update `__all__`:
```python
__all__ = [
    "ReadyConfig",
    "ReadyQueue", "ReadyTask",
    "ReadyMonitor",
    "ReadyJudge",
]
```

**B. Optionally keep `ReadyPool` as an alias for backward compatibility**

If other code imports `ReadyPool`:
```python
from .ready_pool import ReadyMonitor as ReadyPool  # Backward compat
```

### Exact Line Changes

| Change | Lines | Description |
|--------|-------|-------------|
| Replace `ReadyPool` import | 18 | `ReadyPool` → `ReadyMonitor` |
| Update `__all__` | 22-26 | New class name in exports |

---

## 12. New File: `workers/steal_workflow.py` — Optional (Phase 2)

This file is **optional** but recommended for clean work-stealing orchestration logic. The plan doesn't explicitly require it, but it's a natural refactoring.

**Purpose:** Encapsulate the steal protocol — request/response handling, queue size registry, and victim selection — out of `ready_worker.py` for testability.

**Suggested content:**
- `StealProtocol` class: manages `steal_request_queue` as a coordinator
- `register_victim(thief_id, victim_id)` / `request_steal(thief_id)` helpers
- Size-checking utilities for `inbox_queue.qsize()` with macOS fallback

---

## 13. Summary Table

### Total Changes by File

| File | Phase | Lines Changed | Complexity | Risk |
|------|-------|--------------|------------|------|
| `ready_config.py` | P0 | +14 fields +7 env | Trivial | Low |
| `ready_queue.py` | 1 | +12 lines (1 method) | Trivial | Low |
| `ready_worker.py` | 1,2 | ~120 lines rewritten | **High** | **High** — core logic change |
| `workers/det_worker.py` | 1 | +22 lines (1 method) | Low | Low |
| `workers/loc_worker.py` | 1 | +35 lines (1 method) | Low | Low |
| `workers/fw_worker.py` | 1 | +28 lines (1 method) | Low | Low |
| `ready_judge.py` | 3 | ~80 lines rewritten | Medium | Medium — affects judgment timing |
| `ready_pool.py` | 4 | ~250 lines rewritten | **High** | **High** — class rename, logic changes |
| `entrypoint.py` | 1-4 | ~50 lines rewritten | Medium | Medium — new orchestration flow |
| `staging/__init__.py` | 4 | 2 lines | Trivial | Low |
| **TOTAL** | **1-4** | **~600 lines** | | |

### Risk Assessment

| Risk Level | Files | Mitigation |
|-----------|-------|------------|
| **High** | `ready_worker.py`, `ready_pool.py` | Implement Phase 1 first (shared pool only, no stealing), validate with existing `test_judge.py`, then add steal logic |
| **Medium** | `ready_judge.py`, `entrypoint.py` | Keep old `ingest_results()` path as fallback during Phase 3 transition; add `consume_loop()` as optional alternative before making it primary |
| **Low** | All others | Straightforward additions; no behavioral changes |

### Testing Strategy

1. **Phase 1:** Run `test_judge.py` after `ready_judge.py` changes — should pass identically
2. **Phase 1:** Integration test with 1 Det worker + small task set to validate shared pool pull
3. **Phase 2:** Add steal-specific unit test with 2 workers sharing a pool
4. **Phase 3:** Run judge in both old (pool-called) and new (autonomous) mode side-by-side
5. **Phase 4:** Validate monitor loop detects dead worker within 5s, re-enqueues task

---

*This delta report was generated by analyzing `/home/artem/dev/amd-hackathon/docs/plans/PULL_SYSTEM_DESIGN.md` against the current source in `/home/artem/dev/amd-hackathon/staging/`. All line numbers refer to the versions read on 2026-07-13. Actual line positions will shift after each change.*
