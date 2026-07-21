# Pull-Based Pool System Design

> **Project:** AMD ACT II Hackathon — Track 1 Token Efficient Routing Agent  
> **Analyzed System:** `/home/artem/dev/amd-hackathon/staging/`  
> **Date:** 2026-07-13  
> **Role:** Industrial Engineering / Systems Architecture Analysis  
> **Goal:** Transform the current hybrid push/pool system into a **pure pull-based pool** with work stealing.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Architecture Analysis](#2-current-architecture-analysis)
3. [What Should Be Pulling From What?](#3-what-should-be-pulling-from-what)
4. [Work Stealing Architecture](#4-work-stealing-architecture)
5. [Coordinator Role](#5-coordinator-role)
6. [Pull Contracts](#6-pull-contracts)
7. [Edge Case Analysis](#7-edge-case-analysis)
8. [Recommended Implementation: File-by-File Changes](#8-recommended-implementation-file-by-file-changes)
9. [Code Patterns](#9-code-patterns)
10. [Migration Path](#10-migration-path)

---

## 1. Executive Summary

The current staging system at `staging/` implements a **hybrid push-pull model** that exhibits significant industrial engineering inefficiencies:

- **Per-worker private queues** (push from pool → worker) cause load imbalance: deterministic workers finish in milliseconds while the local LLM worker takes seconds, yet idle deterministic workers cannot help with queued tasks.
- **The pool acts as a central dispatcher** (push model), assigning tasks to workers via round-robin distribution, which is statically fair but dynamically inefficient.
- **Workers pull from their private queue** but have **no mechanism to pull from other workers' queues** (no work stealing).
- **The judge is passive** — it only receives results when the pool calls `ingest_results()`, rather than actively pulling.

**The core problem is a classic producer-consumer imbalance** familiar from industrial assembly lines: when stations have different processing times, a single shared buffer with pull-based access (Kanban-style) dramatically outperforms fixed buffer assignments.

### Recommended Transformation

Replace the current **N private queues + round-robin push** with a **single shared task pool + pull-based dispatch + work stealing + autonomous judge pull**. This converts the pool from a dispatcher/scheduler into a **monitor/health-checker**, eliminates idling, and gracefully degrades under constraints.

---

## 2. Current Architecture Analysis

### 2.1 Current Data Flow Diagram

```
entrypoint.py
  │
  ├── classify_batch() ──→ ReadyQueue (thread-safe queue.Queue, 8 category sub-queues)
  │
  └── ReadyPool.start()
        │
        ├── Creates per-worker private multiprocessing.Queues
        ├── Creates shared category multiprocessing.Queues (unused after startup)
        ├── Spawns 4 workers (2 Det + 1 Loc + 1 FW)
        │
        └── ReadyPool.dispatch_loop()
              │
              ├── Drains ReadyQueue → round-robin → per-worker private queues  (PUSH)
              │
              ├── Loop:
              │     ├── judge.ingest_results(results_queue)      (POLL from pool)
              │     ├── judge.ready_to_judge(tid) → judge(tid)  (pool-driven judgment)
              │     ├── deadline adaptation
              │     └── sleep(0.1)
              │
              └── shutdown()
```

### 2.2 Pull Relationships (Current State)

| Puller | Pulls What | From Where | When | Empty Behavior |
|--------|-----------|------------|------|---------------|
| `Worker._pull_task()` | `ReadyTask` | Private queue (1st), shared cat queues (2nd) | In loop (every 0.2s) | `time.sleep(0.2)` then retry |
| `Pool.dispatch_loop()` | Results (via `judge.ingest_results`) | `_results_queue` (multiprocessing.Queue) | Every loop iteration (~0.1s) | `_queue.Empty` → continue |
| `Pool.dispatch_loop()` | Ready state | Worker `ready_flag` | Once at startup (blocking) | Wait until set (block 30s max) |

### 2.3 Key Inefficiencies Identified

1. **Load imbalance**: With 19 tasks and 4 workers, round-robin gives ~5 tasks each. Det workers finish in ~0.1s total; Loc worker takes ~25s per task (=125s total). Det workers sit idle for ~99.9% of their lifetime.

2. **No work stealing**: The `_pull_task()` method checks private queue first, then shared queues, then falls back to a blocking get on private queue. It **never checks other workers' queues**. An idle worker has no way to consume tasks from a busy worker's queue.

3. **Push-based distribution is rigid**: Round-robin doesn't account for actual worker throughput. A worker that processes tasks 100× faster gets the same number of tasks as a worker that's 100× slower.

4. **Judge is a passive consumer**: The judge doesn't actively pull — the pool must periodically call `ingest_results()`. If the pool's loop is delayed (e.g., by logging), results accumulate in the queue.

5. **Shared category queues are populated but never filled**: `self._category_queues` (multiprocessing.Queue per category) are created but never have tasks put into them from the pool side. They're only populated if a worker puts tasks back (which never happens). They're a vestigial structure.

6. **Flow control is poll-based**: Workers poll every 0.2s for new tasks. The pool polls every 0.1s for results. This adds latency and wastes CPU on context switching.

---

## 3. What Should Be Pulling From What?

### 3.1 Correct Pull Relationships

```
                    ┌──────────────────────┐
                    │    Shared Task Pool   │  ← Single source of truth
                    │   (multiprocessing    │     for unstarted tasks
                    │    .Queue or .Simple- │
                    │    Queue)             │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        ┌──────────┐    ┌──────────┐    ┌──────────┐
        │ Worker A │    │ Worker B │    │ Worker C │
        │ (pulls)  │    │ (pulls)  │    │ (pulls)  │
        └─────┬────┘    └─────┬────┘    └─────┬────┘
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                    ┌──────────────────┐
                    │  Results Pool    │  ← Shared results from all workers
                    │ (multiprocessing │
                    │  .Queue)         │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Judge (pulls    │  ← Actively pulls completed results
                    │  results)        │     on its own schedule
                    └──────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Pool Monitor   │  ← Pulls health status, does NOT
                    │  (health check)  │     dispatch tasks anymore
                    └──────────────────┘
```

### 3.2 The Five Pull Relationships

| # | Puller | Pulls What | From Where | Motivation |
|---|--------|-----------|------------|------------|
| **P1** | Idle Worker | Next unstarted task | **Shared Task Pool** | Workers should autonomously grab the next task when free |
| **P2** | Idle Worker | Stealable task | **Busy Worker's in-progress queue** | Work stealing: idle workers help with partially-processed tasks |
| **P3** | Judge | Completed answers | **Results Queue** | Judge autonomously pulls answers and decides when to judge |
| **P4** | Pool Monitor | Worker health status | **Worker processes** | Pool checks `is_alive()`, exit codes, heartbeat |
| **P5** | Pool Monitor/Scheduler | Judgment decisions | **Judge** | Pool (or entrypoint) pulls final decisions for output |

### 3.3 What Should NOT Pull What

| Incorrect Relationship | Why |
|------------------------|-----|
| Pool pushing tasks to per-worker queues | Causes load imbalance; removes worker autonomy |
| Workers pushing "ready for next task" signals | Adds unnecessary signal latency; workers should just pull |
| Judge being called by pool | Judge should run on its own cadence, not be polled |
| Workers writing directly to judge | Would couple workers to judge; results queue is the correct boundary |

---

## 4. Work Stealing Architecture

### 4.1 Why Work Stealing Is Critical Here

The workers have **wildly different processing speeds**:

| Worker Type | Per-Task Time (5 tries) | Relative Speed |
|-------------|------------------------|----------------|
| Det Worker | ~1-5 ms | **40,000× faster** than Loc |
| FW Worker | ~2-10 s (network) | **~10× slower** than Det |
| Loc Worker | ~20-30 s (inference) | **~40× slower** than FW |

With 19 tasks and round-robin distribution:
- Each Det worker gets ~5 tasks → finishes in ~25ms total
- Loc worker gets ~5 tasks → takes ~125s total
- **Det workers idle for ~125s** while Loc churns

Work stealing would let the Det workers (and FW if it finishes early) pull tasks from the Loc worker's backlog.

### 4.2 Architecture: Two-Queue Design per Worker

Each worker operates with **two multiprocessing.Queues**:

```
┌──────────────────────────────────────────────────┐
│                  Worker Instance                   │
│                                                    │
│  ┌─────────────────┐      ┌────────────────────┐  │
│  │   inbox_queue    │      │  stolen_queue      │  │
│  │ (primary tasks)  │      │ (work-stolen tasks)│  │
│  │  ┌───┬───┬───┐   │      │  ┌───┐            │  │
│  │  │ T │ T │ T │   │      │  │ T │            │  │
│  │  └───┴───┴───┘   │      │  └───┘            │  │
│  └────────┬─────────┘      └────────┬───────────┘  │
│           │                         │              │
│           └─────────┬───────────────┘              │
│                     ▼                              │
│           ┌─────────────────┐                      │
│           │  Pull Strategy  │                      │
│           │  1. stolen_queue│                      │
│           │  2. inbox_queue │                      │
│           │  3. task_pool   │                      │
│           │  4. steal_offer │                      │
│           └─────────────────┘                      │
└──────────────────────────────────────────────────┘
```

### 4.3 Work Stealing Protocol

#### Phase 1: A Worker Finishes Its Assigned Tasks

When a worker's `inbox_queue` is empty and the shared task pool is empty:

1. **Announce availability** on a shared `steal_request_queue` (multiprocessing.Queue):
   ```python
   steal_request_queue.put_nowait({
       "thief_id": worker_id,
       "worker_type": worker_type,
       "capabilities": category_whitelist,
       "timestamp": time.monotonic(),
   })
   ```

2. **Busy workers periodically check** `steal_request_queue`:
   ```python
   # Before pulling their next task (or every ~5s during processing):
   try:
       request = steal_request_queue.get_nowait()
       # If we have tasks in our inbox, offer one
       if not self.inbox_queue.empty():
           task = self.inbox_queue.get_nowait()
           self._offer_task(request["thief_id"], task)
   except queue.Empty:
       pass
   ```

3. **Victim offers a task**: The busy worker puts one task from its inbox into the thief's `stolen_queue`:
   ```python
   def _offer_task(self, thief_id: str, task: ReadyTask) -> None:
       thief_queue = victim_queues[thief_id]  # pre-registered at startup
       thief_queue.put_nowait(task)
       logger.info("[steal] %s stole task %s from %s", thief_id, task.task_id, self.worker_id)
   ```

#### Phase 2: Worker Pull Priority

The `_pull_task()` method is reordered:

```python
def _pull_task(self) -> Optional[ReadyTask]:
    """Pull next task with work-stealing awareness."""
    import queue as _queue

    # 1. Stolen tasks first (highest priority — they were transferred to us)
    try:
        return self.stolen_queue.get_nowait()
    except _queue.Empty:
        pass

    # 2. Private inbox (originally assigned tasks)
    try:
        return self.inbox_queue.get_nowait()
    except _queue.Empty:
        pass

    # 3. Shared task pool (unstarted tasks from ReadyQueue)
    # This is the main pull point for the new architecture
    try:
        return self.task_pool.get_nowait()
    except (AttributeError, _queue.Empty):
        pass

    # 4. Attempt work stealing (we're idle, try to find work)
    return self._attempt_steal()
```

#### Phase 3: The `_attempt_steal()` Method

```python
def _attempt_steal(self, timeout: float = 1.0) -> Optional[ReadyTask]:
    """Try to steal work from other workers.
    
    Strategy: Round-robin through known workers, checking their inbox
    queue sizes via shared manager dict.
    """
    # Get current queue sizes from shared state
    for victim_id in self._known_workers:
        if victim_id == self.worker_id:
            continue
        # Check if victim has tasks to spare
        size = self._queue_sizes.get(victim_id, 0)
        if size > 1:  # Victim has >1 task: steal one
            steal_msg = {
                "thief_id": self.worker_id,
                "victim_id": victim_id,
                "timestamp": time.monotonic(),
            }
            self._steal_request_queue.put(steal_msg)
            # Wait briefly for the victim to respond
            try:
                return self.stolen_queue.get(timeout=0.5)
            except _queue.Empty:
                continue
    return None
```

### 4.4 Centralized vs Decentralized Stealing

| Approach | Mechanism | Pros | Cons |
|----------|-----------|------|------|
| **Decentralized** (recommended) | Workers directly read/write each other's queues via registered queue references | No single point of failure; scales linearly | Needs shared queue registry; slightly more complexity |
| **Pool-mediated** | Pool detects idle worker, reassigns tasks | Simpler to implement | Pool becomes bottleneck; adds latency |

**Recommendation: Decentralized with a shared `steal_request_queue`.** This avoids the pool as a middleman while still having a central coordination point for steal requests.

### 4.5 Work Stealing Thresholds

Not all tasks should be stealable. Define a **steal threshold**:

```python
STEAL_THRESHOLD = 2  # Only steal from workers with >2 tasks in their inbox
```

This prevents thrashing where workers steal from each other constantly. If every worker has 1 task, no stealing occurs — each worker finishes its own.

### 4.6 Task Rebalancing Under Deadline Pressure

When `remaining < 60s`, the **deadline adaptation** mode should bypass the normal work-stealing protocol and do **aggressive rebalancing**:

```python
def _deadline_rebalance(worker_queues: dict, pool: ...):
    """Under deadline pressure: collect all unstarted tasks and redistribute
    to the fastest available workers."""
    # Get speed rankings
    speed_order = ["deterministic", "fireworks", "local"]
    
    # Collect all unstarted tasks from inboxes
    all_tasks = []
    for wid, q in worker_queues.items():
        try:
            while True:
                all_tasks.append(q.get_nowait())
        except queue.Empty:
            pass
    
    # Redistribute to fastest workers first
    for task in all_tasks:
        assigned = False
        for wtype in speed_order:
            idle_workers = [w for w in pool.workers 
                          if w.worker_type == wtype and not w.busy_flag.value]
            if idle_workers:
                idle_workers[0].inbox_queue.put_nowait(task)
                assigned = True
                break
        if not assigned:
            pool.task_pool.put_nowait(task)  # Back to shared pool
```

---

## 5. Coordinator Role

### 5.1 Should There Be a Central Coordinator?

**Yes, but with a narrow mandate.** The current `ReadyPool` tries to be everything: dispatcher, scheduler, health monitor, deadline enforcer, and judge driver. This violates the **Single Responsibility Principle** and creates a coordination bottleneck.

### 5.2 Proposed: Three Independent Loops

```
┌──────────────────────────────────────────────────────────┐
│                     entrypoint.py                         │
│                                                           │
│  1. Load config, read tasks, classify, build queue       │
│  2. Start worker processes                               │
│  3. Start judge process (or use judge in separate loop)  │
│  4. Start pool monitor (dedicated health checker)         │
│  5. Wait for completion OR deadline, judge, output        │
└──────────────────────────────────────────────────────────┘
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Pool Monitor    │ │    Judge        │ │   Workers        │
│  (health check)  │ │  (autonomous)   │ │  (autonomous)    │
│                  │ │                 │ │                  │
│  Pulls:          │ │  Pulls:         │ │  Pulls:          │
│  - is_alive()   │ │  - results_q    │ │  - task_pool     │
│  - exit codes   │ │  Pulls when:    │ │  - stolen tasks  │
│  - heartbeat    │ │  - 0.1s loop    │ │  Processes:      │
│  - queue size   │ │  Decides:       │ │  - tasks async   │
│  Alerts on:     │ │  - ready_to_    │ │  Pushes:         │
│  - dead worker  │ │    judge(tid)   │ │  - results_q     │
│  - stuck busy   │ │  - judge(tid)   │ │                  │
│  - near deadln  │ │  Outputs:       │ │                  │
│                  │ │  - _judged{}   │ │                  │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### 5.3 Coordinator Responsibilities (Narrow)

The pool should only be a **health monitor**:

1. **Worker health pull**: Every 5s, check `p.is_alive()` and `p.exitcode` for each worker
2. **Stuck busy-flag detection**: If `busy_flag == 1` but worker hasn't produced results in 60s, reset flag and reassign task
3. **Deadline supervision**: Monitor `remaining_time` and broadcast `deadline_approaching` signal to all workers
4. **Reaping dead workers**: Remove dead workers from tracking lists, update `available_count`

The pool should **NOT**:
- Dispatch tasks to individual workers (workers pull from shared pool)
- Call judge methods (judge runs independently)
- Manage per-worker queues (workers share a task pool + steal protocol)

### 5.4 Alternative: Fully Decentralized (No Coordinator)

Each component could pull independently with no central coordinator:

- Workers pull from shared task pool → process → push to results queue
- Judge pulls from results queue → judges → signals completion
- Entrypoint polls judge for completion → writes output

**Problem**: No health monitoring, no crash recovery, no deadline enforcement that requires global state. The entrypoint needs at least minimal coordination to know when to stop.

**Recommendation**: Keep a lightweight coordinator (the `ReadyPool` renamed to `ReadyMonitor`) that only handles health, deadline, and restart logic. The heavy lifting (dispatch, result collection, judging) is decentralized.

---

## 6. Pull Contracts

### 6.1 Contract P1: Worker ← Shared Task Pool

| Parameter | Specification |
|-----------|--------------|
| **What** | `ReadyTask` — a classified, unstarted task |
| **From** | `task_pool: multiprocessing.Queue` (single shared queue, NOT per-worker) |
| **When** | Worker becomes idle (finishes previous task, pushes results) |
| **How** | `task_pool.get_nowait()` — non-blocking fast path |
| **Empty** | Fall through to work-stealing protocol, then sleep(0.1) and retry |
| **Block variant** | `task_pool.get(timeout=1.0)` for the "no work available anywhere" case |
| **Contract** | The task is **removed** from the pool when pulled. If worker crashes mid-task, the task is orphaned (handled by pool monitor → re-enqueue). |

**Key change from current**: NO per-worker private queues. One shared `multiprocessing.Queue` is the only source of unstarted tasks.

### 6.2 Contract P2: Idle Worker ← Busy Worker (Steal)

| Parameter | Specification |
|-----------|--------------|
| **What** | `ReadyTask` — a task that was originally pulled by another worker but hasn't been started yet |
| **From** | Victim's `inbox_queue` via thief's `stolen_queue` |
| **When** | Thief's task_pool is empty AND victim has >STEAL_THRESHOLD tasks in queue |
| **How** | Thief puts steal request on `steal_request_queue`; victim responds with task on thief's `stolen_queue` |
| **Empty** | Victims have ≤1 task each → no stealing possible → sleep |
| **Signal** | Two queues: `steal_request_queue` (shared, for requests) and per-worker `stolen_queue` (for received tasks) |
| **Timeout** | Thief waits 0.5s max for stolen task, then tries next victim |
| **Contract** | Stolen task is **removed** from victim's queue and **added** to thief's. No task is processed by both. |

### 6.3 Contract P3: Judge ← Results Queue

| Parameter | Specification |
|-----------|--------------|
| **What** | `dict` — a single answer from a worker: `{worker_id, worker_type, task_id, answer, timing_ms, prompt, category}` |
| **From** | `results_queue: multiprocessing.Queue` (shared, same as current) |
| **When** | Continuous loop in a **dedicated judge thread or process** every ~50ms |
| **How** | `results_queue.get_nowait()` in a drain loop |
| **Empty** | No new results → check pending task timeouts → sleep(0.05) |
| **On arrival** | `judge.add_answer(answer)` → check `ready_to_judge(tid)` if enough votes → `judge(tid)` |
| **Contract** | Judge autonomously tracks `_task_answers`, `_judged`, and `_task_first_answer_time`. No pool involvement needed. |

**Key improvement**: Judge runs in its own loop, not as a subroutine of the pool. The pool no longer calls `judge.ingest_results()`.

### 6.4 Contract P4: Pool Monitor ← Worker Process

| Parameter | Specification |
|-----------|--------------|
| **What** | Health status: alive, exit code, busy_flag value, heartbeat timestamp |
| **From** | `p.is_alive()`, `p.exitcode`, `busy_flag: multiprocessing.Value`, `heartbeat: multiprocessing.Value` |
| **When** | Every 5 seconds in the monitor loop |
| **How** | Direct process attribute polling + shared memory reads |
| **Empty/Dead** | Worker dead → check `_pending_map` for orphaned tasks → re-enqueue to task_pool → remove worker from tracking |
| **Stuck** | `busy_flag == 1` and `heartbeat` unchanged for 60s → assume hung → terminate → re-enqueue task |
| **Contract** | Pool monitor is the ONLY component that mutates worker tracking data. It does NOT touch tasks or results. |

### 6.5 Contract P5: Scheduler (Entrypoint or Judge Driver) ← Judge

| Parameter | Specification |
|-----------|--------------|
| **What** | Judgment decision: `{task_id, answer, strategy}` |
| **From** | `judge._judged` dict (accessible via method) |
| **When** | After all tasks judged OR deadline reached OR no more workers alive |
| **How** | `judge.judge_all()` returns final results |
| **Empty** | No judged tasks → fall through with empty output |
| **Contract** | Entrypoint polls judge periodically (every 0.5s) to check `judge.total_judged`. When `total_judged >= total_tasks` or deadline, call `judge.judge_all()` for remaining tasks. |

---

## 7. Edge Case Analysis

### 7.1 Worker Crash During Processing

**Scenario**: Worker pulls task from shared pool, sets `busy_flag=1`, starts `process()`, then crashes (OOM, segfault, exception).

**Current behavior**:  
- Task removed from queue (pulled), results never pushed  
- `busy_flag` stuck at 1 forever  
- Pool never notices (no `is_alive()` check at this granularity)  
- Task silently dropped from output

**Pull-based behavior**:  
1. Pool monitor detects `is_alive() == False` in <5s  
2. Pool checks `_pending_map[worker_id]` to find the orphaned task  
3. Pool re-enqueues task to shared `task_pool`  
4. Pool resets `busy_flag` for that worker  
5. Pool removes worker from `_known_workers` and `_worker_queues`  
6. Next idle worker pulls the task from shared pool  
7. If all workers dead and tasks remain → `judge_all()` produces empty answers

**Mitigation**: Add a **heartbeat** mechanism:
```python
# In worker's main loop:
while self._running:
    self.heartbeat.value = int(time.monotonic())  # Updated every iteration
    task = self._pull_task()
    ...
```

Pool checks: if `busy_flag == 1` and `(now - heartbeat.value) > 60`, assume hung and terminate.

### 7.2 All Workers Busy But Tasks Pending

**Scenario**: 4 workers all processing tasks. 15 tasks remain in shared pool. No available workers.

**Current behavior**: Tasks sit in per-worker queues (some grow, some stay empty). No rebalancing.

**Pull-based behavior**:  
1. Tasks sit in the shared `task_pool` queue  
2. As soon as ANY worker finishes its current task and returns to idle, it pulls the next task  
3. No rebalancing needed — it's a fair queue by construction  
4. If workers have very different speeds, the fast ones pull more tasks naturally (unintentional but correct behavior)

**Risk**: A slow worker could keep pulling tasks while a fast worker sits idle, if the slow worker pulls before the fast one. In practice, the fast worker finishes and is waiting to pull before the slow worker finishes its first task — so fast workers pull more.

**Guarantee**: The shared queue is FIFO. Workers pull from it atomically. A fast worker will complete tasks and be back to pull the next FIFO task before a slow worker finishes its single task.

### 7.3 Deadline Approaching With Unfinished Tasks

**Scenario**: 30s remaining, 10 tasks still not judged.

**Pull-based behavior**:  
1. Pool monitor detects `remaining < 30s` → sets global `deadline_emergency` flag  
2. Workers check this flag before pulling new tasks: if set, they skip the 5-try loop and do a **single try** (fast mode)  
3. Judge checks `deadline_emergency` and reduces `judgment_votes` threshold to 1  
4. At `remaining < 5s`: pool monitor forces workers to abort current task (via thread/process signal) and pushes whatever results they have  
5. Entrypoint: `judge_all()` produces best-effort answers for all tasks

**Critical protocol**: The `deadline_emergency` flag is a `multiprocessing.Value` written by the monitor and read by all workers + judge. No queue needed — shared memory is atomic for a boolean.

### 7.4 Fireworks API Unavailable (No Key)

**Scenario**: `FIREWORKS_API_KEY` is empty or invalid.

**Current behavior**: FW worker pulls tasks, tries all 5 calls (each times out with 401), returns empty answers. Wastes ~2-5s total.

**Pull-based behavior**:  
1. FW worker's `initialize()` checks for API key  
2. If missing: sets `self._available = False`, signals `ready_flag = 1` (ready, but won't process)  
3. In `_pull_task()`: if `!_available`, never pulls from shared pool  
4. Instead: enters a **passive mode** where it only does work-stealing for tasks that need any LLM (not deterministic)  
5. Even better: skip FW workers entirely in `_build_worker_plan()` if no API key → save process slot

**Optimal**: Skip FW workers if no key. The config check at pool construction:
```python
if not config.fw_api_key:
    logger.warning("No FIREWORKS_API_KEY — skipping FW workers")
    config.fw_workers = 0
```

### 7.5 Local Model Load Failure

**Scenario**: Model file not found at `loc_model_path`, or `llama_cpp` import fails, or OOM during model load.

**Current behavior**: Worker catches exception in `initialize()`, logs error, returns. Worker process exits. Pool never notices (no health check at this point).

**Pull-based behavior**:  
1. Worker `initialize()` fails → worker exits with `exitcode != 0`  
2. Pool monitor detects dead worker <5s later  
3. Pool logs warning, removes from tracking, adjusts `available_count`  
4. Tasks that would have gone to Loc worker now go to FW + Det workers  
5. If `judgment_votes` cannot be met (only 2 workers left), judge auto-reduces threshold

**Graceful degradation chain**:
```
LocWorker init fail → exit code 1 → monitor detects → {pool adjusts, tasks rebalanced}
→ remaining workers pull from shared pool → judge sees fewer votes → reduces threshold
→ output with fewer votes → lower confidence but non-empty
```

### 7.6 Det Workers With Empty/None Answers

**Scenario**: Deterministic solver returns `None`, empty string, or nonsensical answer for a task.

**Current behavior**: Worker returns `""` → judge sees empty → counts as vote for "all empty" → waits for all 3 worker types → eventually declares "all_failed".

**Pull-based behavior**:  
1. Judge's `_is_degenerate()` function already handles this  
2. Empty answer is degenerate → excluded from voting groups  
3. If `judgment_votes` count includes empty answers, judge still waits for target count  
4. **Improvement**: Count ONLY non-degenerate answers toward `ready_to_judge()` threshold  
5. If all workers return empty → `all_failed` strategy → empty answer in output

**Deeper fix**: The judge should have a **minimum non-degenerate threshold**:
```python
def ready_to_judge(self, task_id: str) -> bool:
    answers = self._task_answers.get(task_id, [])
    non_degenerate = [a for a in answers if not _is_degenerate(a.get("answer", ""))]
    
    # Primary: enough non-degenerate answers
    if len(non_degenerate) >= self.config.judgment_votes:
        return True
    
    # Secondary: total answers cover all worker types (all failed)
    if len(answers) >= self.total_expected_answers:
        return True
    
    # Fallback: timeout since first answer
    ...
```

### 7.7 Judge Waiting Forever for Missing Worker Type

**Scenario**: LocWorker fails to initialize. Only Det+Fw workers remain. Judge's `_KNOWN_WORKER_TYPES = {"deterministic", "local", "fireworks"}` waits for "local" type forever.

**Current behavior**: Judge's `ready_to_judge()` checks if all 3 worker types have contributed. If one type is permanently absent, tasks with all-empty answers never get judged until the 30s timeout fires.

**Pull-based behavior**:  
1. Judge receives worker types from answers as they arrive  
2. `_KNOWN_WORKER_TYPES` should be **dynamic**, not hardcoded:
   ```python
   # Track actual worker types that have submitted at least one answer
   self._active_worker_types: set[str] = set()
   
   def add_answer(self, answer: dict) -> None:
       wt = self._get_worker_type(answer)
       self._active_worker_types.add(wt)
       ...
   ```
3. `ready_to_judge()` uses `_active_worker_types` instead of hardcoded set:
   ```python
   if task_types == self._active_worker_types:
       return True  # All active types contributed
   ```
4. Pool monitor communicates active workers to judge via shared value or queue

### 7.8 Mixed Worker Speeds (Det ms vs Loc seconds)

**Scenario**: Det workers finish in ~5ms per task. Loc worker takes ~25s per task. With 19 tasks and shared pull pool:

**Pull-based behavior**:  
1. All 19 tasks in shared pool  
2. All 4 workers pull tasks concurrently  
3. Det worker pulls task 1 → finishes in 5ms → pushes results → pulls task 5 → finishes → ...  
4. Det workers churn through tasks at ~200 tasks/second  
5. Loc worker pulls task 2 → takes 25s  
6. By the time Loc finishes task 2, all 19 tasks have been processed by Det+FW workers  
7. Loc's result for task 2 arrives to judge along with Det's results for all 19 tasks  
8. Judge already judged most tasks; Loc's result is extra data used for cross-validation

**Result**: The fast workers dominate throughput. The Loc worker adds diversity but doesn't block the pipeline. The system is naturally load-balanced by speed.

**If Det workers aren't suitable for all tasks**: The `category_whitelist` in `_build_whitelist()` determines which categories each worker type handles. If Det only handles `["math", "sentiment", "factual"]` but most tasks are `["code_gen", "logic"]`, the Det workers can't help with those. The shared pool still ensures Loc+FW handle those categories.

**Recommendation**: Track per-category task counts and make the shared pool category-aware:
```python
# Worker pulls from pool with category filter:
def _pull_task(self) -> Optional[ReadyTask]:
    # Try to pull a task matching our whitelist
    for cat in self.category_whitelist:
        task = self._task_pool_by_category.get(cat)
        if task:
            return task
    # Fallback: any task
    return self._task_pool.get_nowait()
```

---

## 8. Recommended Implementation: File-by-File Changes

### 8.1 `ready_pool.py` → Transform to `ReadyMonitor`

**Current role**: Dispatcher + scheduler + health checker + judge driver  
**New role**: Health monitor only + startup coordinator

**Changes needed:**

```python
class ReadyMonitor:
    """
    Lightweight health monitor for the pull-based pool system.
    
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
    
    def __init__(self, config: ReadyConfig):
        self.config = config
        self.task_pool = multiprocessing.Queue()  # Single shared pool
        self.results_queue = multiprocessing.Queue()
        self.steal_request_queue = multiprocessing.Queue()
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
```

**Deleted methods:**
- `_enqueue_remaining()` — no more per-worker distribution
- `start()` queue enqueue logic — replaced by putting all tasks into `task_pool`
- `dispatch_loop()` — replaced by independent loops

**Simplified start():**
```python
def start(self, queue: ReadyQueue) -> None:
    """Populate shared task pool and spawn workers."""
    self._running.set()
    
    # Drain ReadyQueue into shared task pool
    while not queue.empty:
        task = queue.dequeue_any(preferred_categories=[])
        if task is None:
            break
        self.task_pool.put_nowait(task)
    
    # Spawn workers (same as current, but pass task_pool instead of per-worker queues)
    workers_to_start = self._build_worker_plan()
    for wid, wtype, worker_cls, index in workers_to_start:
        inbox = multiprocessing.Queue()
        stolen = multiprocessing.Queue()
        self._inbox_queues[wid] = inbox
        self._stolen_queues[wid] = stolen
        
        busy_flag = multiprocessing.Value('b', 0)
        heartbeat = multiprocessing.Value('d', time.monotonic())
        
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
            ...
        )
        ...
```

**New monitor loop (replaces dispatch_loop):**
```python
def monitor_loop(self, judge, deadline: float) -> None:
    """Health monitoring loop — runs in main process."""
    while self._running.is_set():
        now = time.monotonic()
        remaining = deadline - now
        
        # 1. Check worker health
        for i, p in enumerate(self._processes):
            if not p.is_alive():
                self._handle_dead_worker(i)
        
        # 2. Check for stuck workers (heartbeat timeout)
        for i, hb in enumerate(self._heartbeats):
            if self._busy_flags[i].value and (now - hb.value) > 60:
                self._handle_stuck_worker(i)
        
        # 3. Deadline emergency broadcast
        if remaining < 30 and not self._deadline_emergency.value:
            self._deadline_emergency.value = 1
            logger.warning("[monitor] DEADLINE EMERGENCY — forcing fast mode")
        elif remaining >= 30 and self._deadline_emergency.value:
            self._deadline_emergency.value = 0  # Reset if somehow recovered
        
        # 4. Log progress
        if now - self._last_log >= 10.0:
            ...
        
        # 5. Check completion
        if judge.total_judged >= self._total_tasks:
            break
        if remaining <= 0:
            break
        if all(not p.is_alive() for p in self._processes):
            break
        
        time.sleep(5.0)  # Check every 5 seconds (not 0.1s — no longer dispatching)
```

### 8.2 `ready_worker.py` — Transform Worker Base Class

**Current**: Worker has `task_queue` (private), `task_queues` (shared category queues — unused).  
**New**: Worker has `task_pool` (shared), `inbox_queue` (private), `stolen_queue` (steal target), `steal_request_queue` (shared).

```python
class ReadyWorker(ABC):
    def __init__(
        self,
        worker_id: str, worker_type: str, config: ReadyConfig,
        task_pool,           # multiprocessing.Queue — shared task pool
        results_queue,       # multiprocessing.Queue — shared results
        steal_request_queue, # multiprocessing.Queue — steal requests
        stolen_queue,        # multiprocessing.Queue — tasks stolen FOR us
        inbox_queue,         # multiprocessing.Queue — tasks assigned TO us
        busy_flag,           # multiprocessing.Value
        heartbeat,           # multiprocessing.Value (float timestamp)
        deadline_emergency,  # multiprocessing.Value (bool)
        category_whitelist=None,
        **kwargs,
    ):
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
        self._known_workers: list[str] = []  # populated at startup
```

**New `_pull_task()` method (the heart of the pull system):**
```python
def _pull_task(self) -> Optional[ReadyTask]:
    """Pull next task with work stealing.
    
    Priority: stolen > inbox > task_pool > steal
    """
    import queue as _queue
    
    # 1. Check stolen queue (fastest path — already transferred to us)
    try:
        return self.stolen_queue.get_nowait()
    except _queue.Empty:
        pass
    
    # 2. Check inbox (tasks explicitly assigned to us)
    try:
        return self.inbox_queue.get_nowait()
    except _queue.Empty:
        pass
    
    # 3. Pull from shared task pool (main pull point)
    try:
        task = self.task_pool.get_nowait()
        if task is not None:
            return task
    except _queue.Empty:
        pass
    
    # 4. Attempt work stealing (we're fully idle)
    return self._attempt_steal()
```

**New `_attempt_steal()` method:**
```python
def _attempt_steal(self) -> Optional[ReadyTask]:
    """Try to steal work from a busy worker.
    
    Sends a steal request to the shared queue, waits briefly for a response.
    """
    import queue as _queue
    
    # Check if there's a clear steal target first
    # (In production, this would check a shared size registry)
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
        return self.stolen_queue.get(timeout=0.5)
    except _queue.Empty:
        return None
```

**Worker's main loop with heartbeat:**
```python
def run(self) -> None:
    """Main loop with heartbeat for crash detection."""
    try:
        self.initialize()
    except Exception as exc:
        logger.error("[%s] Init failed: %s", self.worker_id, exc)
        return
    
    if self.ready_flag is not None:
        self.ready_flag.value = 1
    
    while self._running:
        # Update heartbeat FIRST — shows we're alive even when busy
        self.heartbeat.value = time.monotonic()
        
        task = self._pull_task()
        if task is None:
            time.sleep(0.2)
            continue
        
        self.busy_flag.value = 1
        
        # Check deadline_emergency for "fast mode"
        if self.deadline_emergency.value:
            # Single try instead of full judgment_votes loop
            answers = self._process_single(task)
        else:
            try:
                answers = self.process(task)  # Full judgment_votes loop
            except Exception as exc:
                logger.exception(...)
                answers = []
        
        self._push_results(task, answers)
        self.busy_flag.value = 0
        task.status = "judged"
    
    self.shutdown()
```

### 8.3 `ready_judge.py` — Make Judge Autonomous

**Current**: Judge is a passive object called by the pool.  
**New**: Judge can run independently with its own result-pulling loop.

**Add autonomous result ingestion:**
```python
class ReadyJudge:
    def __init__(self, config: ReadyConfig):
        # ... existing init ...
        self._active_worker_types: set[str] = set()  # Dynamic, not hardcoded
        self.total_expected_answers = config.judgment_votes  # Can be reduced under pressure
        self.deadline_emergency = None  # Set externally by monitor
    
    def add_answer(self, answer: dict) -> None:
        """Record one worker's answer with dynamic worker type tracking."""
        tid = answer.get("task_id")
        if not tid or "answer" not in answer:
            return
        
        wt = self._get_worker_type(answer)
        self._active_worker_types.add(wt)  # Dynamic tracking
        
        self._task_answers[tid].append(answer)
        if tid not in self._task_first_answer_time:
            self._task_first_answer_time[tid] = time.monotonic()
    
    def ready_to_judge(self, task_id: str) -> bool:
        """Check if enough votes collected.
        
        Uses dynamic worker types instead of hardcoded set.
        Reduces threshold under deadline emergency.
        """
        answers = self._task_answers.get(task_id, [])
        non_degenerate = [a for a in answers 
                         if not _is_degenerate(a.get("answer", ""))]
        
        # Determine effective threshold
        threshold = self.config.judgment_votes
        if self.deadline_emergency and self.deadline_emergency.value:
            threshold = max(1, threshold // 2)  # Reduce to ~3 under pressure
        
        # Primary: enough non-degenerate answers
        if len(non_degenerate) >= threshold:
            return True
        
        # Secondary: all active worker types contributed (all failed)
        if len(answers) >= self.config.judgment_votes:
            task_types = set(self._get_worker_type(a) for a in answers)
            if len(task_types) >= min(2, len(self._active_worker_types)):
                # At least 2 types tried, all empty → declare failure
                all_empty = all(
                    _is_degenerate(a.get("answer", "")) 
                    for a in answers
                )
                if all_empty:
                    return True
        
        # Timeout fallback (reduced under deadline)
        timeout = 15.0 if (self.deadline_emergency and 
                          self.deadline_emergency.value) else 30.0
        first_time = self._task_first_answer_time.get(task_id)
        if first_time is not None and (time.monotonic() - first_time) >= timeout:
            return True
        
        return False
```

**Add `consume_loop()` for autonomous operation:**
```python
def consume_loop(self, results_queue, deadline_emergency=None, 
                 stop_event=None) -> None:
    """Autonomous loop: pull results, judge, repeat.
    
    Can run in its own thread or be called periodically.
    """
    self.deadline_emergency = deadline_emergency
    
    while not (stop_event and stop_event.is_set()):
        self.ingest_results(results_queue)
        
        # Try to judge any ready tasks
        for tid in list(self.pending_tasks):
            if self.ready_to_judge(tid):
                answer, meta = self.judge(tid)
        
        time.sleep(0.05)  # 50ms poll interval (faster than current 100ms)
```

### 8.4 `ready_queue.py` — Add Category-Aware Pool Interface

**Current**: `ReadyQueue` with `dequeue_any()` and per-category queues.  
**New**: Keep `ReadyQueue` for initial population, add category-aware pool reader.

```python
class ReadyQueue:
    # ... existing code for classify-time enqueue ...
    
    def drain_to_pool(self, task_pool: multiprocessing.Queue) -> int:
        """Drain all tasks into a shared multiprocessing pool.
        
        Used at startup to populate the worker-facing pool.
        """
        count = 0
        while not self.empty:
            task = self.dequeue_any(preferred_categories=[])
            if task is None:
                break
            task_pool.put_nowait(task)
            count += 1
        return count
```

### 8.5 `ready_config.py` — Add Pull System Tuning Parameters

```python
@dataclass
class ReadyConfig:
    # ... existing fields ...
    
    # ── Pull system tuning ──
    steal_threshold: int = 2              # Min tasks in victim's queue to enable stealing
    steal_timeout_s: float = 0.5          # Max wait for stolen task
    heartbeat_timeout_s: float = 60.0     # Max seconds without heartbeat before kill
    monitor_interval_s: float = 5.0       # Health check interval
    judge_poll_interval_s: float = 0.05   # Judge result pull interval
    emergency_vote_reduction: int = 2     # Divide judgment_votes by this under emergency
    per_worker_inbox_size: int = 3        # Max tasks in a worker's inbox before pool stops feeding it
```

### 8.6 `workers/det_worker.py`, `loc_worker.py`, `fw_worker.py` — Minor Adjustments

**Changes needed in each:**
1. Constructor: accept `task_pool`, `stolen_queue`, `inbox_queue`, `steal_request_queue`, `heartbeat`, `deadline_emergency` instead of `task_queue`, `task_queues`
2. No behavioral changes in `process()` — workers still do 5-try temperature sweep
3. Add `_process_single()` method for deadline emergency fast mode:
   ```python
   def _process_single(self, task: ReadyTask) -> list[dict]:
       """Single fast try for deadline emergency mode."""
       # Same as process() but only one iteration
       answer = self._solver.solve(...)  # Single try
       return [answer]
   ```

### 8.7 `entrypoint.py` — Orchestrate the New Pull System

```python
def main() -> None:
    config = ReadyConfig.from_env()
    deadline = time.monotonic() + config.deadline_s
    
    # Read, classify, build queue (same as before)
    tasks = _read_tasks()
    classified = classify_batch([t.get("prompt", "") for t in tasks])
    queue = ReadyQueue()
    # ... build ready_tasks, enqueue_batch ...
    
    # Create components
    judge = ReadyJudge(config)
    monitor = ReadyMonitor(config)
    
    # Start: populates task_pool, spawns workers
    monitor.start(queue)  # Internally drains queue to task_pool
    
    # Run: three independent loops in main process
    # (Could also use threads or async; here we do sequential for simplicity)
    deadline_emergency = monitor._deadline_emergency
    stop_event = multiprocessing.Event()
    
    # Start judge consumer thread
    import threading
    judge_thread = threading.Thread(
        target=judge.consume_loop,
        args=(monitor.results_queue, deadline_emergency, stop_event),
        daemon=True,
    )
    judge_thread.start()
    
    # Monitor loop (blocking in main thread)
    monitor.monitor_loop(judge, deadline)
    
    # Stop judge
    stop_event.set()
    judge_thread.join(timeout=2.0)
    
    # Collect final results
    judge.ingest_results(monitor.results_queue)  # Final drain
    final_results = judge.judge_all()
    
    # Write output (same as before)
    output = [{"task_id": r["task_id"], "answer": r["answer"]} for r in final_results]
    _write_output(output)
    
    monitor.shutdown()
```

---

## 9. Code Patterns

### 9.1 How Workers Signal Completion and Pull Next Task

```python
# Pattern: Worker run() loop with continuous pull
def run(self):
    self.initialize()
    self.ready_flag.value = 1
    
    while self._running:
        self.heartbeat.value = time.monotonic()  # I'm alive
        
        # PULL — worker drives the work cycle
        task = self._pull_task()
        if task is None:
            time.sleep(0.2)
            continue
        
        # PROCESS
        self.busy_flag.value = 1
        try:
            if self.deadline_emergency.value:
                answers = self._emergency_process(task)
            else:
                answers = self.process(task)
        except Exception:
            answers = []
        
        # PUSH results — always done, even on failure
        self._push_results(task, answers)
        self.busy_flag.value = 0

# Pattern: Task pull with steal fallback
def _pull_task(self) -> Optional[ReadyTask]:
    import queue as _queue
    
    # Priority 1: Stolen tasks (fastest)
    try:
        return self.stolen_queue.get_nowait()
    except _queue.Empty:
        pass
    
    # Priority 2: Inbox (manually assigned)
    try:
        return self.inbox_queue.get_nowait()
    except _queue.Empty:
        pass
    
    # Priority 3: Shared pool (main pull)
    try:
        return self.task_pool.get_nowait()
    except _queue.Empty:
        pass
    
    # Priority 4: Steal from others
    return self._attempt_steal()
```

### 9.2 How the Pool Detects Idle Workers and Redistributes Work

```python
# Pattern: Pool monitor detects idle workers and triggers steal
def monitor_loop(self, judge, deadline):
    while self._running.is_set():
        idle_workers = []
        busy_counts = {}
        
        for i, wid in enumerate(self._worker_ids):
            if not self._busy_flags[i].value:
                idle_workers.append(wid)
            else:
                # Track busy workers' queue sizes
                inbox = self._inbox_queues[wid]
                try:
                    busy_counts[wid] = inbox.qsize()
                except NotImplementedError:
                    pass  # macOS qsize() may raise
        
        # If we have idle workers AND tasks in the task pool,
        # no action needed — they'll pull on their own.
        
        # If we have idle workers but task pool is empty,
        # AND busy workers have multiple queued tasks,
        # broadcast a "redistribute" signal:
        if idle_workers and self.task_pool.empty():
            for wid, size in busy_counts.items():
                if size > self.config.steal_threshold:
                    # Victim has surplus — advertise to idle workers
                    self._steal_opportunity.value = 1
                    break
        
        time.sleep(self.config.monitor_interval_s)
```

### 9.3 How the Judge Pulls Completed Tasks and Decides When to Judge

```python
# Pattern: Judge pulls results and judges autonomously
def consume_loop(self, results_queue, deadline_emergency, stop_event):
    """Run in a daemon thread, pulling results and judging."""
    while not (stop_event and stop_event.is_set()):
        # PULL results
        ingested = self.ingest_results(results_queue)
        
        # DECIDE for each pending task
        for tid in list(self.pending_tasks):
            if self.ready_to_judge(tid):
                answer, meta = self.judge(tid)
        
        # Sleep briefly if no new results
        if ingested == 0:
            time.sleep(0.05)

# Pattern: Judgment readiness with deadline awareness
def ready_to_judge(self, task_id: str) -> bool:
    answers = self._task_answers.get(task_id, [])
    non_degenerate = [a for a in answers 
                     if not _is_degenerate(a.get("answer", ""))]
    
    # Dynamic threshold under deadline pressure
    threshold = self.config.judgment_votes
    if self.deadline_emergency and self.deadline_emergency.value:
        threshold = max(1, threshold // 2)
    
    # Enough good answers
    if len(non_degenerate) >= threshold:
        return True
    
    # All workers tried, all empty
    if len(answers) >= threshold:
        if all(_is_degenerate(a.get("answer", "")) for a in answers):
            return True
    
    # Timeout
    timeout = 15.0 if (self.deadline_emergency and 
                      self.deadline_emergency.value) else 30.0
    first_time = self._task_first_answer_time.get(task_id)
    if first_time and (time.monotonic() - first_time) >= timeout:
        return True
    
    return False
```

### 9.4 How the System Degrades Under Pressure

```python
# Pattern: Graceful degradation chain

# Level 0: Normal mode (remaining > 60s)
# - Workers do full 5-try temperature sweep
# - Judge waits for 5 votes
# - Full work stealing enabled
# - All worker types active

# Level 1: Mild pressure (30s < remaining <= 60s)
if remaining <= 60 and remaining > 30:
    self._deadline_emergency.value = 0  # Not yet emergency
    # Judge still uses 5-vote threshold
    # Workers still do full loop
    # Pool monitor checks more frequently: 2s instead of 5s

# Level 2: Emergency (remaining <= 30s)
if remaining <= 30:
    self._deadline_emergency.value = 1  # Emergency flag
    judge.deadline_emergency = self._deadline_emergency
    
    # Workers read emergency flag and switch to single-try mode
    # Judge halves judgment threshold
    # Pool checks every 1s
    
    # Optional: force workers to abort current task
    for i, p in enumerate(self._processes):
        if self._busy_flags[i].value and p.is_alive():
            # Can't force-abort cleanly in Python multiprocessing
            # Best effort: send SIGUSR1 or set abort flag
            pass

# Level 3: Critical (remaining <= 5s)
if remaining <= 5:
    # Skip any remaining task dispatch
    # Judge all pending tasks with whatever we have
    for tid in judge.pending_tasks:
        if judge.count_answers(tid) >= 1:
            judge.judge(tid)
        else:
            # No answers at all — produce empty
            judge._judged[tid] = {
                "strategy": "deadline_emergency", "answer": ""
            }
    
    # Force output write even if incomplete
    _write_output(judge.judge_all())

# Level 4: Worker count reduced
# If a worker crashes and isn't replaced:
active_workers = sum(1 for p in self._processes if p.is_alive())
if active_workers < self.config.total_workers:
    # Adjust expected answer count
    remaining_tasks = self._total_tasks - judge.total_judged
    max_possible_answers = remaining_tasks * active_workers * self.config.judgment_votes
    judge.total_expected_answers = min(
        self.config.judgment_votes,
        max_possible_answers // max(1, remaining_tasks)
    )
```

---

## 10. Migration Path

### Phase 1: Shared Task Pool (Minimal Change)

1. Add `task_pool` (shared `multiprocessing.Queue`) to worker constructor  
2. Replace per-worker private queues with `task_pool` for new task distribution  
3. Keep `judge.ingest_results()` being called by pool (for now)  
4. Remove round-robin distribution; workers pull from `task_pool` directly  
5. **Result**: Immediate load balancing. Fast workers naturally process more tasks.

### Phase 2: Work Stealing

1. Add `steal_request_queue` to worker constructor  
2. Add `stolen_queue` per worker  
3. Implement `_attempt_steal()` in `ReadyWorker`  
4. Implement victim response in worker's pre-pull check  
5. **Result**: Idle workers can grab tasks from busy workers' inboxes.

### Phase 3: Autonomous Judge

1. Extract judge result-pulling into `consume_loop()`  
2. Run judge in a separate thread (daemon)  
3. Pool no longer calls `judge.ingest_results()`  
4. **Result**: Judge runs on its own cadence, decoupled from pool.

### Phase 4: Pool Monitor (Final Form)

1. Rename `ReadyPool` to `ReadyMonitor` (or add `ReadyMonitor` as a new class)  
2. Remove dispatch/assignment logic  
3. Add heartbeat tracking to workers  
4. Add orphaned task re-enqueue on crash  
5. Add deadline emergency flag and pressure protocol  
6. **Result**: The pull-based system is complete.

### Backward Compatibility

Throughout all phases, the `ReadyTask` dataclass, `ReadyConfig` class, and output contract (`/output/results.json`) remain unchanged. The judge's fuzzy-match voting logic is unaffected. Only the _flow_ of tasks between components changes.

---

## Appendix A: Shared State Diagram (Complete Pull System)

```
                            ┌──────────────────┐
                            │   ReadyQueue      │
                            │  (classify-time)  │
                            └────────┬─────────┘
                                     │ drain_to_pool()
                                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   Shared Task Pool                           │
│              multiprocessing.Queue()                         │
│  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐  │
│  │ T1   │ T2   │ T3   │ T4   │ T5   │ T6   │ ...  │ TN   │  │
│  └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘  │
└─────────────────────────────────────────────────────────────┘
         │          │          │          │
  pull() │   pull() │  pull() │  pull() │
         ▼          ▼          ▼          ▼
   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
   │DetWkr 0│ │DetWkr 1│ │LocWkr 0│ │FwWkr 0 │
   │(fast)  │ │(fast)  │ │(slow)  │ │(med)   │
   └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘
       │          │          │          │
       └──────────┼──────────┼──────────┘
                  ▼          ▼
         ┌──────────────────────┐
         │   Results Queue      │
         │  multiprocessing.Q   │
         └──────────┬───────────┘
                    │ ingest_results()
                    ▼
         ┌──────────────────────┐
         │       Judge          │
         │  (autonomous thread) │
         │  consume_loop()      │
         └──────────┬───────────┘
                    │ judge_all()
                    ▼
         ┌──────────────────────┐
         │  entrypoint (output) │
         └──────────────────────┘

         ┌─────────────────────────────────────┐
         │          Pool Monitor               │
         │  (health checks, re-enqueue,        │
         │   deadline_emergency, reaping)       │
         └─────────────────────────────────────┘
              │  │  │  │  │  │  │  │
              ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼
         Process alive? heartbeat? busy_flag?

         ┌─────────────────────────────────────┐
         │      Steal Request Queue            │
         │  multiprocessing.Queue()             │
         └─────────────────────────────────────┘
              ▲           │           ▲
              │  request  │  respond  │  request
         ┌────┴────┐ ┌───┴────┐ ┌───┴────┐
         │DetWkr 0 │ │DetWkr 1│ │LocWkr 0│
         │(thief)  │ │(thief) │ │(victim)│
         └─────────┘ └────────┘ └────────┘
```

## Appendix B: Performance Projections

| Metric | Current (Push) | Pull System | Improvement |
|--------|---------------|-------------|-------------|
| Total time (19 tasks, 4 workers, mixed speeds) | ~130s (Loc bottleneck) | ~30-40s (Det+FW pull majority) | **3-4× faster** |
| Worker idle time (Det workers) | ~99% | ~40-60% (steal non-Det tasks when idle) | **2.5× better utilization** |
| Worker idle time (FW worker) | ~50% | ~20-30% (steals from Loc) | **2× better utilization** |
| Tasks lost on worker crash | **Yes** (silent) | **No** (re-enqueued via monitor) | **Zero-loss** |
| Judge response latency | ~100ms (pool-bound) | ~50ms (autonomous) | **2× faster judgment** |
| Code complexity | Medium | **Lower** (no dispatcher, simpler lifecycle) | **Simpler** |

---

*This document is the output of a systems architecture analysis. Implementation should be validated against the actual codebase and tested under grader constraints before deployment.*
