# Staging System — Reliability Review Under Grader Constraints

**Reviewer:** Hermes Agent  
**Date:** 2026-07-13  
**Scope:** `/home/artem/dev/amd-hackathon/staging/` and `docs/plans/PARALLEL_SUBMIT_PLAN.md`  
**Container constraints:** 2 vCPU, 4 GB RAM, 600s deadline, linux/amd64, 30s per-task timeout  
**Model:** Qwen2.5-1.5B Q4_K_M GGUF (~1.1GB)

---

## Executive Summary

The staging system has a viable architecture but contains **4 critical (P0) bugs** that can silently drop tasks, **5 high-severity (P1) gaps** in robustness, and several (P2) hardening items. The largest architectural mismatch: **the code implements single-worker 5-try temperature sweep, NOT the 5-worker diversity plan described in the design.** This means majority voting is far weaker than intended.

---

## Risk Matrix Legend

| Severity | Description |
|----------|-------------|
| **Critical** | Task loss, silent data corruption, guaranteed crash |
| **High** | Degraded results under real conditions, partial data loss |
| **Medium** | Suboptimal behavior, inefficiency, minor edge-case issues |
| **Low** | Cosmetic, future-proofing, hardening |

| Likelihood | Description |
|------------|-------------|
| **Certain** | Will happen every time under normal conditions |
| **Likely** | Will happen under common scenarios (any non-trivial task count) |
| **Possible** | Happens under specific conditions (network issues, specific deadlines) |
| **Unlikely** | Requires unusual environment/configuration |
| **Rare** | Edge case that's technically possible but improbable |

---

## 1. Container Constraints Analysis

### 1.1 RAM Budget per Worker

| Component | Memory | Notes |
|-----------|--------|-------|
| OS + Python runtime | ~250-350 MB | Container base, Python interpreter |
| Main process (entrypoint) | ~100-150 MB | Config, queues, judge, classifier |
| 1× LocWorker (llama-cpp) | ~1500-1700 MB | Model (1.1GB) + 2048 ctx + process overhead |
| 1× FwWorker (urllib) | ~50-80 MB | No model, just HTTP client |
| 2× DetWorker | ~60-120 MB total | ~30-60 MB each, pure Python |
| **Total (default config)** | **~2.1-2.5 GB** | ✅ **Safe — within 4GB** |
| Headroom | **~1.5-1.9 GB** | For buffer bloat, OOM safety margin |

**If `STAGING_LOC_WORKERS=2`**: 2 × 1.6GB = 3.2GB + other processes = ~3.9GB → **OOM risk.** The config defaults to 1 but nothing enforces this.

### 1.2 CPU Contention

**Default config**: 4 workers (1 FW + 1 Loc + 2 Det) + 1 main process = 5 processes for 2 vCPUs.

| Worker | CPU Profile | Impact on 2 vCPUs |
|--------|-------------|-------------------|
| LocWorker (n_threads=2) | CPU-bound during inference | Saturates both cores during inference runs |
| FwWorker | I/O-bound (HTTP wait) | Negligible — mostly waiting on network |
| DetWorker | CPU-brief (µs-scale) | Negligible — instant execution |
| Main process | Sleeps 0.1s between iterations | Negligible |

**Real contention**: When LocWorker is inferring with 2 threads on 2 vCPUs, it fully saturates CPU. Other processes are preempted. Since FwWorker and DetWorker are mostly I/O-bound or instant, this is **acceptable**. However, if `n_threads` exceeds 2, oversubscription is guaranteed.

**If 2 LocWorkers are configured**: 4 inference threads on 2 vCPUs → heavy oversubscription, context switching overhead, and likely slower per-inference latency.

### 1.3 Swap / OOM Risk Under Peak Load

**Peak scenario**: All workers firing simultaneously:
- LocWorker doing 5× inference (model loaded in RAM)
- FwWorker in middle of HTTP call
- 2 DetWorkers processing
- Main process in loop iteration

Memory at peak: ~2.5 GB. **Safe.** No swap (containers rarely have swap). The 1.5 GB headroom is the OOM buffer.

**Risk factors:**
- If `LOC_WORKERS=2` → high OOM risk (P1)
- If model file is significantly larger than 1.1GB → OOM risk
- If `N_CTX` is large (8192+) → increased context memory in llama.cpp

### 1.4 Memory Budget Formula

```
Safe workers within 4GB:
  Max 1 × LocWorker (1.6GB fixed)
  + up to 2 × FwWorker (100MB each)
  + up to 2 × DetWorker (100MB total)
  = ~2.0 GB workers + ~0.5 GB overhead = 2.5 GB
  Margin: 1.5 GB
```

**No in-code enforcement.** The `ReadyConfig.from_env()` reads values blindly from env vars. Nothing prevents `STAGING_LOC_WORKERS=3`.

---

## 2. Startup Time

### 2.1 Timeline

```
 0.0s — entrypoint starts
 0.1s — config loaded, tasks read
 0.5s — bulk classify (19 tasks: ~0.5ms/task)
 1.0s — queue built
 1.5s — pool.start() called, 4 processes spawned
          FW worker: initialize() → FireworksSolver() instant (0.01s)
          Det workers: initialize() → import solvers (0.1s)
          LocWorker: initialize() → Llama(model) → ~8-12s
12.0s — LocWorker ready, first tasks being processed
```

**Startup overhead: ~12s out of 600s.** Acceptable (2%).

**Deadline adaptation triggers at 60s remaining (t=540s). Effective processing window: 528s.**

### 2.2 Worker Initialization Failure

**ImportError** during `_build_worker_plan()`:
- Caught at pool level per worker type
- That worker type is simply skipped
- Pool continues with remaining workers
- ✅ Graceful

**Exception during `initialize()`** inside worker process:
```python
# ready_worker.py:90-92
except Exception as exc:
    logger.error("[%s] Initialization failed: %s", self.worker_id, exc)
    return  # worker process exits
```
- Worker process dies silently
- Pool never notices (no `is_alive()` check)
- `busy_flag` stays at 0 (safe, worker never marked busy)
- Tasks that would have been processed by this worker go to other workers
- ✅ Graceful for task processing, but pool's `available_count` still counts the dead worker

### 2.3 Model File Missing

`LocWorker.initialize()`:
```python
self._llm = Llama(model_path=self.config.loc_model_path, ...)
```
- `Llama()` raises `FileNotFoundError` if model file doesn't exist
- Caught by generic `except Exception` in `worker.run()`
- Worker exits; logged but not detected by pool
- ✅ Graceful (other workers handle tasks)

---

## 3. Worker Crash Recovery  ⚠️ P0 Issues

### 3.1 Worker Process Dies Mid-Task

**Scenario**: Worker pulls task, starts `process()`, then OOM-killed or segfaults.

**What happens:**
1. Worker process terminates (SIGKILL from OOM, SIGSEGV, etc.)
2. No exception handler fires (can't catch a kill signal)
3. `busy_flag` stays at `1` (stuck forever)
4. Task is removed from the shared queue (pulled by worker) but never completed
5. No results pushed to `results_queue`
6. Pool never checks `p.is_alive()`
7. `judge._task_answers` never gets entries for this task
8. `judge.total_judged` never reaches `total_tasks`
9. Pool loop continues until deadline
10. `judge_all()` iterates `_task_answers` — **task not there → silently skipped**
11. **Task is missing from output** — grader never sees it

| Risk | Severity | Likelihood | 
|------|----------|------------|
| **Orphaned task silently dropped** | **Critical** | **Possible** (OOM, segfault in llama.cpp) |

### 3.2 No Watchdog

The pool has `self._processes: list[multiprocessing.Process]` but never checks `p.is_alive()` in the dispatch loop. Dead workers:
- Are not detected
- Are not restarted
- Are counted in `available_count` (if busy_flag=0) or block it (if busy_flag=1)
- Stay in the process list forever

**No health check, no restart, no replacement.**

### 3.3 `busy_flag` Stuck at 1

If a crash happens while `busy_flag.value = 1` (during `process()` call):
- The `finally: self.busy_flag.value = 0` never executes
- Flag stays 1 forever
- Pool's logging shows "X busy workers" permanently
- `available_count` property under-reports available workers

This is informational only (no dispatch logic relies on it), but it's a symptom of the deeper watchdog gap.

### 3.4 Orphaned Task Chain

With `judgment_votes=5` and a worker crash mid-5-tries:
- If 2 answers were submitted and worker crashes before the remaining 3
- The `_push_results` is never called (crashed in process)
- Actually, looking at the code flow: answers are APPENDED to a list in `process()`, then returned. If the crash happens before the return, NO answers are submitted.
- So patterns: all 5 answers submitted, or 0 answers submitted. No partial submission.
- This means a crash during processing loses ALL 5 answers for that task.
- **The judge never sees this task.**

---

## 4. Fireworks API Reliability

### 4.1 Missing API Key

**Code path:**
1. `ReadyConfig.fw_api_key = os.environ.get("FIREWORKS_API_KEY", "")` → empty string
2. `FwWorker.initialize()` → `FireworksSolver(api_key="")` → logs warning
3. `FwWorker.process()` → calls `solver.solve()` → HTTP 401 → `HTTPError` raised → caught as generic `Exception` → returns `""` for each try
4. All 5 answers are empty strings
5. Judge treats them as degenerate → other workers' results used

✅ **Graceful degradation.** However, the FwWorker still pulls tasks from the queue and wastes 5× timeout trying and failing. With a missing API key, it would be better to skip FwWorkers entirely or make them return instantly.

### 4.2 Rate Limiting (429s) — ❌ No Exponential Backoff

The `FireworksSolver.solve()` in `agent/solvers/fireworks.py`:
```python
except urllib.error.HTTPError as e:
    logger.error(...)
    raise  # re-raised immediately — NO retry, NO backoff
```

Handling in `FwWorker.process()`:
```python
except Exception as exc:
    logger.warning(...)
    answer = ""  # empty string for that try
```

**On a 429**: the try returns "" immediately. With 5 sequential tries, if the API is rate-limited, all 5 could fail. The worker wastes no time (exception is fast) but contributes no useful answers.

**Consequence under sustained rate limiting**: All FW workers return empty answers. Load shifts to LocWorker and DetWorkers. Since each worker produces 5 answers per task, a single worker can still provide 5 answers. But those 5 answers are from ONE model (same worker) instead of diverse workers.

| Risk | Severity | Likelihood |
|------|----------|------------|
| All 5 FW tries fail under rate limiting | Medium | Possible (if Fireworks enforces aggressive rate limits) |
| No backoff can cause cascading retries under overload | Low | Unlikely with small task counts |

### 4.3 Network Timeouts

The 30s per-try timeout is passed to `urllib.request.urlopen(req, timeout=timeout)`. With 5 sequential tries, a single task could take up to 150s from one worker. If ALL API calls time out:
- 5 × 30s = 150s wasted per task from FW workers
- With 100 tasks and 1 FW worker: 100 × 150s = 15,000s (impossible within 600s)
- But only ~4 tasks get pulled by FW worker in practice (time budget limits throughput)
- Other workers process remaining tasks

**Acceptable** — slow network just reduces FW's contribution, it doesn't block the system.

### 4.4 Fireworks Down

If Fireworks returns 5xx for all requests:
- FW workers return empty answers for all 5 tries
- Remaining workers (LocWorker + DetWorkers) handle all tasks
- Each remaining worker produces 5 answers per task (temperature sweep)
- Judge still works with 5 same-model answers

✅ **Graceful degradation**. No single point of failure.

---

## 5. Deadline Edge Cases  ⚠️

### 5.1 Extremely Short Deadline (<60s)

**What happens:**
1. `deadline = time.monotonic() + short_value`
2. Pool loop starts, `remaining <= 60` immediately
3. Force-judging kicks in: any task with ≥1 answer is judged immediately
4. But no tasks have completed yet (startup ~12s)
5. No tasks are force-judged (none have even 1 answer)
6. Loop continues until `remaining <= 0`
7. At deadline: force-judge pending tasks again — still none
8. `judge_all()` iterates `_task_answers` — **empty**
9. **Results: empty list — grader sees 0 answers**

| Risk | Severity | Likelihood |
|------|----------|------------|
| Tasks dispatched but not yet completed are **silently dropped** from output | **Critical** | **Possible** (if deadline < startup time + first inference) |

**This is the same orphaned-task bug as section 3.1**, triggered by deadline instead of crash.

### 5.2 DEADLINE_S = 0

`ReadyConfig.deadline_s = float(os.environ.get("DEADLINE_S", "600"))`

If grader sets `DEADLINE_S=0`: deadline is immediate. Pool loop immediately hits `remaining <= 0` → no tasks judged → empty output.

The grader likely doesn't do this, but if it does, the system produces empty results with no warning other than a log message.

### 5.3 SIGTERM Before Output Write

**Flow:**
```python
pool.dispatch_loop(queue, judge, deadline)
# <--- SIGTERM here --->
final_results = judge.judge_all()
output = [...]  # strip _judgment
_write_output(output)  # atomic write
pool.shutdown()
```

If the container orchestrator sends SIGTERM at the 600s timeout:
- If dispatched_loop hasn't broken out yet (deadline check), the SIGTERM handler is not installed in the main process
- Main process dies → daemon workers are killed
- `_write_output` NEVER CALLED
- No `/output/results.json` → **grader sees missing file → error**

| Risk | Severity | Likelihood |
|------|----------|------------|
| Output file never written if SIGTERM arrives before write | **High** | **Possible** (deadline timing race) |

**Mitigation in the loop**: The deadline check (`remaining <= 0 → break`) should trigger BEFORE the orchestrator's 600s timeout, since the code computes `deadline = monotonic() + config.deadline_s`. So the break should happen ~1-5ms before SIGTERM. But this is a race with no safety margin.

### 5.4 Worker Mid-5-Tries and SIGTERM

`_handle_sigterm` sets `self._running = False`. The main loop checks `_running` at the top. The current `process()` call runs to completion (all 5 tries finish) before the loop re-checks. So SIGTERM during `process()` does NOT interrupt the current task — all 5 answers are produced. ✅

But if `process()` takes too long (e.g., each try blocks on I/O for 30s), the worker ignores SIGTERM for up to 150s. The pool's `shutdown(timeout=5.0)` would need to `terminate()` or `kill()`.

---

## 6. Queue Overflow / Backpressure

### 6.1 multiprocessing.Queue Size Limit

`multiprocessing.Queue()` with no maxsize argument → unbounded (limited by system memory).

- 100 tasks as `ReadyTask` objects: ~50KB (mostly the prompt text)
- No explicit limit → no `queue.Full` risk

✅ **Low risk.**

### 6.2 Results Queue Backup

Also unbounded. With 100 tasks × 4 workers × 5 answers = 2000 results. Each result dict ≈ 512 bytes → ~1MB. Fine.

### 6.3 Worker Blocks on Full Queue → Deadlock

Since queues are unbounded, `put_nowait()` never blocks or raises `Full` (the underlying `multiprocessing.Queue` uses a buffer thread that never fills). **Deadlock risk is theoretical only.** But if someone adds `maxsize=...` later without handling `queue.Full`, workers would crash → see section 3.1.

---

## 7. Output Integrity

### 7.1 Atomic Write Pattern

```python
tmp = "/output/results.json.tmp"
with open(tmp, "w") as f:
    json.dump(results, f, ensure_ascii=False)
os.replace(tmp, "/output/results.json")
```

`os.replace()` is atomic on POSIX (same-filesystem rename). If killed between write and replace:
- `.tmp` file exists but `results.json` doesn't
- Grader looking for `results.json` finds nothing → error

**Window size**: microseconds. **Risk**: very low, but combined with the SIGTERM timing issue (section 5.3), the container could be killed before ANY write attempt, leaving no output.

### 7.2 Extra _judgment Metadata

The entrypoint correctly strips `_judgment`:

```python
output = [{"task_id": r["task_id"], "answer": r["answer"]} for r in final_results]
```

The internal `judge_all()` includes `_judgment` in its dicts, but the output is stripped before writing. ✅ **Clean contract.**

---

## 8. Architectural Gaps Found

### 8.1 ⚠️ CRITICAL: Single-Worker Architecture, Not 5-Worker Plan

**The design document describes:**
> 5 different workers (fw_1, fw_2, loc_1, loc_2, det_1) each process the same task once, producing 5 diverse answers from different models.

**The code implements:**
> Each task is pulled by ONE worker from the shared queue. That worker generates 5 answers via temperature sweep (0.1→0.9). All 5 answers come from the same model/system.

**Consequences:**
- No cross-model diversity in voting
- Model bias affects all 5 answers consistently
- Judge aggregates same-model temperature variants, which are highly correlated
- This undermines the entire "majority vote" premise of the design

### 8.2 ⚠️ HIGH: No Health Monitoring

- Pool never calls `p.is_alive()`
- Dead workers are not detected
- No watchdog or heartbeat mechanism
- Workers with stuck `busy_flag=1` permanently pollute availability tracking

### 8.3 MEDIUM: `queue.Queue` vs `multiprocessing.Queue` Confusion

`ReadyQueue` uses `queue.Queue` (thread-safe only), but it's only used in the main process before workers start. The pool transfers to `multiprocessing.Queue`. This works but is fragile — if someone later uses `ReadyQueue` across processes, it will silently break.

### 8.4 LOW: Classifier Divergence

`ready_classifier.py` is a simplified pure-regex classifier, NOT the same as `agent.category_filter`. Classification results may differ from the main pipeline, causing suboptimal worker routing. This affects performance but not correctness.

---

## 9. Recommendations

### P0 — Must Fix Before Deployment

| # | Issue | Recommendation | Files to Change |
|---|-------|----------------|-----------------|
| 1 | **Orphaned tasks lost on worker crash** | Add `is_alive()` check in dispatch loop. If a worker dies, requeue its task and remove it from pool. | `ready_pool.py` |
| 2 | **Orphaned tasks lost at deadline** | Track all dispatched tasks (`_pending_dispatched`). At deadline, iterate this set and produce empty answers for any task not in `judge._task_answers`. | `ready_pool.py`, `entrypoint.py` |
| 3 | **SIGTERM before output write** | Install SIGTERM handler in main process that writes output immediately. Also use `atexit.register`. | `entrypoint.py` |
| 4 | **`busy_flag` stuck on crash** | Use `multiprocessing.Queue` with sentinel/heartbeat pattern, or use `Process.exitcode` to detect crashes. Remove dead workers from `_busy_flags`. | `ready_pool.py`, `ready_worker.py` |

### P1 — High Priority

| # | Issue | Recommendation | Files to Change |
|---|-------|----------------|-----------------|
| 5 | **Single-worker 5-try instead of 5-worker diversity** | The fundamental architecture should be: dispatch ONE task to MULTIPLE workers (not one worker does 5 tries). Each worker does 1 try. Collect 1 answer per worker, up to 5 workers. | `ready_pool.py`, `ready_worker.py`, `loc_worker.py`, `fw_worker.py`, `det_worker.py` |
| 6 | **No memory budget enforcement** | Add a check in `ReadyConfig.from_env()` (or `ReadyPool.start()`) that warns/caps total workers based on 4GB budget. Max 1 LocWorker. | `ready_config.py` or `ready_pool.py` |
| 7 | **No watchdog for worker health** | Add periodic `p.is_alive()` check in dispatch loop (every 10 seconds). If a worker died: log, update `_processes`, remove from `_busy_flags`. | `ready_pool.py` |
| 8 | **FW worker wastes time with no API key** | If `fw_api_key` is empty, skip creating FW workers entirely in `_build_worker_plan()`. | `ready_pool.py` |
| 9 | **Output file not written under SIGTERM race** | Write output to a known path EARLY (even if empty), then overwrite. The grader can at least find the file. | `entrypoint.py` |

### P2 — Medium Priority

| # | Issue | Recommendation | Files to Change |
|---|-------|----------------|-----------------|
| 10 | **No exponential backoff for 429s** | Add retry with exponential backoff in `FireworksSolver.solve()` for HTTP 429 and 5xx. | `agent/solvers/fireworks.py` (outside staging but critical) |
| 11 | **Deadline=0 edge case** | Cap `deadline_s` at minimum 30s if set below that. Add fast-path: if deadline < startup budget, skip workers and produce fallback answers. | `ready_config.py`, `entrypoint.py` |
| 12 | **Race: tasks enqueued after deadline** | In `dispatch_loop`, check `remaining > 10` before starting a new batch. If remaining is too small, skip dispatch and just judge what's available. | `ready_pool.py` |
| 13 | **`queue.Queue` in ReadyQueue is not process-safe** | Either document that ReadyQueue is main-process-only, or replace with `multiprocessing.Queue`. | `ready_queue.py` |
| 14 | **Worker processes count in `available_count` after death** | Remove dead workers from tracking lists so `available_count` is accurate. | `ready_pool.py` |
| 15 | **LocWorker crash during initialize leaves no log in pool** | The pool should capture worker stderr or check exit codes to surface initialization failures. | `ready_pool.py` |
| 16 | **No timeout on worker `initialize()`** | If `Llama()` hangs (e.g., corrupt file), the worker never signals readiness. Add a startup timeout in `worker.run()`. | `ready_worker.py` |

---

## 10. Corrected Architecture: Multi-Worker Dispatch

The single most impactful change is converting from "1 worker does 5 tries" to "5 workers each do 1 try" (issue P1-5). This preserves the plan's diversity goal:

**Current (broken diversity):**
```
Task → queue → Worker_A pulls it
  Worker_A: try_1 (T=0.1), try_2 (T=0.3), ..., try_5 (T=0.9)
  All 5 answers from same model → highly correlated
```

**Required (real diversity):**
```
Task → queue → Worker_A pulls it → try_1 (answer 1)
Task → queue → Worker_B pulls it → try_1 (answer 2)  
Task → queue → Worker_C pulls it → try_1 (answer 3)
...
Judge collects 5 answers from 5 different workers/models
```

To implement: each worker does 1 try per task, and the task must be enqueued multiple times (or the pool dispatches the same task to N workers). This requires tracking which workers have processed which task to avoid duplicate work by the same worker.

---

## 11. Priority Summary

| Priority | Count | Key Items |
|----------|-------|-----------|
| **P0** | 4 | Orphaned tasks (crash), orphaned tasks (deadline), SIGTERM output loss, stuck busy_flag |
| **P1** | 5 | Single-worker architecture, no memory enforcement, no watchdog, API key waste, output race |
| **P2** | 7 | No backoff, deadline=0 edge case, late enqueue race, process-safe queue, tracking, init failures, startup timeout |

---

## Appendix: Default Env Var Safety Check

| Env Var | Default | Safe? | Notes |
|---------|---------|-------|-------|
| `STAGING_FW_WORKERS=1` | 1 | ✅ | Single FW worker ~80MB |
| `STAGING_LOC_WORKERS=1` | 1 | ✅ | ~1.6GB — don't increase |
| `STAGING_DET_WORKERS=2` | 2 | ✅ | ~100MB total |
| `STAGING_JUDGMENT_VOTES=5` | 5 | ✅ | Per-task target |
| `STAGING_WORKER_TIMEOUT=30` | 30 | ✅ | Per-try timeout |
| `N_THREADS=2` | 2 | ✅ | Matches 2 vCPU |
| `N_CTX=2048` | 2048 | ✅ | Reasonable for 1.5B model |
| `N_GPU_LAYERS=0` | 0 | ✅ | CPU-only |
