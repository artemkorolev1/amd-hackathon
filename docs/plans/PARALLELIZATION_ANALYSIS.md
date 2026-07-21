# Staging Pool Parallelization Model — Analysis

> Based on source code audit of `staging/ready_pool.py`, `staging/ready_worker.py`,
> `staging/ready_config.py`, `staging/entrypoint.py`, `staging/ready_judge.py`,
> `staging/workers/loc_worker.py`, `staging/workers/det_worker.py`,
> `staging/workers/fw_worker.py`, `Dockerfile`, `Dockerfile.gpu`, and `Dockerfile.staging`.

---

## 1. Does the current architecture match the user's mental model?

**Yes, the core shape matches, but with three meaningful gaps.**

| Mental Model | Actual Implementation | Match? |
|---|---|---|
| Shared task pool | `multiprocessing.Queue` (line 47, `ready_pool.py`) | ✅ |
| Workers pull autonomously | `_pull_task()` loop (lines 171–199, `ready_worker.py`) | ✅ |
| Workers grab tasks freely | `task_pool.get_nowait()` — no dispatch coordinator | ✅ |
| "Up to 4 workers" | Default spawns **6 workers** (2 Det + 3 Loc + 1 FW) | ❌ Off by 2 |
| Workers grab one, finish, grab next | Each worker processes **5 sequential tries** per task before releasing | ⚠️ Subtask is true, but a task occupies a worker for 5× the single-inference time |
| Fast workers handle more | No actual work-stealing implementation exists (see Q4) | ❌ Steal is a no-op |

The ready-barrier in `start()` (lines 151–157 of `ready_pool.py`) is a positive addition not
explicitly in the mental model: the monitor waits up to 30 s for all workers to signal
`ready_flag=1` before releasing any to pull. Without this barrier, fast-initializing workers
(DetWorker, FwWorker, ~0 s) would drain the pool before slow-initializing LocWorkers
(~1–3 s model load) are ready to compete.

---

## 2. Maximum safe worker count

### CPU submission container (2 vCPU, 4 GB RAM, `Dockerfile.staging`)

| Process | Memory estimate | CPU impact |
|---|---|---|
| Main process (monitor + judge) | ~80 MB | Negligible |
| DetWorker (×2) | ~10 MB each (solvers only) | Sub‑ms per call |
| FwWorker (×1) | ~20 MB (HTTP client) | Network‑bound, near‑zero CPU |
| LocWorker (×1), 1.5B Q4_K_M GGUF | ~1,300 MB (1,100 MB mmapped model + ~200 MB runtime/KV cache) | ~1 CPU thread per inference |
| LocWorker (×2) | ~2,600 MB | 2 CPU threads → saturates both vCPUs |
| LocWorker (×3) | ~3,900 MB | Over 4 GB → OOM risk |

**Safe maximum for 4 GB RAM / 2 vCPU: 2 LocWorkers total**, preferably 1 LocWorker
to leave headroom. Mix recommendation in Q6.

**Safe maximum for 8 GB VRAM / RTX A4000 (GPU dev, `Dockerfile.gpu`):**
- Each LocWorker with `n_gpu_layers=-1`: ~1.1 GB VRAM (model) + ~300 MB (KV cache, CUDA overhead) ≈ 1.4 GB/worker
- **3 LocWorkers = ~4.2 GB VRAM** — comfortably within 8 GB
- Add DetWorkers + FwWorker — they use zero VRAM
- **Safe maximum: 3 LocWorkers + extras**, limited by the GPU's compute capacity
  (llama.cpp serializes per‑process GPU work; 3 concurrent processes sharing GPU via
  CUDA streams may contend for compute units)

---

## 3. Bottlenecks — workers idle while tasks wait

### a) No category filtering on `task_pool.get_nowait()` — wasted pulls

The `category_whitelist` passed to each worker is **only used in `_attempt_steal()`** (line 212,
`ready_worker.py`). The primary `task_pool.get_nowait()` call (line 192) has **no category filter**.
A DetWorker that pulls a `code_gen` task has no solver for it and returns 5 empty answers.
The task is consumed from the pool but produces worthless output.

**Impact:** Up to 37.5 % of task-pool pulls could be wastes on a mixed workload,
depending on the category distribution.

### b) Sequential 5‑try processing blocks the worker

Each worker runs `judgment_votes=5` sequential LLM calls per task (e.g., `loc_worker.py`
lines 79–106). The worker is `busy_flag=1` for the entire duration. A single task
can block a LocWorker for ~5 s. During that time, no other task can be picked up by
that worker. This is by design, but it means:
- 6 workers × 1 task each = 6 tasks in flight max
- Duration of each worker's busy period = 5× single‑inference time
- Fast workers (Det, ~1 ms total) finish and go idle while LocWorkers are still churning

### c) Ready barrier delays fast workers

The `ready_flag` barrier in `pool.start()` (lines 151–157) blocks **all workers** until the
slowest initializes. With LocWorkers taking ~1–3 s each to load GGUF models, DetWorkers
and FwWorker sit idle for that time. This is a deliberate trade‑off to prevent the fast
workers from draining the pool, but it adds latency to first-task pickup.

### d) Results‑queue dual consumption race

Both the **judge daemon thread** (`judge.consume_loop`, `entrypoint.py` line 268) and the
**monitor loop** (`pool.monitor_loop` line 208) call `results_queue.get_nowait()`.
`multiprocessing.Queue` is thread/process‑safe, but the split consumption means answers
may arrive at the judge in a bursty pattern — some consumed by the 5‑s monitor tick,
others by the 50‑ms judge poll. This is not a correctness bug, but it makes timing analysis
harder and could mask true processing delays.

### e) Steal protocol is a no‑op (see Q4) — idle workers stay idle

After the shared `task_pool` is empty, workers who finish early cannot redistribute work
from still‑busy workers. They spin with 0.2 s `sleep()` + 0.5 s steal‑request timeout,
doing nothing.

---

## 4. Does pull‑based dispatch (+ work stealing) achieve the described behavior?

**Partially. The pull dispatch works; the work stealing is a stub with no implementation.**

### Pull dispatch — works correctly
Workers call `_pull_task()` in a loop:
```
1. stolen_queue.get_nowait()   ← never has items (see below)
2. inbox_queue.get_nowait()    ← unused in current workflow
3. task_pool.get_nowait()      ← the real workhorse
4. steal_request_queue.put() + stolen_queue.get(timeout=0.5)
```
Steps 3 is effective. Workers drain the shared pool in order of their arrival
(which correlates with initialization speed). Fast workers (DetWorker) finish their
tasks quickly and come back for more, achieving a **de facto load‑balanced distribution**
as long as tasks remain in the pool.

### Work stealing — incomplete mechanism

The steal protocol in `_attempt_steal()` (lines 201–222, `ready_worker.py`):
1. Posts a steal request to `steal_request_queue`
2. Waits 0.5 s on its own `stolen_queue` for a response

**No code exists that reads `steal_request_queue` and pushes tasks into any `stolen_queue`.**
The `steal_request_queue` is created in `pool.__init__()` (line 48) and workers write to it,
but neither the monitor loop nor any other component consumes it. The `stolen_queue`
per‑worker queues likewise receive no writes.

**Consequence:** After the shared pool is empty, every worker goes through:
- `task_pool.get_nowait()` → `_queue.Empty` → `_attempt_steal()`
- Posts a steal request (ignored) → `stolen_queue.get(timeout=0.5)` → timeout → `None`
- Sleeps 0.2 s → repeats

This wastes 0.5–0.7 s per idle cycle per worker. With 6 workers idle, that's ~3–4 s
aggregate wasted wall time per cycle. For a 600‑s deadline, this overhead is negligible,
but it means the steal mechanism does **nothing** to help fast workers pick up slack
when slow workers are stuck on long tasks.

### Does the pull dispatch achieve "fast workers naturally handle more tasks"?

**Yes, for as long as the shared `task_pool` has items.** DetWorkers (~1 ms/task) will
cycle through many tasks while a single LocWorker is still crunching its first task.
This works correctly without any central dispatch logic.

When the pool is empty and some workers are still busy, fast workers idle. This is where
the user's mental model expects work stealing to redistribute tasks from busy workers,
but it doesn't happen.

---

## 5. Realistic max concurrent tasks in flight

| Scenario | Workers | Concurrent tasks in flight | Bottleneck |
|---|---|---|---|
| CPU submission (current: 6 workers) | 2 Det + 3 Loc + 1 FW | **4** (2 "heavy" LLM + 2 light) | 2 vCPU saturated by LocWorkers; 3 LocWorkers OOM |
| CPU submission (recommended: 4 workers) | 2 Det + 1 Loc + 1 FW | **4** (1 heavy + 3 light) | LocWorker saturates 1 CPU core |
| CPU submission (alt: 4 workers) | 2 Det + 2 Loc | **3** (2 heavy + 1 light) — but RAM at limit | 2 LocWorkers saturate both CPU cores |
| GPU dev (3 Loc + 1 FW + 2 Det) | 3 Loc + 1 FW + 2 Det | **6** (3 heavy on GPU + 3 light) | VRAM and GPU compute shared |

"Concurrent tasks in flight" = number of workers that currently hold a task from the pool.
Each such task is being processed with 5 sequential tries by that one worker.

**Key insight:** The 5‑try-per-task model means a task does NOT receive parallel processing
across workers. One worker owns one task and produces all 5 answers. The "in flight" count
is therefore exactly the number of workers that have successfully pulled a task.

**Practical ceiling for CPU submission: 4.** Limited by CPU cores (2) and RAM (4 GB).
You can have 6 workers alive, but only ~2 can make meaningful LLM progress simultaneously.

---

## 6. Cap worker count to 4 for CPU submission?

**Yes.** The default of 6 workers (2 Det + 3 Loc + 1 FW, `ready_config.py`) is
unsafe for the grader container's 4 GB RAM — 3 LocWorkers alone need ~4.5 GB.

### Recommended mix: 2 Det + 1 Loc + 1 FW

| Worker | Count | Rationale |
|---|---|---|
| DetWorker | 2 | Covers `math`, `factual`, `sentiment`, `summarization`, `ner` — near‑instant, zero CPU |
| LocWorker | 1 | Covers `code_gen`, `code_debug`, `logic` — uses ~1.3 GB, one CPU thread |
| FwWorker | 1 | Covers all categories, handles escalation — network‑bound, zero CPU |

**Memory estimate:** ~1,500 MB (LocWorker) + 30 MB (Det×2) + 20 MB (FW) + 80 MB (main) ≈
**1.6 GB total** — well within 4 GB with margin for OS and buffering.

**CPU impact:** Only the single LocWorker contends for a CPU core during inference.
DetWorkers are sub‑ms; FwWorker is IO‑bound. The 2 vCPUs are sufficient.

### Why not 2 Det + 2 Loc?

- 2 LocWorkers = ~2.6–3.0 GB RSS. With main process + overhead, total approaches 3.5 GB.
  Tight and leaves no headroom for OS page cache, tmp files, or memory spikes.
- 2 LocWorkers saturate both vCPUs, leaving zero room for DetWorker calls or the
  monitor/judge to run promptly. Latency spikes are likely.
- Only justified if the workload is dominated by categories that NEED local inference
  (`code_gen`, `code_debug`, `logic`) AND FW API is unavailable.

### Config change needed

The default `loc_workers` count is computed from `len(loc_model_configs)` (line 114,
`ready_config.py`). To deploy 1 LocWorker while keeping 3 model configs available for
GPU dev, the submission image should set `STAGING_LOC_WORKERS=1` (hardcoded in env)
or filter `loc_model_configs` to one entry. The simplest fix for CPU submission:

```yaml
# In Dockerfile.staging env:
ENV STAGING_LOC_WORKERS=1
ENV STAGING_LOC_MODEL_CONFIGS='[{"id": "qwen2.5-instruct", "path": "/models/qwen2.5-1.5b-instruct-q4_k_m.gguf", "categories": ["factual","logic","math","summarization","code_debug","ner","code_gen","sentiment"]}]'
```

For GPU dev (`Dockerfile.gpu`), keep the default 3‑model config for maximum diversity.

---

## 7. Deadlock risks with shared queue + steal protocol

### Risk 1: 🔴 `results_queue.put_nowait()` can crash a worker

In `_push_results()` (line 240, `ready_worker.py`), `results_queue.put_nowait(a)` is called
without a `try/except`. Python's `multiprocessing.Queue` is backed by a `Pipe` + buffer;
if the buffer fills up faster than the judge can drain it (e.g., all 6 workers finish near‑simultaneously
while the judge is in a 50‑ms sleep), `put_nowait` raises `queue.Full`. This propagates up
to `run()` (line 137–147) into the outer `except Exception` block, which logs but does not
re‑enqueue the task. **The task is lost from the pool.**

**Mitigation needed:** Wrap `results_queue.put_nowait` in a `try/except _queue.Full` with
retry logic or a timeout‑based `put()` fallback.

### Risk 2: 🟡 Worker death → task permanently lost

When a worker dies (crash, OOM kill, segfault), the `ReadyTask` object it pulled from
`task_pool` is gone (consumed from the `multiprocessing.Queue`). The monitor detects the
death (line 190, `ready_pool.py`) and logs orphaned task IDs, but the re‑enqueue comment
on line 314 says explicitly:

> "Best-effort re-enqueue: we can't recover the task object because it was consumed by
> the worker."

The task must be re‑created from the original input. This is not implemented. The deadline
drain (line 248–253) produces empty answers for incomplete tasks. **This is acceptable
for the hackathon (a crashed worker is exceptional) but is a data‑loss path.**

### Risk 3: 🟡 Steal‑request queue memory leak

`steal_request_queue` receives `put_nowait()` calls from every worker every time it becomes
idle with an empty `stolen_queue`. Since nothing reads from this queue, requests accumulate
indefinitely. `multiprocessing.Queue` is unbounded (maxsize=0), so this is a slow memory
leak rather than a block. At ~200 bytes per request × 6 workers × ~10 idle cycles/s × 600 s =
~7 MB leaked per run — negligible, but sloppy.

### Risk 4: 🟢 Spurious wake‑up loop on empty pool

When `task_pool` is empty, all workers spin: `get_nowait()` → `Empty` → `_attempt_steal()` →
`timeout` → `sleep(0.2)` → repeat. This is a busy‑wait pattern (0.2 s sleep makes it tolerable),
not a deadlock. But it means **100 % of workers can be "idle" (spinning in sleep/steal‑timeout)
while the few busy workers are still processing tasks.** The steal mechanism should redistribute
tasks but doesn't.

### Risk 5: 🟢 Daemon thread abrupt termination

The judge thread is a `daemon=True` thread (entrypoint.py line 270). If the main process
exits without cleanly stopping it, results still in `results_queue` are lost. The `finally` block
(lines 279–284) sets `stop_event`, joins with 2‑s timeout, and does a final drain — this is
well‑handled.

### Summary

| Risk | Severity | Impact | Needs Fix? |
|---|---|---|---|
| `put_nowait` crash on results_queue | 🔴 High | Task + answers lost, worker crashes | Yes — add try/except with retry |
| Worker death → task lost | 🟡 Medium | 1 task produces empty answer per crash | Acceptable for hackathon |
| Steal‑request memory leak | 🟢 Low | ~7 MB per run | Clean up eventually |
| Idle spin on empty pool | 🟢 Low | ~700 ms wasted per idle cycle per worker | Fix steal mechanism to unblock |
| Daemon thread loss | 🟢 Low | Edge case on abrupt exit | Already handled |

### Recommended fixes (in priority order)

1. **Wrap `results_queue.put_nowait()` in try/except `queue.Full`** — retry with
   `results_queue.put(timeout=1.0)`. Prevents worker crashes.
2. **Implement the steal‑response side** — add a monitor‑loop tick that reads
   `steal_request_queue` and, for each request, checks if any busy worker has a task
   exceeding a duration threshold (e.g., >10 s), then pushes that task (via `inbox_queue`
   or `stolen_queue`) to the requesting idle worker. Or simpler: just have workers
   re‑check `task_pool` more aggressively without the broken steal dance.
3. **Limit default workers to 4 for CPU submission** — change `ready_config.py` defaults
   or `Dockerfile.staging` env vars. The current 6-worker default is unsafe for 4 GB RAM.

---

## Appendix: Category dispatch coverage matrix

| Category | DetWorker | LocWorker (qwen2.5‑instruct) | LocWorker (qwen2.5‑coder) | LocWorker (gemma‑3) | FwWorker |
|---|---|---|---|---|---|
| math | ✅ solve_arithmetic | ✅ | ❌ | ❌ | ✅ |
| factual | ✅ solve_factual_qa | ✅ | ❌ | ❌ | ✅ |
| sentiment | ✅ solve_sentiment | ✅ | ❌ | ✅ | ✅ |
| summarization | ✅ solve_summarization | ✅ | ❌ | ❌ | ✅ |
| code_debug | ✅ solve_code_debugging | ✅ | ✅ | ✅ | ✅ |
| ner | ❌ | ✅ | ✅ | ❌ | ✅ |
| code_gen | ❌ | ❌ | ✅ | ✅ | ✅ |
| logic | ❌ | ✅ | ❌ | ❌ | ✅ |

A single LocWorker using the `qwen2.5-instruct` model covers 6 of 8 categories.
The `code_gen` and `ner` categories are not covered by `qwen2.5-instruct` (per config
whitelist), so they must go to FwWorker. Two LocWorkers (instruct + coder) cover all 8.

---

*Analysis produced from source code audit on 2026-07-13.*
