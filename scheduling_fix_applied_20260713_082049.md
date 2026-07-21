# Scheduling Fix: Per-Worker-Type Task Partitioning

**Date:** $(date)
**Author:** Hermes Agent (automated fix)

## Problem

When `det` (deterministic) workers and `local` (GPU) workers share the same
category task pools in `staging/ready_pool.py`, the `det` workers process
tasks in microseconds while `local` models take ~500ms per task. This causes
`det` workers to drain **all** category pools before `local` workers get a
chance to pull any tasks.

The existing judge fix (`ready_to_judge` waiting for local answers) doesn't
help because local models never receive tasks in the first place.

## Root Cause

`staging/ready_pool.py` — `ReadyPool.start()` drained all tasks into shared
per-category pools (`_category_pools: dict[str, multiprocessing.Queue]`).
All worker types pulled from the same pools. Fast workers consumed
everything before slow workers could react.

## Solution: Option C — Round-Robin Partition with Backstop

Three files were modified to implement per-worker-type task partitioning:

### 1. `staging/ready_config.py` — New configuration parameter

```python
reservation_timeout_s: float = 30.0  # env: STAGING_RESERVATION_TIMEOUT_S
```

Controls how long a task waits in its reserved worker-type pool before being
released to the shared overflow pool (backstop).

### 2. `staging/ready_pool.py` — Core partitioning logic

**Before:**
- `_category_pools: dict[str, Queue]` — single per-category pools shared by all
- `start()` — tasks drained from ReadyQueue into shared `_category_pools`
- `monitor_loop()` — health checks, orphan re-enqueue, deadline broadcast

**After:**
- `_type_pools: dict[str, dict[str, Queue]]` — per-worker-type per-category pools
- `_task_type: dict[str, str]` — tracks which worker_type each task is assigned to
- `_task_enqueued: dict[str, float]` — timestamp for backstop timeout calculation
- `_task_map: dict[str, ReadyTask]` — keeps task references for backstop release
- `_backstop_released: set[str]` — prevents duplicate release to overflow

**Partition algorithm (`start()`):**
1. Collect distinct worker types from the worker plan
2. Create `_type_pools[wtype][cat] = Queue()` for each worker type × category
3. Spawn workers, each receiving ONLY its own type's pools as `category_pools`
4. Drain ReadyQueue tasks round-robin using `itertools.cycle(worker_types)`
5. Each task goes into exactly one type's pool — fast workers cannot touch
   other types' reserved tasks

**Backstop algorithm (`monitor_loop()` step 2b):**
1. Every monitor cycle, check for tasks still in `_pending_map` with
   `assigned == None` (not yet pulled by any worker)
2. If a task has been waiting longer than `reservation_timeout_s` (30s),
   release it to the shared overflow pool (`self.task_pool`)
3. Any worker can then pull it from overflow, ensuring no task gets stuck
   indefinitely due to a slow/unavailable worker type

### 3. `staging/ready_worker.py` — Always check overflow pool

**Before:**
```python
if self.category_pools and self.category_whitelist:
    # check category pools
elif self.task_pool is not None:
    # fallback: check shared pool
```

The `elif` meant workers skipped the shared pool entirely when
`category_pools` was populated, which broke the backstop.

**After:**
```python
# 3. Pull from per-type reserved pools
if self.category_pools and self.category_whitelist:
    for cat in self.category_whitelist:
        pool = self.category_pools.get(cat)
        ...

# 4. ALWAYS check overflow pool (backstop)
if self.task_pool is not None:
    task = self.task_pool.get_nowait()
    ...

# 5. Attempt work stealing
```

The overflow check is now unconditional — all workers check the shared
overflow after their reserved pools, before attempting work stealing.

## Verification

```
✓ staging/ready_config.py — syntax OK
✓ staging/ready_pool.py — syntax OK
✓ staging/ready_worker.py — syntax OK
✓ All imports OK
✓ reservation_timeout_s = 30.0
✓ All new tracking attrs present in ReadyPool.__init__
✓ start() uses round-robin type partition
✓ monitor_loop() has backstop release logic
✓ Worker overflow check is independent (not elif)
```

## Files Modified

| File | Lines Changed | Description |
|------|---------------|-------------|
| `staging/ready_config.py` | +2 | Added `reservation_timeout_s` config |
| `staging/ready_pool.py` | ~80 | Per-type pools, round-robin drain, backstop, tracking attrs |
| `staging/ready_worker.py` | ~15 | Made overflow check unconditional (not elif) |

## Files Not Modified (no changes needed)

- `staging/ready_queue.py` — Queue logic unchanged
- `staging/entrypoint.py` — Orchestration unchanged
- `staging/ready_judge.py` — Judgment logic unchanged
- `staging/ready_classifier.py` — Classification unchanged
- `staging/workers/det_worker.py` — Worker implementations unchanged
- `staging/workers/loc_worker.py`
- `staging/workers/fw_worker.py`

## Data Flow After Fix

```
ReadyQueue
    │
    ▼  (round-robin)
┌─────────┬─────────┬────────────┐
│ det     │ local   │ fireworks  │  ← _type_pools
│ pools   │ pools   │ pools      │
└────┬────┴────┬────┴─────┬──────┘
     │         │          │
     ▼         ▼          ▼
det_worker  loc_worker  fw_worker   ← each pulls ONLY from own type's pool
     │         │          │
     │         │          │
     └─── shared overflow (task_pool) ← backstop after 30s
                    │
               any worker can pull
```

## Interaction with Existing Judge Fix

The judge's `ready_to_judge` method already requires at least one answer from
a `local` worker type before judging (to prevent det-worker speed dominance).
With this partitioning fix, local workers are now **guaranteed** to receive
tasks, so the judge fix can actually function.
