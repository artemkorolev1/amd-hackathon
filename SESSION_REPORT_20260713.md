# Session Report — July 13, 2026

## What Was Built

### Staging Pull System (Complete)

The staging parallel worker system at `staging/` was fully built and GPU-tested:

- **ReadyPool → ReadyMonitor** — Shared task pool, workers pull autonomously, monitor loop does health checks every 5s
- **ReadyWorker** — Pull-based dispatch with 4-priority: stolen > inbox > category pools > steal. Heartbeat per iteration for crash detection. `_process_single()` for deadline emergency mode.
- **ReadyJudge** — Autonomous `consume_loop()` in daemon thread. Dynamic `_active_worker_types` tracking instead of hardcoded types. Deadline-aware vote threshold halving.
- **Per-category queues** — Workers only pull from their whitelisted category pools (8 categories, one `multiprocessing.Queue` each)
- **6 workers**: 2 DetWorker + 3 LocWorker (Qwen2.5 instruct, Qwen2.5 coder, Gemma-3) + 1 FwWorker
- **Dockerfile** updated to include all 3 models, staging/, runner/, scripts/, and dispatcher.py with STAGING_ENABLED env var routing

### Runner Tools (Complete)

Wraps agent.Pipeline for local eval runs — completely separate from the staging container path:

- `runner/batch_runner.py` — ProcessPoolExecutor, per-task 30s timeout, memory-aware worker capping
- `runner/evaluate.py` — fuzzy_match cascade, 3-sheet XLSX reports
- `runner/deploy.py` — Docker build/tag/push/verify automation

### Fixes Applied

| Fix | What | Status |
|-----|------|--------|
| Startup race | Tasks drain into pool AFTER all workers signal ready (moved drain after ready_flag barrier) | ✅ |
| Category pool reference | Pools created in __init__ BEFORE spawning workers (was creating a second set after spawn, workers had stale empty refs) | ✅ |
| DetWorker solver imports | Added solve_code_generation, solve_ner, solve_logic imports | ✅ |
| Model path symlink | `/models/` symlinked to local project's `models/` | ✅ |
| Stale /input/tasks.json | Deleted leftover test file that was overriding sys.argv[1] | ✅ |

### Plans Created

| File | Content |
|------|---------|
| `docs/plans/PULL_SYSTEM_DESIGN.md` | 1472-line pull-based architecture design |
| `docs/plans/PULL_SYSTEM_DELTA.md` | 1144-line file-by-file implementation delta |
| `docs/plans/INTEGRATION_AUDIT.md` | 589-line full module inventory + dependency graph + 5-phase integration plan |
| `docs/plans/MODULES_NOT_INTEGRATED.md` | 218-line list of 31 agent/ modules not in any production path |
| `docs/plans/FIREWORKS_INTEGRATION.md` | 289-line Fireworks API integration status |
| `docs/plans/PARALLELIZATION_ANALYSIS.md` | 313-line analysis of worker counts, bottlenecks, deadlock risks |
| `docs/plans/TOOLS_INTEGRATION_CHECK.md` | Tools reachable from staging workers vs theoretical |

## Current Performance

**10 tasks on GPU (factual_combined_80.json):**
- 10/10 judged, 7 majority_3plus, 3 all_failed
- 4.6 GB VRAM with all 3 LocWorkers loaded
- ~5s total (DetWorkers dominate, LocWorkers' GPU answers arrive too late)

## Open Issue: DetWorker Speed Dominance

**Root cause:** 2 DetWorkers × 5 temperature sweeps = 10 answers per task, arriving in microseconds. The judge sees 5+ answers and judges immediately. The 3 LocWorkers' GPU inference takes ~2.5s per task — their answers arrive after the task is already judged. They never contribute.

**Fix needed (Option A):** `ready_to_judge()` must require at least one answer from a `local` worker type before judging. This lets GPU models actually contribute their answers.

## What's Next

1. **Implement Option A** — Judge waits for at least one LocWorker answer
2. **Test container build** — `python3 -m runner.deploy --build-only`
3. **Optionally wire Fireworks** — Set FIREWORKS_API_KEY, STAGING_FW_WORKERS=1
4. **Full 80-question GPU run**
5. **Submit CPU container**

## Files Changed This Session

staging/ready_pool.py (343→450 lines)
staging/ready_worker.py (191→258 lines)
staging/ready_judge.py (376→423 lines)
staging/ready_config.py (82→159 lines)
staging/ready_queue.py (139→157 lines)
staging/entrypoint.py (193→324 lines)
staging/__init__.py (26→26 lines)
staging/workers/det_worker.py (89→128 lines)
staging/workers/loc_worker.py (108→143 lines)
staging/workers/fw_worker.py (78→107 lines)
runner/__init__.py (NEW)
runner/batch_runner.py (NEW, 441 lines)
runner/evaluate.py (NEW, 677 lines)
runner/deploy.py (NEW, 449 lines)
dispatcher.py (NEW, 44 lines)
Dockerfile (44→52 lines)
Makefile (59→86 lines)
requirements.txt (14→16 lines)
docs/plans/INTEGRATION_AUDIT.md (NEW)
docs/plans/PULL_SYSTEM_DELTA.md (NEW)
docs/plans/MODULES_NOT_INTEGRATED.md (NEW)
docs/plans/FIREWORKS_INTEGRATION.md (NEW)
docs/plans/PARALLELIZATION_ANALYSIS.md (NEW)
docs/plans/TOOLS_INTEGRATION_CHECK.md (NEW)
