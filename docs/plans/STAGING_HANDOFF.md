# Staging Architecture — Session Handoff Report

**Date:** 2026-07-13
**Project:** AMD ACT II Hackathon — Track 1 Token Efficient Routing Agent
**Root:** `/home/artem/dev/amd-hackathon/`
**Plan:** `docs/plans/PARALLEL_SUBMIT_PLAN.md`

---

## What Was Built

### `staging/` — Parallel Submission Architecture (11 files)

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 24 | Package exports (ReadyConfig, ReadyQueue, ReadyTask, ReadyPool, ReadyJudge) |
| `ready_config.py` | 95 | Configuration from `STAGING_*` env vars |
| `ready_queue.py` | 135 | `ReadyTask` dataclass + `ReadyQueue` (multi-category) |
| `ready_worker.py` | 168 | `ReadyWorker` ABC — 5-try sequential processing, per-category queue pull |
| `ready_pool.py` | 318 | `ReadyPool` — per-category queues, priority routing, watchdog, deadline mgmt |
| `ready_judge.py` | 310 | `ReadyJudge` — fuzzy-match grouping, majority vote, Fireworks escalation |
| `ready_classifier.py` | 198 | Standalone 8-way classifier (pure regex, no agent deps) |
| `entrypoint.py` | 193 | Container entrypoint — read/classify/dispatch/judge/write |
| `workers/__init__.py` | 5 | Worker package init |
| `workers/det_worker.py` | 88 | Deterministic solver worker (2 instances default) |
| `workers/fw_worker.py` | 78 | Fireworks API worker (1 instance, temp sweep 0.1→0.9) |
| `workers/loc_worker.py` | 116 | Local GGUF worker (1 instance, temp sweep) |

### Supporting files

| File | Purpose |
|------|---------|
| `Dockerfile.staging` | Separate container image for staging pipeline |
| `test_judge.py` | 11 unit tests for judge voting logic |
| `test_input.json` | Sample 3-task test fixture |

### Container image

**Tag:** `ghcr.io/artemkorolev1/amd-hackathon-submit:v15-staging`
**Build:** `docker buildx build --platform linux/amd64 -t ... --load -f Dockerfile.staging .`

---

## Architecture (4-Tier Pull Model)

```
/input/tasks.json ──→ ready_classifier (8-way bulk classify)
                         ↓
                    ReadyQueue (per-category Queues)
                         ↓
                    ReadyPool
                   /    |    \    \
           FwWorker  LocWorker  DetWorker  DetWorker
           (FW API)  (GGUF)    (regex)    (regex)
              ↓         ↓         ↓         ↓
          5 tries × temp sweep, push {answer, timing_ms, prompt, category}
                         ↓
                    ReadyJudge
                   /          \
            majority (≥3)    escalate (no majority)
                  ↓                ↓
            accept        Fireworks API (fw_router)
                  ↓                ↓
               /output/results.json
```

**Key design decisions:**
- **Pull model:** Workers poll per-category queues — they only pull categories they can handle (DetWorkers skip code_gen, logic, ner)
- **Priority matrix:** `category_priority` in config controls which worker type handles which category first
- **Sequential 5-tries:** Each worker runs the same task 5 times with temperature sweep (0.1→0.9), sends all to judge
- **Tiered judgment:** ≥3 agree = accept, 2 agree with ≥4 votes = accept, all different = escalate to Fireworks
- **Deadline adaptation:** 5-vote full → 3-vote → 1-vote as deadline approaches
- **Crash watchdog:** Periodic `is_alive()` check, reset stale busy flags, orphaned tasks re-judged

**Default worker count:** 1 Fireworks + 1 Local + 2 Deterministic = 4 workers

---

## P0 Fixes Applied (6 items, 5 files)

All applied and cross-checked. 11/11 tests pass, all files parse, API contracts match.

| ID | Fix | File | What |
|----|-----|------|------|
| **S1** | Signal handler deadlock | `ready_worker.py:151` | Removed `logger.info()` from SIGTERM handler (logging lock + signal = deadlock) |
| **R1** | Per-category queues | `ready_worker.py`, `ready_pool.py` | Workers poll category-specific queues with whitelist; DetWorkers only pull categories they can solve |
| **R2** | Priority matrix wired | `ready_pool.py:85-93` | `_build_whitelist()` reads `config.category_priority` to order each worker's pull priority |
| **S2** | Result validation | `ready_judge.py:111` | `add_answer()` validates `task_id`/`answer` keys vs crashing on malformed dict |
| **R5** | Main SIGTERM handler | `entrypoint.py` | Writes output on SIGTERM/SIGINT; grader always gets a result file |
| **R3+R4+R6** | Watchdog + drain + cleanup | `ready_pool.py` | `is_alive()` crash detection, final queue drain, busy_flag cleanup |

---

## Remaining Issues (P1 — Not Yet Fixed)

| Priority | ID | Issue | File | Source |
|----------|----|-------|------|--------|
| P1 | A3 | Gradual deadline degradation — cliff edge at 60s | `ready_pool.py` | Architecture Review |
| P1 | A4 | No startup timing instrumentation | `ready_worker.py` | Architecture Review |
| P1 | S3 | `config._safe_int()` for env vars — `int("abc")` crashes | `ready_config.py:58` | Safety Review |
| P1 | RL3 | Worker crash → orphaned task not explicitly requeued | `ready_pool.py` | Reliability Review |
| P1 | RL4 | FW worker wastes 150s on 5×30s timeouts with no API key | `workers/fw_worker.py` | Reliability Review |

### P2 Items (Medium Priority)

| ID | Issue | File |
|----|-------|------|
| S5 | `_fallback_best` dead code | `ready_judge.py:308` |
| S6 | `_pull_task` broad exception swallow | `ready_worker.py:127` |
| S7 | Prompt duplication in every result dict | `ready_worker.py:147` |
| S8 | LocWorker shutdown doesn't guarantee memory free | `workers/loc_worker.py:104` |
| S9 | Tests not pytest-compatible | `test_judge.py` |
| S10 | `_TEMPERATURE_SWEEP` duplicated | `fw_worker.py`, `loc_worker.py` |
| RL5 | Deadline=0 edge case | `entrypoint.py` |
| RL6 | Queue backpressure (unbounded Q, theoretical) | `ready_pool.py`, `ready_worker.py` |

---

## Test Results

### Unit tests (11/11 pass)
```
fuzzy_match:          7/7 PASS  (exact, normalized, numeric 1%, token overlap, negatives)
majority_3plus:       PASS  — 3/5 agree → accept
all_different:        PASS  — all different → escalate (or fallback without FW key)
majority_2of3:        PASS  — 2/3 agree → accept
judge_all:            PASS  — batch judge returns correct format
```

### Container smoke test
Runs successfully with 2 deterministic workers. Classification works (standalone classifier). Deterministic solvers fail to load because `agent/solvers/deterministic.py` has `import spacy` at module level and spacy not installed. Tasks get `all_failed` judgment. Full container test needs `requirements.txt` deps resolved.

---

## Three Architecture/Reliability Reviews

All three reviews are saved as reference documents:

| Review | File (in staging/) | Key P0 Findings |
|--------|-------------------|-----------------|
| Architecture | (in this report) | Wrong-worker consumption, dead priority matrix, no gradual deadline |
| Reliability | `RELIABILITY_REVIEW.md` | Orphaned tasks on crash, no SIGTERM handler, stuck busy_flag |
| Safety | (in this report) | Signal handler deadlock, KeyError crash, no final drain, malformed env vars |

---

## File Inventory

```
/home/artem/dev/amd-hackathon/
├── staging/                          # NEW — parallel submission architecture
│   ├── __init__.py                   # Package init
│   ├── entrypoint.py                 # Container entrypoint (replaces harness.py)
│   ├── ready_config.py               # Configuration from env vars
│   ├── ready_queue.py                # Task queue (ReadyTask, ReadyQueue)
│   ├── ready_pool.py                 # Worker pool manager (per-category queues)
│   ├── ready_worker.py               # Worker base class (5-try + category whitelist)
│   ├── ready_judge.py                # Judgment/voting module (majority vote + escalation)
│   ├── ready_classifier.py           # Standalone 8-way classifier (no agent deps)
│   ├── RELIABILITY_REVIEW.md         # Full reliability review report
│   ├── test_judge.py                 # 11 unit tests for judge
│   └── workers/
│       ├── __init__.py
│       ├── fw_worker.py              # Fireworks API worker
│       ├── loc_worker.py             # Local GGUF model worker
│       └── det_worker.py             # Deterministic solver worker
├── Dockerfile.staging                # Separate Dockerfile for staging submission
├── docs/plans/PARALLEL_SUBMIT_PLAN.md# Plan document
├── agent/                            # UNTOUCHED — existing pipeline
├── harness.py                        # UNTOUCHED — existing entrypoint
└── requirements.txt                  # Updated with sympy, spacy, click
```

---

## Next Steps (for next chat)

1. **Resolve container deps** — rebuild with proper spacy/click installation so deterministic solvers load
2. **Apply P1 fixes** — deadline degradation, startup timing, safe config parsing
3. **Container smoke test** — run with real tasks, verify per-category queue routing works
4. **Comparison test** — run both `harness.py` and `staging/entrypoint.py` on same eval set, compare accuracy
5. **Docker push** — push `v15-staging` to GHCR when ready

---

## Rollback Safety

- Zero changes to `agent/` directory
- Zero changes to `harness.py`
- `Dockerfile` (main) untouched — `Dockerfile.staging` is a separate build
- Switching back: rebuild from `Dockerfile`, ignore `staging/`
