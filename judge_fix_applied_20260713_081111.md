# Judge Speed Dominance Fix

## Date
2026-07-13

## Problem
DetWorker Speed Dominance: 2 DetWorkers × 5 temperature sweeps = 10 answers per task, arriving in microseconds. The judge sees 5+ answers and judges immediately. The 3 LocWorkers' GPU inference takes ~2.5s per task — their answers arrive after the task is already judged. They never contribute.

## Root Cause
In `staging/ready_judge.py`, `ready_to_judge()` checked only total answer count (`count < judgment_votes`). With `judgment_votes=5` and 10 microsecond DetWorker answers, the threshold was met instantly, bypassing LocWorkers entirely.

## Fix Applied
**File**: `staging/ready_judge.py`, method `ReadyJudge.ready_to_judge()`

### Change
Added a guard in the primary judgment path (when enough non-degenerate answers exist) requiring at least one answer from a `local` worker type before returning `True`:

```python
has_local = any(
    self._get_worker_type(a) == "local"
    for a in answers
)
if has_local or self.config.loc_workers == 0:
    return True
logger.debug(
    "[judge] Waiting for local worker answer on %s "
    "(have %d/%d non-degenerate, %d total) — "
    "will fall back to timeout if local never arrives",
    task_id, len(non_degenerate), threshold, count,
)
# Fall through to timeout / all-empty checks below
```

### Safety net
- **No local workers configured** (`loc_workers == 0`): skip the check entirely
- **Timeout fallback** (30s, 15s under deadline emergency): forces judgment if local workers never arrive
- **All-empty degenerate fallback**: unchanged, still fires when every active type has tried
- **Deadline emergency final drain** in `monitor_loop()`: still force-judges pending tasks regardless, ensuring no task is stuck forever

### Logging
A `logger.debug()` message is emitted when `ready_to_judge()` is waiting for a local worker answer, aiding debugging/timing analysis.

## Verification
- Syntax check: `python3 -c "import ast; ast.parse(open('staging/ready_judge.py').read()); print('OK')"` → OK
