# Findings Review — Quick Submission Test Plan

Review of 16 findings from two code review agents against the **v14-fix** harness.py pipeline
(`ghcr.io/artemkorolev1/amd-hackathon-submit:v14-fix`).

## Classification Scheme

| Tier | Label | Meaning |
|------|-------|---------|
| 0 | **Include — CRITICAL** | Prevents grader failure (MISSING_TASKS, INVALID_RESULTS_SCHEMA, MODEL_VIOLATION, RUNTIME_ERROR). Must fix before submission. |
| 1 | **Include — HIGH** | Could cause task-level failures or empty answers. Should fix before submission. |
| 2 | **Include — MEDIUM** | Degrades accuracy or correctness. Fix if time permits. |
| 3 | **Defer** | Real issue but not blocking a quick submission test. Address in next iteration. |
| 4 | **Reject** | Not actually a problem — misunderstanding by review agents. |

---

## Finding-by-Finding Analysis

### #1 — `_process()` in harness.py has no try/except around its call in __main__ (CRITICAL)
**Verdict: INCLUDE — CRITICAL**

The `__main__` block (lines 663-687) wraps the loop in `try/finally` but **not** `try/except`. If `_process()` throws an exception (e.g., AttributeError, TypeError), the stack unwinds, the `finally` block writes partial results (missing the current task), and the script exits with code 1. This causes **MISSING_TASKS** for that task position.

**Files to change:** `harness.py`
**Fix:**
```python
try:
    answer = _process(prompt)
except Exception as exc:
    logger.warning("_process failed for task %s: %s", tid, exc)
    answer = ""
results.append({"task_id": tid, "answer": answer})
```
**Test:** Inject a division-by-zero or type error in a mock solver, verify that task gets empty answer instead of being dropped.

---

### #2 — `ALLOWED_MODELS` env var never read by harness.py (CRITICAL)
**Verdict: INCLUDE — CRITICAL**

`ALLOWED_MODELS` is documented as a grader-injected env var but only `agent/config.py` (old `main.py` pipeline) reads it. The active `harness.py` pipeline never references it. `fw_router.py` uses hardcoded model IDs (`kimi-k2p7-code`, `minimax-m3`). If the grader sets `ALLOWED_MODELS` to restrict which models may be called, every Fireworks call will use an unauthorized model → **MODEL_VIOLATION** grader error → possible disqualification.

**Files to change:** `harness.py`, optionally `agent/solvers/fw_router.py`
**Fix:** Add ALLOWED_MODELS parsing to `harness.py` (module-level):
```python
_ALLOWED_MODELS_RAW = os.environ.get("ALLOWED_MODELS", "")
_ALLOWED_MODELS: set[str] = set()
if _ALLOWED_MODELS_RAW.strip():
    for m in _ALLOWED_MODELS_RAW.split(","):
        m = m.strip()
        if m:
            _ALLOWED_MODELS.add(m if "/" in m else f"accounts/fireworks/models/{m}")
```
Then in `_fw_fallback()` and the escalation block, validate `cfg.model_id in _ALLOWED_MODELS` before calling. If no ALLOWED_MODELS set (empty), allow all models (backward compat).
**Test:** Set `ALLOWED_MODELS="accounts/fireworks/models/deepseek-v4-flash"` and verify calls only use that model. Set it to a non-matching model and verify no Fireworks calls are made.

---

### #3 — Complexity model hardcoded path doesn't exist in container (HIGH)
**Verdict: DEFER**

`agent/complexity.py` hardcodes `_MODELS_DIR = "/home/artem/dev/amd-hackathon-shared/classifiers/best_complexity_model"` which doesn't exist in the container. The fallback returns 0.5, which is a constant medium-complexity score. This works correctly — prompts get medium-complexity prompts and token budgets. The heuristic alternative (`complexity_filter.py`) exists but would change behavior. For a quick submission test, the constant 0.5 fallback is harmless — it doesn't cause grader failure or empty outputs.

**Not blocking.** Address after submission when tuning accuracy.

---

### #4 — `_DET_CAT_MAP["factual"] = "other_complex"` prevents deterministic factual QA (HIGH)
**Verdict: REJECT**

The code:
- Line 172: `_DET_CAT_MAP["factual"] = "other_complex"`
- Line 555: `solver_cat = _DET_CAT_MAP[category]` → calls `solve_factual_qa(prompt, "other_complex")`
- `solve_factual_qa()` line 1119: `if category not in ("factual_knowledge", "question_answering", "factual", "other_complex"): return None`

The function **does** accept `"other_complex"` — the review agent missed this. The deterministic factual QA pipeline works correctly end-to-end.

---

### #5 — No `llm is None` check in `_infer()` (HIGH)
**Verdict: DEFER**

`_infer()` (line 341) calls `llm.create_chat_completion()` without checking if `llm` is None. However, the inner `_call()` lambda runs inside a `try/except Exception` block (lines 348-358). If `llm is None`, `_call()` raises `AttributeError` in the worker thread, which is propagated by `future.result()` and caught by `except Exception as exc:` — the function returns `""` gracefully.

No crash occurs. The finding is **factually incorrect** — the error is handled. An explicit guard would be cleaner and slightly faster (avoids thread submission overhead), but it's not a grader failure risk.

**Not blocking.**

---

### #6 — `syntax_ok()` regex for code fence extraction is too narrow (HIGH)
**Verdict: INCLUDE — HIGH**

`syntax_ok()` on line 245: `re.search(r"```(?:python)?\n([\s\S]+?)\n```", code)` has these failures:
1. No match for ````py` or ````python3` language tags
2. Requires newline before closing ```` — fails on no-trailing-newline files
3. `\n` won't match `\r\n` (CRLF) 
4. No allowance for space after `python` before newline

This causes **false negative syntax checks** on valid code → triggers unnecessary Fireworks fallback (wasting time) or returns empty "syntax failed" answers.

**Files to change:** `harness.py`
**Fix:** Replace the regex with a more permissive one:
```python
def syntax_ok(code: str) -> bool:
    m = re.search(r"```\w*\s*\n?([\s\S]+?)(?:\n```|```)", code)
    src = m.group(1).strip() if m else code.strip()
    if not src:
        return False
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False
```
**Test:** Create test cases with ````py`, no trailing newline, CRLF line endings, verify all parse correctly.

---

### #7 — Fireworks escalation calls missing explicit `timeout=` (HIGH)
**Verdict: INCLUDE — HIGH**

The `_fw_fallback()` function (line 471-477) correctly passes `timeout=FIREWORKS_TIMEOUT_S`. But the **hard-case escalation calls** (lines 498-547) for summarization, sentiment, math, logic, code_gen, and code_debug all omit `timeout=`. They use `_fw.solve()`'s default of `timeout=29`. If the Fireworks API is slow, these escalation calls can burn 29s before the local model even runs, leaving almost no time for inference. Since escalation runs BEFORE local, a slow API could exhaust the entire 30s per-task budget.

**Files to change:** `harness.py`
**Fix:** Add `timeout=FIREWORKS_TIMEOUT_S` to all escalation `_fw.solve()` calls. Even better, use a tighter timeout (e.g., 20s) for pre-local escalation since they should be fast (simple models like minimax-m3).
```python
answer = _fw.solve(
    cfg.model_id, cfg.system_prompt, prompt,
    max_tokens=cfg.max_tokens, temperature=cfg.temperature,
    prefill=cfg.prefill, task_type="sentiment",
    timeout=FIREWORKS_TIMEOUT_S,  # ADD THIS
)
```
**Test:** Mock slow Fireworks response, verify escalation times out and falls through to local within the expected window.

---

### #8 — `json.dump` without `ensure_ascii=False` (MEDIUM)
**Verdict: INCLUDE — MEDIUM**

Line 630: `json.dump(results, f)` uses default `ensure_ascii=True`, which escapes all non-ASCII characters (Unicode) to `\uXXXX` sequences. This breaks the grader's substring matching for answers containing Chinese, emoji, accented characters, or mathematical symbols.

**Files to change:** `harness.py`
**Fix:**
```python
json.dump(results, f, ensure_ascii=False)
```
**Test:** Create a task whose answer contains "café" or "π ≈ 3.14", verify the output JSON contains the literal characters, not `\uXXXX` escapes.

---

### #9 — Deterministic sentiment solver never returns "neutral" or "mixed" (MEDIUM)
**Verdict: DEFER**

The deterministic `_classify_sentiment()` returns only "positive", "negative", or None. For neutral texts, it returns None, allowing the LLM to handle it. This is an accuracy limitation, not a grader failure. The LLM handles neutral and mixed cases correctly.

**Not blocking for quick submission test.**

---

### #10 — `INFERENCE_TIMEOUT_S = 60s` too generous (MEDIUM)
**Verdict: INCLUDE — HIGH**

The grader enforces a **30s hard kill per task**. With INFERENCE_TIMEOUT_S=60s:
- If the LLM hangs (e.g., infinite generation loop), it blocks for 60s before the harness detects timeout
- The grader kills the entire container at 30s, potentially without writing the final `/output/results.json`
- All subsequent tasks are lost

Setting to 28s lets the harness gracefully degrade: timeout → empty → Fireworks fallback → move to next task — all before the grader's 30s axe falls.

**Files to change:** `harness.py`
**Fix:**
```python
INFERENCE_TIMEOUT_S = 28.0  # Was 60.0 — must be under grader's 30s hard limit
```
**Test:** Create a prompt that causes the model to enter a long generation loop (e.g., "write a very long story"), verify harness times out at ~28s and returns empty (then Fireworks fallback), not crashing the process.

---

### #11 — `_DOC_HEADER_RE` regex compiled fresh on every `_process()` call (MEDIUM)
**Verdict: DEFER**

The regex on line 384-387 is compiled inside `_process()`, meaning it's recompiled for every task. Impact: ~microseconds per call. Trivially fixable but zero impact on grading.

**Not blocking.**

---

### #12 — `_strip_kimi_preamble` applied unconditionally to all Fireworks code results (MEDIUM)
**Verdict: INCLUDE — MEDIUM**

Lines 549 and 613 call `_strip_kimi_preamble(answer)` on ALL Fireworks code results, even when the model isn't Kimi. The function strips everything before the first ``` fence. For non-Kimi models that return code without fences (e.g., just the function body), this could discard valid code entirely. Currently all code tasks route to `kimi-k2p7-code` via the router, but if ALLOWED_MODELS forces a different model, this would corrupt answers.

**Files to change:** `harness.py`
**Fix:** Only strip preamble for Kimi models:
```python
def _is_kimi_model() -> bool:
    return "kimi" in FIREWORKS_MODEL.lower()

# Then in usage:
if _is_kimi_model():
    answer = _strip_kimi_preamble(answer)
```
Or better, pass the model_id through and check it at the call site.

**Test:** Route code_gen to a non-Kimi model, verify output is not corrupted.

---

### #13 — `_fw` instantiated at module load time (MEDIUM)
**Verdict: REJECT**

`_fw = FireworksSolver()` at line 98 reads env var at import time. In a container, environment variables are set once at container start and never change. This is the standard pattern for configuration. Not a bug.

---

### #14 — Empty prompt returns empty string with no warning (MEDIUM)
**Verdict: DEFER**

The pre_filter returns `bypass` with `direct_answer=""` for empty prompts. The `if s0.direct_answer:` check is falsy for empty strings, so it falls through to classification + LLM call. This could produce an unexpected answer from the LLM but won't cause grader failure. Logging could be added but isn't critical.

**Not blocking.**

---

### #15 — ThreadPoolExecutor never shut down (MEDIUM)
**Verdict: REJECT**

Line 331 creates a `ThreadPoolExecutor` that's never shut down. In an ephemeral container that runs for ~600s max and then terminates, the OS cleans up all resources. No leak. This is the expected pattern for short-lived batch jobs.

---

### #16 — No circuit breaker in harness.py (MEDIUM)
**Verdict: DEFER**

There's no circuit breaker for Fireworks calls. If the API is persistently failing, every task will attempt escalation calls, each burning up to 29s before falling through. For a 19-task, 600s deadline, this could waste significant time. However:
- If no `FIREWORKS_API_KEY` is set, escalation is skipped entirely
- The `_fw_fallback` and escalation blocks wrap calls in try/except, so failures are caught
- For a quick submission test, this is an optimization, not a blocker

**Not blocking for quick test.** Add in accuracy iteration if Fireworks proves unreliable.

---

## Summary Table

| # | Finding | Verdict | Reason |
|---|---------|---------|--------|
| 1 | `_process()` no try/except | **INCLUDE (CRITICAL)** | Causes MISSING_TASKS |
| 2 | `ALLOWED_MODELS` not read | **INCLUDE (CRITICAL)** | Causes MODEL_VIOLATION |
| 3 | Complexity model hardcoded path | DEFER | Falls back to 0.5, works fine |
| 4 | `_DET_CAT_MAP["factual"]` gate | REJECT | Function accepts "other_complex" |
| 5 | `llm is None` in `_infer()` | DEFER | Caught by `except Exception`, no crash |
| 6 | `syntax_ok()` regex too narrow | **INCLUDE (HIGH)** | False syntax fails → empty answers |
| 7 | Fireworks escalation no timeout | **INCLUDE (HIGH)** | Can exhaust 30s task budget |
| 8 | `json.dump` no `ensure_ascii=False` | **INCLUDE (MEDIUM)** | Breaks grader substring matching |
| 9 | Sentiment no neutral/mixed | DEFER | Falls through to LLM, accuracy only |
| 10 | INFERENCE_TIMEOUT_S=60 too generous | **INCLUDE (HIGH)** | Can cause container kill before results |
| 11 | `_DOC_HEADER_RE` recompiled | DEFER | Microsecond optimization |
| 12 | `_strip_kimi_preamble` unconditional | **INCLUDE (MEDIUM)** | Corrupts non-Kimi code answers |
| 13 | `_fw` at module load | REJECT | Standard container pattern |
| 14 | Empty prompt returns empty | DEFER | Edge case, no grader failure |
| 15 | ThreadPoolExecutor never shut down | REJECT | Ephemeral container, no leak |
| 16 | No circuit breaker | DEFER | Optimization, not blocker |

---

## Ranked Build Plan

### Fix Group 1: CRITICAL (grader failure prevention)
**Fix first, test each, then proceed.**

| Step | Fix | Files | What to change | Test procedure |
|------|-----|-------|----------------|----------------|
| 1 | **#2: ALLOWED_MODELS** | `harness.py` | Add env var parsing at module level. Check `cfg.model_id in _ALLOWED_MODELS` before calling Fireworks. If ALLOWED_MODELS is empty, allow all (backward compat). | `ALLOWED_MODELS="nonexistent-model" python3 harness.py < test.json` → verify no Fireworks calls (stderr shows "not in allowed list") |
| 2 | **#1: _process() try/except** | `harness.py` | Wrap `_process(prompt)` call in try/except, on exception append `{"task_id": tid, "answer": ""}` to results. | Inject a synthetic crash in a solver: `if "crash" in prompt: raise RuntimeError("boom")` → verify task result is empty, not dropped |
| 3 | **#10: INFERENCE_TIMEOUT_S** | `harness.py` | Change `INFERENCE_TIMEOUT_S = 60.0` → `28.0` | Run with a slow-generating prompt → verify timeout at ~28s, process continues, result is empty string |

### Fix Group 2: HIGH (task-level failure prevention)

| Step | Fix | Files | What to change | Test procedure |
|------|-----|-------|----------------|----------------|
| 4 | **#7: Fireworks escalation timeout** | `harness.py` | Add `timeout=FIREWORKS_TIMEOUT_S` (or 20s) to all 6 escalation `_fw.solve()` calls | Mock slow API, verify escalation times out and falls through to local before per-task budget exhausted |
| 5 | **#6: syntax_ok() regex** | `harness.py` | Replace regex with permissive version supporting ````py`, CRLF, no-trailing-newline, space after language tag | Create test code blocks with ````py`, no trailing newline, CRLF — verify all parse as valid |
| 6 | **#8: ensure_ascii=False** | `harness.py` | Change `json.dump(results, f)` → `json.dump(results, f, ensure_ascii=False)` | Task with Unicode answer → verify literal chars in output JSON |

### Fix Group 3: MEDIUM (correctness)

| Step | Fix | Files | What to change | Test procedure |
|------|-----|-------|----------------|----------------|
| 7 | **#12: Conditional _strip_kimi_preamble** | `harness.py` | Only call `_strip_kimi_preamble()` when the model is Kimi. Check the model_id from the router config. | Route code_gen to a non-Kimi model → verify output is not preamble-stripped |

### When to Rebuild

1. **After Fix Group 1** — Optionally rebuild and test-push. These are critical fixes.
2. **After ALL fixes (Groups 1-3)** — Final rebuild and push as `v15-fix` or `v14-fix-2`.

### Local iterative testing (no rebuild needed)
Most fixes can be tested locally without rebuilding:
```bash
cd ~/dev/amd-hackathon
python3 harness.py /path/to/test/tasks.json
```
Only the Docker build is needed for the final container push.

### Container rebuild commands
```bash
cd ~/dev/amd-hackathon
docker buildx build --platform linux/amd64 \
  -t ghcr.io/artemkorolev1/amd-hackathon-submit:v15-fix \
  --load .
docker push ghcr.io/artemkorolev1/amd-hackathon-submit:v15-fix
```
