# Bug Fixes Applied — agent/main.py
**Date:** 2026-07-13  
**Branch:** v12d  
**Commit base:** acd4fae  

---

## Bug 1 (P0) — Missing LLM arg for `solve_with_consensus()`

**Problem:**
`solve_with_consensus()` requires `llm` as its first positional argument (see `agent/solvers/local_vote.py:105-106`), but both call sites in `agent/main.py` omitted it. Additionally, `main()` never imported or instantiated a `llama_cpp.Llama` instance to pass.

**Changes:**
1. **Line 22** — Added `Any` to typing imports for the `llm: Any` parameter.
2. **Lines 24-31** — Added `LOCAL_MODEL_PATH` to the imports from `agent.config`.
3. **Line 178** — Changed `_run_pipeline_impl` signature from `(fireworks)` to `(llm: Any, fireworks)`.
4. **Lines 338-339** — Added `llm=llm,` keyword arg to the primary `solve_with_consensus()` call.
5. **Lines 392-393** — Added `llm=llm,` keyword arg to the code-retry `solve_with_consensus()` call.
6. **Lines 422-451** — Rewrote `main()` to load a `llama_cpp.Llama` instance when `LLAMA_ENABLE` is true and the GGUF model file exists, using the same loading pattern from `agent/pipeline.py:331-353`. The `llm` object is then passed to `_run_pipeline_impl(llm, fireworks)`.
7. **Line 461** — Updated `await _run_pipeline_impl(llm, fireworks)` call site.

---

## Bug 2 (P0) — NAKED_CATEGORIES blocking deterministic solvers

**Problem (original):**
`NAKED_CATEGORIES = {"ner", "summarization", "factual", "logic", "math"}` excluded 5 of 7 deterministic categories from ever reaching their solvers. The routing gate at line 159 (`category not in NAKED_CATEGORIES`) reduced coverage to only `{"sentiment", "code_debug"}`.

**Status:**
Already fixed in commit `6065437` ("fix: remove NAKED categories..."). Current state at **line 59** is `NAKED_CATEGORIES: set[str] = set()` — all 7 deterministic categories (`math`, `logic`, `sentiment`, `ner`, `factual`, `code_debug`, `summarization`) correctly reach their solvers. **No further changes needed.**

---

## Bug 3 (P1) — FIREWORKS_CATEGORIES too narrow

**Problem:**
`FIREWORKS_CATEGORIES = {"sentiment"}` meant categories where the 1.5B local model scores 0% (summarization) or <40% (NER, logic) never escalated to Fireworks.

**Change:**
**Line 62** — Expanded from `{"sentiment"}` to `{"sentiment", "summarization", "ner", "logic"}`.

| Category | Local 1.5B score | Now reaches Fireworks |
|----------|------------------|----------------------|
| sentiment | moderate         | ✓ (was already)      |
| summarization | ~0%          | ✓ **NEW**            |
| ner       | <40%             | ✓ **NEW**            |
| logic     | <40%             | ✓ **NEW**            |

---

## Verification

- `python3 -c "import ast; ast.parse(open('agent/main.py').read()); print('OK')"` — **passed**.
- Total: 469 lines, 0 syntax errors.
