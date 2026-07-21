# amd-hackathon — Shared Context

> Shared buffer for parallel Hermes sessions. Protocol:
> - **START:** Read this file to catch up on state
> - **DURING:** Append to Ideas/Decisions as things come up
> - **END:** Update Active Tasks, Blockers, Status before leaving
> - **CONFLICT:** Keep both entries with timestamps — don't delete others

---

## Current Status (as of Jul 15, 2026 — GSM8K math pipeline built)

### Major changes this session (Jul 15)

1. **Fireworks removed** — All API escalation code archived. Pipeline is 100% local-only.
2. **Pipeline GEPA fixed** — Old `routing_table` injection derouted through Fireworks path. Now patches `dynamic_prompts._CATEGORY_PROMPTS` directly.
3. **Factual pipeline GEPA: 100%** — Deterministic FactDB solver handles all 19 factual questions.
4. **Math pipeline GEPA: 31.6%** — Up from 21.1% with `_solve_narrative_math()` integration + last-number post-processing.
5. **Math classifier + tool router** — `math_classifier.py` (13 problem types) + `math_tool_router.py` (routes each type to best solver). Deterministic keyword-based.
6. **GSM8K 8.8K dataset downloaded** — 7,473 train + 1,319 test, MIT licensed. Full reasoning traces with `<<...>>` markers.
7. **SimpleMath 100K downloaded** — 22K word problems + 77K numeric expressions, MIT licensed.
8. **Step classifiers built (deterministic)** — 1-vs-multi, binary cascade, per-position operation type. Surface features cap at ~38% accuracy.

### Key finding: Surface classifiers hit ceiling

Question text features cannot reliably predict step count. 1-step and multi-step problems look identical. Cascade binary classifiers don't help — each individual binary has the same limitation.

### Path forward: ToRA pattern

Microsoft's ToRA (ICLR 2024) pattern is the right direction: LLM generates step-by-step plan + Python expressions → deterministic `PythonExecutor` computes each step. The 1.5B model can generate simple expressions (it cannot solve the full problem), and execution is 100% deterministic. MIT licensed.

### Architecture direction
```
Question → LLM planner (step plan + expressions)
         → PythonExecutor (deterministic computation per step)
         → Result combiner → final answer
```

### Data available
- GSM8K: /tmp/gsm8k_train.parquet (7,473) + /tmp/gsm8k_test.parquet (1,319)
- SimpleMath: /tmp/simplemath_100k.csv (100K, 22K word + 77K expression)
- Training-v3: data/eval/training-v3.json (19 math)
- Math combined: data/eval/math_combined_80.json (94 math)
- Competition math: data/raw/prompt_data/sources/competition_math.jsonl (12,500, no answers)

### Files created this session
- `agent/pipeline_gepa.py` — Pipeline-integrated GEPA (prompts tested through full Pipeline)
- `agent/solvers/math_classifier.py` — 13-type math problem classifier
- `agent/solvers/math_tool_router.py` — Routes math types to best solver
- `agent/solvers/math_step_counter.py` — Step count classifier (5 buckets)
- `agent/solvers/math_step_classifier.py` — Per-position operation type classifier
- `agent/solvers/math_cascade.py` — Cascading classifier pipeline
- `agent/solvers/math_binary_step_classifier.py` — Binary cascade (8 classifiers)
- `agent/solvers/math_one_vs_multi.py` — 1-step vs multi-step classifier
- `docs/pipeline-gepa-design.md` — Pipeline GEPA design doc
- `docs/gepa_results_summary.md` — GEPA results reference
- `archive/` — Archived Fireworks, fw_router, circuit_breaker code

### Pipeline changes this session
- `agent/main.py` — Fireworks removed, `solve_math_word_problems()` integrated
- `agent/pipeline.py` — Fireworks methods removed, math post-processing improved, routing_table dead code
- `agent/config.py` — Fireworks config removed
- `agent/solvers/deterministic.py` — `_solve_narrative_math()` + `solve_gsm8k()` integrated
- `agent/solvers/fw_router.py` — Archived
- `agent/solvers/fireworks.py` — Archived
- `agent/circuit_breaker.py` — Archived


### Cascaded Classifier — 98.2% on training-v2 (1,514 Q), 99.0% on validation-v1

7 bugs fixed last session:
1. **harness.py used primary-only classifier** (no cascade) — fixed
2. **Zebra puzzles routed logic→math** from "Solve:" prefix — fixed with constraint-puzzle guard
3. **`\bprime\b` matched "prime minister"** in competition regex — fixed
4. **Factual guard returned contaminated score** instead of 0 — fixed
5. **NER long tweets scored factual higher** — added NER task guard
6. **"Law and Order" TV show → logic** via `\border\b` — added "who was" factual boost
7. **"how many seasons" classified as math** — fixed factual guard condition

**3 new secondaries:** secondary_qa.py (factual↔summarization), secondary_codeguard.py (MCQ→not code), secondary_nertweet.py (NER tweets+biomedical)

### Code Tool Cascade — wired into pipeline.py ✓

`agent/solvers/code_tool_cascade.py` — binary decision tree for coding tools.
Each node is one yes/no split. Routes to category-specific LLM prompts in `dynamic_prompts.py`.
**Now wired into `pipeline.py` process() method.** Template solvers (two_sum, fib, etc.) return
directly via cascade. Non-template prompts route to LLM with coding_challenge_* prompt keys.

### Solver Routing Accuracy (deterministic-only, pipeline-context eval)

| Category | training-v2 | validation-v2 | Notes |
|----------|:-----------:|:-------------:|-------|
| factual | 90.0% | 92.0% | FactDB |
| sentiment | 76.0% | 68.0% | VADER |
| ner | 94.0% | 80.0% | solve_ner (old regex) |
| code_gen | 2.0% | 6.0% | template solver only (exact fn name) |
| code_debug | 0% | 0% | LLM-only |
| logic | 26.0% | 6.0% | logical_reasoning |
| math | 0% | 2.0% | LLM-only |
| summarization | 0% | 0% | LLM-only |
| **TOTAL** | **36.0%** | **31.8%** | deterministic-only, no LLM |

### Full handoff at docs/handoffs/jul15-session-handoff.md

---

## Active Tasks

| Task | Status |
|------|--------|
| Full pipeline eval with **GPU** (end-to-end answer accuracy) | Not started |
| Evaluate binary cascade tree on Kaggle dataset (616 problems) | Done — all 616 have structured I/O, 201 DS, 98 DP, 112 sort/search |
| Run full pipeline with local LLM (gemma-3-1b / qwen) | Not started |
| Validate code_tool_cascade end-to-end in pipeline context | Done — template solvers return directly, LLM paths routed correctly |

---

## Blockers

- ~~code_tool_cascade not wired into pipeline.process()~~ **FIXED — cascaded wired in Jul 15 session**
- No GPU pipeline eval run yet — all numbers are deterministic-only (no local LLM loaded)
- ~~CONTEXT.md stale with v0-v5 data~~ **Updated Jul 15**

---

## Versions

### Submission Log

| Tag | Code | Docker | Lablab Submit | Grader Check | Result |
|-----|------|--------|---------------|-------------|--------|
| initial | Jul 9 ~13:43 CDT | — | ? | — | — |
| `v0-submitted` | Jul 9 14:15 CDT | Built ~14:15, pushed ~14:20 | Re-saved ~14:20 | **Jul 9, 20:54** | **52.6%** FAILED |
| `v1-no-fireworks` | Jul 9 16:38 CDT | Built ~16:38, pushed ~16:40 | Re-saved ~16:40 | **Never checked** — `:latest` was overwritten by v2 before grader pulled it | Skipped |
| `v2-fireworks-030` | Jul 9 17:21 CDT | Built ~17:21, pushed ~17:25 | Re-saved ~17:50 (as `:latest`) | Never checked — overwritten by v2-tagged | Skipped |
| `v2-tagged` | Jul 9 17:21 CDT (same image) | Tagged as `:v2`, pushed ~20:15 CDT | **Re-saved ~20:15 CDT with `:v2`** | Jul 10, 01:51 | **RUNTIME_ERROR** — container crashed during eval |
| `v3-parse-fix` | Jul 10 02:05 CDT | Built ~02:05, pushed ~02:10 | **Re-saved ~02:10 with `:v3`** | — | Same image as v4 below |
| `v4` | Jul 10 02:15 CDT (same image as v3) | Tagged as `:v4`, pushed ~02:15 | **Re-saved ~02:15 with `:v4`** | **Jul 10, 02:37 CDT** | **42.1%** FAILED — greedy deterministic solvers stole tasks from Fireworks |
| `v5-threshold-010` | Jul 10 02:50 CDT | Built ~02:52, pushed ~02:55 | **Re-save as `:v5`** | Pending | Lowered Fireworks threshold from 0.30 to 0.10 |

**Current GHCR tags:**
- `ghcr.io/artemkorolev1/amd-hackathon-submit:latest` = v5-threshold-010 (sha256:0539df0feb4a)
- `ghcr.io/artemkorolev1/amd-hackathon-submit:v5` = v5-threshold-010 (sha256:0539df0feb4a)
- `ghcr.io/artemkorolev1/amd-hackathon-submit:v2` = v2-fireworks-030 (RUNTIME_ERROR, sha256:c1331085)
- `ghcr.io/artemkorolev1/amd-hackathon-submit:v3` = v3-parse-fix (sha256:4ef4802c)
- `ghcr.io/artemkorolev1/amd-hackathon-submit:v4` = v4 clean resubmission (sha256:4ef4802c)

### Version Details

| Tag | Description | Image |
|-----|-------------|-------|
| `v0-submitted` | Fireworks API + ML classifier + 2 deterministic solvers (arithmetic, logic). python:3.12-slim. | `sha256:39f9282574a1` |
| `v1-no-fireworks` | Fireworks removed. 6 deterministic solvers (math, logic, sentiment, NER, factual QA, code debug). python:3.12-slim. | `sha256:1c48a80f817e` |
| `v2-fireworks-030` | Fireworks re-enabled (threshold 0.30). 6 solvers. **RUNTIME_ERROR — `parse_allowed_models()` TypeError.** | `sha256:c1331085c4fd` |
| `v3-parse-fix` | Fixed `parse_allowed_models(raw: str = "")`. Same code otherwise. **42.1% FAILED (checked Jul 10 02:37 CDT).** | `sha256:4ef4802c882d` |
| `v4` | Clean resubmission tag. Same image as v3. **Same 42.1% — greedy deterministic solvers (sentiment, NER, factual QA, code debug) intercepted Fireworks-bound tasks.** | `sha256:4ef4802c882d` |
| `v5-ensemble` | **CODE ONLY — not submitted.** 3-way ensemble classifier (MiniLM-LR, TF-IDF-LR, deterministic) with majority vote. New: `classifier_ensemble.py`, `classifier_minilm.py`. Updated `hybrid_classifier.py`. Eval: 80% stress, 100% mixed. | — |
| `v6-nvidia-bench` | **CODE ONLY — not submitted.** Benchmarked nvidia/prompt-task-and-complexity-classifier: 25% stress, 30% mixed, 99ms per text, 735MB. Not viable — poor category mapping, too large/slow for Docker. | — |
| `v5-threshold-010` | **SUBMITTED.** Fireworks threshold lowered from 0.30 to 0.10. Same code base as v5-tight-gates. Tagged as `:v5` on GHCR. | `sha256:0539df0feb4a` |
| `v7-complexity` | **CODE ONLY — not submitted.** Unified complexity scorer combining 6 signals: MiniLM confidence, TF-IDF confidence, custom binary classifier (trained on 78 curated questions), Bitmorphic heuristics, length/structure, inverted readability. Weighted ensemble achieves +0.128 delta (simple vs complex separation). New: `agent/complexity.py`. | — |

**Switching:** `git checkout <tag>` to go back to any version.
**Current:** `master` branch = v7-complexity (e2932cd).

---

## Version History

| Version | Location | Score | Status | Notes |
|---------|----------|-------|--------|-------|
| v1 | `/home/artem/amd-hackathon` | **52.6%** | Submitted Jul 9 | Failed 80% gate |
| v2 | `/home/artem/amd-hackathon-v2` | — | In progress, NOT pushed | See v2 section below |

---

## Status

| Version | Result | Notes |
|---------|--------|-------|
| **v0-submitted** `58ae181` | **52.6%** FAILED | Checked Jul 9, 20:54. Original Fireworks + 2 solvers. |
| **v1-no-fireworks** `479cf79` | **SKIPPED** | Never evaluated — `:latest` overwritten before grader pulled. |
| **v2-fireworks-030** `e8ca91e` | **RUNTIME_ERROR** | Checked Jul 10, 01:51. `parse_allowed_models()` called with arg, function took none → TypeError crash. |
| **v3-parse-fix** `24dba2b` | **42.1%** FAILED | Checked Jul 10, 02:37 CDT. Greedy deterministic solvers intercepted Fireworks-bound tasks, giving wrong answers. Fixed in v5-tight-gates. |
| **v4** `24dba2b` (same commit) | **42.1%** FAILED | Same result as v3 — identical image, different tag. |
| **v5-threshold-010** `b72f66d` | **PENDING** | Submitted as `:v5`. Threshold lowered to 0.10. |

**Current GHCR :latest = v5-threshold-010 (sha256:0539df0feb4a).** Submitted and waiting for grader.
**Repo:** `github.com/artemkorolev1/amd-hackathon` = PUBLIC (empty placeholder for form).
**Code repo:** `github.com/artemkorolev1/amd-hackathon-submit` = PRIVATE (real code).
Deadline: **July 11, 2026 12:00 PM EDT**. Submissions: 10/hour limit.

## Check/Score Timeline Log

**Team:** Raiders of Vibehalla | **Project:** Router to Vibehalla
**GHCR:** `ghcr.io/artemkorolev1/amd-hackathon-submit:latest`
**GitHub:** `github.com/artemkorolev1/amd-hackathon`

**How to update:** Paste a fresh leaderboard snapshot. I'll parse it and append to this section.

### Julian Day 187 (Jul 8) — Snapshot 1

| Hour | Events | Running Total |
|------|--------|--------------|
| Jul 08 04:00 | 1 | 2 |
| Jul 08 16:00 | 1 | 3 |
| Jul 08 17:00 | 1 | 4 |

### Julian Day 188 (Jul 9–10) — Snapshot 4 (~00:27 Jul 10, 28 new events since Snapshot 3)

| Hour Window | Events | Running Total | Notes |
|-------------|--------|--------------|-------|
| Jul 09 03:00 | 1 | 5 | |
| Jul 09 08:00 | 1 | 6 | |
| Jul 09 10:00 | 2 | 8 | |
| Jul 09 11:00 | 1 | 9 | |
| Jul 09 12:00 | 1 | 10 | |
| Jul 09 13:00 | 2 | 12 | |
| Jul 09 15:00 | 7 | 19 | |
| Jul 09 16:00 | 4 | 23 | |
| Jul 09 17:00 | 5 | 28 | |
| Jul 09 18:00 | 3 | 31 | |
| Jul 09 19:00 | 2 | 33 | |
| Jul 09 20:00 | 3 | 36 | **Router to Vibehalla checked 20:54, 52.6%** |
| Jul 09 21:00 | 12 | 48 | |
| Jul 09 22:00 | 12 | 60 | |
| Jul 09 23:00 | 29 | 89 | |
| Jul 10 00:00 | 28 | 117 | Continuous overnight — 56/hr rate |

**Running total: ~117 submissions**
**Router to Vibehalla:** Still 52.6% (checked 20:54). **v2 re-save at ~22:50 still pending.**
**New leaderboard entry:** Lazy Lives track 1 - vbest (89.5%, 5,560 tokens) — #08.

**Observations:**
- Grader runs continuously, not in batches. Processing rate varies: peak ~56/hr overnight.
- Submissions cycle through multiple check states: PULL_ERROR → RUNTIME_ERROR → ACCURACY_GATE_FAILED as images get fixed and re-pulled.
- Entries with corrected error types appear to get re-checked promptly after re-save.

### Important: Push vs. Re-save

From Discord intel: **Pushing a new Docker image to GHCR does NOT trigger a re-grade on its own.** You must open the submission form on lablab.ai and click "Update Submission" to bind the grader to the new image.

**Verification:** Check GHCR package download counter. If it increments after re-saving, the grader pulled your image.

---

## Active Tasks

| Task | Who | Status |
|------|-----|--------|
| Retrain v2 classifier (`python train_classifier.py`) | Next session | Pending |
| End-to-end pipeline test with sample tasks | Next session | Pending |
| Dockerfile: add Ollama + Qwen3-4B-2507 | Next session | Pending |
| Push v2 image + submit | Next session | Blocked (needs above) |

---

## Decisions

- **2026-07-10 (v4 session)** v2 RUNTIME_ERROR root cause: `parse_allowed_models()` in config.py took zero args, but main.py v2 commit called it with `raw_models` arg → TypeError crash. Only triggered when grader set FIREWORKS_API_KEY + ALLOWED_MODELS (both non-empty). Local smoke tests never set those env vars so the bug was invisible. Fix: changed signature to `parse_allowed_models(raw: str = "")`.
- **2026-07-10 (v4 session)** Repo visibility: `amd-hackathon` = PUBLIC (empty placeholder for lablab form), `amd-hackathon-submit` = PRIVATE (real code + Docker image staging). GHCR package `ghcr.io/artemkorolev1/amd-hackathon-submit` stays public (required for grader pull). Package visibility is independent of repo visibility.
- **2026-07-10 (v4 session)** Fireworks wiring cross-checked against participant guide + 4 accepted projects (NovaAI, TokenRouter, TokenForge, Hybrid Token Router). Structurally identical: same base image, env var handling, reasoning suppression, urllib approach. No Fireworks credits to test actual API call — but code path matches proven patterns.
- **2026-07-09** Hybrid classifier: TF-IDF + LogisticRegression with deterministic fallback. ML routes at target 80%+.
- **2026-07-09** Research delegated to Claude Code directly (Exa via Agent-Reach). Hermes orchestrates/commits.
- **2026-07-09** Docker image switched from Ubuntu 22.04 + local llama.cpp/Qwen3.5-4B to python:3.12-slim + Fireworks API only. No local model in image (472MB vs 5.7GB).
- **2026-07-09** Fireworks threshold lowered from 0.65 to 0.30 — 3x more tasks route to Fireworks instead of local/fallback.
- **2026-07-09** Four new deterministic solvers added: sentiment (weighted keywords), NER (biomedical + general), factual QA (70+ facts + context extraction), code debugging (10 patterns).
- **2026-07-09** Submission standards defined: python:3.12-slim, GHCR public package, verify imports, smoke test before push. Saved as skill `hackathon-submit-verify`.
- **2026-07-09** From now on: no pushes without explicit permission. Claude Code agents get narrower prompts to avoid scope creep.
- **2026-07-09 (v2 session)** Root cause of 52.6%: answer quality failing LLM judge, NOT classifier. Judge (reportedly gpt-oss-120b) grades content PASS/FAIL. Raw output with no system prompt fails even when logic is correct.
- **2026-07-09 (v2 session)** Three Fireworks model names in routing table were wrong (dots must be `p`: `qwen-3.7-plus`→`qwen3p7-plus`, `kimi-k2.7-code`→`kimi-k2p7-code`; `deepseek-v3.2` doesn't exist → `deepseek-v4-pro`). Fixed in v2.
- **2026-07-09 (v2 session)** v2 strategy: category-specific system prompts + answer post-processing (strip fences, extract final answer line) — highest impact. Classifier improvements secondary.

---

## Blockers

- v2 classifier model (`models/classifier.pkl`) is stale — needs retrain with new `train_classifier.py` before any test or push.
- ALLOWED_MODELS list not published yet — injected at runtime on July 11 launch day. Router handles this dynamically.
- Previous blocker (CPU inference bottleneck) resolved by removing local model from Docker image.

---

## v2 Summary (session 2026-07-09)

### What changed
| File | Change |
|------|--------|
| `src/main.py` | Added system prompts per category, answer post-processing, always writes results.json |
| `src/prompts.py` | NEW — system prompts + post-processing (strip fences, extract Answer: line, sentiment format) |
| `src/routing.py` | Fixed 3 broken model names, added GLM models, improved normalize_model_name |
| `train_classifier.py` | New datasets: MBPP, DebugBench, CoNLL2003, SST-2, tweet_eval (~3800 samples vs 1600) |
| `knowledge/` | 5 research files from 3 parallel agents — do not re-research |

### What still needs doing (next session)
1. `cd /home/artem/amd-hackathon-v2 && source venv/bin/activate && python train_classifier.py` — retrain classifier, verify ≥80%
2. End-to-end test with sample tasks (see `config/sample_tasks.json`)
3. Dockerfile: add Ollama + Qwen3-4B-2507 for zero-token local inference
4. Push image + submit

### Confirmed callable Fireworks models (live-probed July 7)
```
accounts/fireworks/models/gpt-oss-20b        $0.07/$0.30  cheap, NER/sentiment
accounts/fireworks/models/gpt-oss-120b       $0.15/$0.60
accounts/fireworks/models/deepseek-v4-flash  $0.14/$0.28  fast general
accounts/fireworks/models/deepseek-v4-pro    $0.90/$2.70  best reasoning
accounts/fireworks/models/glm-5p1            ~$0.20/$0.80
accounts/fireworks/models/glm-5p2            ~$0.35/$1.20
accounts/fireworks/models/kimi-k2p6          ~$0.75/$3.00
accounts/fireworks/models/kimi-k2p7-code     $0.95/$4.00  best code
accounts/fireworks/models/qwen3-235b-a22b    MoE mid-tier
accounts/fireworks/models/qwen3p7-plus       Qwen 3.7+
```
`kimi-k2p5` is DEPRECATED (removed late June 2026).

---

## Ideas

- Dynamic thread scaling: probe `nproc` at startup, adjust `--threads` and concurrent request pool accordingly
- Multiple llama-server instances sharded by prompt category
- Shared context file as drop-box between sessions (this file)

---

## Future Tasks (from code review 2026-07-09)

### Priority (post-hackathon)
- **Simplify classification pipeline** — 4 layers (ML → deterministic → Bitmorphic → catch-all) accreted rather than designed. ML covers only 8/19 categories. Reconsider the layering.
- **Bump `LLAMA_N_CTX` from 2048 to 8192+** — Qwen3.5-4B supports 32K native. 2048 clips code gen and long-form tasks. Cost: ~1-2 GB more RAM during inference.
- **Remove dead code**: `verify.py` (imported but never called), `LOCAL_CAPABLE_CATEGORIES` (defined, unused), `ML_HIGH_CONFIDENCE_CATEGORIES` (unused), Bitmorphic `route` field (never consumed).

### Quick fixes
- **Close eval sandbox gap in `tools.py`** — `()` + `.` + `[]` are all allowed through regex gate, so `().__class__.__mro__` chain works despite `__builtins__={}`. Switch to `ast.literal_eval` or a safe expression parser.
- **Fix `python_executor` env wipe** — `env={"PYTHONIOENCODING": "utf-8"}` strips PATH/HOME. Merge with `os.environ | {...}`.
- **Hook up `verify()`** — `main.py` imports it, never calls it. Wire it into the solve pipeline.
- **Fix Bitmorphic case-sensitivity bug** — `bitmorphic.py:213` passes `prompt` instead of `lower` to `_keyword_signal`, causing `structured_output` signal to miss uppercase keywords (JSON, CSV, YAML).
- **Remove `--no-warmup` from `start.sh`** — first inference pays cold-start penalty. On 19 sequential tasks, minor but every second counts on deadline.
- **Add checksum verification to Dockerfile model download** — 2.6 GB with zero integrity check.

### Investigate
- `--reasoning off` in `start.sh` vs `reasoning` param in `local.py` — server-level flag makes per-task reasoning switching a no-op. Confirm intent.
- Deterministic solvers run twice (pre-solve loop + inside `_solve_task`). Remove redundancy.
- `classifier_ml.py` default paths point to non-existent `models/` dir — latent crash for bare instantiation.

---

## Resource Probe

Run at startup and log results here:
```
# nproc:
# /proc/cpuinfo cores:
# /proc/meminfo MemTotal:
```
