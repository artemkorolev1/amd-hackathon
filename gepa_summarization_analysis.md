# GEPA Judge+Analyze: text_summarization (v12d baseline)

## 1. JUDGE — Current State

### Accuracy Estimates

| Solver | Accuracy | Notes |
|--------|----------|-------|
| Sumy (extractive) | **0%** (not wired) | Sumy package NOT installed — `_SUMY_AVAILABLE = False`. All 4 strategies in `solve_summarization()` silently return `None`. |
| Lead-biased extractive fallback | ~0% (unreachable) | Only runs if Sumy is available — dead code without it. |
| First-N-sentences fallback | ~0% (unreachable) | Same: inside Sumy strategies. |
| Local LLM (qwen2.5-1.5b) | ~62% | From architecture doc |
| FW (gpt-oss-120b via minimax-m3) | ~92% | From architecture doc |
| **Effective total** | **~62%** | Local only — deterministic solver does nothing |

### Prompt Tier Analysis

In `dynamic_prompts.py` (lines 268-296), summarization has 3 tiers (low/medium/high):

- **Low**: "Summarize the text. Strictly obey ANY length constraint... If no length is given, output at most 2 sentences. No preamble... Output ONLY the summary text."
- **Medium**: Adds "Capture all key points" + ups max to 3 sentences.
- **High**: Adds factual accuracy + "Preserve important names, numbers, and relationships."

**Issues:**
1. All tiers lack entity/keyword emphasis — high tier mentions names/numbers/relationships, but low/medium don't. This hurts XY/XSUM style where the expected answer is a single headline containing specific entities.
2. No differentiation between extractive vs abstractive tasks. CNN/DailyMail expects extractive bullet points; XSUM expects abstractive headline.
3. Length constraints are soft ("at most 2/3 sentences") but many eval prompts say "in exactly 2 sentences" — the prompt should detect and obey exact constraints.
4. No multi-source handling in system prompts — that's only done in `_fireworks_escalate()` for FW.

### Solver Routing (pipeline.py `process` method, line 596)

Current routing through `_run_deterministic()`:

1. `category == "summarization"` and `complexity < simple_max` → iterates ALL deterministic solvers (arithmetic, logic, sentiment, NER, factual, code_debug, summarization) — wasteful
2. `solve_summarization(prompt, "summarization")` called — but Sumy not installed → returns `None` → falls through to local LLM
3. If complexity too high or deterministic fails → goes to `_infer()` (local LLM) or role-based dispatch
4. `_fireworks_escalate()` called only when `role == "api_llm"` or local LLM fails

**Issues:**
- The loop iterates ALL solvers even for summarization — pass-through of all 7 solvers wastes cycles.
- Sumy solver is completely dead without `sumy` package installed.
- FW summarization uses `max_tokens=80` in `_fireworks_escalate` — very tight for multi-source synthesis.
- `_fireworks_escalate()` summary_prompt overrides the system prompt from `fw_router.py`; the FW caveman prompt ("Only the summary obeying prompt's length constraint. No intro.") is completely bypassed for summarization.

### FW Routing (fw_router.py)

- Summarization routes to **minimax-m3** with 500 max_tokens
- Caveman prompt: "Only the summary obeying prompt's length constraint. No intro."
- Long prompt with SOURCE → forced minimax-m3 summarization guard (line 147-152)
- No kimi-k2p7-code routing for summarization (reasoning models not needed)

**Issues:**
- FW routing table routes summarization to minimax-m3, but `_fireworks_escalate()` ignores `cfg.system_prompt` and sends its own hardcoded summary_prompt.
- 500 max_tokens is generous but `_fireworks_escalate()` overrides to `max_tokens=80`.
- No complexity-based model upgrade (e.g., summarization with complexity>0.7 → deepseek).

### Secondary Summarization Classifier

`resolve_summarization()` in `secondary_summarization.py` is a strength:
- 20+ regex patterns for document structure detection
- 4-case overrides for logic→summarization, math→summarization, code_gen→summarization, factual→summarization
- Counter-signals for analytical prompts (solve/prove/explain why/compare)
- Entity density and length scoring

**Verdict:** The secondary classifier is well-designed. The scoring thresholds (doc_score>=4.0, math_score<2.0, etc.) look reasonable.

### Grading Method

`summarization_grade()` in `scripts/grade_answer.py` (lines 146-189):
- 4-signal cascade: fuzzy_match → entity recall (≥50% or ≥2) → keyword overlap (≥40%) → numeric overlap (any shared numbers)
- **But:** NOT wired into `runner/evaluate.py` — that only imports `fuzzy_match` and `grade_answer` from `scripts.evaluate`, not `summarization_grade`.
- `runner/evaluate.py` uses `grade_answer()` (generic fuzzy_match) for all categories including summarization.
- So summarization grading uses the **standard fuzzy_match cascade** which is too strict for free-form summaries.

## 2. ANALYZE — Root Causes

### Critical Issues

| # | Issue | Impact | Root Cause |
|---|-------|--------|------------|
| 1 | **Sumy NOT installed** | Deterministic solver does nothing (0% coverage) | Dependency missing; `solve_summarization()` silently returns None |
| 2 | **summarization_grade() not wired** | Summarization graded with standard fuzzy_match (too strict) | `runner/evaluate.py` imports only `grade_answer`, not `summarization_grade` |
| 3 | **FW max_tokens=80 override** | Multi-source summaries truncated; can't synthesize | `_fireworks_escalate()` hardcodes 80 tokens, ignores fw_router's 500 |
| 4 | **Prompt tiers lack entity focus** | Low/medium tiers don't instruct for entities | Design gap in `dynamic_prompts.py` |

### Medium Issues

| # | Issue | Impact |
|---|-------|--------|
| 5 | `_run_deterministic` iterates ALL solvers | Wastes 6 solver calls per summarization before hitting the right one |
| 6 | No extractive vs abstractive detection | CNN/DM (bullets) and XSUM (headline) get same treatment |
| 7 | `_fireworks_escalate()` hardcodes summary prompt | Bypasses the caveman prompt optimizations in fw_router.py |
| 8 | Length constraints not extracted from prompt | "in exactly 2 sentences" in eval data not parsed | 

### Data Set Breakdown

**summarization_train.json** (366 entries):
- 177 CNN/DailyMail (extractive bullets) — expected answers are 3-bullet summaries
- 189 XSUM (abstractive headlines) — expected answers are 1-sentence headlines
- All hard difficulty
- Prompts: "Summarize the following news article:" (CNN/DM) or "Summarize the following article in 1-2 sentences:" (XSUM)

**summarization_combined_25.json** (25 entries):
- Subset of above, for smaller eval runs

**Heldout sets**: 2 summarization in eval_longform_20, 6 in validation-v3

## 3. PROPOSE — Specific Changes

### P0: Fix Sumy Installation (Critical)

```bash
pip install sumy
```
This immediately activates all 4 solver strategies in `solve_summarization()`:
- Lead-biased LexRank → ensemble voting → individual algorithm chain → first-N-sentences fallback

Estimated accuracy lift: From 0% to ~40-50% on extractive tasks (CNN/DM) where lead-biased extraction works well.

### P1: Wire summarization_grade() into evaluate.py (Critical)

Change `runner/evaluate.py` line 30:
```python
from scripts.evaluate import fuzzy_match, grade_answer
# → from scripts.grade_answer import fuzzy_match, grade_answer, summarization_grade
```

Add summarization grading path (after keyword coverage check, before standard grade_answer):
```python
if category == "summarization":
    passed = summarization_grade(answer, expected)
    reason = "Passed (summarization_grade)" if passed else f"summarization_grade failed"
```
Estimated accuracy lift: +5-10% on abstractive (xsum) tasks where entity/keyword overlap catches matches that strict fuzzy_match misses.

### P2: Remove Summary Prompt Override in _fireworks_escalate (High)

Delete lines 458-471 in `pipeline.py` (hardcoded summary_prompt) and instead pass `cfg.system_prompt` from `fw_router.py`:
```python
if category == "summarization":
    cfg = _fw_route("summarization", prompt, complexity)
    answer = self._fw.solve(
        cfg.model_id, cfg.system_prompt, prompt,
        max_tokens=cfg.max_tokens, temperature=cfg.temperature,
        prefill=cfg.prefill, task_type="summarization",
        timeout=self.cfg.fireworks_timeout_s,
    )
```
This uses the optimized caveman prompt "Only the summary obeying prompt's length constraint. No intro." and respects 500 max_tokens.

### P3: Optimize Prompt Tiers for Entity Extraction (Medium)

Update `dynamic_prompts.py` summarization tiers:
- **Low** (extractive bullet): Add "Capture key names, numbers, and places. Output 1-2 sentences." (currently missing)
- **Medium**: Already mentions "key points" — add "Include exact names and numbers"
- **High**: Already has good entity focus — keep as-is

Also add length constraint extraction:
```python
# Before constructing prompt, extract length constraint
length_match = re.search(r'in (exactly|1-2|at most) (\d+) (sentence|word|bullet)', prompt, re.I)
if length_match:
    constraint = f"Output exactly {length_match.group(2)} {length_match.group(3)}(s)."
    # Append to system prompt
```

### P4: Optimize Solver Routing (Medium)

- Change `_run_deterministic` to skip solvers that can't handle the category:
  ```python
  def _run_deterministic(self, category, prompt):
      dispatch = {
          "summarization": [solve_summarization],
          "math": [solve_arithmetic],
          ...
      }
      for fn in dispatch.get(category, []):
          ...
  ```
- After Sumy is installed, add a pre-filter: if `content < 400 words` AND `no SOURCE markers` AND `no analytical keywords` → Sumy direct. Otherwise → LLM or FW.

### P5: Extractive vs Abstractive Detection (Low-Medium)

Add detection in pipeline:
```python
# CNN/DM style: "Summarize the following news article:" → extractive (bullet points)
# XSUM style: "Summarize the following article in 1-2 sentences:" → abstractive (headline)
is_cnn_dm = bool(re.search(r'Summarize the following news article', prompt, re.I))
is_xsum = bool(re.search(r'Summarize the following article in', prompt, re.I))
```
Route extractive (CNN/DM) to Sumy + local LLM, abstractive (XSUM) to FW directly.

### P6: Routing Threshold Tuning (Medium)

Current architecture doc recommends:
- Input < 400 words AND no SOURCE markers → Sumy (after install)
- SOURCE markers → FW direct
- Long input (>800 words) → FW (local 1.5B truncates)

Implement these thresholds by adding a `can_summarize_extractive()` pre-filter check from `deterministic_filters.py`.

### Estimated Accuracy Impact

| Change | Estimated Lift | Cumulative |
|--------|---------------|------------|
| P0: Install sumy | +0% → ~40% (on extractive) | 62% → ~70% |
| P1: summarization_grade | +5-10% (on abstractive) | ~70% → ~75% |
| P2: Fix FW prompt | +5% (quality on FW) | ~75% → ~78% |
| P3-P5: Prompt/routing | +3-5% | ~78% → ~80% |
| P6: Thresholds | +2% | ~80% → ~82% |

**Target: 80-85%** (up from current estimate of ~62-70%)

## 4. Files to Modify

1. `/home/artem/dev/amd-hackathon/agent/pipeline.py` — lines 454-480 (`_fireworks_escalate`), lines 543-555 (`_run_deterministic`)
2. `/home/artem/dev/amd-hackathon/agent/dynamic_prompts.py` — lines 268-296 (summarization tiers)
3. `/home/artem/dev/amd-hackathon/runner/evaluate.py` — line 30 (import) and lines 222-230 (grading dispatch)
4. `/home/artem/dev/amd-hackathon/agent/solvers/deterministic.py` — line 2987 (`_SUMY_AVAILABLE` will auto-set True after install)
5. Environment: `pip install sumy`
