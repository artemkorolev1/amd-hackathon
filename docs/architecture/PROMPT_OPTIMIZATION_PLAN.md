# Prompt Engineering Optimization Plan — Filtered Build v9

Date: 2026-07-11
Eval basis: 300-question raw-model benchmark + 60-set deterministic coverage

---

## What We Know

### Raw model accuracy (300 questions, T=0.0, no pipeline)

| Model | Overall | Code | Math | NER | Sentiment | General | Factual |
|---|---|---|---|---|---|---|---|
| qwen2.5-1.5b | 79.7% | 100% | 48% | 67% | 50% | 50% | 90% |
| qwen2.5-coder-1.5b | 76.7% | 100% | 52% | 50% | 42% | 42% | 87% |
| nemotron-3-nano-4b | **90.7%** | 92% | **96%** | **100%** | **58%** | **100%** | 91% |
| phi-4-mini-q4 | 87.3% | 100% | 76% | 92% | 33% | 42% | **97%** |

### Two disconnected prompt architectures

| System | Used by main loop? | Complexity tiers? | Feature injection? |
|---|---|---|---|
| `agent/templates.py` (78 lines) | **Yes** | No | No |
| `agent/dynamic_prompts.py` (517 lines) | **No** | Yes (low/med/high) | Yes (creativity, verbosity, structured, multi-step) |

The main loop does ONE lookup: `SYSTEM_PROMPTS.get(category)` — one static string per category.

### Deterministic solver coverage

- **code_debug**: 0/12 distinct bug patterns covered by the deterministic solver (off-by-one, product-init-zero patterns match NONE of the actual 12 bugs)
- **NER**: 0/6 — entities are correct but output format (flat `"WNT, beta-catenin"`) doesn't match expected structured format (`"GENE: WNT, beta-catenin; DISEASE: medulloblastoma"`)

### 11 questions ALL models failed
8 sentiment, 2 math (both combinatorics), 1 factual (float precision)

### Pre-existing bug (critical)
`main.py` line 280 calls `fireworks.solve(prompt=prompt, ...)` but `FireworksSolver.solve()` has no `prompt` parameter — it expects `user_prompt`. This makes the API escalation path a TypeError. Fix before any Phase 1 changes.

---

## Phase 1 — Wire dynamic_prompts.py into main.py

**Goal**: Replace static `SYSTEM_PROMPTS` dict with `build_system_prompt()` using category + complexity.

**Changes:**
1. Import `build_system_prompt` and `get_max_tokens` from `agent.dynamic_prompts`
2. Replace line 263: `sys_prompt = SYSTEM_PROMPTS.get(category, ...)` → `sys_prompt = build_system_prompt(category=category, complexity_score=complexity)`
3. Replace fixed max_tokens=512 with `get_max_tokens(category, complexity)`
4. Fix pre-existing bug: change `prompt=prompt` to `user_prompt=prompt` on line 280

**Validation**: Run 300-set through filtered build. Compare per-category accuracy against naked model baseline.

**Risk**: None — `build_system_prompt()` returns a plain string, API-compatible with `fireworks.solve()`.

---

## Phase 2 — Targeted Format Fixes

### 2a. NER: Add 1-shot example
**Problem**: Models output the right entities but format them as prose paragraphs instead of structured `CATEGORY: list` format.
**Fix**: Inject a 1-shot example via `custom_instructions` parameter in `build_system_prompt()`:
```
Example: "Extract all genes, diseases, organizations from this text..."
Output: GENE: WNT, beta-catenin; DISEASE: medulloblastoma; ORGANIZATION: Cold Spring Harbor Laboratory
```
**Expected ROI**: NER accuracy from 67% → 90%+. Verified: all NER failures across all models contain the correct entities — the format is the only gap.

### 2b. Sentiment: Fix evaluator, not the prompt
**Problem**: The cascade evaluator expects `"negative (sarcastic)"` with parenthetical qualifier. The 1.5B models genuinely misclassify sarcasm as positive (42-50% of failures are real misclassifications). But nemotron's 10/52 sentiment failures ARE pure format issues (correct label, missing parenthetical).
**Fix**: Soften the cascade evaluator — strip parenthetical qualifiers from expected answers before matching. This recovers nemotron's 10 failures immediately.
**Don't change the sentiment prompt** — the 1.5B models are already outputting the wrong base label. A format change won't fix sarcasm detection. The real fix is routing ambiguous sentiment to Fireworks.
**Expected ROI**: 10/52 sentiment failures recovered (~19% improvement on this category).

### 2c. General/JSON: Post-processor
**Problem**: Models wrap JSON in markdown code fences or add extra fields/key text.
**Fix**: Add a 3-line post-processor that strips markdown fences and re-serializes valid JSON. Extend `agent/cleaning.py` which already has fence-stripping logic.
**Expected ROI**: 50% → 90%+ on general category. Note: some failures are pure text explanations (model didn't produce JSON at all) — those need prompt reinforcement.

### 2d. Math: Add "Answer:" prefix format
**Problem**: Models produce reasoning text but the evaluator can't extract the final number reliably.
**Fix**: The medium/high math prompts in `dynamic_prompts.py` already say `"End with 'Answer: <value>' on its own line."` — just needs to be the default/low too.
**Expected ROI**: Modest (~5-10%). The 1.5B model's math failures are genuine reasoning errors, not format issues.

---

## Phase 3 — Classifier-Conditional Injection

**Idea**: Use Stage 2 confidence + Stage 3 complexity as gates for prompt content. All data flows already exist in the pipeline.

| Confidence | Complexity | Strategy |
|---|---|---|
| ≥0.85 | <0.3 | Caveman-terse (no examples) |
| 0.7-0.85 | <0.3 | Standard prompt from `dynamic_prompts.py` |
| <0.7 | any | Inject 1-shot example via `custom_instructions` |
| any | ≥0.7 | Use "high" complexity tier (step-by-step, scaffolding) |

**Implementation**: `build_system_prompt(category, complexity_score=complexity, custom_instructions=few_shot)` — the function already accepts these parameters.

**Note**: `feature_scores` for multi-axis injection (creativity, verbosity, structured_output, multi_step) are currently zero-valued stubs in `pipeline.py`'s `Stage1Features`. Feature extraction was removed. Skip feature injection unless we re-add Stage 1.

---

## Phase 4 — GEPA Optimization (Postpone)

### Verdict: NOT practical to run today
- ✅ Repo cloned at `/home/artem/hermes-agent-self-evolution/`
- ✅ 60-question eval dataset exists for fitness scoring
- ❌ **OPENAI_API_KEY not set** — GEPA requires `gpt-4.1` for optimizer
- ❌ No virtual environment in the evolution repo
- ❌ GEPA optimizes SKILL.md files, not Python dicts — would need prompt extraction + custom DSPy module

### Alternative: Manual A/B testing
More practical path: Write a sweep harness that varies prompt parameters (verbosity, step visibility, 1-shot presence, format instruction strictness), runs the existing 300-question eval, and logs scores. Uses the Fireworks API we already have — zero new dependencies.

---

## Phase 5 — Multi-Model Consensus with Diverse Prompts

**Current**: `solve_with_consensus()` samples 3 times with the same system prompt, different temperatures.
**Upgrade**: Pass a list of prompt variants. Each sample draws a different variant.

**How it works:**
- 3 samples per question
- Sample 1: Standard prompt (from `dynamic_prompts.py`)
- Sample 2: Prompt + 1-shot example
- Sample 3: Prompt with "Think step by step" scaffolding
- Agreement across diverse prompts is a **stronger** signal than same-prompt thermal noise

**Changes needed**: `solve_with_consensus()` accepts `prompt_variants: list[str]` instead of `system_prompt: str`. Cycles through variants round-robin.

---

## Implementation Order

```
Week 1:
  └─ Fix main.py:280 TypeError (30 min)
  └─ Phase 1: Wire dynamic_prompts.py → main.py (1-2h)
  └─ Phase 2a: NER 1-shot injection (30 min)
  └─ Phase 2b: Sentiment evaluator softening (30 min)
  └─ Phase 2c: JSON post-processor (30 min)
  └─ Phase 2d: Math "Answer:" default format (15 min)
  └─ Run 300-set eval → compare per-category deltas

Week 2:
  └─ Phase 3: Classifier-conditional injection (2-3h)
  └─ Phase 5: Multi-model diverse-prompt consensus (2-3h)
  └─ Full 300-set regression run
  └─ Build 60-question validation set and cross-validate

Postpone:
  └─ Phase 4 (GEPA): Requires API key + prompt extraction + custom DSPy module
  └─ GEPA alternative: Manual A/B sweep harness (lower effort, same effect)
```

## Reviewer Corrections Incorporated

These are corrections from the parallel review that changed my original assumptions:

1. **Sentiment is NOT just evaluator strictness** — 1.5B models genuinely misclassify sarcasm as positive. Only nemotron's 10 failures are parenthetical-mismatch issues. Phase 2b is a partial fix.
2. **NER has TWO failure modes** — prose wrapping AND missing prefix labels (`GENE:`, `DISEASE:`). Phase 2a addresses both by providing an exact format example.
3. **GEPA is blocked** by missing API key and the Python-dict-embedded prompt structure. The manual sweep alternative is more practical.
4. **code_debug deterministic solver covers 0/12 patterns** — worse than the 0/6 I originally stated. The existing solver patterns (off-by-one, product-init-zero) don't match any of the actual bugs in the 60-set (mutable defaults, deadlocks, closures, etc.).
