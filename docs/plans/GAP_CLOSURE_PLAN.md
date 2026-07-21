# Gap-Closure Plan — Data & Evaluation Benchmarks

> Generated: 2026-07-11
> Status: Complete — all actions executed
> Author: Hermes Agent

---

## 1. CURRENT STATE

### 1.1 Category Counts (Stage2/Unified — train split)

| Category | Current | Target | Gap | Fill Strategy |
|----------|---------|--------|-----|---------------|
| math | 22,399 | 5K+ | ✅ Adequate | — |
| logic | 19,605 | 5K+ | ✅ Adequate | — |
| factual | 13,510 | 5K+ | ✅ Adequate | — |
| sentiment | 6,005 | 5K+ | ✅ Adequate | — |
| ner | 6,051 | 5K+ | ✅ Adequate | — |
| summarization | 6,105 | 5K+ | ✅ Adequate | — |
| **code_gen** | **213** | **5K+** | **❌ GAP** | **Downloaded 38,798 new items** |
| **code_debug** | **7** | **2K+** | **❌ CRITICAL GAP** | **164 HumanEvalPack + need synthetic** |

### 1.2 Source Files Downloaded This Session

| File | Items | Content | Category |
|------|-------|---------|----------|
| `python_code_instructions_18k_alpaca.jsonl` | 18,612 | Instruction → code (write functions) | code_gen |
| `codealpaca_20k.jsonl` | 20,022 | Instruction → code (general code gen) | code_gen |
| `humanevalpack_gen.jsonl` | 164 | Function spec → solution (HumanEval) | code_gen |
| `humanevalpack_debug.jsonl` | 164 | Buggy code → fixed code (6 bug types) | code_debug |
| **Total new code_gen** | **38,798** | | |
| **Total new code_debug** | **164** | | |

### 1.3 Evaluation Dataset Inventory (HF Benchmarks)

All queried and verified accessible:

| Benchmark | Dataset ID | Items | Tool Coverage | In eval_deterministic.py? |
|-----------|-----------|-------|---------------|--------------------------|
| GSM8K | openai/gsm8k | 1,319 | arithmetic → math | ✅ Yes |
| SVAMP | nguyen-brat/svamp | 1,000 | arithmetic → math | ✅ Yes |
| MathQA | allenai/math_qa | 4,475 | arithmetic → math | ✅ Yes |
| LogiQA | lucasmccabe/logiqa | 7,376 | logic → propositional logic | ✅ Yes |
| SST-2 | stanfordnlp/sst2 | 872 | sentiment | ✅ Yes |
| IMDB | stanfordnlp/imdb | 25,000 | sentiment | ✅ Yes |
| NCBI Disease | ncbi/ncbi_disease | 941 | NER | ✅ Yes |
| SQuAD 2.0 | rajpurkar/squad_v2 | 11,873 | factual_qa | ✅ Yes |
| HumanEval | openai/openai_humaneval | 164 | code_gen | ✅ Yes (as debug) |
| HumanEvalPack | bigcode/humanevalpack | 164 | code_debug + code_gen | ✅ Yes (as debug) |
| MBPP | google-research-datasets/mbpp | 500 | code_gen | ❌ Not yet |
| XSum | EdinburghNLP/xsum | 11,334 | summarization | ❌ Not yet |
| CNN/DailyMail | abisee/cnn_dailymail | 11,490 | summarization | ❌ Not yet |
| MATH | qwedsacf/competition_math | 12,500 | math | ❌ Not yet |

---

## 2. CODE_GEN GAP CLOSURE

### 2.1 Dataset Research

Three datasets were researched per the task:

| Dataset | Available? | License | Format | Match to code_gen? |
|---------|-----------|---------|--------|-------------------|
| **iamtarun/python_code_instructions_18k_alpaca** | ✅ Yes (18,612 items) | No explicit license (public HF, similar CC-BY) | `instruction` + `input` + `output` | **Excellent** — write functions from natural language specs |
| **codeparrot/apps** | ⚠️ Script-based, deprecated HF loader (5K items) | MIT | Competitive programming problems | Good but harder to load |
| **sahil2801/CodeAlpaca-20k** | ✅ Yes (20,022 items) | CC-BY-4.0 | `instruction` + `input` + `output` | **Excellent** — general code generation tasks |

### 2.2 What Was Downloaded

**Primary: `iamtarun/python_code_instructions_18k_alpaca`**
- 18,612 items — write Python functions from instruction specs
- Format: `instruction` (spec), `input` (optional test data), `output` (Python solution)
- Saved to: `sources/python_code_instructions_18k_alpaca.jsonl`

**Secondary: `sahil2801/CodeAlpaca-20k`**
- 20,022 items — general code generation (algorithms, data structures, scripting)
- License: CC-BY-4.0 (safe)
- Format: `instruction` + `input` + `output` (all code tasks)
- Saved to: `sources/codealpaca_20k.jsonl`

**Tertiary: `bigcode/humanevalpack` (code_gen split)**
- 164 items — HumanEval function-level code generation
- Saved to: `sources/humanevalpack_gen.jsonl`

**Total code_gen added: 38,798 items** (exceeds 5K target by 7.7x)

### 2.3 Recommended Integration

1. Map all three source files to unified format: `prompt`, `expected_answer`, `category="code_gen"`, `source`
2. Distribute across train/val/test (e.g., 80/10/10)
3. The existing Magpie-Phi3 mapping (50K items already in stage2) also contributes some code_gen items — check `task_category` for `code_generation` matches

---

## 3. CODE_DEBUG GAP CLOSURE

### 3.1 Dataset Research

| Dataset | Available? | Items | Notes |
|---------|-----------|-------|-------|
| **bigcode/humanevalpack** | ✅ Yes | 164 | 6 bug types, canonical + buggy code, tests |
| **iamtarun/python_code_instructions_18k_alpaca** | ✅ Yes | 18,612 | Could inject synthetic bugs |
| **sahil2801/CodeAlpaca-20k** | ✅ Yes | 20,022 | Could inject synthetic bugs |
| Rtian/DebugBench | Hard to access | ~722 | Small, not verified accessible |

### 3.2 What Was Downloaded

**`bigcode/humanevalpack` (debug split)**
- 164 items — each has `buggy_solution` (buggy code), `canonical_solution` (fixed), `bug_type`, `tests`
- Bug types: missing logic (33), operator misuse (25), excess logic (31), variable misuse (23), value misuse (44), function misuse (8)
- Saved to: `sources/humanevalpack_debug.jsonl`
- **This is the seed set — 164 is below the 2K target**

### 3.3 Code Debug Synthetic Generation Strategy

To reach 2K+ items, generate synthetic code_debug prompts:

| Method | Est. Yield | Effort | Description |
|--------|-----------|--------|-------------|
| **Bug injection into code_gen** | **~3,000–5,000** | Medium | Take working code from CodeAlpaca-20k or python_code_instructions, inject bugs using 6 bug types from HumanEvalPack taxonomy |
| **HumanEvalPack cross-product** | ~984 | Low | Take 164 canonical solutions × 6 bug types each (but some combos don't make sense) |
| **LLM-generated debug prompts** | ~2,000 | High | Use Kimi K2.7 to generate buggy code pairs |

**Recommended approach (immediate):** Create a `generate_debug_data.py` script that:
1. Selects 2,000 working code snippets from `python_code_instructions_18k_alpaca`
2. For each, randomly picks a bug type from HumanEvalPack's 6 types
3. Applies a transformation to inject that bug:
   - *operator misuse*: `==` ↔ `=`, `+` ↔ `*`, etc.
   - *variable misuse*: off-by-one in loop indices, wrong variable name
   - *missing logic*: remove return/conditional branch
   - *value misuse*: wrong initial value, wrong comparison constant
   - *excess logic*: dead code that shadows correct result
   - *function misuse*: wrong function call, wrong parameter order
4. Validates that injected code is syntactically valid Python
5. Stores (buggy_code, fixed_code, bug_type, original_source)

**Implementation timeline:** ~1 day for the script, with immediate 2K+ yield.

---

## 4. EVALUATION BENCHMARK VERIFICATION

### 4.1 Current eval_deterministic.py Coverage

| Solver | Datasets Loaded | Items | Adequate? |
|--------|----------------|-------|-----------|
| arithmetic | GSM8K (1,319), SVAMP (1,000), MathQA (4,475) | ~6,794 | ✅ Yes |
| logic | LogiQA (7,376) | 7,376 | ✅ Yes |
| sentiment | SST-2 (872), IMDB (25,000) | 25,872 | ✅ Yes |
| ner | NCBI Disease (941) | 941 | ✅ Yes |
| factual_qa | SQuAD 2.0 (11,873) | 11,873 | ✅ Yes |
| code_debugging | HumanEval (164) | 164 | ⚠️ Small but OK for patterns |

### 4.2 NEW Tool Eval Benchmarks Needed

These tools exist in `upgrade_deterministic.py` but NOT yet in `eval_deterministic.py`:

| New Tool | Existing Func | Recommended Eval Dataset | Items | Status |
|----------|--------------|-------------------------|-------|--------|
| Code generation solver | (planned) | **HumanEval** (already loaded), **MBPP** | 164 + 500 = 664 | ⚠️ MBPP not yet in eval script |
| Summarization extractive solver | (planned, see DETERMINISTIC_ROUTING_ARCHITECTURE.md gap #4) | **XSum**, **CNN/DailyMail** | 11,334 + 11,490 = 22,824 | ❌ Not in eval script at all |
| Narrative math solver | `_solve_narrative_math` | **GSM8K** (already loaded), **SVAMP**, **competition_math** | 2,319 + 12,500 = 14,819 | ⚠️ competition_math not in eval script |
| Propositional logic solver | `_solve_truth_table`, `_is_propositional_task` | **LogiQA** (already loaded), **allenai/ZebraLogicBench** | 7,376 + ~1,000 = ~8,376 | ⚠️ ZebraLogic not in eval script |

### 4.3 Verified: Each Eval Dataset Has Required Fields

| Dataset | prompt/task | expected/ground truth | category label |
|---------|------------|----------------------|----------------|
| GSM8K | `question` ✅ | `answer` (after `####`) ✅ | Implicit (math) ✅ |
| SVAMP | `question` ✅ | `answer` ✅ | Implicit (math) ✅ |
| MathQA | `Problem` + `options` ✅ | `correct` letter → option ✅ | Implicit (math) ✅ |
| LogiQA | `context` + `query` + `options` ✅ | `correct` index ✅ | Implicit (logic) ✅ |
| SST-2 | `sentence` ✅ | `label` (0/1) ✅ | Implicit (sentiment) ✅ |
| IMDB | `text` ✅ | `label` (0/1) ✅ | Implicit (sentiment) ✅ |
| NCBI Disease | `tokens` ✅ | `ner_tags` ✅ | Implicit (NER) ✅ |
| SQuAD 2.0 | `context` + `question` ✅ | `answers.text[0]` ✅ | Implicit (factual QA) ✅ |
| HumanEval | `prompt` (function spec) ✅ | `canonical_solution` ✅ | Implicit (code) ✅ |
| HumanEvalPack | `buggy_solution` + `docstring` ✅ | `canonical_solution` ✅ | `bug_type` ✅ |
| XSum | `document` ✅ | `summary` ✅ | Implicit (summarization) ✅ |
| CNN/DailyMail | `article` ✅ | `highlights` ✅ | Implicit (summarization) ✅ |
| MBPP | `text` (problem desc) ✅ | `code` + `test_list` ✅ | Implicit (code) ✅ |

---

## 5. ADDITIONAL EVAL DATA GAPS

### 5.1 What Additional Eval Data Is Needed

| Tool | Missing Eval Data | Action Needed | Priority | 
|------|------------------|--------------|----------|
| **Summarization solver** | XSum and CNN/DailyMail loaders need to be added to `eval_deterministic.py` | Add new loader functions + register in SOLVERS | HIGH |
| **Code generation solver** | MBPP loader needs to be added (HumanEval already there) | Add MBPP to code_gen solver entry in SOLVERS | MEDIUM |
| **Narrative math solver** | competition_math loaded as evaluation dataset | Add loader to arithmetic solver block | LOW (GSM8K/SVAMP already cover this) |
| **Propositional logic solver** | ZebraLogicBench as a harder logic eval | Add loader for allenai/ZebraLogicBench | LOW (LogiQA already covers) |

### 5.2 Summary: Eval Data Deliverables

| File | What to Add | Target |
|------|------------|--------|
| `eval_deterministic.py` | Add new SOLVER entry for `code_generation` | Use HumanEval + MBPP (664 items) |
| `eval_deterministic.py` | Add new SOLVER entry for `summarization` | Use XSum + CNN/DailyMail (22.8K items) |
| `eval_deterministic.py` | Extend `arithmetic` with `competition_math` | +12.5K math items |
| `eval_deterministic.py` | Extend `logic` with `ZebraLogicBench` | +~1K harder logic items |
| `eval_deterministic.py` | Register `solve_summarization`, `solve_code_generation`, `solve_narrative_math`, `solve_propositional_logic` from upgrade script | 4 new solvers |

---

## 6. ACTIONABLE RECOMMENDATIONS

### Immediate (done this session):
1. ✅ Downloaded `python_code_instructions_18k_alpaca` → 18,612 code_gen items
2. ✅ Downloaded `codealpaca_20k` → 20,022 code_gen items
3. ✅ Downloaded `humanevalpack` debug split → 164 code_debug items (seed data)
4. ✅ Downloaded `humanevalpack` gen split → 164 code_gen items
5. ✅ Verified 13 HF evaluation benchmarks are accessible with correct field structure
6. ✅ Created this gap-closure document

### Next steps (recommended):

| # | Task | Owner | Est. Effort |
|---|------|-------|-------------|
| 1 | **Create synthetic code_debug generator** (`scripts/generate_debug_data.py`) | Pipeline | 1 day |
| 2 | **Integrate new sources into stage2 pipeline** via `integrate_datasets.py` update | Pipeline | 0.5 day |
| 3 | **Add completions** for competition_math (12,500) to magpie_phi3 (50,527) for extra coverage | Pipeline | 0.5 day |
| 4 | **Add summarization eval** to `eval_deterministic.py` (XSum + CNN/DM) | Eval | 1 day |
| 5 | **Add MBPP eval** for code_gen to `eval_deterministic.py` | Eval | 0.5 day |
| 6 | **Register new solvers** from `upgrade_deterministic.py` into deterministic.py | Solvers | 1 day |
| 7 | **Run full eval** with all 9+ solvers against all datasets | Eval | 0.5 day |



---

## 7. QUALITY CONTROL ARCHITECTURE — Cross-Cutting

> This section defines per-stage QC gates that validate stage output before it flows downstream.
> Each gate is **pure Python stdlib** (re, math) — no ML imports.
> All thresholds are **tunable via labeled validation data**.

### 7.1 Overall Architecture

```
Prompt → Stage 0 → QC0 → Stage 1 → QC1 → Stage 2 → QC2 → Stage 3 → QC3 → Solver
                         ↓        ↓        ↓        ↓        ↓
                    ┌─────────────────────────────────────────────┐
                    │         Global QC Aggregator                │
                    │  (composite confidence from all gates)      │
                    └──┬──────────────────────────────────────────┘
                       │ if ANY gate FLAG or FAIL
                       ▼
                 Fallback: escalate to Fireworks / flag as UNCERTAIN
```

Each stage runs independently. After it finishes, its QC gate validates the output.
If QC fails → route is marked UNCERTAIN → falls through to Fireworks instead.
No cascading garbage: a bad Stage 0 output doesn't poison Stage 1.

### 7.2 Per-Stage QC Gates

**QC0 — Pre-filter Validation** (file: `agent/qc0.py`)
| Check | What it detects | Tunable parameter |
|-------|----------------|-------------------|
| Bypass keyword verification | Greeting bypass fires without actual greeting words | `min_greeting_matches: int = 1` |
| Code-indicator validation | Route-to-code fires without def/class/return/fences | `min_code_markers: int = 1` |
| Arithmetic purity | Route-to-math fires on mixed text+numbers | `max_non_numeric_ratio: float = 0.2` |
| Summarization keyword check | Route-to-summary fires without summarize/gist/brief | `min_summary_keywords: int = 1` |
| **Validation data:** stage0 split has `code_present`, `calculation_needed`, `factual_lookup` as ground truth.

**QC1 — Feature Consistency** (file: `agent/qc1.py`)
| Check | What it detects | Tunable parameter |
|-------|----------------|-------------------|
| Creativity × structure conflict | Both creativity > 0.5 AND structured_output = True (rarely co-occur) | `max_conflict_sum: float = 1.0` |
| Verbosity length sanity | 5-word prompt flagged as "essay" or 500-word prompt as "short" | `verbosity_bins = {'short': (0,25), 'medium': (25,100), 'long': (100,300), 'essay': (300,inf)}` |
| Multi-step verification | multi_step True but < 2 explicit action verbs present | `min_action_verbs: int = 2` |
| **Validation data:** stage1 split has ground truth `creativity_score`, `verbosity_level`, `structured_output`, `multi_step_reasoning` features.

**QC2 — Category Confidence** (file: `agent/qc2.py`)
| Check | What it detects | Tunable parameter |
|-------|----------------|-------------------|
| Score margin Δ | (top - second) / top — narrow margin = uncertain | `margin_min: float = 0.15` |
| Category probe | "Must-have" keyword check per category (e.g. code_gen must have write/create/generate) | `probe_min_score: float = 0.3` |
| Anti-pattern registry | Patterns that contradict the prediction (e.g. factual + "write a function") | `anti_pattern_penalty: float = 0.4` |
| Fallback detection | All scores ≤ 0 (defaulted to factual) | `zero_score_flag: bool = True` |
| Logic structural override | If factual wins but logic has structural puzzle signals (numbered constraints, relational verbs) | `logic_override_margin: float = 0.3` |
| Sentiment/summary verb-class | If sentiment/summary tied, check verb vocabulary ratio | `verb_class_ratio: float = 0.6` |
| **Validation data:** stage2 split (134K items) with 8-way category labels.

**QC3 — Complexity Sanity** (file: `agent/qc3.py`)
| Check | What it detects | Tunable parameter |
|-------|----------------|-------------------|
| Per-category range bounds | math = 0.0 but has numbers; greeting = 0.9 | `min_complexity[c]`, `max_complexity[c]` per category |
| Signal coherence | Do fired signals agree on direction or contradict? | `min_coherence: float = 0.25` |
| Length-correlation | Very short prompts (<15 words) unlikely > 0.3 | `length_penalty_threshold: int = 15` |
| Provenance check | If complexity score uses heuristic defaults (not tuned for this category), lower confidence | `default_confidence_penalty: float = 0.2` |
| **Validation data:** stage3 split (77K items) with ground truth complexity scores.

### 7.3 Fine-Tuning Methodology

Each QC gate exposes its tunable parameters as a config dict. The fine-tuning loop:

```
for each QC gate:
    load labeled validation data for that stage
    load current default parameters
    
    for each tunable parameter p:
        sweep candidate_values = [v0, v1, v2, ...]
        best_f1 = 0
        best_val = default
        
        for val in candidate_values:
            set p = val
            run gate on all validation items
            compute precision, recall, F1 against ground truth
            if F1 > best_f1 and precision >= 0.90:  # prefer high precision
                best_f1 = F1
                best_val = val
        
        set p = best_val
    
    run gate on held-out test set
    report before/after per-category and overall
    
    write best parameters to agent/qc_config.py
```

**Optimization target**: Precision over recall. Better to FLAG a correct prediction as uncertain (cost: one extra Fireworks call) than to FAIL an incorrect prediction that gets accepted (cost: wrong answer on leaderboard).

### 7.4 Validation Set Construction

Use existing labeled data from shared storage:

| Stage | Data path | Train | Val | Test | QC ground truth field |
|-------|-----------|-------|-----|------|-----------------------|
| 0 | `shared/prompt_data/stage0/` | 47,668 | 15,891 | 15,894 | `code_present`, `calculation_needed`, `factual_lookup` |
| 1 | `shared/prompt_data/stage1/` | 47,668 | 15,891 | 15,894 | `creativity_score`, `verbosity_level`, `structured_output`, `multi_step_reasoning` |
| 2 | `shared/prompt_data/stage2/` | 80,626 | 26,876 | 26,880 | `label_8way` |
| 3 | `shared/prompt_data/stage3/` | 46,478 | 15,493 | 15,495 | `complexity` |

Each split stratified by category to avoid imbalance bias. Test set held out until final evaluation.

### 7.5 Implementation Files

| File | Purpose |
|------|---------|
| `agent/qc0.py` | Stage 0 QC gate — pre-filter validation |
| `agent/qc1.py` | Stage 1 QC gate — feature consistency |
| `agent/qc2.py` | Stage 2 QC gate — category confidence composite |
| `agent/qc3.py` | Stage 3 QC gate — complexity sanity check |
| `agent/qc_config.py` | Tunable thresholds per stage (generated by fine-tuning) |
| `agent/qc_aggregator.py` | Global QC receiver — aggregates all gate verdicts |
| `fine_tune_qc.py` | Fine-tuning script — sweeps thresholds against validation data |
| `validate_qc.py` | Runs all QC gates against test sets and reports |

### 7.6 Fine-Tuning Priority

| Priority | Gate | Why | Est. impact |
|----------|------|-----|-------------|
| **P0** | QC2 — Category confidence | Directly prevents misrouting (v5's 42.1% root cause) | Highest |
| **P1** | QC0 — Pre-filter validation | Catches false bypasses that skip all downstream processing | High |
| **P2** | QC3 — Complexity sanity | Prevents wrong model selection based on bad complexity score | Medium |
| **P3** | QC1 — Feature consistency | Primarily diagnostic; feature errors less likely to cause wrong answers | Lower |

### 7.7 Saved Research (for knowledge base)

The following research findings from July 11 were saved for reference but NOT implemented (heuristics-only decision):

- **NLI models** (`cross-encoder/nli-deberta-v3-xsmall` at 270MB) — could verify routing via entailment, not used to keep Docker slim
- **Self-consistency calibration** (frugal-router pattern) — sample local model k=3, measure agreement, escalate if low. Requires local model in container.
- **Solver confidence gating** — each deterministic solver returns `confident=True/False` at parse time. Pure code change, no model needed. This IS part of the QC approach.
- **Per-category calibrated thresholds** — different margin thresholds per category (FACTUAL=0.6, MATH=0.6, CODE=0.4, SENTIMENT=0.0). Implementable in `qc_config.py`.

---

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Code_debug synthetic data quality issues | Medium | High | Validate by running unit tests against fixed code; spot-check 10% |
| License issues with iamtarun dataset (no explicit license) | Low | Medium | Prefer CodeAlpaca-20k (CC-BY-4.0) as primary; use iamtarun as supplement only |
| HumanEvalPack only has 164 items | High (fact) | Medium | Use as seed for synthetic generation; 164 patterns × 12 variants each = 1,968 |

---

## 7. FILES CREATED / MODIFIED THIS SESSION

### Downloaded (to `/home/artem/dev/amd-hackathon-shared/prompt_data/sources/`):
- `python_code_instructions_18k_alpaca.jsonl` — 18,612 code_gen items
- `codealpaca_20k.jsonl` — 20,022 code_gen items
- `humanevalpack_gen.jsonl` — 164 code_gen items (HumanEval functions)
- `humanevalpack_debug.jsonl` — 164 code_debug items (buggy + fixed code)

### Copied (to `/home/artem/dev/amd-hackathon/prompt_data/sources/`):
- Same 4 files as above (for local project access)

### Created:
- `GAP_CLOSURE_PLAN.md` — this document

### Verified (accessible HF datasets):
- openai/gsm8k, nguyen-brat/svamp, allenai/math_qa, lucasmccabe/logiqa
- stanfordnlp/sst2, stanfordnlp/imdb, ncbi/ncbi_disease, rajpurkar/squad_v2
- openai/openai_humaneval, bigcode/humanevalpack, google-research-datasets/mbpp
- EdinburghNLP/xsum, abisee/cnn_dailymail, allenai/ZebraLogicBench
- qwedsacf/competition_math (already downloaded locally as sources file)
