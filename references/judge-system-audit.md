# Judge / QC / Answer-Verification System — Complete Audit

> Generated: 2026-07-13
> Codebase: `/home/artem/dev/amd-hackathon/`

---

## 1. QC Gate — `agent/solvers/verify.py`

**File:** `/home/artem/dev/amd-hackathon/agent/solvers/verify.py` (462 lines)

### Checks Performed

| Check | Function | What It Does | Thresholds |
|-------|----------|-------------|------------|
| **Empty/null** | `verify()` line 375 | Returns fail if answer is empty or whitespace-only | `not answer or not answer.strip()` |
| **Hedge words** | `_has_hedge()` | 15 regex patterns against lowercase text | Any match → fail |
| **Degenerate repetition** | `_is_degenerate()` | Single word dominates >50% of tokens | `most > len(words) * 0.5`, requires ≥3 words |
| **Too short** | `_is_too_short()` | Strip length < 2 chars | `len(text.strip()) < 2` |
| **Too long** | `_is_too_long()` | Max 8000 chars (default), 4000 for summaries | `len(text) > max_chars` |
| **Code syntax** | `_valid_python()` | AST parse with fragment normalization | Syntax error → fail |
| **Code lint** | `format_and_lint()` | Black formatting + ruff linting | Per-category tolerance |
| **Code imports** | `_has_import_statement()` | `import`/`from` detection | Any import → fail (safety) |

### Hedge Pattern List (15 patterns)
```
"i don't know", "i do not know", "i cannot", "i can't",
"as an ai", "unable to", "no information", "is not provided",
"does not contain", "cannot answer", "cannot provide",
"not enough information", "insufficient", "sorry", "the text does not"
```

### Category-Specific Behavior

- **`code_gen`** / **`code_debug`**: Extracts code block, validates Python syntax, runs black + ruff. For `code_gen`: ANY lint errors → fail. For `code_debug` fragments: >5 lint errors → fail, else relaxed.
- **`summarization`**: Max 4000 characters (vs 8000 default).
- **All other categories**: Only hedge/degenerate/short checks.

### Return Type: `VerifyResult`
```python
@dataclass
class VerifyResult:
    passed: bool
    reason: str = ""
    details: dict = field(default_factory=dict)
```

### `verify_strict()` — Extended Version
Additional constraint checks: expected_sentence count, max_words limit, expected_bullet count. Used by validation harness.

---

## 2. Official Grader — `scripts/grade_answer.py`

**File:** `/home/artem/dev/amd-hackathon/scripts/grade_answer.py` (189 lines)

### `fuzzy_match(answer, expected) → bool`

4-strategy cascade, tried in order:

| Strategy # | Check | Details |
|-----------|-------|---------|
| **1. Exact** | Case-insensitive string equality | `a_low == e_low` |
| **2. Substring** | Expected in answer (or answer in expected if ≥3 chars) | `e_low in a_low` or `a_low in e_low` (if `len(a) >= 3`) |
| **3. Numeric 1%** | Extract numbers, pairwise compare within 1% tolerance | `abs(na[i] - ne[i]) / abs(ne[i]) <= 0.01` (handles zero) |
| **4. Token overlap** | Common stopwords removed. Short expected (<50 chars): ≥50% overlap. Long: ≥30% | `len(overlap) >= len(e_tokens) * threshold` |

### `grade_answer(answer, expected) → (bool, str)`
Returns `(passed, reason_string)`. Reasons include "Passed", "Empty answer", "Agent error: ...", "numeric mismatch X vs Y", or "expected: ..., got: ...".

### `summarization_grade(output, expected) → bool`
Separate, more lenient grader for summarization. 4-signal cascade:
1. `fuzzy_match` (catches near-exact)
2. **Entity recall**: Capitalized named entities, ≥50% recall or ≥2 entities matching
3. **Keyword overlap**: Significant words (≥4 chars), ≥40% overlap
4. **Numeric overlap**: Any shared numbers

### Helper Functions
- `extract_numbers(s) → List[float]`: Extract all decimal numbers from string
- `tokenize(s) → set`: Lowercase, split on non-alphanumeric, return non-empty tokens

---

## 3. `scripts/evaluate.py` — CLI Evaluation Harness

**File:** `/home/artem/dev/amd-hackathon/scripts/evaluate.py` (221 lines)

- Re-exports `fuzzy_match`, `grade_answer`, `tokenize`, `extract_numbers` from `grade_answer.py`
- Reads tasks.txt and ground_truth.txt, runs agent via subprocess
- Calls `grade_answer()` for each task
- **84.2% gate**: Returns exit code 0 if accuracy ≥ 84.2%, else 1
- Used by `grade_results.py` and `scripts/grade_v12e.py`

---

## 4. Pipeline QC Integration

### `agent/main.py` — Filtered Pipeline v9 (470 lines)

**Lines 244-267: QC Gate on Deterministic Answers**
```python
if ans is not None and ans.strip():
    qc_result = qc_verify(ans, category=category)
    if qc_result and not qc_result.passed:
        # QC failed — treat as unanswered, escalate to API
        needs_api = True
        ans = None
    else:
        # Also runs format_and_lint for code categories (info-only)
```

**Lines 356-407: Code Quality Retry Loop**
For `code_gen`/`code_debug`, retries up to 2 times if lint errors found:
1. Runs `format_and_lint()` on answer
2. If lint errors → construct retry prompt with error feedback
3. Retry via Fireworks (if available) then local model
4. Loop breaks if no lint errors or retries exhausted

### `agent/pipeline.py` — Pipeline Class (864 lines)

**Lines 761-769: Consensus + QC in `process()`**
```python
result = solve_with_consensus(...)
answer = result["majority_answer"]
v = verify(answer, category)
if not v.passed or result["agreement_score"] < 0.5:
    fw = self._fw_fallback(...)
```

### QC Fail → Fireworks Escalation Flow
```
verify() fails → needs_api = True → Fireworks API call
```

---

## 5. Self-Consistency Voting — `agent/solvers/local_vote.py`

**File:** `/home/artem/dev/amd-hackathon/agent/solvers/local_vote.py` (215 lines)

### `solve_with_consensus()` — Ensemble Voting

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 3 | Number of samples |
| `temperature` | 0.1 (first), 0.7 (rest) | Diverse sampling |
| `timeout_per_sample` | 30.0s | Per-sample timeout |

### Category-Aware Normalization (`normalize_answer()`)

| Category | Normalization |
|----------|--------------|
| `math` | Extract last numeric value |
| `sentiment` | Map to canonical label (positive/negative/neutral) |
| `ner` | Extract proper nouns, sort, join with semicolons |
| `code_*` | Raw (no normalization) |
| `logic`/`factual` | Strip punctuation, lowercase |
| `summarization` | Raw (never voted on) |

### Return Value
```python
{
    "majority_answer": str,    # First raw sample matching majority normalized form
    "agreement_score": float,  # Fraction of samples agreeing (0.0–1.0)
    "all_answers": list[str],  # All raw sample outputs
    "samples": list[str],      # Normalized versions
}
```

### Usage in Pipeline
- `agent/main.py` (line 339): `solve_with_consensus()` is the primary local inference path
- `agent/pipeline.py` (line 751): Used when `consensus_samples > 1` and category is in `consensus_categories = {"math", "sentiment", "ner"}`
- Agreement score threshold for accepting: `>= 0.5` (line 762 in pipeline.py)

### Config (`agent/config.py`)
```python
CONSENSUS_SAMPLES = 1           # Currently disabled (single sample)
CONSENSUS_THRESHOLDS = {
    "ner": 0.6, "sentiment": 0.5, "code_debug": 0.6, "code_gen": 0.6,
    "math": 0.75, "logic": 0.75, "summarization": 0.0, "factual": 0.5, "general": 0.5,
}
```

---

## 6. Format Normalizer — `agent/solvers/format_normalizer.py`

**File:** `/home/artem/dev/amd-hackathon/agent/solvers/format_normalizer.py` (342 lines)

### `normalize_sentiment_output(raw_output) → (label, confidence)`

12-step cascade for extracting sentiment labels:
1. Direct match → `high` confidence
2. Abbreviation (pos/neg/neut/mix) → `medium`
3. Strip markdown → `high`
4. Strip punctuation → `high`
5. Negation rules (not positive → negative) → `medium`
6. Mixed detection (both X and Y, X/Y) → `high`
7. Single keyword → `high`
8. Multiple keywords (last wins) → `medium`
9. Sentence-by-sentence extraction → `high`
10. Levenshtein typo correction (dist ≤ 2) → `medium`
11. Abbreviation in words → `medium`
12. `unknown` / `low`

Labels: `positive`, `negative`, `neutral`, `mixed`, `unknown`
Confidence levels: `high`, `medium`, `low`

**Usage:** Called from `EvaluationAgent._evaluate_cell()` (evaluation_agent.py line 260) for GEPA cell evaluation.

---

## 7. Quality Config — `agent/quality_config.py`

**File:** `/home/artem/dev/amd-hackathon/agent/quality_config.py` (60 lines)

### Per-Category QC Thresholds (from classifier metrics)

| Category | Metric | Threshold | Precision |
|----------|--------|-----------|-----------|
| `code_debug` | top3_gap | 2.6667 | insufficient data |
| `code_gen` | margin | 0.6 | 1.000 |
| `factual` | inverse_active | 1.0 | 1.000 |
| `logic` | top3_gap | 2.6667 | insufficient data |
| `math` | max_score | 2.0 | 0.911 |
| `ner` | top_over_avg | 4.4444 | 0.922 |
| `sentiment` | top3_gap | 2.6667 | insufficient data |
| `summarization` | top3_gap | 2.6667 | insufficient data |

**Global fallback:** `top3_gap` at 2.6667 threshold.

**Policy:**
```python
QC_POLICY = {
    "on_fail": "escalate",
    "min_accept_precision": 0.85,
    "fallback_solvers": True,
}
```

**Note:** This config exists but is NOT directly wired to verify.py or the pipeline. It appears to be generated by `fine_tune_qc.py` for classifier-level QC, not answer-level QC.

---

## 8. Code Quality Cell — `agent/cells/code_quality.py`

**File:** `/home/artem/dev/amd-hackathon/agent/cells/code_quality.py` (204 lines)

Separate cell that runs `ruff check` on generated code. Not directly used in the main pipeline QC gate — this is a GEPA cell for workflow orchestration.

- `check(code) → QualityReport`: JSON-parseable ruff violations
- `fix(code) → str`: Auto-fixes code with `ruff --fix`
- `check_with_fix(code) → (QualityReport, str)`: Combined check + auto-fix

`QualityReport` fields: `passed`, `errors`, `warnings`, `fixable_count`, `formatted`, `summary`, `tool`, `elapsed_ms`

---

## 9. Evaluation Agent — `agent/evaluation_agent.py`

**File:** `/home/artem/dev/amd-hackathon/agent/evaluation_agent.py` (313 lines)

GEPA evaluation runner. Uses its own `fuzzy_match()` function (independent copy, similar logic) to score cell outputs against expected answers.

### `_compute_extra_metrics()` — Format Compliance by Category

| Category | Check |
|----------|-------|
| `math` | Contains a digit |
| `code_*` | Valid JSON |
| `factual` | ≤120 words |
| `ner` | Has entity labels (PERSON:/ORG:/LOC:/DATE:) |
| `sentiment` | Label in {positive, negative, neutral, mixed} |
| `summarization` | 1-5 sentences |

### Scoring
Returns cells with: `accuracy`, `avg_output_tokens`, `avg_latency_ms`, `format_compliance`, `details` (per-question breakdown).

---

## 10. Deterministic Solvers — `agent/solvers/deterministic.py`

**File:** `/home/artem/dev/amd-hackathon/agent/solvers/deterministic.py` (3279 lines)

**No hard-coded correctness checks.** Solvers return answers or `None` (meaning "uncertain"). They don't evaluate or grade their own output. When a solver returns a non-None answer, the pipeline treats it as correct and returns it without verification for certain code paths. However, in `agent/main.py` (line 244-267), verify() IS called on deterministic answers.

Solvers included: `solve_arithmetic`, `solve_logic`, `solve_sentiment`, `solve_ner`, `solve_factual_qa`, `solve_code_debugging`, `solve_summarization`.

---

## 11. Grader Scripts

### `scripts/grade_v12e.py` (78 lines)
Pipes output from agent runs through `grade_answer()` from `scripts/evaluate.py`. Produces per-category accuracy + JSON results file in `scripts/eval_results/`.

### `grade_results.py` (131 lines)
Grades multi-model Excel results against ground truth JSONs. Uses `fuzzy_match` from `scripts/evaluate.py`. Reports per-model, per-category, and per-difficulty accuracy. 84.2% gate check.

### `comprehensive_gpu_eval.py` (481 lines)
Full pipeline eval runner. Calls `grade_answer()` for all categories EXCEPT summarization (uses `summarization_grade()`). Also checks `accept` list for multiple valid answers. Builds JSON reports with `correct`/`reason` per item.

### `analyze_summarization_grading.py` (443 lines)
Deep analysis of why `fuzzy_match` fails for summarization. Compares with entity recall, keyword overlap, ROUGE-1 F1, numeric overlap. Proposes `summarization_grade()` as replacement.

---

## 12. Eval Result Data Files

### Key Files

| File | Items | Accuracy | Source |
|------|-------|----------|--------|
| `eval_results/comprehensive_eval_qwen2.5-1.5b-instruct_20260713_083928.json` | 366 | 75.1% | GPU eval |
| `eval_results/archive/nemotron_naked_hurt_categories.json` | 199 | 95.0% | Nemotron (naked) |
| `eval_results/archive/stage_qc_hedge_detect_summary.json` | 60 | N/A | QC stage test |

### Result Format
Every eval result entry contains:
```json
{
    "task_id": "q-...",
    "category": "math",
    "prompt": "...",
    "expected": "...",
    "answer": "...",
    "timing_ms": 2028.3,
    "correct": true,
    "reason": "Passed",
    "source": "build-A-40"
}
```
No separate "judge_decision", "score", or "fuzzy_match_score" fields — correctness is binary via `grade_answer()`.

---

## 13. Confidence Scoring

**There is no unified confidence scoring system.** The following piecemeal confidence signals exist:

1. **Consensus agreement score** (`local_vote.py`): `agreement_score` (0.0–1.0) — fraction of samples agreeing
2. **Format normalizer confidence** (`format_normalizer.py`): `"high"/"medium"/"low"` for sentiment extraction certainty
3. **Classifier confidence** (`classifier.py`): Category classification scores (used for routing, not answer grading)
4. **QC pass/fail** (`verify.py`): Binary, no confidence scale
5. **Grading** (`grade_answer.py`): Binary pass/fail, not a confidence score

---

## 14. Data Flow Diagram (Text)

```
User Prompt
    │
    ▼
Stage 0 (pre-filter) ──bypass──► Direct Answer (no QC)
    │
    ▼
Stage 2 (8-way classifier) → category, confidence
    │
    ▼
Stage 3 (per-category complexity)
    │
    ▼
Stage 4 (decision table)
    │
    ├── Simple + Covered → Deterministic Solver
    │       │
    │       ▼
    │   verify() ← QC Gate
    │       │ pass ──► Return answer
    │       │ fail ──► escalate to API
    │
    └── Complex/Uncovered → API Path
            │
            ▼
        Consensus Voting (local_vote.py)
            │
            ▼
        verify() + agreement check
            │ pass ──► Return answer
            │ fail ──► Fireworks escalation
            │
            ▼
        Fireworks API (fallback)
            │
            ▼
        Code retry loop (up to 2x for code tasks)
            │
            ▼
        Return answer (possibly empty)
```

---

## 15. Summary of All Judge/QC Code Locations

| # | File | Function/Purpose | Type |
|---|------|------------------|------|
| 1 | `agent/solvers/verify.py` | `verify()` — QC gate | Pre-answer quality check |
| 2 | `agent/solvers/verify.py` | `verify_strict()` — Extended QC | Pre-answer quality |
| 3 | `agent/solvers/verify.py` | `format_and_lint()` — Black + ruff | Code validation |
| 4 | `scripts/grade_answer.py` | `fuzzy_match()` — 4-strategy cascade | Post-hoc grading |
| 5 | `scripts/grade_answer.py` | `grade_answer()` — Pass/fail grader | Post-hoc grading |
| 6 | `scripts/grade_answer.py` | `summarization_grade()` — Entity/keyword/num | Post-hoc grading |
| 7 | `scripts/evaluate.py` | CLI harness using grade_answer | Post-hoc grading |
| 8 | `agent/main.py` | QC gate on deterministic answers | Pipeline integration |
| 9 | `agent/main.py` | Code quality retry loop | Pipeline integration |
| 10 | `agent/pipeline.py` | Consensus + verify() → Fireworks | Pipeline integration |
| 11 | `agent/pipeline.py` | Code_gen syntax check fallback | Pipeline integration |
| 12 | `agent/solvers/local_vote.py` | `solve_with_consensus()` — Voting | Ensemble/consensus |
| 13 | `agent/solvers/local_vote.py` | `normalize_answer()` — Category norm | Vote normalization |
| 14 | `agent/solvers/format_normalizer.py` | `normalize_sentiment_output()` | Format normalization |
| 15 | `agent/quality_config.py` | Per-category QC thresholds | Configuration |
| 16 | `agent/config.py` | `CONSENSUS_THRESHOLDS` | Configuration |
| 17 | `agent/evaluation_agent.py` | Cell evaluation with fuzzy_match + format | GEPA evaluation |
| 18 | `agent/cells/code_quality.py` | ruff-based code quality | Cell-level QC |
| 19 | `comprehensive_gpu_eval.py` | `grade_answer()` with accept lists | Batch grading |
| 20 | `grade_results.py` | Multi-model Excel grading | Batch grading |
| 21 | `scripts/grade_v12e.py` | Pipeline grader | Batch grading |
| 22 | `analyze_summarization_grading.py` | Summarization grading analysis | Analysis |

---

## 16. Key Findings

1. **Two separate grading systems exist.** The **QC gate** (`verify.py`) checks answer quality (hedging, too-short, code validity) before returning. The **official grader** (`grade_answer.py`) compares answers to ground truth after the fact. They serve different purposes and do not share logic.

2. **No unified confidence score.** There is no single "model confidence" that follows an answer through the pipeline. The closest alternative is the consensus agreement score (0.0-1.0) from local_vote.py, but it's only computed when `CONSENSUS_SAMPLES > 1` (currently disabled — set to 1).

3. **Summarization has a separate grading path.** `summarization_grade()` in `grade_answer.py` uses entity recall + keyword overlap + number overlap, while all other categories use `fuzzy_match()`. This was added because `fuzzy_match()` doesn't work well for free-form summaries.

4. **QC gate is used differently in the two pipeline implementations.** In `agent/main.py`, QC failure on deterministic answers triggers `needs_api = True` (escalation to Fireworks/local). In `agent/pipeline.py`, QC failure triggers `_fw_fallback()` directly, and consensus agreement < 0.5 also triggers fallback.

5. **Code quality retry loop** exists only in `agent/main.py` (lines 356-407). It retries code generation up to 2 times with lint error feedback.

6. **No per-question judge scores** in eval results. The `correct` field is binary. There's no `fuzzy_match_score`, `judge_decision`, or confidence field stored alongside results.

7. **quality_config.py** exists but isn't wired to the main verify.py/pipeline. It appears to be for a classifier-level QC system that was never fully integrated.
