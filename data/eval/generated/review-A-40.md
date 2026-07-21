# Review Report: build-A-40.json vs heldout_40.json Standard

**Date:** 2026-07-13  
**Reviewer:** Hermes Agent  
**Set:** build-A-40.json (40 questions)  
**Reference:** heldout_40.json (40 questions)

---

## 1. Format Check

### Top-Level Fields
All 40 entries have the correct top-level fields: `task_id`, `category`, `prompt`, `gold`. No missing or extra fields found.

### Gold Sub-Key Structure

| Category | Expected Keys | Status |
|---|---|---|
| `sentiment_classification` | `answer` | ✅ All correct |
| `code_debugging` | `function`, `tests`, `_reference` | ✅ All 3 entries have all three |
| `code_generation` | `function`, `tests` (+ optional `check_code`, `context`, `_reference`) | ✅ All 8 have `function` and `tests` |
| `factual_knowledge` | `answer` (+ optional `accept`) | ✅ All 4 have `answer`; 3 have `accept` list |
| `logical_reasoning` | `answer` (+ optional `accept`) | ✅ All 5 have `answer`; 1 has `accept` list |
| `text_summarization` | `keywords`, `min_coverage` | ✅ All 3 have both |
| `math_reasoning` | `answer` | ✅ All 14 have `answer` |

**Verdict: PASS** — No structural issues.

---

## 2. Category Distribution

Expected vs Actual:

| Category | Expected | Actual | Status |
|---|---|---|---|
| `math_reasoning` | 14 | 14 | ✅ |
| `code_generation` | 8 | 8 | ✅ |
| `factual_knowledge` | 4 | 4 | ✅ |
| `logical_reasoning` | 5 | 5 | ✅ |
| `code_debugging` | 3 | 3 | ✅ |
| `sentiment_classification` | 3 | 3 | ✅ |
| `text_summarization` | 3 | 3 | ✅ |

No unexpected or missing categories. Task IDs follow a clean `{abbr}-10x` naming convention.

**Verdict: PASS**

---

## 3. Prompt Sanity

### Empty/Whitespace Checks
All 40 prompts are non-empty and contain meaningful content. **PASS**

### Length Check
No prompt exceeds 800 characters. The longest build prompt is 497 chars (text summarization).  

Comparison with reference prompt lengths:

| Category | Ref Avg | Build Avg | Diff |
|---|---|---|---|
| `code_debugging` | 184.0 | 170.7 | Comparable |
| `code_generation` | 526.6 | **111.8** | ⚠️ Build is 4.7× shorter |
| `factual_knowledge` | 59.5 | 36.5 | Shorter (simpler facts) |
| `logical_reasoning` | 154.6 | 136.8 | Comparable |
| `math_reasoning` | 274.4 | **102.6** | ⚠️ Build is 2.7× shorter |
| `sentiment_classification` | 205.0 | 123.0 | Shorter |
| `text_summarization` | 522.3 | 487.7 | Comparable |

### Placeholder Text
No prompts contain `TODO`, `FIXME`, `REPLACE`, or `[insert]` placeholders. **PASS**

### Verbatim Reference Copy
No prompts are exact copies of reference prompts. Fuzzy check found 3 cases with similarity >0.90 (see §6 Originality below), but these are structurally similar "next number in sequence" templates with different sequences — not verbatim copies.

**Verdict: PASS** (with concerns about difficulty level — see §5)

---

## 4. Answer/Gold Sanity

### Sentiment Classification (3 entries)
- Classes present: `positive`, `negative`, `neutral` ✅
- All 3 classes are represented exactly once. No invalid answers.

### Math Reasoning (14 entries)
- All answers are numeric (int or float). ✅
- **All 14 answers were verified correct** by manual calculation:
  - `gsm8k-101`: 3×$0.50 + 2×$0.75 = **$3.00** ✅
  - `gsm8k-102`: 60×2.5 = **150 mi** ✅
  - `gsm8k-103`: 24 × ⅔ = **16 boys** ✅
  - `gsm8k-104`: 120 - 30 = 90; 90 - 30 = **60 marbles** ✅
  - `gsm8k-105`: (8-6)/8 = **0.25** ✅
  - `gsm8k-106`: 12×5 = **60 cm²** ✅
  - `gsm8k-107`: 35÷5 = **7 weeks** ✅
  - `gsm8k-108`: 45×8 = **360 widgets** ✅
  - `gsm8k-109`: (15+8)/5 = **$4.60** ✅
  - `gsm8k-110`: 15-22 = **-7°C** ✅
  - `gsm8k-111`: 24÷3 = **8 bags** ✅
  - `gsm8k-112`: 2×7 = **14 miles** ✅
  - `gsm8k-113`: 25×0.8 = **$20** ✅
  - `gsm8k-114`: 72-20-40 = **12** ✅

### Code Generation (8 entries)
- All have `function` name and `tests` list with at least 2 test cases. ✅
- Test arguments and expected values appear correct for the described functions.
- **No `check_code`, `context`, or `_reference` fields** — these are simpler "write a function from a description" tasks (gen-101 through gen-108 style), analogous to reference entries `gen-2` and `gen-4`. This is an acceptable format for basic code_gen tasks, but means tests are not executable in a self-contained manner (see §7).

### Code Debugging (3 entries)
- All have `function`, `tests`, and `_reference`. ✅
- **All 3 reference solutions were executed against their test cases and pass 100%:**
  - `debug-101` (sum_to_n): all 3 tests pass ✅
  - `debug-102` (find_max): all 3 tests pass (including all-negative list edge case) ✅
  - `debug-103` (reverse_string): all 3 tests pass ✅

### Text Summarization (3 entries)
- All have non-empty `keywords` lists with 4 keywords each. ✅
- All have `min_coverage: 0.5`. ✅
- All keywords appear within their respective source texts. ✅

### Factual Knowledge (4 entries)
- All have non-empty `answer` fields. ✅
- 3 of 4 have `accept` lists with valid alternates. ✅
- `xfact-101` (Au) has `accept: ["Au"]` which is a self-reference — the `accept` list duplicates the exact `answer`. This is harmless but redundant.

### Logical Reasoning (5 entries)
- All have correct answers verified:
  - `xlogic-101`: 3,6,11,18,27 → **38** (+3,+5,+7,+9,+11) ✅
  - `xlogic-102`: 2,5,10,17,26 → **37** (+3,+5,+7,+9,+11) ✅
  - `xlogic-103`: Snail problem → **day 9** (12m well, +4/-3 daily) ✅
  - `xlogic-104`: 1,2,4,7,11,16 → **22** (+1,+2,+3,+4,+5,+6) ✅
  - `xlogic-105`: Box stacking → **green** ✅

**Verdict: PASS** — All answers are correct and well-formed.

---

## 5. Difficulty Assessment

### Overall: Build-A-40 is significantly easier than heldout_40

| Category | Reference Difficulty | Build Difficulty | Assessment |
|---|---|---|---|
| `math_reasoning` | Multi-step word problems (e.g., 5 workers producing toys, chalk conservation — avg 274 chars) | Single-step arithmetic (e.g., "3 apples × $0.50", "train at 60 mph for 2.5h" — avg 103 chars) | ⚠️ **Too easy** — comparable to grade 3-4 math vs reference's grade 5-6 |
| `code_generation` | Complex functions with edge cases (correct_bracketing, valid_date, mod_power, max_fill — avg 527 chars) | Simple single-purpose functions (sum_list, max_of_three, reverse_string — avg 112 chars) | ⚠️ **Too easy** — 1.5B models can handle both, but these are trivial |
| `factual_knowledge` | Mix of easy and obscure facts (Mount Tambora, Roald Amundsen — avg 60 chars) | Very basic facts (Mars, Mt Everest, George Orwell, chemical symbol of gold — avg 37 chars) | ⚠️ **Too easy** — all are among the most well-known facts |
| `logical_reasoning` | Interesting puzzles (books stacking, number sequences with trick answers like 94 from squares) | Standard arithmetic sequences (+2,+3,+4 / +3,+5,+7) and simple snail problem | ⚠️ **Slightly easy** but reasonable |
| `code_debugging` | Off-by-one and formula bugs (is_prime returns True on factor, C→F formula wrong — avg 184 chars) | Similar off-by-one, initialization bugs (sum_to_n range(n), find_max initial 0, reverse_string slice) | ✅ **About right** — good variety of bug types |
| `sentiment_classification` | Subtle/mixed sentiment (e.g., "hate admitting...but genuinely improved" — avg 205 chars) | Very obvious sentiment ("food was cold and service slow", "weather is perfect" — avg 123 chars) | ⚠️ **Too easy** — no ambiguity or nuance |
| `text_summarization` | Moderately complex passages (Hagia Sophia, Voyager, Deep Blue — avg 522 chars) | Similar complexity passages (Apollo 11, internet history, Panama Canal — avg 488 chars) | ✅ **About right** — comparable length and complexity |

### Rating by Category (1=too easy, 5=just right, 10=too hard)

| Category | Rating | Notes |
|---|---|---|
| math_reasoning | 2/10 | Vastly oversimplified vs reference |
| code_generation | 1/10 | Trivial one-liner functions |
| factual_knowledge | 2/10 | Extremely basic facts |
| logical_reasoning | 6/10 | Slightly easier but reasonable |
| code_debugging | 7/10 | Good variety, appropriate complexity |
| sentiment_classification | 3/10 | No nuance, all obvious |
| text_summarization | 5/10 | Comparable quality |

**Overall difficulty rating: 3.5/10** — The set is significantly easier than the reference, especially in math and code generation. A 1.5B model would likely achieve very high scores with minimal effort, making the benchmark less useful for distinguishing model capabilities.

---

## 6. Originality

### Exact Duplicates
No prompts are exact verbatim copies of reference prompts. ✅

### Fuzzy Overlaps (>0.90 similarity)
Three minor overlaps were detected, all in `logical_reasoning`:
1. **xlogic-101 (sim 0.915) ≈ xlogic-2**: Both are "next number in sequence" questions with the same template but different sequences (3,6,11,18,27 vs 2,6,12,20,30). Acceptable — it's a standard template.
2. **xlogic-102 (sim 0.915) ≈ xlogic-2**: Same template, different sequence (2,5,10,17,26). Acceptable.
3. **xlogic-104 (sim 0.906) ≈ xlogic-5**: Same template, different sequence (1,2,4,7,11,16 vs 1,1,2,3,5,8,13). Acceptable.

These are structurally similar question formats, not copied content. The reference itself uses this pattern (xlogic-2, xlogic-5, xlogic-14 all follow the same "next number" template). **No originality issues.**

### Patterns
- Build uses naming convention `{abbr}-10x` (e.g., `gsm8k-101`) while reference uses `{abbr}-x` (e.g., `gsm8k-2`). Clean separation.
- Build code generation tasks are entirely original (not copied from HumanEval or reference).

**Verdict: PASS** — No plagiarism concerns.

---

## 7. Broken Entries

### Runtime Failures
None found. All 40 entries are structurally sound and well-formed.

### Tests Executed Successfully
- All 3 code_debugging reference solutions execute and pass all tests. ✅
- All 14 math answers are numerically verified. ✅
- All 5 logic answers are conceptually verified. ✅

### Potential Issues (non-blocking)

1. **xfact-101 `accept: ["Au"]`** — The `accept` list duplicates the exact `answer`. This is slightly redundant but not broken; using `accept` to list alternate valid forms would be more useful (e.g., if the answer were "gold", accept could be ["Au", "Gold"]). As written, "Au" is the answer and the only accepted alternative is "Au" — making `accept` functionally a no-op.

2. **code_generation missing executable tests** — Unlike the reference's HumanEval-style entries (humaneval-1, etc.) which have `check_code` with `def check(candidate):` that can be run, the build's gen-101 through gen-108 only have `tests` arrays with args/expected. This means the generation tests require a separate evaluation harness to execute. The reference's `gen-2` and `gen-4` entries follow the same pattern, so this is consistent — but it does make automated evaluation less straightforward.

3. **No negative-sentiment edge cases** — The only negative sentiment example ("The food was cold and the service was slow") is very direct. No examples of mixed or sarcastic negative sentiment (cf. reference's xsent-13 which has the model distinguish between stated dislike and genuine praise).

4. **No edge-case math problems** — All math problems produce clean, positive integer answers (except gsm8k-110 which is -7.0 and gsm8k-109 which is 4.6). There are no problems requiring fractions, remainders, or the model to handle ambiguity. Reference problems like gsm8k-48 (chalk conservation) involve multi-step logic and separate cases.

**Verdict: PASS** — No outright broken entries.

---

## 8. Overall Score

### Rating: **5/10**

This is a mixed set. It nails the basics (format, structure, answer correctness) but falls short on ambition.

### Top 3 Issues That Need Fixing

1. **🔴 CRITICAL: Math and code_gen are severely undertuned for 1.5B models**  
   The math problems are single-step arithmetic (avg 103 chars vs reference's 274 chars). The code generation tasks are trivial one-liner functions (avg 112 chars vs reference's 527 chars). A 1.5B model will breeze through these. **Recommendation:** Add multi-step word problems for math and more complex function-writing tasks (e.g., string parsing, data transformation) for code generation.

2. **🟡 MODERATE: Sentiment lacks nuance**  
   All 3 examples are extremely obvious ("cold food = negative", "perfect weather = positive"). 1.5B models can handle this, but the benchmark would be more discriminative with at least one ambiguous or mixed-review sentiment. **Recommendation:** Replace one entry with a review containing both positive and negative elements (cf. reference's xsent-13).

3. **🟡 MODERATE: No self-contained test infrastructure for code_gen**  
   The gen entries lack `check_code`/`_reference` fields, meaning tests can't be executed without an external harness. While `gen-2` and `gen-4` in the reference share this limitation, adding these fields would make the benchmark more robust and self-contained. **Recommendation:** Add `check_code` with `def check(candidate):` wrapper and `_reference` with a reference implementation for each code_generation entry.

### Other Minor Notes
- Consider increasing `text_summarization` keyword count or raising `min_coverage` above 0.5 for greater discrimination.
- The `logical_reasoning` category is the best-constructed in this set — good variety (3 sequences, 1 snail, 1 stack) at appropriate difficulty.
- The `code_debugging` category is well-constructed with classic, pedagogically sound bugs.
- Task ID naming (`gsm8k-101` etc.) cleanly distinguishes from the reference (`gsm8k-2` etc.).

---

## Summary

| Check | Result |
|---|---|
| Format | ✅ All 40 entries well-structured |
| Category Distribution | ✅ Exact match to spec |
| Prompt Sanity | ✅ No empty/placeholder prompts |
| Answer Correctness | ✅ All answers verified |
| Code Test Validity | ✅ All debugging tests pass |
| Originality | ✅ No plagiarism detected |
| Difficulty | ⚠️ Significantly easier than reference (3.5/10) |
| Broken Entries | ✅ None found |
| **Overall** | **5/10 — Needs difficulty rebalancing** |
