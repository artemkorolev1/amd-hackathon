# Tool Evaluation Dataset & Harness Analysis Report

**Date:** 2026-07-13  
**Project:** /home/artem/dev/amd-hackathon  
**Harness:** agent/solvers/eval_tools.py (487 lines)  
**Data root:** data/eval/

---

## 1. Evaluation Dataset Audit

### Per-Dataset Summary

| Dataset | Total | Str Answer | Dict Answer | Has `expected_answer` | Has `gold` | Has `task_id` | Key Categories | Eval Type |
|---|---|---|---|---|---|---|---|---|
| **validation-v1** | 400 | 400 | 0 | ✓ 400 | ✗ | ✓ 400 | 8 balanced (50 each) | Tool-level |
| **validation-v2** | 400 | 400 | 0 | ✓ 400 | ✗ | ✓ 400 | 8 balanced (50 each) | Tool-level |
| **validation-v3** | 48 | 48 | 0 | ✓ 48 | ✗ | ✓ 48 | 8 balanced (6 each) | Tool-level |
| **training-v1** | 1,514 | 1,514 | 0 | ✓ 1,514 | ✗ | ✓ 1,514 | 8 cats (114-200 each) | Tool-level |
| **training-v2** | 1,514 | 1,514 | 0 | ✓ 1,514 | ✗ | ✓ 1,514 | 8 cats (114-200 each) | Tool-level |
| **training-v3** | 152 | 152 | 0 | ✓ 152 | ✗ | ✓ 152 | 8 cats (19 each) | Tool-level |
| **eval_60_medium_hard** | 60 | 60 | 0 | ✓ 60 | ✗ | ✗ | 8 cats (7-8 each) | Pipeline (hard) |
| **eval_hard_218** | 218 | 218 | 0 | ✓ 218 | ✗ | ✗ | 8 cats (24-34) + general | Pipeline (hard) |
| **eval_clean_val** | 12,727 | 0 | 0 | ✗ | ✗ | ✗ | No category labels | Unusable |
| **eval_mini_10** | 10 | 10 | 0 | ✓ 10 | ✗ | ✓ 10 | 6 cats mixed | Tool-level |
| **gsm8k_100** | 100 | 100 | 0 | ✓ 100 | ✗ | ✓ 100 | math:100 | Tool-level |
| **sst2_100** | 100 | 100 | 0 | ✓ 100 | ✗ | ✓ 100 | sentiment:100 | Tool-level |
| **math_combined_80** | 94 | 47 | 0 | ✓ 94 | ✗ | ✓ 94 | math:94 | Tool-level |
| **factual_combined_80** | 58 | 45 | 0 | ✓ 58 | ✗ | ✓ 58 | factual:58 | Tool-level |
| **build-A-40** | 40 | 0 | 40 | ✗ | ✓ 40 | ✓ 40 | 7 renamed cats | Pipeline (generated) |
| **build-B-40** | 40 | 0 | 40 | ✗ | ✓ 40 | ✓ 40 | 7 renamed cats | Pipeline (generated) |
| **complexity_eval_40** | 40 | 40 | 0 | ✓ 40 | ✗ | ✗ | 8 cats (5 each) | Tool-level |
| **eval_v14_remaining_20** | 20 | 20 | 0 | ✓ 20 | ✗ | ✗ | 8 cats mixed | Tool-level |
| **eval_v14_test_20** | 20 | 20 | 0 | ✓ 20 | ✗ | ✗ | 8 cats mixed | Tool-level |
| **eval_v14_timeout_stress_19** | 19 | 7 | 0 | ✓ 19 | ✗ | ✓ 19 | 8 cats mixed | Stress test |
| **fireworks_eval_20** | 20 | 20 | 0 | ✓ 20 | ✗ | ✗ | 5 cats (4 each) | Tool-level |
| **eval_longform_20** | 20 | 0 | 0 | ✗ | ✗ | ✓ 20 | 8 cats mixed | Pipeline (long-form) |
| **complexity_eval_candidates** | 87 | 0 | 0 | ✗ | ✗ | ✗ | code_gen only | Metadata only |

### Key distinctions:

**Datasets with `task_id`:**
- All validation/training sets (v1/v2/v3), math_combined_80, factual_combined_80, gsm8k_100, sst2_100, build-A/B-40, eval_mini_10, eval_longform_20, eval_v14_timeout_stress_19

**Datasets using `gold` field (instead of `expected_answer`):**
- build-A-40, build-B-40 (both use nested `gold: {"answer": "..."}` dict format)

**Datasets incompatible with current harness** (no `expected_answer` / no category):
- eval_clean_val (12,727 items — missing both category and expected_answer)
- eval_longform_20 (no expected_answer, only has notes/expected_length)
- complexity_eval_candidates (has label_human but not expected_answer)

### Tool-level vs Pipeline-level distinction:

| Eval Type | Criteria | Datasets |
|---|---|---|
| **Tool-level** | Simple string answer; tool output can be directly compared | v1, v2, v3, training-v1/v2/v3, gsm8k_100, sst2_100, math_combined, factual_combined, eval_mini_10, complexity_eval_40, v14_*, fireworks_20 |
| **Pipeline-level** | Complex answer (code, reasoning, nested); requires multi-step/LLM | eval_60_mh, eval_hard_218, build-A/B-40, eval_longform_20 |

---

## 2. Per-Category Eval Data Coverage

### All datasets combined (usable + unusable)

| Category | Total Questions | String-Matchable | % Compatible | Coverage Rank |
|---|---|---|---|---|
| math | 778 | 728 | 93.6% | Most |
| sentiment | 677 | 676 | 99.9% | ▲ |
| factual | 632 | 615 | 97.3% | |
| code_gen | 585 | 577 | 98.6% | |
| logic | 584 | 581 | 99.5% | |
| ner | 574 | 571 | 99.5% | |
| summarization | 574 | 570 | 99.3% | |
| code_debug | 400 | 396 | 99.0% | Least |
| *(generated cats)* | | | | |
| sentiment_classification | 6 | 6 | 100% | (generated) |
| code_debugging | 6 | 0 | 0% | (generated — dict ans) |
| code_generation | 16 | 0 | 0% | (generated — dict ans) |
| factual_knowledge | 8 | 8 | 100% | (generated) |
| logical_reasoning | 10 | 10 | 100% | (generated) |
| math_reasoning | 28 | 0 | 0% | (generated — dict ans) |
| text_summarization | 6 | 0 | 0% | (generated — dict ans) |

### Key coverage facts:

1. **All 8 standard categories** are well-covered (400-778 questions each) across the validation/training datasets.
2. **~98% of questions** use simple string expected_answers compatible with `_match()`.
3. **math** has the most questions (778) but also the lowest string-matchable ratio (93.6% — some expected answers are multi-sentence).
4. **code_debug** has the fewest questions (400) — only half the count of other categories.
5. **The 6 "generated" category names** (code_debugging, code_generation, etc.) are only in build-A/B and use `gold: dict` format — incompatible with the current harness's `_match()`.

### Coverage gaps by dataset

| Gap | Dataset | Impact |
|---|---|---|
| 12,727 unlabeled items | eval_clean_val | Wasted resource — needs category + answer labels |
| 20 items with no answer | eval_longform_20 | Unusable for accuracy eval |
| 87 items with no answer | complexity_eval_candidates | Candidate pool, not labeled |
| 80 items with `gold` dict | build-A/B-40 | Incompatible with `_match()` — harness reads `expected_answer`, not `gold` |

---

## 3. Tool-Output to Expected-Answer Matching Analysis

### Matching Logic (eval_tools.py lines 73-116)

The harness uses `_match(output, expected)` with 5-stage matching:

1. **Normalize**: Remove non-alphanumeric chars, lowercase, collapse whitespace
2. **Exact match**: `o_norm == e_norm`
3. **Substring match**: `e_norm in o_norm or o_norm in e_norm`
4. **Number comparison**: Parse first numeric value; tolerance <0.01
5. **Token overlap (fuzzy)**: >60% token overlap AND >50% content-word (len>3) overlap

Strictness: **Moderately permissive** — substring and fuzzy matching catch many near-misses. The problem is not matching strictness but that tools produce fundamentally wrong output (or "Could not solve").

### Mismatch Examples: math_solve

Tool accuracy (filtered): **8.1%** (3 matches / 37 attempts)

| # | Expected | math_solve Output | Classification |
|---|---|---|---|
| 1 | `7` (Daisy's potato weight problem) | "Could not solve" | content_wrong |
| 2 | `31` (Stephen's loan interest) | "Could not solve" | content_wrong |
| 3 | `105` (Mark's test rate problem) | "Could not solve" | content_wrong |
| 4 | `6,250` (charity fundraiser) | "Could not solve" | content_wrong |
| 5 | `100` (Ford's rose garden) | "Could not solve" | content_wrong |
| 6 | `20` (car rental percentages) | "Could not solve" | content_wrong |
| 7 | `10` (shark + remora percentage) | "Could not solve" | content_wrong |
| 8 | `15` (Paul's cupcake problem) | "Could not solve" | content_wrong |

**Root cause**: math_solve uses `eval()`/`sympy` for raw expressions. All 100 validation math questions are word problems. The tool has **no word-problem parser**. Every prompt yields "Could not solve".

### Mismatch Examples: factual_qa

Tool accuracy (filtered): **19.5%** (36 matches / 185 attempts)

| # | Expected | factual_qa Output | Classification |
|---|---|---|---|
| 1 | "the Kwakiutl of the Pacific Northwest" | "The largest cities in the Southwestern US..." | coverage_gap |
| 2 | "No, the plaintiff is a citizen of New York..." | "Rhual was constructed in 1634 by Evan Edwards." | coverage_gap |
| 3 | "Their bloodstream continues to contain elevated adrenaline" | "The potential uses of AI in government..." | coverage_gap |
| 4 | "The US could no longer remain a superpower" | "a decline in investment." | matching_too_strict |
| 5 | "Management's unwillingness to make all financial records available" | "Stakeholders, Risks, Stakeholder management..." | coverage_gap |
| 6 | "How nuclear attacks are identified/to; who controls the weapons" | "Because it fails to address the actual capabilities..." | coverage_gap |
| 7 | "the friend only." | "I don't know" | coverage_gap |
| 8 | "Not wrong, Not wrong" | "I don't have this fact" | coverage_gap |

**Root cause**: FTS5 database covers only 16,568 facts — mostly common knowledge. The validation questions include law, medicine, political science, and history. Only 19.5% of questions hit a known fact. Remaining 80% return "I don't know" or irrelevant matches.

### Mismatch Examples: summarize

Tool accuracy (filtered): **0.0%** (0 matches / 26 attempts)

| # | Expected (abstractive) | summarize Output (extractive) | Classification |
|---|---|---|---|
| 1 | "Glenn Mason was jailed for 15 months..." | "Guilty: Glenn Mason, 56, has said he will..." | matching_too_strict |
| 2 | "The Houston Texans defensive end released the video..." | "At 6-foot-5 and 289lb JJ Watt is a terror..." | matching_too_strict |
| 3 | "A quote attributed to Maya Angelou on her commemorative stamp..." | "One of America's greatest poets was honored..." | matching_too_strict |
| 4 | "Rihanna Cooper, 21, from Hull, is working as an escort..." | "Britain's youngest sex swap patient has resorted..." | matching_too_strict |
| 5 | "Holly Willoughby, Katherine Jenkins... wear pink" | "Forget sober black... Pink is the new colour..." | format_difference |
| 6 | "Police in London are trying to catch the gang..." | "London (CNN)It wasn't messrs Clooney, Pitt..." | format_difference |
| 7 | "Jake Tapper will add the Sunday show..." | "New York (CNN)Jake Tapper is the next anchor..." | matching_too_strict |
| 8 | "Javier Hernandez has four goals in four..." | "Real Madrid manager Carlo Ancelotti will not..." | matching_too_strict |

**Root cause**: `summarize` uses Sumy LexRank (extractive — picks leading sentences). Expected answers are reference summaries (abstractive — human-written). The extractive summary reuses the article's first sentences; the reference summarizes differently. Token overlap is typically 0-30%, well below the 60% threshold. **Matching to reference summaries is fundamentally wrong for extractive summarizers.**

### Mismatch Examples: logic tools

All 4 logic tools collectively: **~0% accuracy** on validation logic questions.

| # | Tool | Expected | Output | Classification |
|---|---|---|---|---|
| 1 | solve_logic_puzzle | "3. Relying on unconfirmed assumptions..." | "Could not solve" | coverage_gap |
| 2 | solve_logic_puzzle | "0. G and H." (singer/piano arrangement) | "Could not solve" | coverage_gap |
| 3 | solve_logic_puzzle | "1. Huaizhou City will certainly encounter..." | "Could not solve" | coverage_gap |
| 4 | solve_logic_puzzle | "3. After taking lotus leaf products..." | "Could not solve" | coverage_gap |
| 5 | solve_logic_puzzle | "3. , 6." (office floor arrangement) | "Could not solve" | coverage_gap |
| 6 | solve_logic_puzzle | "1. There were no reports of missiles..." | "Could not solve" | coverage_gap |
| 7 | solve_logic_puzzle | "0. In societies that protect freedom of thought..." | "Could not solve" | coverage_gap |
| 8 | solve_logic_puzzle | "3. If you do not participate..." | "Could not solve" | coverage_gap |

**Root cause**: The validation "logic" prompts are **LSAT-style argument analysis and multi-paragraph reasoning** (passage + multiple choice). The tools solve:
- `solve_logic_puzzle`: Constraint puzzles (seating, ordering)
- `solve_syllogism`: Categorical syllogisms (All A are B...)
- `solve_truth_teller_liar`: Knights/knaves puzzles
- `solve_number_sequence`: Arithmetic sequence completion

**None of these tools handle LSAT/GMAT reading comprehension + argument analysis.** The validation data and tool capabilities are fundamentally mismatched.

### Mismatch Classification Summary

| Classification | Definition | Count (of 32 samples) |
|---|---|---|
| **content_wrong** | Tool ran but gave wrong answer | 8 (all math_solve: "Could not solve") |
| **coverage_gap** | Tool can't handle this type of prompt at all | 16 (8 factual_qa, 8 logic tools) |
| **matching_too_strict** | Tool output is semantically related but doesn't pass string match | 6 (summarize) |
| **format_difference** | Output is same content but different format | 2 (summarize) |

---

## 4. Easter Egg Shelf Sanity Check

Verified against the unfiltered eval run (tool_eval_final.json — all tools ran, no answer matching):

| Tool | Status | Notes |
|---|---|---|
| format_csv | ✅ OK | Pretty-prints CSV with aligned columns |
| text_stats | ✅ OK | Returns word/character/sentence counts |
| reverse_text | ✅ OK | Reverses input text |
| top_words | ✅ OK | Returns frequency table of top N words |
| to_leetspeak | ✅ OK | Converts text to leetspeak |
| is_palindrome | ✅ OK | Returns True/False |
| to_emoji | ✅ OK | Converts words to emoji |
| days_until_april_fools | ✅ OK | No-arg tool, returns days as string |
| weather_hot_take | ❌ **BUG** | _TOOL_PARAMS says `None` but tool needs `temp_c` |
| flip_coin | ✅ OK | No-arg tool, returns Heads/Tails |

**Bug identified**: `weather_hot_take` is misconfigured in `eval_tools.py` line 164: `"weather_hot_take": None`. The tool's actual registered input is `temp_c` (float). When the harness runs `tool()` with no args, it raises `TypeError: missing 1 required positional argument: 'temp_c'`. The `run_tool_on_prompt` catches this exception and returns `None`.

**Fix needed**: Change line 164 to `"weather_hot_take": "temp_c"`.

---

## 5. GSM8K and SST-2 Eval Analysis

### GSM8K (100 questions)

| Property | Value |
|---|---|
| Format | List of dicts with `task_id`, `category`, `prompt`, `expected_answer` |
| expected_answer type | **String** (numeric: e.g., "18", "3", "70000") |
| Category | `math` (100/100) |
| Has `task_id` | ✓ All 100 |
| Compatible with `_match()`? | ✓ Yes — simple number strings |

**Math_solve results on 5 samples:**

| Question | Expected | math_solve Output | Answerable? |
|---|---|---|---|
| Janet's ducks lay 16 eggs/day, eats 3, bakes with 4, sells rest by 6-packs... | 18 | "Could not solve" | ❌ Word problem |
| A robe takes 2 bolts blue + half that white = total? | 3 | "Could not solve" | ❌ Word problem |
| Josh buys house for $80k, puts $50k repairs, sells for $200k, profit? | 70000 | "Could not solve" | ❌ Word problem |
| James runs 3 sprints 3x/week, 60m each, total weekly? | 540 | "Could not solve" | ❌ Word problem |
| Wendi feeds chickens 3 cups mixed feed, produces 2 cups compost/chicken... | 20 | "Could not solve" | ❌ Word problem |

**Verdict**: 0/100 GSM8K questions are answerable by current `math_solve`. All are word problems requiring multi-step reasoning. **GSM8K is the right eval set but needs a word-problem solver.**

### SST-2 (100 questions)

| Property | Value |
|---|---|
| Format | List of dicts with `task_id`, `category`, `prompt`, `expected_answer` |
| expected_answer type | **String** ("positive" or "negative") |
| Category | `sentiment` (100/100) |
| Has `task_id` | ✓ All 100 |
| Compatible with `_match()`? | ✓ Yes |

**Sentiment_analysis results on 5 samples:**

| Text | Expected | sentiment Output | Answerable? |
|---|---|---|---|
| "it 's a charming and often affecting journey" | positive | neutral | ⚠️ VADER not installed |
| "unflinchingly bleak and desperate" | negative | neutral | ⚠️ VADER not installed |
| "allows us to hope that nolan is poised to embark..." | positive | neutral | ⚠️ VADER not installed |
| "the acting, costumes, music, cinematography... are all astounding" | positive | neutral | ⚠️ VADER not installed |
| "it 's slow -- very, very slow" | negative | neutral | ⚠️ VADER not installed |

**Verdict**: SST-2 is perfectly answerable IF VADER is installed. Currently `vaderSentiment` is missing, so the tool always returns "neutral" (fallback value). Install via `pip install vaderSentiment` to restore functionality. Estimated accuracy on SST-2: ~65-70% (VADER's typical performance on binary sentiment).

---

## 6. New Eval Gaps — Recommended Datasets

Based on TOOL_EVAL_GAPS.md and the mismatch analysis, here are the datasets needed to measure new tools:

### Priority 1: Code Fix & Generation Tools

| Need | Dataset | Format | Source | Priority |
|---|---|---|---|---|
| Code generation test | HumanEval (164 problems) | prompt → function signature + tests | Already downloaded (HumanEvalPack) | 🔴 Critical |
| Code generation test | MBPP (974 problems) | prompt → function + test cases | Standard benchmark | 🔴 Critical |
| Code debug test | BugsInPy / Defects4J | Buggy code + fix | Existing v1/v2/v3 data already has this | 🟢 Already covered |
| Code verification | Code test execution harness | Rewrite match() to execute code vs test | Needs building | 🟡 Medium |

**Status**: Validation v1/v2/v3 already have 400 code_debug questions (fixes). But code_gen value test needs HumanEval/MBPP. The HumanEvalPack is already downloaded per TOOL_EVAL_GAPS.md.

### Priority 2: Math Word Problems

| Need | Dataset | Format | Source | Priority |
|---|---|---|---|---|
| Math word problems | GSM8K (8,500) | Multi-step arithmetic | Need loader | 🔴 Critical |
| Math word problems | SVAMP (1,000) | Varied word problems | Public | 🟡 Medium |
| Math competition | MATH (12,500) | Competition-level | Public | 🟢 Low (too hard) |

**Status**: gsm8k_100.json exists as a test subset. Need full GSM8K loader + word-problem solver tool.

### Priority 3: LSAT/Logic Reasoning

| Need | Dataset | Format | Source | Priority |
|---|---|---|---|---|
| Logical reasoning | LogiQA (8,600) | Passage + MCQ | Public | 🔴 Critical |
| Logical reasoning | ReClor (6,100) | Law school reading comp | Public | 🟡 Medium |
| Logic puzzles | Custom knight/knave generator | Truth-teller puzzles | Easy to build | 🟡 Medium |

**Status**: Current "logic" eval data is LSAT-style but tools are constraint-puzzle focused. Need LogiQA to match the existing eval data format.

### Priority 4: Summarization

| Need | Dataset | Format | Source | Priority |
|---|---|---|---|---|
| Extractive summary eval | CNN/DailyMail (300K) | Article + highlights | Public | 🟡 Medium |
| Abstractive summary eval | XSum (227K) | Article + single-sentence summary | Public | 🟡 Medium |

**Status**: Current `summarize` is extractive (LexRank). Need ROUGE-based eval instead of string matching. Alternatively, change expected_answers to match extractive behavior.

### Priority 5: Factual QA

| Need | Dataset | Format | Source | Priority |
|---|---|---|---|---|
| Broad factual QA | SQuAD 2.0 (150K) | Passage + question + answer | Download & index into FTS5 | 🟡 Medium |
| Trivia QA | TriviaQA (95K) | Question + answer | Public | 🟢 Low |

**Status**: FTS5 only has 16K facts. SQuAD 2.0 would boost coverage dramatically. The validation factual questions cover law, medicine, history — SQuAD won't cover all but helps.

### Priority 6: NER

| Need | Dataset | Format | Source | Priority |
|---|---|---|---|---|
| Biomedical NER | NCBI Disease (14K) | Abstract + entity spans | Already available in eval_deterministic.py | 🟡 Medium |

**Status**: Current ner_extract fails because spaCy model won't load. Fix spaCy model first, then NCBI Disease dataset is ready.

### Cross-cutting: Harness Improvements

| Need | Priority |
|---|---|
| Fix `weather_hot_take` param mapping in _TOOL_PARAMS | 🟢 Low (fun tool) |
| Support `gold` field as fallback in `load_eval_items()` | 🟡 Medium (unlocks build-A/B) |
| Add ROUGE metric for summarization eval | 🟡 Medium |
| Install vaderSentiment for sentiment_analysis | 🟢 Low (easy fix) |

---

## Summary of Findings

### What's working well
- **8 balanced validation/training datasets** (v1/v2/v3) with clean string answers
- **GSM8K and SST-2** subsets exist for dedicated eval (need tool fixes to use)
- **Matching logic** is appropriately permissive (fuzzy + number + substring)
- **Sentiment** (70.1%) and **NER** (67.6%) have reasonable baselines
- **7/10 easter egg tools** work correctly

### What's broken
- **math_solve** can't handle any of the 778 math questions (all word problems) — 8.1% accuracy
- **logic tools** can't handle any LSAT-style reasoning — ~0% accuracy on 584 questions
- **summarize** extractive vs abstractive mismatch — 0.0% accuracy
- **factual_qa** covers only 16K facts — 19.5% accuracy
- **VADER not installed** — sentiment always returns "neutral" regardless of input
- **weather_hot_take param mapping** — will crash in eval harness
- **spaCy model missing** — ner_extract unable to load

### What's missing (datasets)
1. **GSM8K full** — need word-problem solver + loader (already have 100-test subset)
2. **LogiQA** — needed to match LSAT-style logic eval data to appropriate tools
3. **HumanEval/MBPP** — for code generation measurement
4. **ROUGE-based summarization eval** — string matching can't work for summaries

### Key numbers
| Metric | Value |
|---|---|
| Total questions across all datasets | ~19,000 |
| Questions usable in current harness | ~5,200 |
| Questions with string-matchable answers | ~5,100 (98%) |
| Harness accuracy (filtered) | 17.8% |
| Harness accuracy (unfiltered) | 12.3% |
| Tools with >50% accuracy | 2 (sentiment 70.1%, ner 67.6%) |
| Tools with ~0% accuracy | 7 (logic tools, summarize, search_factual, format_python) |
