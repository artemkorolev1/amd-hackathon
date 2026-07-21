# Binary Reject Cascade — Category Router Plan

**Date:** 2026-07-15
**Goal:** Replace the 8-way scoring classifier with a cascade of binary reject classifiers. Each node answers one yes/no question. Rejects cascade down. No multi-way decisions.

---

## The Pattern

A **binary reject classifier** answers "Is this my category?" If yes → route to that category's chain. If no → pass to the next classifier. Each classifier is trained independently to recognize its own category and reject everything else.

**Why this over the current 8-way scorer:**
- Simpler decision boundaries per classifier (one-vs-rest, not one-vs-seven)
- Each classifier can be optimized independently (precision vs recall per category)
- Adding/removing categories requires zero retraining of other classifiers
- Cascade order can route to priority categories first (high-value, high-confidence)

**Key rule:** Each split must be a clean boundary — no adjacent comparisons. If it needs a three-way split, add two binary nodes.

---

## Current State

| Category | Status | Notes |
|----------|--------|-------|
| **math** | In progress — step count cascade | Existing `math_binary_step_classifier.py` uses 8 classifiers with adjacent comparisons (1-vs-2, 2-vs-3). Switching to "N-step vs N+-step" pattern. |
| **code** (gen + debug) | In progress — code tool cascade | Existing `code_tool_cascade.py` binary tree. Needs category-level + sub-category cascade. |
| **logic** | Not started | Currently handled by 8-way scorer + `secondary_reasoning.py` |
| **factual** | Not started | Currently handled by 8-way scorer + `secondary_factual.py` |
| **sentiment** | Not started | Currently handled by 8-way scorer + VADER |
| **ner** | Not started | Currently handled by 8-way scorer + `prototype_ner_v3` |
| **summarization** | Not started | Currently handled by 8-way scorer + Sumy |

---

## Phase 0: Math Step Count Cascade (in progress)

Replace the 8 adjacent binary classifiers with a depth-first reject cascade:

```
                      ┌─────────────────┐
                      │  Is this math?   │──NO──→ reject to next category
                      │ (category check) │
                      └────────┬────────┘
                               │ YES
                               ▼
                    ┌─────────────────────┐
                    │  1-step vs multi-step│──YES→ bucket 1
                    │  (one_vs_multi)     │
                    └──────────┬──────────┘
                               │ NO (2+ steps)
                               ▼
                    ┌──────────────────────────┐
                    │  2-step vs 3+-steps       │──YES→ bucket 2
                    │  (two_vs_three_plus)     │
                    └───────────┬──────────────┘
                                │ NO (3+ steps)
                                ▼
                    ┌──────────────────────────┐
                    │  3-step vs 4+-steps       │──YES→ bucket 3
                    │  (three_vs_four_plus)    │
                    └───────────┬──────────────┘
                                │ NO (4+ steps)
                                ▼
                              bucket 4+
```

**Key change from current code:** Replace `is_one_vs_multi` (which currently checks 1-step vs 2+ with an 8-rule conf threshold) with a cleaner, higher-confidence reject classifier. The cascade stops at the first YES — no parallel voting.

**What to build:**
1. Rewrite `is_one_vs_multi()` — clear binary: "Is this definitely 1 step?" High precision target. Low recall is OK — rejects get caught by next classifier.
2. Rewrite `is_two_vs_three_plus()` — "Is this definitely 2 steps?"
3. Rewrite `is_three_vs_four_plus()` — "Is this definitely 3 steps?"
4. Remove `is_four_vs_five_plus`, `is_low_vs_high`, `is_five_plus`, `is_complex_narrative`, `is_rate_problem` — no longer needed in cascade form.
5. New `predict_step_count()` uses cascade (first match wins) instead of confidence-weighted voting.

**Evaluation:** Per-classifier precision/recall on GSM8K train split. Target: each node >80% precision (reject early, don't misroute).

---

## Phase 1: Code Category Cascade (in progress)

Two-level cascade:

```
                    ┌─────────────────────────┐
                    │  Is this code-related?   │──NO──→ reject to next category
                    │  (code vs everything)   │
                    └───────────┬─────────────┘
                                │ YES
                                ▼
                    ┌───────────────────────────┐
                    │  Is this code_debug?      │──YES→ debug cascade
                    │  (debug vs generation)   │
                    └───────────┬───────────────┘
                                │ NO → code_gen
                                ▼
                    ┌───────────────────────────┐
                    │  Has template match?      │──YES→ template solver
                    │  (algorithm name / fn)   │
                    └───────────┬───────────────┘
                                │ NO
                                ▼
                    ┌───────────────────────────┐
                    │  Has structured I/O?      │──YES→ sub-cascade (DS/DP/etc)
                    │  (JSON examples, tests)  │
                    └───────────┬───────────────┘
                                │ NO
                                ▼
                              LLM
```

**What to build:**
1. Category-level `is_code()` classifier — strong code signals (def/return/import/```), suppresses code-like patterns in factual/logic.
2. Keep `secondary_code.py` for debug-vs-gen disambiguation.
3. Keep `code_tool_cascade.py` routing tree for template/DS/DP/sort-search dispatch.
4. Add evaluation: per-node precision/recall on code vs non-code held-out set.

---

## Phase 2: Reasoning Cascade (logic + math)

```
                    ┌──────────────────────────┐
                    │  Is this math?           │──YES→ math cascade
                    │  (explicit calc/eqns)   │      (Phase 0)
                    └──────────┬───────────────┘
                               │ NO
                               ▼
                    ┌──────────────────────────┐
                    │  Is this logic?          │──YES→ logic cascade
                    │  (puzzles, syllogisms)  │
                    └──────────┬───────────────┘
                               │ NO → reject to next category
```

**Key insight from the existing 8-way scorer:** Math and logic have a known confusion zone (logic puzzles with numbers getting classified as math). The math classifier should reject logic-looking patterns (named entities + constraints), and the logic classifier should reject pure arithmetic.

**What to build:**
1. `is_math()` — strong math signals only. Explicit calc/solve/equation/formula keywords. Suppress for: named-entity puzzles, factual lookup with numbers, SQuAD-format QA. Target: high precision (>90%), moderate recall is fine — logic/math confusions fall through to the other classifier.
2. `is_logic()` — strong logic signals. Named-entity constraint puzzles (3+ names + each/different), syllogisms (all/some/no X are Y), truth-teller (knight/knave), clue-based logic puzzles. Also catches proof/justification and complexity analysis.
3. Within logic sub-cascade:
   - Is it a zebra puzzle? (3+ names, constraints, each/different) → zebra solver
   - Is it a truth-teller puzzle? (knight/knave) → truth-table solver
   - Else → LLM
4. Within math sub-cascade (Phase 0 already covers step count):
   - Is it pure arithmetic? (expression with +-*/ no narrative) → arithmetic solver
   - Is it a word problem? → SymPy or LLM

**Evaluation:** Math vs logic confusion rate (currently the biggest gap). Route to wrong solver = wasted compute.

---

## Phase 3: Text Cascade (sentiment + NER + summarization)

```
                    ┌──────────────────────────────┐
                    │  Is this text analysis?      │──NO──→ reject to factual
                    │  (sentiment/NER/summarize)  │
                    └──────────────┬───────────────┘
                                   │ YES
                                   ▼
                    ┌──────────────────────────────┐
                    │  Is this sentiment?          │──YES→ VADER or LLM
                    │  (positive/negative/neutral) │
                    └──────────────┬───────────────┘
                                   │ NO
                                   ▼
                    ┌──────────────────────────────┐
                    │  Is this NER?                │──YES→ prototype_ner_v3
                    │  (extract entities/types)   │
                    └──────────────┬───────────────┘
                                   │ NO → summarization
                                   ▼
                              Sumy or LLM
```

**What to build:**
1. `is_text_analysis()` — strong signals: explicit extraction/analysis/classification keywords. Suppresses for: factual QA with sentiment-like questions ("How does X feel about Y?" is factual, not sentiment), factual with entity-like mentions.
2. `is_sentiment()` — sentiment-specific keywords (positive/negative/opinion/review/feeling). Very narrow match — everything else rejects to NER/summarization.
3. `is_ner()` — entity extraction verbs (extract/find/identify/list/tag entities/names/types). If `{@...@}` markers present → bypass to prototype_ner_v3 directly.
4. Default → summarization (Sumy extractive or LLM).

**Key insight from current system:** Local LLMs get ~54% on sentiment and ~54-67% on NER — these categories NEED deterministic solver routing (VADER, prototype_ner_v3) or FW. The binary cascade must prioritize getting the category right for these, because misrouting sentiment → NER solver wastes the question.

---

## Phase 4: Factual Knowledge Cascade

```
                    ┌──────────────────────────────┐
                    │  Is this factual QA?         │──NO──→ reject to LLM fallback
                    │  (questions with answerable  │
                    │   lookup patterns)           │
                    └──────────────┬───────────────┘
                                   │ YES
                                   ▼
                    ┌──────────────────────────────┐
                    │  FactDB has the answer?      │──YES→ direct answer
                    │  (BM25 ≥ 6.0)               │
                    └──────────────┬───────────────┘
                                   │ NO
                                   ▼
                              LLM or FW
```

**What to build:**
1. `is_factual()` — question-word starters (what/who/when/where/why/how), SQuAD format (Context: + Question:), knowledge lookup verbs (define/explain/describe/meaning/history/capital). Suppresses for: math-y "how many" questions that are actually calculation.
2. Remainder (everything the cascade rejected) → LLM fallback (or general router).

---

## Overall Cascade Order

The reject cascade order determines priority — earlier classifiers get first crack:

```
Level 1: code?           (highest precision, deterministic solver exists)
Level 2: math?           (deterministic solver exists, clear signals)
Level 3: logic?          (deterministic solver exists for puzzles)
Level 4: sentiment?      (VADER deterministic, but narrow)
Level 5: NER?            (prototype_ner_v3, F1=0.96)
Level 6: summarization?  (Sumy extractive)
Level 7: factual?        (FactDB, 17K facts)
Level 8: → LLM fallback
```

Order rationale: Categories with high-confidence deterministic solvers go first. Code has the strongest signal (fence markers, def/return). Math has clear numeric signals. Logic overlaps with math so it goes after. Sentiment/NER/summarization are text categories that look similar. Factual is the broadest net — catches everything with question words. Everything that rejects all the way down gets routed to the local LLM.

---

## Implementation Order

1. **Phase 0: Math step count cascade** — rewrite binary classifiers to use reject-first-match pattern (not parallel voting)
2. **Phase 1: Code category cascade** — add top-level `is_code()` reject node
3. **Phase 2+3+4: Remaining categories** — build `is_math()`, `is_logic()`, `is_text()`, `is_sentiment()`, `is_ner()`, `is_factual()` reject classifiers
4. **Integration:** Wire reject cascade into agent/main.py replacing the 8-way `classify_category()` call
5. **Benchmark:** Compare cascade accuracy vs current 8-way scorer on eval_all_300

---

## Test & Validation Plan

Per classifier:
- Precision: % of YES predictions that are correct (minimize false positives)
- Recall: % of category-owner items that get YES (minimize false negatives — items falling through to wrong category)
- F1: Harmonic mean of precision and recall

Acceptance criteria for each binary node:
- **Higher-precision nodes (code, math, sentiment)**: precision > 90%, recall can be lower
- **Broad-catch nodes (factual, summarization)**: recall > 85%, precision can be lower
- **Confusion-critical nodes (logic vs math)**: both precision > 80% and recall > 80%

End-to-end: Overall category accuracy must match or exceed current 85-86% on eval_all_300.
