# AMD Hackathon — Comprehensive Evaluation Dataset Audit Report

**Date:** 2026-07-11  
**Auditor:** Hermes Agent  
**Project root:** `/home/artem/dev/amd-hackathon`

---

## 1. Dataset Inventory & Completeness

| Dataset | Expected | Actual | Status |
|---|---|---|---|
| `eval_all_300.json` | 300 | **300** ✅ | Correct |
| `eval_hard_100.json` | 100 | **218** ⚠️ | **118 extra** (name implies 100) |
| `eval_medium_100.json` | 100 | **28** ⚠️ | **72 missing** (name implies 100) |
| `eval_simple_100.json` | 100 | **0** 🚫 | **Empty `[]` (2 bytes)** — broken |
| `eval_mixed_20.json` | 20 | 20 ✅ | Correct |
| `eval_stress_test_20.json` | 20 | 20 ✅ | Correct |
| `temp_questions_*.json` (8 files) | 100 total | **100 total** ✅ | 12-13 per file |
| `complexity_training_data.json` | — | **78** | Hand-curated, OK |
| `training_data_4way.json` | — | **30,328** | 4-way classifier data |
| `training_data.json` | — | **58,704** | 8-way classifier data |

---

## 2. Category Distribution

### Eval sets use *abbreviated* category names (NOT matching classifiers)

| Eval name | Classifier name | Count across all evals |
|---|---|---|
| `factual` | `factual_knowledge` | 99 |
| `math` | `math_reasoning` | 91 |
| `code_gen` | `code_generation` | 83 |
| `logic` | `logical_reasoning` | 79 |
| `code_debug` | `code_debugging` | 58 |
| `ner` | `named_entity_recognition` | 52 |
| `sentiment` | `sentiment_classification` | 52 |
| `summarization` | `text_summarisation` | 52 |
| **`general`** | **⚠️ NO MAPPING** | **20** |

### `eval_all_300.json` category breakdown

| Category | Count | % |
|---|---|---|
| factual | 67 | 22.3% |
| math | 46 | 15.3% |
| code_gen | 39 | 13.0% |
| logic | 38 | 12.7% |
| code_debug | 26 | 8.7% |
| ner | 24 | 8.0% |
| sentiment | 24 | 8.0% |
| summarization | 24 | 8.0% |
| **general** | **12** | **4.0%** |

### Critical: The "task_bank" portion (82 unique items) is missing 4 categories

The 82 items in `eval_all_300.json` that are NOT in `eval_hard_100.json` cover only:
- factual: 41 (50%), math: 16 (19.5%), general: 11 (13.4%), code_gen: 10 (12.2%), logic: 4 (4.9%)

**Missing entirely:** summarization, code_debug, ner, sentiment.

---

## 3. Difficulty / Complexity Distribution

### `eval_all_300.json`

| Difficulty | Count | % |
|---|---|---|
| hard | 218 | 72.7% |
| easy | 46 | 15.3% |
| medium | 28 | 9.3% |
| ambiguous | 8 | 2.7% |

**Heavily skewed toward "hard".** Easy and medium are under-represented.

### Other sets

| Dataset | Difficulty |
|---|---|
| `eval_hard_100.json` | 100% hard (all 218 items) |
| `eval_medium_100.json` | 100% medium (28 items) |
| `eval_simple_100.json` | (empty — no data) |
| `eval_stress_test_20.json` | hard=8, medium=7, easy=4, trick=1 |
| `eval_mixed_20.json` | **No difficulty field at all** ⚠️ |

### `complexity_training_data.json` (0/1 labels)

| Complexity | Count | % |
|---|---|---|
| 0 (simple) | 47 | 60.3% |
| 1 (complex) | 31 | 39.7% |

Good balance, but only 78 items total.

---

## 4. Dataset Overlap (Structural Independence)

### eval_hard_100.json and temp_questions_*.json are SUBSETS of eval_all_300.json

```
eval_all_300.json (300)
  ├── 218 items also in eval_hard_100.json  ← 100% overlap
  ├── 100 items also in temp_questions_*.json  ← 100% overlap
  └── 82 items unique (the "task_bank")
```

**Impact:** You cannot use `eval_hard_100.json` or `temp_questions_*.json` as independent hold-out or validation sets — every item already appears in `eval_all_300.json`. Any metric computed on them is not independent.

---

## 5. Label Consistency Issues

### 5a. Answer leakage — 52/300 items have answer in prompt

Mostly code_debug items where the prompt includes the function signature/body that also appears in the expected answer. This means the answer can be trivially "extracted" from the prompt rather than requiring genuine reasoning.

Also: **20 items** across `eval_medium_100.json` and `eval_all_300.json` use `"42"` as the answer — a popular culture reference that can be guessed without understanding.

### 5b. `general` category — orphan category

20 items across eval datasets use `category: "general"`, which maps to **none** of the 8 classifier categories. These are mostly "return JSON" formatting tasks. They are untestable against the 8-way classifier and their purpose in the eval set is unclear.

### 5c. `complexity_training_data.json` — one category inconsistency

Uses `"sentiment"` instead of `"sentiment_classification"`. All other categories match the canonical names.

---

## 6. Training Data Imbalance (Critical)

### 8-way classifier (`training_data.json`): ~61x imbalance

| Label | Count | % |
|---|---|---|
| math_reasoning | 10,000 | 17.03% |
| named_entity_recognition | 10,000 | 17.03% |
| text_summarisation | 10,000 | 17.03% |
| factual_knowledge | 10,000 | 17.03% |
| sentiment_classification | 10,000 | 17.03% |
| logical_reasoning | 8,376 | 14.27% |
| **code_debugging** | **164** | **0.28%** |
| **code_generation** | **164** | **0.28%** |

### 4-way classifier (`training_data_4way.json`): ~30.5x imbalance

| Label | Count | % |
|---|---|---|
| reasoning | 10,000 | 32.97% |
| text | 10,000 | 32.97% |
| knowledge | 10,000 | 32.97% |
| **code** | **328** | **1.08%** |

**Impact:** Both classifiers will be effectively blind to code-related tasks. The `code` class gets <1% of training data in the 4-way model, and code-related classes get <0.6% in the 8-way model. Any routing decision involving code will be near-random.

---

## 7. Source Quality

| Source type | Count across evals |
|---|---|
| AI-generated (`claude-code-hard-v1` + `syn-*`) | 546 |
| Hand-curated | 19 |
| Benchmark-derived (SQuAD, MathQA, etc.) | 21 |

The vast majority (93%) of eval data is AI-generated, not hand-curated. The 200 `claude-code-hard-v1` items dominate and are all marked "hard". Only 40 items across all datasets are from real benchmarks or hand curation.

---

## 8. Critical Gaps Summary

| Priority | Gap | Impact |
|---|---|---|
| **P1** | `eval_simple_100.json` is **empty** (2 bytes) | No easy test set exists |
| **P1** | `eval_medium_100.json` has **28/100 items** | Can't evaluate medium difficulty properly |
| **P1** | 8-way training data: **code=0.56%** vs others | Classifier blind to code tasks |
| **P1** | 4-way training data: **code=1.08%** vs others | Classifier blind to code tasks |
| **P2** | Category naming **inconsistent** across eval/classifier | Need normalization layer or rename |
| **P2** | **20 "general" items** with no classifier mapping | Unusable for 8-way evaluation |
| **P2** | **Answer leakage** in 52/300 items | Inflates code_debug accuracy metrics |
| **P3** | **72.7% hard** in primary eval set | Can't measure easy/medium performance |
| **P3** | `eval_hard_100.json`/`temp_questions_*` are **subsets** | Not independent validation sets |
| **P3** | `eval_mixed_20.json` has **no difficulty field** | Can't stratify results by difficulty |
| **P3** | `complexity_training_data.json` — `"sentiment"` vs `"sentiment_classification"` | Minor mapping inconsistency |
| **P3** | 20 items answer="42" (guessing risk) | Trivial to "guess" without understanding |

---

## 9. Recommended Actions

### Must-do:
1. **Restore `eval_simple_100.json`** — Create 100 easy items covering all 8 categories
2. **Complete `eval_medium_100.json`** — Add 72 more medium-difficulty items covering missing categories (factual, code_debug, ner, sentiment, summarization)
3. **Fix training data imbalance** — Synthetically generate or collect ~10,000 code-related items for both 4-way and 8-way training sets
4. **Normalize category names** — Add a mapping layer or rename all eval categories to match `categories.json`

### Should-do:
5. **Remove/remap `general` items** — Either drop them from eval or reclassify under existing 8 categories
6. **Fix answer leakage** — Remove answer text from prompts in code_debug items
7. **Add `difficulty` field** to `eval_mixed_20.json`
8. **Clarify dataset roles** — Rename `eval_hard_100.json` to `eval_hard_selection.json` and document it's a subset of `eval_all_300.json`

### Nice-to-have:
9. **Diversify sources** — Add more hand-curated and benchmark-derived items (currently 93% AI-generated)
10. **Collect more easy items** — Currently only 46 easy items exist in the entire eval ecosystem
---

## Files Created

- `/home/artem/dev/amd-hackathon/audit_eval_sets.py` — Primary audit script
- `/home/artem/dev/amd-hackathon/audit_eval_deep.py` — Deeper overlap, leakage, and training data analysis
- `/home/artem/dev/amd-hackathon/audit_eval_summary.py` — Final summary and recommendations
- `/home/artem/dev/amd-hackathon/audit_eval_final.py` — Unique item analysis and gap catalog
- `/home/artem/dev/amd-hackathon/eval-dataset-audit.md` — **This report**
