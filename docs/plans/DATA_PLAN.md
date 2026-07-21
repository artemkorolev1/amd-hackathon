# Updated Data Plan — AMD ACT II Hackathon Track 1
> Generated: 2026-07-11
> Pipeline: Stage 0 (Pre-filters) → Stage 1 (Multi-axis) → Stage 2 (8-way Category) → Stage 3 (Per-category Complexity) → Stage 4 (Solver Selection)

---

## 1. CURRENT INVENTORY (on disk)

### 1.1 Prompt Data Pipeline (`/home/artem/dev/amd-hackathon/prompt_data/`)

| Split | Items | Source |
|-------|-------|--------|
| `train.jsonl` | 10,824 | Dolly-15K + Alpaca-GPT4 + eval300 + temp_questions |
| `val.jsonl` | 1,353 | same |
| `test.jsonl` | 1,354 | same |
| **Total** | **13,531** | |

**8-way distribution (train):**
```
factual:        8,156 (75.4%)  ← heavily dominant
ner:            1,233 (11.4%)
summarization:    981 ( 9.1%)
code_gen:         307 ( 2.8%)
math:              43 ( 0.4%)
logic:             40 ( 0.4%)
sentiment:         33 ( 0.3%)
code_debug:        31 ( 0.3%)
```

### 1.2 ML Training Data (`classifiers/training_data.json`)

| Category | Items | Source HF Datasets |
|----------|-------|--------------------|
| math_reasoning | 10,000 | GSM8K |
| named_entity_recognition | 10,000 | CoNLL2003 |
| text_summarisation | 10,000 | XSum |
| factual_knowledge | 10,000 | MMLU |
| sentiment_classification | 10,000 | SST-2 |
| logical_reasoning | 8,376 | LogiQA |
| code_generation | 164 | HumanEval |
| code_debugging | 164 | HumanEvalPack |
| **Total** | **58,704** | |

**Problem:** Imbalanced — code categories severely under-represented. Labels are old names (need mapping).

### 1.3 Complexity Training Data (`classifiers/complexity_training_data.json`)

| Items | Source |
|-------|--------|
| **78** | Hand-curated |

**Problem:** CRITICALLY SMALL — 78 items is not enough to train any complexity model. This is the single biggest gap.

### 1.4 Evaluation Data

| File | Items | Purpose |
|------|-------|---------|
| `eval_all_300.json` | 300 | 8-category eval (balanced: factual=67, math=46, code_gen=39, logic=38, code_debug=26, ner=24, sentiment=24, summarization=24, general=12) |
| `eval_simple_100.json` | 100 | Easy prompts |
| `eval_medium_100.json` | 100 | Medium prompts |
| `eval_hard_100.json` | 100 | Hard prompts |
| `eval_60_balanced.json` | 60 | Balanced 8-way |

---

## 2. DISCOVERED DATASETS — DETAILED ANALYSIS

### 2.1 ⭐ Magpie-Align/Magpie-Phi3-Pro-1M-v0.1
**URL:** https://huggingface.co/datasets/Magpie-Align/Magpie-Phi3-Pro-1M-v0.1
**Size:** 1,000,000 examples
**Labels:** `instruction`, `task_category`, `difficulty` (5 levels: very easy, easy, medium, hard, very hard), `response`, `intent`, `knowledge`
**Labels for prompt:** instruction ✓, category ✓ (task_category), difficulty ✓ (5-level ordinal)
**Pipeline fit:**
- **Stage 0 (Pre-filters):** ✗ Not applicable
- **Stage 1 (Multi-axis):** ✗ No multi-axis features
- **Stage 2 (8-way Category):** ✅ YES — primary source. Magpie categories (LLM-generated) need mapping to 8-way
- **Stage 3 (Per-category Complexity):** ✅ YES — 5 difficulty levels per category. Best source for per-category complexity
- **Stage 4 (Solver Selection):** ✗ No solver performance data
**Priority:** **#1 — DOWNLOAD FIRST**
**Est. usable items:** 200K–500K (after category mapping + filtering short instructions)

### 2.2 ⭐ wesley7137/question_complexity_classification
**URL:** https://huggingface.co/datasets/wesley7137/question_complexity_classification
**Size:** ~14,049 examples
**Labels:** `question` (text), `rating` (complexity 0.0–1.0 continuous)
**Pipeline fit:**
- **Stage 0 (Pre-filters):** ✗
- **Stage 1 (Multi-axis):** ✅ Complexity score maps to overall difficulty axis
- **Stage 2 (8-way Category):** ✗ No category labels
- **Stage 3 (Per-category Complexity):** ⚠️ PARTIAL — continuous 0-1 scores are perfect but unlabeled by category. Needs co-training or weak labels
- **Stage 4 (Solver Selection):** ✗
**Priority:** **#2 — DOWNLOAD SECOND** (14K continuous complexity labels are rare and valuable)
**Est. usable items:** 14K (all)

### 2.3 ⭐ competition_math (MATH Dataset)
**URL:** https://huggingface.co/datasets/competition_math
**Size:** 7,500 train + 5,000 test = 12,500
**Labels:** `problem` (text), `type` (7 math categories), `level` (difficulty 1–5), `solution`
**Pipeline fit:**
- **Stage 0 (Pre-filters):** ✅ Can validate pre-filter for calculation/math detection
- **Stage 1 (Multi-axis):** ✗
- **Stage 2 (8-way Category):** ✅ YES — all items are math. Also has 7 sub-categories (Algebra, Geometry, etc.)
- **Stage 3 (Per-category Complexity):** ✅ YES — 5-level difficulty for math category. Best source for math-specific complexity
- **Stage 4 (Solver Selection):** ✗
**Priority:** **#3 — DOWNLOAD THIRD**
**Est. usable items:** 12,500 (all, for math). Sub-category mapping: Algebra/Prealgebra/Intermediate Algebra → math, Geometry/Counting/Number Theory/Precalculus → math + logic hybrid

### 2.4 ⭐ anasnassar/llm-query-complexity-benchmark
**URL:** https://huggingface.co/datasets/anasnassar/llm-query-complexity-benchmark
**Size:** 4,800 train + 1,200 test = 6,000 (perfectly balanced)
**Labels:** Queries with LOW, MEDIUM, HIGH complexity + 10 domains (hpc, mathematics, statistics_ml, physics_chemistry, engineering, life_sciences, cs_software, philosophy_ethics, social_sciences, history_culture)
**Pipeline fit:**
- **Stage 0 (Pre-filters):** ✗
- **Stage 1 (Multi-axis):** ✅ Has domain labels useful for feature diversity
- **Stage 2 (8-way Category):** ⚠️ PARTIAL — 10 domains don't map cleanly to 8 categories, but can re-label via LLM
- **Stage 3 (Per-category Complexity):** ✅ YES — designed for routing tier selection. 3-tier complexity
- **Stage 4 (Solver Selection):** ✅ YES — was built for STREAM routing framework
**Priority:** **#4 — DOWNLOAD FOURTH**
**Est. usable items:** 6,000 (all, balanced)

### 2.5 ⭐ CARROT-LLM-Routing/SPROUT
**URL:** https://huggingface.co/datasets/CARROT-LLM-Routing/SPROUT
**Size:** 30,968 train + 6,636 val + 6,637 test = 44,241
**Labels:** `prompt`, `golden_answer`, `dataset`, `dataset_level`, scores from 13+ models (GPT-4o, Claude 3.5, Llama variants, Mixtral, etc.)
**Pipeline fit:**
- **Stage 0 (Pre-filters):** ✗
- **Stage 1 (Multi-axis):** ✗
- **Stage 2 (8-way Category):** ⚠️ Has `dataset` field that indicates source (MMLU, etc.). Can map to 8 categories
- **Stage 3 (Per-category Complexity):** ✅ YES — `dataset_level` may indicate difficulty tier. Score differences across models show difficulty
- **Stage 4 (Solver Selection):** ✅ PRIME — actual scores from 13+ model sizes directly inform routing decisions
**Priority:** **#5 — DOWNLOAD FIFTH**
**Est. usable items:** 44K (all). Best source for stage 4 training.

### 2.6 lime-nlp/MATH_Difficulty
**URL:** https://huggingface.co/datasets/lime-nlp/MATH_Difficulty
**Size:** ~105,000 rows
**Labels:** `problem`, `ground_truth`, `difficulty` (original MATH level 1–5), `solved_percentage` (continuous 0–100%)
**Pipeline fit:**
- **Stage 3 (Per-category Complexity — Math):** ✅ YES — both discrete (1-5) and continuous (0-100) difficulty for math
**Priority:** **#6 — DOWNLOAD SIXTH** (math-specific, supplements competition_math)

### 2.7 lime-nlp/GSM8K_Difficulty
**URL:** https://huggingface.co/datasets/lime-nlp/GSM8K_Difficulty
**Size:** ~40,000 rows
**Labels:** `problem`, `ground_truth`, `solved_percentage` (difficulty 0–100%)
**Pipeline fit:**
- **Stage 3 (Per-category Complexity — Math):** ✅ YES — continuous difficulty for math word problems
**Priority:** **#7** (math-specific)

### 2.8 rokokot/question-type-and-complexity
**URL:** https://huggingface.co/datasets/rokokot/question-type-and-complexity
**Size:** ~8,280 (7,400 train + 440 val + 440 test)
**Labels:** `text`, `language`, `question_type` (binary: content/polar), `complexity_score` (continuous), linguistic features (avg_links_len, lexical_density, n_tokens, etc.)
**Pipeline fit:**
- **Stage 1 (Multi-axis):** ✅ YES — linguistic features (avg_links_len, avg_max_depth, lexical_density) directly map to complexity axes
- **Stage 3 (Per-category Complexity):** ⚠️ Binary categories (content/polar) don't map to 8-way but continuous complexity useful
**Priority:** **#8** (useful for multi-axis feature engineering, limited categories)

### 2.9 agentlans/prompt-difficulty
**URL:** https://huggingface.co/datasets/agentlans/prompt-difficulty
**Size:** ~100K prompts
**Labels:** Difficulty scores (1-7 scale, aggregated from 8+ LLMs)
**Pipeline fit:**
- **Stage 3 (Complexity):** ✅ YES — general complexity scores
**Priority:** **#9** (no category labels)

### 2.10 SupraLabs Prompt-Routing-Dataset
**URL:** https://huggingface.co/datasets/SupraLabs/Prompt-Routing-Dataset
**Size:** 992 prompts
**Labels:** Complexity 1-5
**Pipeline fit:**
- **Stage 3 (Complexity):** ✅ YES — directly for routing
**Priority:** **#10** (very small, but directly applicable and quick to integrate)

### 2.11 JanW42/prompt-difficulty-mean
**URL:** https://huggingface.co/datasets/JanW42/prompt-difficulty-mean
**Size:** 79,806 train + 9,976 test + 9,976 val = ~100K
**Labels:** `text`, `labels` (difficulty float)
**Pipeline fit:**
- **Stage 3 (Complexity):** ✅ YES
**Priority:** **#11** (no category labels)

### 2.12 shivenkk/modelrouter (GitHub Reference)
**URL:** https://github.com/shivenkk/modelrouter
**Type:** Pre-trained DistilBERT classifier (67M params, 96% accuracy) + methodology
**Not a dataset** — it's a trained model + evaluation set (671 queries)
**Value:** Reference architecture for 4-class difficulty classification. ~30ms CPU inference validates DistilBERT approach.
**Priority:** **REFERENCE** — don't download as data, use as architecture inspiration

### 2.13 Existing Data (already on disk)
**Dolly-15K:** 15,010 raw → ~11K usable after category mapping ✓
**Alpaca-GPT4:** ~52K total → ~2K usable (only 9 categories mapped to our 8-way) ✓
**Both already downloaded** by `download_prompt_data.py`

---

## 3. CATEGORY MAPPING: Magpie-Phi3 → 8 Hackathon Categories

Magpie uses free-text `task_category` labels. Based on the PRM (Process Reward Model) annotation methodology, common categories include:

| Magpie Category (likely) | Hackathon 8-Way | Confidence | Mapping Logic |
|--------------------------|-----------------|------------|---------------|
| code_generation, code_writing, programming | code_gen | High | Direct match |
| code_debugging, bug_fixing, error_correction | code_debug | High | Direct match |
| math, arithmetic, algebra, geometry, calculus | math | High | Direct match |
| logical_reasoning, deductive_reasoning, puzzle | logic | Medium | Includes logic puzzles and reasoning |
| sentiment_analysis, emotion_detection | sentiment | High | Direct match |
| summarization, text_summarization | summarization | High | Direct match |
| named_entity_recognition, information_extraction | ner | Medium | NER + IE map to ner category |
| factual_knowledge, trivia, general_qa, open_qa, closed_qa | factual | High | All factual QA types |
| creative_writing, brainstorming, translation | → factual | Low | Default fallback |
| safety, alignment, ethical | → factual | Low | Not in our 8 categories — fallback |
| roleplaying, conversation | → factual | Low | Default fallback |

**Estimated yield after mapping:** 400K–600K items with clear category mapping.
**For ambiguous items:** Use LLM-as-judge to re-label or fall back to "factual".

---

## 4. PIPELINE STAGE FIT MATRIX

| Dataset | Stage 0 Pre-filters | Stage 1 Multi-axis | Stage 2 8-way Category | Stage 3 Per-cat Complexity | Stage 4 Solver Select |
|---------|:-------------------:|:------------------:|:----------------------:|:--------------------------:|:--------------------:|
| **Magpie-Phi3 (1M)** | ✗ | ✗ | ✅✅ Primary | ✅✅ Primary | ✗ |
| **wesley7137 (14K)** | ✗ | ✅ | ✗ | ✅ (no cat) | ✗ |
| **competition_math (12.5K)** | ✅ (calc filter) | ✗ | ✅ (math only) | ✅✅ Math diffs | ✗ |
| **anasnassar (6K)** | ✗ | ✅ (domains) | ⚠️ (10 domains) | ✅ 3-tier | ✅ (routing) |
| **CARROT/SPROUT (44K)** | ✗ | ✗ | ⚠️ (via dataset field) | ⚠️ (implicit) | ✅✅ Prime |
| **lime-nlp/MATH (105K)** | ✗ | ✗ | ⚠️ (math only) | ✅✅ Continuous | ✗ |
| **lime-nlp/GSM8K (40K)** | ✗ | ✗ | ⚠️ (math only) | ✅ Continuous | ✗ |
| **rokokot (8K)** | ✗ | ✅✅ Features | ✗ | ✅ | ✗ |
| **agentlans (100K)** | ✗ | ✗ | ✗ | ✅ | ✗ |
| **SupraLabs (992)** | ✗ | ✗ | ✗ | ✅ | ✗ |
| **JanW42 (100K)** | ✗ | ✗ | ✗ | ✅ | ✗ |
| **Existing training_data (58K)** | ✗ | ✗ | ✅ Already used | ✗ | ✗ |
| **Existing prompt_data (13.5K)** | ✗ | ✗ | ✅ Already used | ✗ | ✗ |

**Legend:** ✅ = Strong fit; ⚠️ = Partial fit; ✗ = Not applicable

---

## 5. WEAK LABELS FOR STAGE 0 (PRE-FILTERS)

Stage 0 pre-filters are regex-based trivial detectors:
- **code detection**: ``` or `def ` or `function ` → ~100% precise
- **calculation detection**: `\d+ *[+\-*/] *\d+` or `calculate X` → ~95%+ precise
- **factual lookup**: `who is`, `what is`, `capital of`, `date of` → ~90% precise

**Weak labels available from existing data:**

| Pre-filter | Items w/ weak label | Source |
|-----------|--------------------|--------|
| Code detection | 58K (from training_data.json: code_gen + code_debug = 328; plus HumanEval 164) | existing |
| Calculation detection | 12.5K (competition_math) | **needs download** |
| Factual lookup | ~10K (MMLU from training_data.json) | existing |
| Math keywords | ~40K (GSM8K_Difficulty) | **needs download** |

**Recommendation:** Stage 0 pre-filters are trivial regexes. Weak labels from existing data (training_data.json) are sufficient for validation. No additional download needed specifically for Stage 0.

---

## 6. BEST SOURCE FOR PER-CATEGORY COMPLEXITY LABELS

This is the **biggest gap** in the current pipeline. Currently only 78 hand-curated items.

| Dataset | Category Coverage | Complexity Scale | Items | Priority |
|---------|-----------------|------------------|-------|----------|
| **Magpie-Phi3** | All 8 categories (after mapping) | 5 levels (1-5) | 200K–600K | ⭐⭐⭐ BEST |
| **competition_math** | math only | 5 levels (1-5) + solutions | 12,500 | ⭐⭐ |
| **wesley7137** | No categories (all types) | Continuous 0-1 | 14,049 | ⭐⭐ |
| **lime-nlp/MATH_Difficulty** | math only | 5 levels + continuous 0-100% | 105,000 | ⭐ |
| **anasnassar** | 10 domains (6 map to our 8) | 3 tiers (low/med/high) | 6,000 | ⭐ |

**Best strategy:** USE MAGPIE-PHI3 as the primary source for per-category complexity. Map difficulty levels to 0-1 scale:
- very_easy → 0.1
- easy → 0.3
- medium → 0.5
- hard → 0.7
- very_hard → 0.9

Supplement math complexity from competition_math (level 1-5) and lime-nlp datasets.

---

## 7. DOWNLOAD PRIORITY ORDER

| Order | Dataset | Size | Why | Est. Download Time |
|-------|---------|------|-----|-------------------|
| **1** | **Magpie-Phi3-Pro-1M** | 1M items | Primary source for both category + complexity labels | 5-10 min (HF datasets, ~5GB) |
| **2** | **wesley7137/question_complexity** | 14K | Best continuous complexity (0-1), no download client needed | <1 min |
| **3** | **competition_math** | 12.5K | Math category + difficulty 1-5 | <1 min |
| **4** | **anasnassar/llm-query-complexity** | 6K | Designed for routing, balanced | <1 min |
| **5** | **CARROT-LLM-Routing/SPROUT** | 44K | Model scores for solver selection | 2-3 min |
| **6** | **lime-nlp/MATH_Difficulty** | 105K | Math continuous complexity | 1-2 min |
| **7** | **lime-nlp/GSM8K_Difficulty** | 40K | Math word problem complexity | 1 min |
| **8** | **rokokot/question-type** | 8K | Linguistic features for multi-axis | <1 min |
| **9** | **agentlans/prompt-difficulty** | 100K | General difficulty scores | 2-3 min |
| **10** | **SupraLabs Prompt-Routing** | 992 | Quick integration, routing-specific | <30s |
| **11** | **JanW42/prompt-difficulty-mean** | 100K | Difficulty mean scores | 2-3 min |

---

## 8. DATA GENERATION PLAN (fills gaps)

For categories/axes NOT covered by any dataset:

| Missing Data | Generation Method | Est. Yield | Priority |
|-------------|------------------|-----------|----------|
| Per-category complexity for logic, NER, sentiment, summarization, code_debug | Use Kimi K2.7 to rate difficulty of existing training_data items (58K already have category labels) | 58K rated items | HIGH — quickest path |
| Multi-axis features (creativity, verbosity, structured output, multi-step reasoning) | Programmatic feature extraction from existing prompts (length, question marks, bullet list patterns, instruction verbs) | All 13.5K + 58K items | MEDIUM — can be computed, not downloaded |
| Solver selection training data | Use CARROT/SPROUT model scores + simulate with prompt_data items sent to both small and large models | 44K (CARROT) | MEDIUM |
| Validation set for per-category complexity | Use eval sets (simple/medium/hard 100 each) + hand-label 10 per category | 300-500 | LOW — baseline exists |

---

## 9. RECOMMENDED ACTION PLAN

### Phase 1 (Immediate — Complete Today)
1. ✅ Download **Magpie-Phi3-Pro-1M** (~5-10 min)
2. ✅ Download **wesley7137/question_complexity_classification** (~30s)
3. ✅ Download **competition_math** (~30s)
4. ✅ Download **anasnassar/llm-query-complexity-benchmark** (~30s)
5. Write mapping script: `map_magpie_categories.py` → maps Magpie categories to 8 hackathon categories
6. Write integration script: `integrate_datasets.py` → merges all new datasets with prompt_data/

### Phase 2 (Build Stage 3 — Per-Category Complexity)
1. Download **CARROT/SPROUT** (~2-3 min)
2. Download **lime-nlp/MATH_Difficulty** and **GSM8K_Difficulty** (~2 min)
3. Use LLM (Kimi K2.7) to rate difficulty of existing training_data items (58K)
4. Build per-category complexity regressor using all sources

### Phase 3 (Polish and Validate)
1. Download **agentlans/prompt-difficulty** and **JanW42** for general complexity
2. Cross-validate complexity predictions on eval sets
3. Create balanced train/val/test splits across all 8 categories
4. Document final data inventory in DATA_INVENTORY.md

---

## 10. PROJECTED FINAL DATA INVENTORY

| Source | Items | Categories | Complexity | Stage Fit |
|--------|-------|-----------|-----------|-----------|
| Magpie-Phi3-Pro-1M (mapped) | ~400K | 8-way (all) | 5-level | Stage 2, Stage 3 |
| Existing training_data.json | 58,704 | 8-way (all), imbalanced | none | Stage 2 (already used) |
| Existing prompt_data | 13,531 | 8-way (all), imbalanced | none | Stage 2 (already used) |
| wesley7137 complexity | 14,049 | none | continuous 0-1 | Stage 1, Stage 3 |
| competition_math | 12,500 | math (7 sub-cats) | 1-5 level | Stage 2, Stage 3 |
| CARROT/SPROUT | 44,241 | via dataset field | implicit | Stage 4 |
| anasnassar complexity | 6,000 | 10 domains | 3-tier | Stage 1, Stage 3, Stage 4 |
| lime-nlp/MATH_Difficulty | 105,000 | math only | 1-5 + continuous | Stage 3 |
| lime-nlp/GSM8K_Difficulty | 40,000 | math only | continuous | Stage 3 |
| rokokot linguistic | 8,280 | binary | continuous + features | Stage 1 |
| agentlans difficulty | 100,000 | none | 1-7 scale | Stage 3 |
| SupraLabs routing | 992 | none | 1-5 | Stage 3 |
| JanW42 difficulty | 100,000 | none | continuous | Stage 3 |
| **TOTAL** | **~895K–1.1M** | | | |

### Category Balance Target (Stage 2 Classifier)

| Category | Current | Target (after Magpie mapping) |
|----------|---------|-------------------------------|
| factual | 8,156 (train) | 50-100K |
| math | 43 (train) | 50-100K |
| code_gen | 307 (train) | 50-100K |
| logic | 40 (train) | 40-80K |
| code_debug | 31 (train) | 20-40K |
| ner | 1,233 (train) | 30-60K |
| sentiment | 33 (train) | 30-60K |
| summarization | 981 (train) | 30-60K |

Magpie-Phi3 alone can provide enough items to balance all 8 categories.
