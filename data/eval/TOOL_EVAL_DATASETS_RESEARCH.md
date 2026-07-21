# Tool Evaluation Datasets Research

**Project:** 8-way prompt classifier with 25 deterministic solver tools  
**Goal:** Find labeled test sets that can evaluate the ACCURACY of each solver tool (not routing/classifier accuracy)

---

## Executive Summary

We already have ground-truth eval data from 10 known benchmark sources in our existing training-v3.json/validation-v3.json files (GSM8K, SST-2, NQ-Open, XSum, MBPP, HumanEvalPack, LogiQA, tweetner7, zebra puzzles, WNUT2017). Below is a broader map of external datasets — from Kaggle, HuggingFace, and our hackathon repos — organized by tool category.

---

## 1. Factual Tools (5 tools)

### `factual_qa` — SQLite FTS5 fact DB (Dolly 15K + MMLU + common knowledge)

**Available Datasets with Ground Truth Answers:**

| Dataset | Source | URL | Size | Items | License | Download |
|---------|--------|-----|------|-------|---------|----------|
| Databricks Dolly 15K | Kaggle | https://www.kaggle.com/datasets/databricks/databricks-dolly-15k | 4.7 MB | 15K | CC BY-SA 3.0 | ✓ |
| NQ-Open (Natural Questions) | HF (already used) | https://huggingface.co/datasets/nq_open | ~50 MB | ~90K | CC BY-SA 3.0 | ✓ |
| SubjQA (Question Answering) | Kaggle | https://www.kaggle.com/datasets/arashnic/subjqa-question-answering-dataset | 11 MB | ~10K | CC0 | ✓ |
| BoolQ (Yes/No QA) | Kaggle | https://www.kaggle.com/datasets/thedevastator/unlock-logical-thinking-with-the-boolq-dataset | 3 MB | 15,942 | CC0 | ✓ |
| QuAIL (Reading Comprehension) | Kaggle | https://www.kaggle.com/datasets/thedevastator/introducing-quail-a-comprehensive-reading-compre | 1.6 MB | 15K | CC0 | ✓ |

**Recommendation for `factual_qa`:** Use NQ-Open (already working) + Dolly 15K for diverse factual QA. BoolQ adds yes/no coverage.

### `spell_check` / `list_misspellings` — SymSpell correction

| Dataset | Source | URL | Size | Items | License | Download |
|---------|--------|-----|------|-------|---------|----------|
| Spelling Corrector (Norvig) | Kaggle | https://www.kaggle.com/datasets/bittlingmayer/spelling | 2.5 MB | ~1M words | Unknown | ✓ |
| Grammatical Error Detection | Kaggle | https://www.kaggle.com/datasets/vipin20/nlp-word-correction | 1.8 MB | ~5K | Unknown | ✓ |
| English Grammar Error Dataset | Kaggle | https://www.kaggle.com/datasets/colabsss/english-grammar-error-dataset | 0.5 MB | ~2K | CC0 | ✓ |

**Recommendation:** Norvig's spelling dataset has misspelled→correct pairs. Perfect for SymSpell eval.

### `search_web` / `search_factual` — DuckDuckGo search

Hard to eval offline since they require live web access. Consider creating a **frozen corpus** of 50-100 queries with known top-1 answers from Wikipedia/DDG. Or skip systematic eval and only test offline with NQ-Open questions that have known answers.

---

## 2. Sentiment Tool (1 tool)

### `sentiment_analysis` — VADER-based (currently 70.4% accuracy)

| Dataset | Source | URL | Size | Items | License | Download |
|---------|--------|-----|------|-------|---------|----------|
| SST-2 (Stanford Sentiment) | HF (already used) | https://huggingface.co/datasets/glue/viewer/sst2 | ~7 MB | 67K | CC0 | ✓ |
| Rotten Tomatoes Movie Reviews | Kaggle | https://www.kaggle.com/datasets/thedevastator/movie-review-data-set-from-rotten-tomatoes | 0.5 MB | ~10K | CC0 | ✓ |
| IMDB Reviews | HF | https://huggingface.co/datasets/imdb | ~80 MB | 50K | Other | ✓ |

**Already evaled:** SST-2 (19 questions in training-v3). **Add:** IMDB for cross-domain eval.

---

## 3. Summarization Tool (1 tool)

### `summarize` — Sumy-based extractive (6 algorithms + ensemble)

| Dataset | Source | URL | Size | Items | License | Download |
|---------|--------|-----|------|-------|---------|----------|
| CNN/DailyMail News Summarization | Kaggle | https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail | 503 MB | ~300K | CC0 | ✓ |
| XSum | HF (already used) | https://huggingface.co/datasets/xsum | ~600 MB | 226K | CC BY-NC-SA 4.0 | ✓ |
| Text for Summarize NLP Task | Kaggle | https://www.kaggle.com/datasets/radimkzl/text-for-summarize-nlpllm-task | 489 MB | ~50K | CC0 | ✓ |

**Already used:** XSum (19 in training-v3). **Recommendation:** CNN/DailyMail is the standard benchmark. Extract a 100-item test subset with reference summaries for ROUGE scoring.

---

## 4. Math Tool (1 tool)

### `math_solve` — SymPy + calculator (5/8 complex archetypes)

| Dataset | Source | URL | Size | Items | License | Download |
|---------|--------|-----|------|-------|---------|----------|
| GSM8K (Grade School Math 8K) | Kaggle | https://www.kaggle.com/datasets/johnsonhk88/gsm8k-grade-school-math-8k-dataset-for-llm | 4.9 MB | 8K | Apache 2.0 | ✓ |
| MathQA (Math Problems) | Kaggle | https://www.kaggle.com/datasets/thedevastator/dataset-for-solving-math-word-problems | 6.6 MB | ~37K | CC0 | ✓ |
| AIMO External Dataset | Kaggle | https://www.kaggle.com/datasets/alejopaullier/aimo-external-dataset | 4.3 MB | ~1K | MIT | ✓ |
| MGSM (Multilingual GSM) | Kaggle | https://www.kaggle.com/datasets/open-benchmarks/mgsm-multilingual-grade-school-math-benchmark | 0.3 MB | 1,250 | Unknown | ✓ |
| Math-CoT (4-Step Reasoning) | Kaggle | https://www.kaggle.com/datasets/trinhduc041/nckh-processed-data | 11.5 MB | ~7.5K | Apache 2.0 | ✓ |
| SigmaDolphin | Kaggle | https://www.kaggle.com/datasets/saurabhshahane/sigmadolphin | 3.4 MB | ~30K | Other | ✓ |
| MATH Dataset (Competition Math) | HF | https://huggingface.co/datasets/competition_math | ~5 MB | 12,500 | MIT | ✓ |
| SVAMP | HF | https://huggingface.co/datasets/svamp | ~1 MB | 1,000 | MIT | ✓ |

**Already used:** GSM8K (19 in training-v3). **Add:** MATH for hard problems, SVAMP for robustness.

---

## 5. Logic Tools (4 tools)

### `solve_logic_puzzle` — python-constraint

| Dataset | Source | URL | Size | Items | License | Download |
|---------|--------|-----|------|-------|---------|----------|
| Logic Grid Deduction Dataset | Kaggle | https://www.kaggle.com/datasets/reyesenrique/logic-grid-deduction-dataset | 0.006 MB | ~50 | Apache 2.0 | ✓ |
| Logic Grid Deduction Advanced 100 | Kaggle | https://www.kaggle.com/datasets/reyesenrique/logic-grid-deduction-advanced-100 | 0.003 MB | 100 | Apache 2.0 | ✓ |
| Aetherian Concord Narrative Logic Puzzles | Kaggle | https://www.kaggle.com/datasets/kaziaishikuzzaman/aetherian-concord-narrative-logic-puzzles-dataset | 1.6 MB | ~500 | Apache 2.0 | ✓ |
| ProSAT — Alice in Wonderland Puzzles | Kaggle | https://www.kaggle.com/datasets/habanwer/nemotron-atlas-solver-augmented-traces-script | 3.5 MB | ~9500 | CC BY-SA 4.0 | ✓ |

### `solve_syllogism` — Venn-like set logic

No dedicated Kaggle/HF datasets found. **Recommendation:** Use existing LogiQA dataset (already in training-v3) which includes syllogism-style questions. Also generate synthetic syllogism test cases.

### `solve_truth_teller_liar` — Knight/knave puzzles

No standard datasets. **Recommendation:** Create synthetic test cases (easy: 20 puzzles with known solutions). The zebra puzzle datasets (already used: "zebra" in training-v3) are similar.

### `solve_number_sequence` — 6 pattern types

| Dataset | Source | URL | Size | Items | License | Download |
|---------|--------|-----|------|-------|---------|----------|
| AI Agent Reasoning: Microsoft P3 Puzzles | Kaggle | https://www.kaggle.com/datasets/amanatar/ai-agent-reasoning-traces-250-microsoft-p3 | 0.017 MB | 250 | CC BY 4.0 | ✓ |
| STARC-50 (Abstraction & Reasoning) | Kaggle | https://www.kaggle.com/datasets/komilparmar/starc-50 | 0.07 MB | 50 | MIT | ✓ |

**Recommendation:** OEIS has a downloadable list of integer sequences (oeis.org). Can extract first-N terms as input and next term as expected answer.

---

## 6. Code Tools (2 tools)

### `format_python` — black+ruff linting

Evaluation dataset: Code snippets with known formatting errors. **Recommendation:** Take well-formatted Python files from HumanEval/MBPP, intentionally add formatting violations (wrong indentation, spacing, line length), run through formatter, and verify output matches original.

### `execute_code_safe` — RestrictedPython sandbox

| Dataset | Source | URL | Size | Items | License | Download |
|---------|--------|-----|------|-------|---------|----------|
| OpenAI HumanEval Code Gen | Kaggle | https://www.kaggle.com/datasets/thedevastator/openai-humaneval-code-gen | 0.045 MB | 164 | CC0 | ✓ |
| Coding Questions Dataset | Kaggle | https://www.kaggle.com/datasets/guitaristboy/coding-questions-dataset | 0.13 MB | ~500 | MIT | ✓ |
| PyTRAIN:CLEAN (Reasoning+Code) | Kaggle | https://www.kaggle.com/datasets/vishnuobi/pytrainclean-jsonl-datasetreasoning-finalcode | 0.096 MB | ~1K | CC BY-SA 4.0 | ✓ |

**Already used:** HumanEvalPack (19 in training-v3), MBPP (19). **Add:** Direct HumanEval for code generation eval.

---

## 7. NER Tool (1 tool)

### `ner_extract` — spaCy NER

| Dataset | Source | URL | Size | Items | License | Download |
|---------|--------|-----|------|-------|---------|----------|
| CoNLL-2003 (English) | Kaggle | https://www.kaggle.com/datasets/alaakhaled/conll003-englishversion | 0.98 MB | ~35K | CC0 | ✓ |
| BC5CDR (BioCreative V - Biomedical) | Kaggle | https://www.kaggle.com/datasets/madjakul/bc5cdr-iob | 0.8 MB | ~1,500 | Unknown | ✓ |
| MultiCoNER 2 (Multilingual) | Kaggle | https://www.kaggle.com/datasets/cryptexcode/multiconer-2 | 192 MB | ~30K | CC BY-SA 4.0 | ✓ |
| NCBI Disease | HF | https://huggingface.co/datasets/ncbi_disease | ~5 MB | 6,932 | CC0 | ✓ |

**Already used:** tweetner7 (18 in training-v3), WNUT2017 (1). **Add:** CoNLL-2003 (standard benchmark), NCBI Disease for biomedical.

---

## 8. Fun Tools (10 tools — lower priority)

These don't need dedicated eval datasets:
- **CSV formatter** — Can compare against known CSV parsing spec
- **Text stats / reverse text / palindrome check** — Trivially verifiable
- **Word cloud** — Visual, subjective
- **Leetspeak** — Can generate synthetic test cases
- **Coin flip** — Random, just test it produces valid output
- **Emoji translator** — Can compare against Unicode standard mapping
- **Weather hot take** — Humor, subjective
- **April Fools countdown** — Date math, trivially testable

---

## Hackathon Repo Findings

Checked repos: `/home/artem/dev/amd-garbage/scripts-v12e/repos/`

| Repo | Dataset File | Description |
|------|-------------|-------------|
| amd-hack-track1-router/bench/ | `tasks_extended.json` (251 lines) | Prompts only, no answers — **not usable for tool eval** |
| amd-hackathon-agent/eval/ | `test_tasks.json` (82 lines) | Prompts only, no answers — **not usable for tool eval** |
| amd-routing-agent/input/ | `tasks.json` (16 items) | General prompts, no ground truth — **not usable** |

**None of the hackathon repos contain labeled evaluation datasets** — they only have input prompts without ground-truth answers. All our ground-truth data comes from our own `data/eval/` directory.

---

## Existing Local Datasets (Repurposable for Tool Eval)

| File | Items | Tool Category | Ground Truth? | Notes |
|------|-------|---------------|---------------|-------|
| training-v3.json | 152 | All 8 categories | ✓ expected_answer | 19 per category, from GSM8K/SST-2/NQ-Open/XSum/MBPP/HumanEvalPack/LogiQA |
| validation-v3.json | 48 | All 8 categories | ✓ expected_answer | 6 per category, held-out companion |
| factual_combined_80.json | 80 | factual | ✓ expected_answer | Combined from training sources |
| math_combined_80.json | 80 | math | ✓ expected_answer | Combined from training sources |
| ner_all_models.json | NER results | NER | ✓ expected per entity | Has expected vs got comparison |
| heldout_40.json | 40 | 7 categories | ✓ gold.answer | Reference from hackathon team |
| build-A-40.json / build-B-40.json | 80 | 7 categories | ✓ gold.answer | Custom built |
| primary/eval_60_medium_hard.json | 60 | All | ✓ expected_answer | Medium-hard difficulty |
| primary/eval_hard_218.json | 218 | All | ✓ expected_answer | Hard questions |
| primary/eval_clean_val.json | ~3K+ | All | ✓ expected_answer | Large validation set |

---

## Summary Table of Recommended External Datasets

| # | Dataset Name | Source | URL | Tool(s) | Items | Ground Truth | License |
|---|-------------|--------|-----|--------|-------|------------|---------|
| 1 | GSM8K | Kaggle/HF | Kaggle: johnsonhk88/gsm8k... | math_solve | 8,792 | ✓ Answers | Apache 2.0 |
| 2 | MATH | HF | competition_math | math_solve | 12,500 | ✓ Solutions | MIT |
| 3 | SVAMP | HF | svamp | math_solve | 1,000 | ✓ Answers | MIT |
| 4 | MathQA | Kaggle | thedevastator/dataset-for-solving-math-word-problems | math_solve | 37K | ✓ | CC0 |
| 5 | SST-2 | HF | glue/sst2 | sentiment_analysis | 67K | ✓ Labels | CC0 |
| 6 | IMDB | HF | imdb | sentiment_analysis | 50K | ✓ Labels | Other |
| 7 | Rotten Tomatoes | Kaggle | thedevastator/movie-review-data-set-from-rotten-tomatoes | sentiment_analysis | 10K | ✓ | CC0 |
| 8 | CNN/DailyMail | Kaggle | gowrishankarp/newspaper-text-summarization-cnn-dailymail | summarization | 300K | ✓ Highlights | CC0 |
| 9 | XSum | HF | xsum | summarization | 226K | ✓ Summary | CC BY-NC-SA |
| 10 | NQ-Open | HF | nq_open | factual_qa | 90K | ✓ Answers | CC BY-SA |
| 11 | BoolQ | Kaggle | thedevastator/unlock-logical-thinking-with-the-boolq-dataset | factual_qa | 15,942 | ✓ Yes/No | CC0 |
| 12 | Dolly 15K | Kaggle | databricks/databricks-dolly-15k | factual_qa | 15K | ✓ Answers | CC BY-SA |
| 13 | HumanEval | Kaggle | thedevastator/openai-humaneval-code-gen | execute_code_safe | 164 | ✓ Tests | CC0 |
| 14 | MBPP | HF | mbpp | execute_code_safe | 974 | ✓ Tests | CC BY-SA |
| 15 | CoNLL-2003 | Kaggle | alaakhaled/conll003-englishversion | ner_extract | 35K | ✓ Entities | CC0 |
| 16 | NCBI Disease | HF | ncbi_disease | ner_extract | 6,932 | ✓ Entities | CC0 |
| 17 | LogiQA | HF | logiqa | solve_syllogism / solve_logic_puzzle | 8,678 | ✓ Answers | MIT |
| 18 | Logic Grid Deduction | Kaggle | reyesenrique/logic-grid-deduction-dataset | solve_logic_puzzle | 50+100 | ✓ Solutions | Apache 2.0 |
| 19 | Norvig Spelling | Kaggle | bittlingmayer/spelling | spell_check / list_misspellings | ~1M words | ✓ Corrections | Unknown |
| 20 | P3 Puzzles (Microsoft) | Kaggle | amanatar/ai-agent-reasoning-traces-250-microsoft-p3 | solve_number_sequence | 250 | ✓ Solutions | CC BY 4.0 |

---

## Key Recommendations

1. **Highest ROI for accuracy eval:** GSM8K (math), SST-2 (sentiment), CoNLL-2003 (NER), CNN/DailyMail (summary), Norvig Spelling (spelling), HumanEval (code)

2. **Already integrated in training-v3:** GSM8K, SST-2, NQ-Open, XSum, MBPP, HumanEvalPack, LogiQA, tweetner7 — but these were used for *classifier* training, not tool accuracy eval. Can **reuse** the same datasets for tool accuracy by running the tool on each question and comparing against expected_answer.

3. **Gaps (no suitable datasets found):**
   - `solve_truth_teller_liar` — Need to generate synthetic knight/knave puzzles
   - `solve_number_sequence` — Could use OEIS for pattern extrapolation
   - `solve_syllogism` — LogiQA partially covers this; supplement with synthetic
   - `search_web` / `search_factual` — Need a curated frozen query-document set or rely on offline FTS5 eval

4. **How to use:** For each tool, iterate over the recommended dataset, call the tool with the question as input, compare output to expected_answer via fuzzy-match (same grading strategy as the hackathon).
