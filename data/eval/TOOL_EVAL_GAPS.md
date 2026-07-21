# Tool Evaluation Gap Analysis — 2026-07-13

## Baseline Accuracy (from eval_tools.py)

| Tool | Attempts | Matches | Accuracy | What's tested |
|------|:--------:|:-------:|:--------:|---------------|
| sentiment_analysis | 30 | 16 | **53.3%** | VADER on validation general sentiment |
| factual_qa | 177 | 36 | **20.3%** | FTS5 fact DB (limited coverage) |
| spell_check | 177 | 83 | **46.9%** | Incidental matching on factual prompts |
| math_solve | 83 | 2 | **2.4%** | Word problems, not raw expressions |
| summarize | 30 | 2 | **6.7%** | Extractive vs reference mismatch |
| solve_logic_puzzle | 140 | 0 | **0.0%** | LSAT-style reasoning, not puzzles |
| solve_syllogism | 0 | 0 | **N/A** | Never reached |
| solve_truth_teller_liar | 140 | 0 | **0.0%** | LSAT-style reasoning, not knights |
| solve_number_sequence | 140 | 2 | **1.4%** | Not matching logic prompts |
| ner_extract | 0 | 0 | **0.0%** | spaCy model not loaded |
| format_python | — | — | **N/A** | Not an answer-producing tool |
| execute_code_safe | — | — | **N/A** | Not an answer-producing tool |

## Critical Gaps

### 1. Code answer tools (code_debug, code_gen)
**Issue:** No tool produces "fixed code" or "generated code" as output.
**Need:** Code generation solver (12 templates from winning repo) + behavioral verification.
**Eval data:** HumanEval (164), MBPP (1K) — already have HumanEvalPack downloaded.

### 2. Math word problems (math)
**Issue:** 97.6% of math prompts are word problems, not raw expressions.
**Need:** Word-problem accumulator (narrative math solver) from winning repo pattern.
**Eval data:** GSM8K (8.5K), SVAMP (200), MATH (12.5K) — need GSM8K loader.

### 3. LSAT-style logic (logic)
**Issue:** 100% of logic prompts are multi-paragraph reasoning with multiple choice.
**Need:** A tool that can parse and answer LSAT/GMAT-style logical reasoning questions.
**Eval data:** LogiQA (8.6K) — need to investigate if our pattern works there.

### 4. NER (ner)
**Issue:** spaCy model fails to load (torch dependency issue or missing model).
**Need:** Fix lazy loading or use lightweight alternative (spacy en_core_web_sm download).
**Eval data:** NCBI Disease (14K) — already available in eval_deterministic.py.

### 5. Factual QA coverage (factual)
**Issue:** FTS5 only covers 16,568 facts — can't answer most validation factual questions.
**Need:** Load SQuAD v2, expand common knowledge, use web search as fallback.
**Eval data:** SQuAD 2.0 (150K) — download and index into FTS5.

### 6. Summarization (summarization)
**Issue:** Extractive summary rarely matches reference summary exactly.
**Need:** Loosen matching for extractive vs abstractive summaries, or add post-processing.
**Eval data:** XSum (227K), CNN/DailyMail (300K) — need ROUGE-based eval.

### 7. Tools that need dedicated eval datasets
| Tool | Need | Recommended dataset |
|------|------|--------------------|
| spell_check | Text with known typos | symspellpy's built-in test set |
| solve_number_sequence | Number pattern completion | OEIS subsets |
| solve_truth_teller_liar | Knight/knave puzzles | Custom synthetic (easy to generate) |
| solve_syllogism | Syllogism test cases | Custom from logic textbooks |
