# AMD ACT II Hackathon — Expected Answer Length & Grader Analysis Report

## Data Sources Analyzed

| Dataset | Questions | Format | Source |
|---------|-----------|--------|--------|
| eval_60_medium_hard.json | 60 | `{questions: [{category, prompt, expected_answer, difficulty}]}` | "Medium-hard" subset |
| eval_comprehensive_20260711_235828.json | 300 | Per-question records with category_label, difficulty, expected_answer, final_answer, `correct` field | Full eval pipeline run |

---

## 1. Grader Analysis: `evaluate.py` `fuzzy_match()`

The official grader at `/home/artem/dev/amd-hackathon/evaluate.py` uses a **chain of four strategies**, applied in order. The **first match wins**.

### Strategy 1: Exact (case-insensitive)
```python
answer.lower() == expected.lower()
```

### Strategy 2: Substring
```python
expected in answer       → PASS  # expected found anywhere in answer
if len(answer) >= 3 and answer in expected → PASS
```
**Key insight**: If your output CONTAINS the expected answer text verbatim, you pass. This is very forgiving for code/NER tasks where the expected answer is structured text that could appear embedded in a larger response.

### Strategy 3: Numeric tolerance (±1%)
- Same-length number lists → pairwise compare within 1%
- Single number in expected → any number in answer within 1%
- Handles zero-division edge case
- **Key insight**: Math answers just need the correct numeric value. The surrounding text is irrelevant.

### Strategy 4: Token overlap with stopword filtering
Stopwords removed: `the, a, an, is, to, of, in, and, that, for, it, on, with, as, at, by, or, be`
```
Short expected (< 50 chars):  threshold = 0.5  (need ≥50% of content tokens)
Long expected (≥ 50 chars):   threshold = 0.3  (need ≥30% of content tokens)
```
**Key insight**: For long answers (most summarization, factual hard, code_gen), you only need ~1/3 of the key content tokens to match. This is very permissive — as long as you hit the main nouns, verbs, and domain terms, you pass.

### What the grader rewards
| Answer style | Passes? | Why |
|---|---|---|
| Concise, direct answer | ✅ | Substring or exact match |
| Verbose + embedded answer | ✅ | Substring match captures the expected text |
| Explanation only (no answer) | ❌ | No token overlap with expected |
| Wrong numeric value | ❌ | Fails numeric check |
| Right concept, wrong terminology | ❌ | Fails token overlap |
| Partial content with key terms | ✅ | Token overlap ≥ 30% for long answers |

**Bottom line**: The grader heavily rewards answers that CONTAIN the expected tokens/numbers/text, even if wrapped in extra explanation. For code tasks, outputting the exact expected function body inside a larger response still passes (substring match). For factual/summarization, you need ~1/3 of the key content tokens.

---

## 2. Expected Answer Length Analysis

### Dataset characteristics

**60-Q dataset**: All medium/hard difficulty. Expected answers are **verbose** (full paragraphs, code implementations, structured lists). Representative of what a "real" complex evaluation looks like.

**300-Q dataset**: Mixed difficulty. Easy/ambiguous/medium questions have **very short expected answers** (1-3 tokens, often multiple-choice labels like "a) 9:2" or "c) 4 hours"). Hard questions have longer expected answers.

### Token budget recommendations

Token counts use the heuristic: `tokens = max(1, int(word_count * 1.3))` — approximating GPT-family tokenization for English prose.

#### Code Generation (code_gen)

| Difficulty | n | Avg Tokens | Min | Max | Budget* | Avg Prompt Tokens |
|-----------|---|-----------|-----|-----|--------|------------------|
| medium | 5 | 59 | 40 | 111 | **135** | 72 |
| hard | 3 | 114 | 92 | 128 | **155** | 79 |

**Format**: Full Python function implementations with imports, type hints, docstrings.
**300-Q note**: Medium includes 10 trivial "def function_name" answers (2 tokens) — these are function-signature-only expected answers.
**Recommendation**: Hard code_gen needs ~130-160 tokens for the function body. With explanation/context, budget 200-300 tokens total output.

#### Code Debugging (code_debug)

| Difficulty | n | Avg Tokens | Min | Max | Budget* | Avg Prompt Tokens |
|-----------|---|-----------|-----|-----|--------|------------------|
| medium | 5 | 35 | 18 | 74 | **90** | 91 |
| hard | 2 | 36 | 31 | 41 | **50** | 114 |

**Format**: Short corrected code snippet (usually 3-10 lines). The fix is minimal (changing `<` to `<=`, moving `count = 0` into `__init__`).
**Recommendation**: 50-90 tokens for the code fix. Pipeline's system prompt requests "Output ONLY the fully corrected function inside ```python ... ```".

#### Math

| Difficulty | n | Avg Tokens | Min | Max | Budget* | Avg Prompt Tokens |
|-----------|---|-----------|-----|-----|--------|------------------|
| easy | 7 | 1 | 1 | 1 | **30** | 7 |
| medium | 7 | 45 | 29 | 63 | **75** | 35 |
| hard | 1 | 115 | 115 | 115 | **140** | 26 |

**Format**: Easy = multiple-choice letter only (e.g. "a) 9:2"). Medium = worked solution with explanation. Hard = full derivation.
**Recommendation**: Medium math needs ~50-80 tokens (result + explanation). For hard in 60-Q dataset, the expected answer includes full worked solution (~115 tok). With numeric tolerance, just providing the right number + explanation is enough.

#### Logic

| Difficulty | n | Avg Tokens | Min | Max | Budget* | Avg Prompt Tokens |
|-----------|---|-----------|-----|-----|--------|------------------|
| medium | 4 | 79 | 59 | 94 | **115** | 79 |
| hard | 4 | 151 | 119 | 195 | **235** | 116 |
| ambiguous | 2 | 1 | 1 | 1 | **30** | 12 |

**Format**: Medium = step-by-step deduction (e.g. truth-teller puzzles). Hard = complex grid puzzles with full attribution table.
**Recommendation**: Hard logic needs the most tokens of any category (~200 tok expected). The answer must contain the full solution mapping. Budget generously.

#### Factual

| Difficulty | n | Avg Tokens | Min | Max | Budget* | Avg Prompt Tokens |
|-----------|---|-----------|-----|-----|--------|------------------|
| easy | 35 | 1 | 1 | 3 | **30** | 8 |
| medium | 3 | 91 | 85 | 98 | **120** | 113 |
| hard | 5 | 126 | 105 | 156 | **190** | 129 |
| ambiguous | 6 | 1 | 1 | 1 | **30** | 8 |

**Format**: Easy = single-word factoid answer. Medium = 2-3 sentence explanation (e.g. carbon dating). Hard = multi-paragraph with multiple sub-answers (e.g. BBB barriers + strategies).
**Recommendation**: Hard factual requires ~130-190 tokens. The prompts are also long (~130 tok). This is one of the most demanding categories.

#### Sentiment

| Difficulty | n | Avg Tokens | Min | Max | Budget* | Avg Prompt Tokens |
|-----------|---|-----------|-----|-----|--------|------------------|
| medium | 5 | 3 | 1 | 10 | **30** | 64 |
| hard | 2 | 14 | 10 | 18 | **25** | 72 |

**Format**: Single label (positive/negative/neutral/mixed) sometimes with parenthetical qualifier like "negative (sarcastic)" or "mixed/negative".
**Note**: 300-Q hard answers are all 1-3 tokens (just the label). 60-Q hard includes parenthetical justification.
**Recommendation**: 15-30 tokens is plenty. The grader passes on substring — as long as "negative" is in your output, you pass the sentiment check.

#### Named Entity Recognition (NER)

| Difficulty | n | Avg Tokens | Min | Max | Budget* | Avg Prompt Tokens |
|-----------|---|-----------|-----|-----|--------|------------------|
| medium | 3 | 69 | 57 | 75 | **90** | 136 |
| hard | 4 | 61 | 48 | 81 | **100** | 146 |

**Format**: Structured entity lists: "PERSONS: ... | ORGANIZATIONS: ... | LOCATIONS: ..." or formatted per domain.
**Recommendation**: 50-100 tokens for the entity list. The grader passes via substring match — if all entity names appear somewhere in your answer, you pass. Having structured output ensures max overlap.

#### Summarization

| Difficulty | n | Avg Tokens | Min | Max | Budget* | Avg Prompt Tokens |
|-----------|---|-----------|-----|-----|--------|------------------|
| medium | 3 | 54 | 46 | 62 | **75** | 134 |
| hard | 4 | 71 | 59 | 80 | **100** | 134 |

**Format**: 2-3 sentence balanced summaries synthesizing multiple sources.
**Note**: Prompts are very long (~134 tok) because they embed full source texts.
**Recommendation**: 75-100 tokens for the summary. The grader uses token overlap at 30% threshold (summarization expected answers are ≥50 chars, so 30% threshold). You need ~30% of key nouns/verbs to match.

#### General

| Difficulty | n | Avg Tokens | Min | Max | Budget* | Avg Prompt Tokens |
|-----------|---|-----------|-----|-----|--------|------------------|
| easy | 4 | 1 | 1 | 1 | **30** | 6 |
| medium | 7 | 2 | 1 | 2 | **30** | 5 |
| hard | 1 | 1 | 1 | 1 | **30** | 7 |

**Format**: JSON snippets or very short answers (e.g. `{"result": 42}`). Minimal category.
**Recommendation**: 30 tok budget is more than sufficient.

---

## 3. Key Findings & Implications

### A. Pipeline behavior (from 300-Q eval)
The pipeline's `final_answer` is **significantly longer** than the `expected_answer`:

| Category | Expected Avg Tokens | Pipeline Final Avg Tokens | Ratio |
|----------|-------------------|-------------------------|-------|
| code_gen hard | 130 | 356 | 2.7x |
| code_debug hard | 53 | 265 | 5.0x |
| factual hard | 83 | 300 | 3.6x |
| logic hard | 18 | 332 | 18x |
| sentiment hard | 2 | 120 | 60x |
| summarization hard | 63 | 258 | 4.1x |

The pipeline's LLM generates verbose chain-of-thought reasoning alongside the answer, which bloats output by 3-18x. The grader's substring match means many of these still pass because the expected answer text appears within the verbose output — but the token waste is enormous.

### B. Categories where concise works

For **sentiment**, **math easy/hard**, **NER**, and **general**: short, direct answers pass the grader easily (substring or numeric match). No need for verbose explanations.

### C. Categories needing comprehensive answers

For **logic hard**, **factual hard**, **code_gen hard**, and **summarization hard**: the expected answers are inherently long (60-200 tokens). The grader expects ~30% token overlap, so you need to cover most key content terms.

### D. What the grader's token overlap means in practice

For a typical **factual hard** question (expected ~126 tokens, ~680 chars):
- Stopwords removed → ~85-95 content tokens remain
- Need 30% = ~25-30 of those tokens to appear in answer
- This is lenient: ~25 matching keywords is enough even if phrasing differs

For a **summarization** question (expected ~71 tokens, ~380 chars):
- ~50-60 content tokens after stopword filtering
- Need 30% = ~15-18 content token matches
- As long as you mention the key entities and numbers from both sides of the debate, you pass

---

## 4. Token Budget Recommendations Summary Table

| Category | Difficulty | Answer Tokens (Avg) | Recommend Budget | Prompt Tokens (Avg) |
|----------|-----------|-------------------|-----------------|-------------------|
| code_gen | medium | 59 | **135** | 72 |
| code_gen | hard | 114 | **155** | 79 |
| code_debug | medium | 35 | **90** | 91 |
| code_debug | hard | 36 | **50** | 114 |
| math | medium | 45 | **75** | 35 |
| math | hard | 115 | **140** | 26 |
| logic | medium | 79 | **115** | 79 |
| logic | hard | 151 | **235** | 116 |
| factual | medium | 91 | **120** | 113 |
| factual | hard | 126 | **190** | 129 |
| sentiment | medium | 3 | **30** | 64 |
| sentiment | hard | 14 | **25** | 72 |
| ner | medium | 69 | **90** | 136 |
| ner | hard | 61 | **100** | 146 |
| summarization | medium | 54 | **75** | 134 |
| summarization | hard | 71 | **100** | 134 |
| general | any | 1-2 | **30** | 5-7 |
| math | easy | 1 | **30** | 7 |
| factual | easy | 1 | **30** | 8 |

**Budget formula**: `max(30, max_expected_tokens * 1.2)` for categories where explanation may be needed (sentiment, general, math easy). For categories where the expected answer IS the output (code_gen, code_debug, ner), budget = `max_expected_tokens * 1.2` rounded up.

### Files analyzed
- `/home/artem/dev/amd-hackathon/eval_60_medium_hard.json` (60 questions, Format B)
- `/home/artem/dev/amd-hackathon-filtered-build/eval_results/eval_comprehensive_20260711_235828.json` (300 questions)
- `/home/artem/dev/amd-hackathon/evaluate.py` (official grader with fuzzy_match)
- Generated analysis files: none persisted (ran inline Python analysis)
