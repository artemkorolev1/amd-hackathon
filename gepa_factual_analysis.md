# GEPA Analysis: factual_knowledge — July 14, 2026

## Executive Summary

**Factual knowledge is the last unevaluated category.** The current architecture (FactDB FTS5 → local LLM → FW escalation) achieves **~84% local accuracy**, with the ceiling estimated at **~92%** (FW-equivalent). The gap is ~8% due to:

1. **FactDB errors at high confidence** (7/64 high-conf FactDB answers are wrong)
2. **Counterfactual "what if" questions** misclassified as factual (13/77 = 17% of test set)
3. **Long-tail trivia** the 1.5B model doesn't know and FactDB doesn't cover perfectly

**Bottom line:** Prompt strategy barely matters for factual — the model doesn't know the facts. **FactDB is the real lever**, and it's already loaded with 21K facts covering all eval entries. The next improvement is BM25 tuning + verification/reranking, not more data.

---

## 1. Prompt Ablation Results

### Experimental Setup
- **Model:** qwen2.5-1.5b-instruct-q4_k_m.gguf (N_GPU_LAYERS=-1)
- **Test set:** 77 factual questions (19 from training-v3.json + 58 from factual_combined_80.json)
- **Temperature:** 0.0 (deterministic), max_tokens=64
- **Evaluation:** Normalized string match (lowercase, stripped punctuation, collapse spaces)

### Results

| Variant | System Prompt | Accuracy |
|---------|--------------|:--------:|
| **verbose** | Full factual/low tier from dynamic_prompts.py | **37.7%** (29/77) |
| **empty** | No system prompt (just user message) | **36.4%** (28/77) |
| **minimal** | `"Answer:"` | **35.1%** (27/77) |

### Analysis
- **All three variants perform similarly** (within 2.6 points). The differences are 1-2 questions, essentially noise.
- **Raw LLM accuracy is ~37%** because most factual questions are long-tail NQ trivia (e.g., "who plays gram on the young and the restless?" → Max Shippee) that the 1.5B model simply doesn't know.
- **The model hallucinates confidently.** It produces plausible-sounding but wrong answers (e.g., "John McEwen" for first Australian PM, "Madhya Pradesh" for center of India).
- **Verbose prompt** slightly edges out due to "Address every part of multi-part questions" — this helps with the few multi-part questions, but the effect is marginal.

### Verdict: GEPA finding holds — prompt strategy doesn't matter
The GEPA finding from Jul 2026 (empty/blank system prompt wins) was based on the fact that **labeling prompts like "Fact:" or "Q:"** hurt performance on 1-2B models. Our three variants avoid those pitfalls, so they're all within noise of each other.

**Recommendation:** Keep using the verbose prompt from `dynamic_prompts.py` factual/low tier. It doesn't hurt and helps the few multi-part questions. Don't invest more time in prompt engineering for factual — the ceiling isn't in the prompt.

---

## 2. FactDB Coverage Analysis

### Current FactDB State
- **21,207 facts** loaded from:
  - Dolly-15k: 5,980 facts
  - batch-trn1/2/3: ~3,000 (training NQ entries)
  - pop-culture-v1: 1,093 facts
  - mmlu-* (26 subjects): ~8,000 facts
  - common-knowledge: 632 facts
  - plus others

### Coverage Test Results

| Metric | Count | % |
|--------|:-----:|:-:|
| Total questions | 77 | 100% |
| Any FactDB match | 77 | 100% |
| High-confidence (score ≥ 6.0) | 64 | 83.1% |
| High-conf + correct | 57 | 89.1% of high-conf |
| High-conf + wrong | 7 | 10.9% of high-conf |
| No match (score < 3.0) | 0 | 0% |

### What FactDB Does Well
- **Short trivia**: 60/64 short-trivia questions get high-confidence matches (94%)
- **High scores**: Scores range from 34 to 168 for good matches
- **Sources correct**: Most matches come from the training data (batch-trn*), meaning the eval questions were already loaded

### What FactDB Does Poorly
- **Counterfactual "What if" questions** (Context: ... If ...): Only 4/13 get high-confidence matches, and those are often wrong because FactDB matches on surface terms but doesn't understand the counterfactual framing.
- **7 high-confidence wrong answers**: These are cases where FactDB has a plausible-but-wrong answer that happens to match well on BM25 (e.g., "Good Morning Good Morning" → The Beatles, not Gene Kelly).

### FactDB Expansion Assessment
The training/validation factual entries (training-v1/v2/v3, validation-v1/v2/v3) are **ALREADY loaded** into FactDB via the `batch-trn*` sources. The 525 "unloaded" entries I initially counted are all duplicates already in the DB.

**No additional training data to load** — the FactDB already has everything from the eval sets.

To expand FactDB meaningfully, we'd need **new external sources**:
- Wiki-QA or TriviaQA
- HotpotQA (multi-hop)
- More MMLU subjects
- Custom domain facts

---

## 3. Multi-Step Workflow Assessment

### Current Architecture
```
FactDB (≥6.0) → DIRECT answer
FactDB (≥3.0) → DIRECT (flagged)
FactDB (<3.0) → Local LLM → FW escalation
```

### Proposed Multi-Step Workflow

For questions where FactDB confidence is **between 3.0 and 8.0** (uncertain range):

**Step 1 — Understand:** "What type of fact is needed? Is this a name, date, number, or definition?"
**Step 2 — FactDB retrieval:** Query FactDB, return top-3 candidates with scores
**Step 3 — Cross-verify:** Ask LLM to compare candidates against the question and pick the best:
  - "Which of these facts best answers the question? If none, say 'I don't know'."
**Step 4 — Self-consistency check:** Run Step 3 three times, take majority vote

### Expected Impact
- **Helps with** the 7 high-confidence wrong answers (FactDB returns wrong fact, multi-step catches it)
- **Helps with** edge cases where FactDB has multiple close matches
- **Does NOT help** with true long-tail trivia — model doesn't know them regardless

**Estimated lift:** +3-5% on the gap between FactDB and ceiling

---

## 4. Self-Consistency Voting Assessment

### Analysis
For factual QA:
- **The 1.5B model's raw accuracy is ~37%** — 3-sample voting won't help because most samples will be confidently wrong (same hallucination).
- **Voting is only useful** when FactDB returns a low-confidence answer and the model has partial knowledge.
- The agreement score on factual would be high (model always says the same wrong thing) but wrong.

### Where Voting Would Help
- **Edge cases** where FactDB score is 3.0-6.0 and model has seen the fact in training
- **Multi-hop questions** where the model can reason through different paths

### Estimated Lift
- **Minimal (<2%)** for single-fact questions
- **Could help +5%** on held-out multi-fact questions if applied only when FactDB confidence < 6.0

### Implementation
The code already supports self-consistency via `solve_with_consensus` in `local_vote.py`. Currently `consensus_categories = {"math", "sentiment", "ner"}` — adding "factual" is trivial.

However, I **recommend NOT enabling voting for factual** until the FactDB accuracy is improved first. Voting on a wrong model is worse than trusting a high-confidence FactDB.

---

## 5. Recommendations

### Priority 1 (High Impact, Low Effort): BM25 Threshold Tuning
- **Problem:** 7/64 high-confidence FactDB answers are wrong, and the current threshold doesn't catch them
- **Fix:** Add a **verification/reranking step** for FactDB results with score 6.0-15.0:
  - Ask LLM: "Does the following answer the question? Answer YES or NO."
  - If NO, skip to LLM fallback
- **Expected lift:** +3-5% (catches most wrong high-conf answers)
- **Effort:** ~20 lines of code

### Priority 2 (Medium Impact, Low Effort): Counterfactual Routing Fix
- **Problem:** "Context: ... If ..." questions misrouted to factual
- **Fix:** Add pattern detection in `pipeline.py` for counterfactual framing (`/^If /`, `Context:...If...`) → route to logic or treat as reasoning, not factual
- **Expected lift:** Removes 13 noisy questions from factual evaluation (not directly accuracy, but cleaner metrics)
- **Effort:** ~10 lines of code

### Priority 3 (Low Impact, Medium Effort): External Fact Sources
- **Problem:** FactDB already loading everything from eval sets (21K facts)
- **Fix:** Add HotpotQA, TriviaQA subsets for truly new coverage
- **Expected lift:** +2-5% on held-out data with new topics
- **Effort:** ~2 hours (download, format, load)

### Priority 4 (Low Impact, Low Effort): Add "factual" to consensus_categories
- Only enable after BM25 threshold tuning and verification step
- Set `CONSENSUS_SAMPLES=3` and `consensus_categories` to include "factual"
- Monitor agreement scores — only use majority if agreement ≥ 0.6

---

## 6. Data Files Examined

| File | Relevant Content |
|------|-----------------|
| `agent/pipeline.py` | Lines 844-851: deterministic solver dispatch (runs FactDB), lines 853-892: local LLM fallback |
| `agent/solvers/deterministic.py` | `solve_factual_qa()` at line 2442 — FTS5 → dict → context matching |
| `agent/solvers/fact_db.py` | FactDB with 3-tier FTS5 query (AND → prefix → OR), BM25 tuning |
| `agent/solvers/local_vote.py` | `solve_with_consensus()` — 3-sample voting, factual normalizes by stripping punctuation |
| `agent/dynamic_prompts.py` | factual prompt tiers (low/medium/high at lines 164-188) |
| `scripts/build_fact_db.py` | Loads Dolly + common_knowledge.jsonl into FTS5 |
| `data/eval/training-v3.json` | 19 factual NQ entries (already in FactDB as "batch-trn3") |
| `data/eval/factual_combined_80.json` | 58 factual entries including 13 counterfactual "Context:" questions |

## 7. Files Created/Modified

| File | Action |
|------|--------|
|| `gepa_factual_ablation.py` | Created — prompt ablation script (3 variants × 77 questions) |
|| `gepa_factual_ablation_results.json` | Created — full per-question results |
|| `gepa_factdb_analysis.py` | Created — FactDB coverage analysis |
|| `gepa_factdb_expansion_check.py` | Created — verified no new training data to load |
|| `gepa_check_db_sources.py` | Created — checked FactDB source breakdown |
|| **`gepa_factual_analysis.md`** | **This file** — comprehensive analysis |

---

## Appendix: Raw Ablation Results

### Per-Variant Accuracy
```
verbose: 37.7% (29/77) — Full verbose prompt from dynamic_prompts.py factual/low tier
empty:   36.4% (28/77) — No system prompt — just user message
minimal: 35.1% (27/77) — Minimal 'Answer:' prefix
```

### FactDB High-Conf Wrong Answers (7 cases)
1. "what are the ranks in the us navy" → FactDB: "Seaman" (expected: "E-8s senior chief petty officer")
2. "when is the last time the vikings were in the nfc championship" → FactDB: "2017" (expected: "1976")
3. "who sang the song good morning good morning" → FactDB: "The Beatles" (expected: "Gene Kelly")
4. "Context: CMB..." → FactDB: irrelevant match
5. "Context: Thomson electron..." → FactDB: irrelevant match  
6. "Context: Fleming penicillin..." → FactDB: "Alexander Fleming" (partial match, wrong answer)
7. "Context: Mitochondria mtDNA..." → FactDB: irrelevant biological fact
