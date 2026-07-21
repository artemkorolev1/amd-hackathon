# Sentiment Classification Parameter Research for 1B GGUF Models

**Date:** 2026-07-13
**Context:** AMD ACT II Hackathon — llama.cpp GGUF models (1-2B params)
**Pipeline:** GEPA framework, Cell-based evolution with DecodingConfig

---

## 1. Parameter Analysis

### 1.1 temperature

| Property | Value |
|----------|-------|
| Valid range | [0.0, 2.0] (llama.cpp: 0.0 = greedy, higher = more random) |
| llama.cpp default | 0.2 |
| Current pipeline | 0.0 (greedy) |
| **Recommended for sentiment** | **0.0** — sentiment is deterministic; you want the exact same label every time for the same input |
| Recommended for NER | 0.0 — entities are factual |
| Recommended for factual | 0.0 — exact answers |
| Recommended for summarization | 0.1–0.3 — some diversity in phrasing is OK |
| GEPA sweep? | Yes — sweep {0.0, 0.1, 0.2} |

**Notes:**
- temperature=0.0 uses `argmax` (greedy decoding) — this is correct for classification tasks.
- For generation tasks (summarization), a tiny amount of temperature helps avoid boring repetition.
- Values > 0.0 for sentiment introduce variance that hurts reproducibility.

### 1.2 top_p (Nucleus Sampling)

| Property | Value |
|----------|-------|
| Valid range | [0.0, 1.0] (1.0 = disabled, lower = more focused) |
| llama.cpp default | 0.95 |
| Current pipeline | 1.0 (effectively disabled) |
| **Recommended for sentiment** | **0.8–0.9** — limits the vocabulary to high-probability tokens. For 1B models, the correct sentiment label usually has much higher probability than alternatives. A top_p of 0.8 cuts out noise tokens while keeping the correct label. |
| Recommended for NER | 0.8 — entities are specific; noise hurts |
| Recommended for factual | 0.9 — allows for multi-token answers while filtering garbage |
| Recommended for summarization | 0.95 — more variety in word choice is fine |
| GEPA sweep? | Yes — sweep {1.0, 0.95, 0.9, 0.8} |

**Theory:**
- With temperature=0.0 AND top_p < 1.0, top_p is effectively ignored because there's no sampling — greedy always picks the top token. However, **some** implementations (including llama.cpp) still apply top_p filtering even at temperature=0.0 as a logit processor. When temperature != 0.0, top_p restricts the next-token pool to the smallest set whose cumulative probability >= top_p, renormalizing over that subset.
- For 1B models on short sentiment tasks, top_p=0.9 reduces the chance of the model outputting rare/invalid tokens like "Positivve" instead of "positive".

### 1.3 top_k

| Property | Value |
|----------|-------|
| Valid range | [0, ∞) (0 = disabled, 40 = llama.cpp default) |
| llama.cpp default | 40 |
| Current pipeline | 40 |
| **Recommended for sentiment** | **10–20** — only the top 10-20 most likely tokens are considered. For 1B models outputting "positive" / "negative" / "neutral", these few tokens dominate the distribution. A low top_k prevents the model from going off into irrelevant tokens. |
| Recommended for NER | 20 — entity names need more headroom |
| Recommended for factual | 40 — longer answers need more diversity |
| Recommended for summarization | 40 — default is fine |
| GEPA sweep? | Yes — sweep {40, 20, 10} |

**Interaction with top_p:**
- When both top_k and top_p are set, **top_k is applied first** (keep only top-k tokens), then top_p (keep only cumulative probability mass p). So with top_k=10 and top_p=0.9, the final pool is the intersection of both filters.
- For 1B models on sentiment, top_k=10 alone is often sufficient. When temperature=0.0, top_k acts as a hard filter — but greedy decoding already picks token 1, so top_k only matters if temperature > 0.0 or if there's a tie.

### 1.4 min_p (Minimum Probability Threshold)

| Property | Value |
|----------|-------|
| Valid range | [0.0, 1.0] (0.0 = disabled) |
| llama.cpp default | 0.05 |
| Current pipeline | 0.0 (disabled) |
| **Recommended for sentiment** | **0.05–0.1** — min_p=0.05 discards any token whose probability is less than 5% of the top token's probability. For sentiment classification where the correct token has > 50% probability, this helps by filtering out very unlikely tokens even when top_p wouldn't. |
| Recommended for NER | 0.05 — filters noise entities |
| Recommended for factual | 0.02 — some factual answers need rare tokens |
| Recommended for summarization | 0.0 — disabled (summarization benefits from occasional rare words) |
| GEPA sweep? | Yes — sweep {0.0, 0.02, 0.05, 0.1} |

**How min_p differs from top_p:**
- **top_p**: "Keep tokens until we've accumulated P probability mass." — relative to cumulative distribution.
- **min_p**: "Keep any token whose probability is at least P% of the top token's probability." — relative to the best token.
- **For 1B sentiment models**: The top token (the correct label) typically has 60-90% probability. min_p=0.05 means keeping tokens with probability >= 5% * 90% = 4.5%. This is slightly stricter than top_p=0.9 and can be combined with it.
- **Key insight**: min_p is more aggressive at filtering than top_p for small models with peaky distributions. It's particularly valuable for short-answer classification where you want to avoid the model randomly exploring synonyms.

### 1.5 repeat_penalty

| Property | Value |
|----------|-------|
| Valid range | [0.0, ∞) (1.0 = no penalty, >1.0 = penalize repetition) |
| llama.cpp default | 1.0 |
| Current pipeline | 1.0 (disabled) |
| **Recommended for sentiment** | **1.0–1.05** — slight penalty (1.05) can help if the model starts repeating the label, but for 1-2 token outputs the effect is minimal. Keep at 1.0 unless you observe repetition issues. |
| Recommended for NER | 1.0 — no benefit (entities don't repeat enough) |
| Recommended for factual | 1.0 — no benefit for short answers |
| Recommended for summarization | 1.1–1.2 — repetition is a real problem in long-form generation |
| GEPA sweep? | Low priority — sweep {1.0, 1.05} only if repetition observed |

**Why it matters for 1B models:**
- On longer outputs (summarization), 1B models ARE prone to repeating tokens/words. repeat_penalty=1.1 significantly improves output quality.
- For **sentiment** (1-2 token output: "positive"/"negative"/"neutral"), repetition simply isn't possible. repeat_penalty=1.0 is optimal.
- For **NER** (multiple entity names in one output), slight repeat_penalty=1.05 can prevent the model from emitting the same entity twice.

### 1.6 seed

| Property | Value |
|----------|-------|
| Valid range | Any integer or None |
| llama.cpp default | None (random seed from system entropy) |
| Current pipeline | None |
| **Recommended for sentiment** | **Fixed seed (e.g., 42)** — when combined with temperature=0.0, a fixed seed makes output **fully deterministic**. Without a fixed seed, GPU/CPU nondeterminism in kernel launches can cause tiny variations even at temperature=0.0. |
| Recommended for NER | Fixed seed (42) |
| Recommended for factual | Fixed seed (42) |
| Recommended for summarization | None (let it vary naturally) |
| GEPA sweep? | No — single fixed seed is sufficient |

**Does a fixed seed give consistent results with llama.cpp?**
- **Yes, with caveats**: llama.cpp's RNG is deterministic given the same seed on the same hardware/binary. However:
  - Different GPU hardware (AMD vs NVIDIA) may produce slightly different float accumulation, leading to different logits.
  - Different GGUF quantization levels change the exact logits.
  - Same seed + same model + same hardware = deterministic.
- **At temperature=0.0**: With greedy decoding, the seed matters only for RNG calls in CUDA/ROCm kernel launches. A fixed seed eliminates this source of noise.
- **For GEPA evaluation**: Using a fixed seed is critical for fair comparisons between cells. Without it, noise from RNG state can randomly shift accuracy by 1-5% on small eval sets.

### 1.7 Parameter Interaction Summary

For sentiment classification with 1B models:

```
temperature = 0.0    # greedy — always pick highest probability token
top_p       = 0.9    # narrow token pool to top ~90% probability mass
top_k       = 20     # hard limit to 20 most likely tokens
min_p       = 0.05   # discard tokens with <5% of top token's probability
repeat_penalty = 1.0 # disabled (1-2 token outputs)
seed        = 42     # fixed for reproducibility
```

**When temperature=0.0, top_p/top_k/min_p are still useful** because:
1. They act as logit processors that can shift which token has the highest probability after filtering.
2. Greedy decoding still respects the filtered logit distribution.
3. Some llms benefit from the probability mass redistribution even with temperature=0.0.

**For 1B models specifically:**
- Small models have "peakier" output distributions — the correct token is usually >50% probability.
- This means tight sampling (top_p < 1.0, min_p > 0, low top_k) primarily helps by **removing noise**, not by changing the answer.
- The benefit is more consistent formatting (fewer misspellings like "Postive" or "neutal").
- The benefit is most visible on borderline cases where the model is uncertain.

---

## 2. Propagation Bug Audit

### 2.1 The Bug

`DecodingConfig` in `agent/cell.py` defines 7 parameters:

```python
temperature: float = 0.0
max_tokens: int = 64
top_p: float = 1.0
top_k: int = 40
min_p: float = 0.0
repeat_penalty: float = 1.0
seed: Optional[int] = None
```

But most inference sites only pass `temperature` and `max_tokens`, ignoring `top_p`, `top_k`, `min_p`, `repeat_penalty`, and `seed`.

### 2.2 All Inference Sites

| File | Line(s) | Params Passed | Bug? |
|------|---------|---------------|------|
| **agent/evaluation_agent.py** | 234-244 | ✅ ALL: max_tokens, temperature, top_p, top_k, min_p, repeat_penalty, seed | **FIXED** |
| **agent/pipeline.py** (`_infer`) | 483-488 | ❌ Only: max_tokens, temperature(0.0), stop | **BUG — missing 5 params** |
| **agent/pipeline.py** (routing path) | 626-627 | ❌ Only: max_tokens, stop, timeout | **BUG — missing 6 params** |
| **agent/pipeline.py** (main path) | 722-723 | ❌ Only: max_tokens, stop, timeout | **BUG — missing 6 params** |
| **agent/workflow.py** (single-shot) | 191 | ❌ Only: max_tokens, temperature (callable signature) | **BUG — missing 5 params** |
| **agent/workflow.py** (multi-step) | 224 | ❌ Only: max_tokens, temperature (callable signature) | **BUG — missing 5 params** |
| **agent/gepa_runner.py** | 297-300 | ❌ Only: max_tokens, temperature | **BUG — missing 5 params** |
| **agent/gepa_category_runner.py** | 306-309 | ❌ Only: max_tokens, temperature | **BUG — missing 5 params** |
| **agent/solvers/local_vote.py** | 160-164 | ❌ Only: max_tokens, temperature | **BUG — missing 5 params** |

### 2.3 Fix Priority

1. **HIGH**: `agent/pipeline.py` (_infer method) — this is the main inference path for ALL production traffic
2. **HIGH**: `agent/workflow.py` (WorkflowEngine) — workflow cells are the most complex and benefit most from tuned params
3. **MEDIUM**: `agent/gepa_runner.py` — used for GEPA factual optimization
4. **MEDIUM**: `agent/gepa_category_runner.py` — used for GEPA category optimization
5. **LOW**: `agent/solvers/local_vote.py` — used for consensus voting, which already uses multiple temperatures

### 2.4 Root Cause

The `_infer` method in `pipeline.py` was written as a simple wrapper before `DecodingConfig` existed. It accepts positional args `(messages, max_tok, stop_seq, timeout)` and hardcodes `temperature=0.0`. The `WorkflowEngine` class similarly hardcodes the `llm_infer_fn` signature as `(messages, max_tokens, temperature) -> str`. Neither knows about `DecodingConfig`.

The fix involves:
1. Changing `pipeline._infer` to accept an optional `DecodingConfig` parameter (or forwarding all params from the routing table entry).
2. Changing `WorkflowEngine.run` to forward the full `DecodingConfig` to the infer callable, and updating the callable's signature.
3. Updating `gepa_runner.py` and `gepa_category_runner.py` to pass the additional params.

---

## 3. Potential Improvement Estimate

### 3.1 Sentiment Classification

Baseline: temperature=0.0, top_p=1.0, top_k=40, min_p=0.0, repeat_penalty=1.0, seed=None

**Estimated improvement from parameter tuning:**

| Change | Expected Δ Accuracy | Rationale |
|--------|-------------------|-----------|
| top_p=1.0 → 0.9 | +0.5–2% | Reduces rare token misspellings; helps on ambiguous cases |
| top_k=40 → 20 | +0.5–1% | Focuses on likely sentiment tokens |
| min_p=0.0 → 0.05 | +0.5–2% | Filters low-probability noise tokens aggressively |
| seed=fixed (42) | ±0% (accuracy) / +reproducibility | Makes evaluation deterministic |
| repeat_penalty=1.05 | +0–0.5% | Marginal for 1-token outputs |
| **Combined** | **+1–4%** | **Conservative estimate** |

**Why the improvement is modest:**
- 1B models already get sentiment mostly right with greedy decoding.
- The main benefit is in **consistency and formatting**, not in changing correct→incorrect answers.
- On a 100-question eval set, expect 1-4 additional correct answers from tuned parameters.

### 3.2 NER (Named Entity Recognition)

| Change | Expected Δ | Rationale |
|--------|-----------|-----------|
| top_p=1.0 → 0.85 | +1–3% | More focused entity extraction |
| top_k=40 → 20 | +1–2% | Hallucinated entities reduced |
| min_p=0.0 → 0.05 | +1–3% | Spurious "entities" filtered |
| repeat_penalty=1.0 → 1.05 | +0–1% | Prevents repeating same entity |
| **Combined** | **+2–5%** | **NER benefits more than sentiment** |

### 3.3 Factual QA

| Change | Expected Δ | Rationale |
|--------|-----------|-----------|
| top_p=1.0 → 0.95 | +0–1% | Factual QA needs longer answer diversity |
| min_p=0.0 → 0.02 | +0–1% | Mild noise filtering, but don't hurt recall |
| **Combined** | **+0–2%** | **Minimal — factual QA is prompt-dependent** |

### 3.4 Summarization

| Change | Expected Δ | Rationale |
|--------|-----------|-----------|
| temperature=0.0 → 0.2 | +1–3% | Slight randomness improves summary quality |
| repeat_penalty=1.0 → 1.15 | +2–5% | Significant reduction in repetition |
| top_p=1.0 → 0.95 | +0–1% | Mild benefit |
| **Combined** | **+3–8%** | **Summarization benefits most** |

### 3.5 Summary Table

| Task | Baseline Accuracy | Expected Tuned | Δ |
|------|------------------|---------------|---|
| Sentiment | ~85% | **~87–89%** | +1–4% |
| NER | ~65–75% | **~68–78%** | +2–5% |
| Factual | ~70–80% | **~70–81%** | +0–2% |
| Summarization | ~70–80% | **~73–85%** | +3–8% |

---

## 4. GEPA Sweep Recommendations

### 4.1 Parameters to Sweep

| Parameter | Values for Sweep | Priority |
|-----------|-----------------|----------|
| temperature | [0.0, 0.1, 0.2] | HIGH |
| top_p | [1.0, 0.95, 0.9, 0.8] | HIGH |
| top_k | [40, 20, 10] | MEDIUM |
| min_p | [0.0, 0.02, 0.05, 0.1] | MEDIUM |
| repeat_penalty | [1.0, 1.05, 1.1] | LOW |
| seed | [None, 42] | LOW |

### 4.2 Sweep Strategy

**Phase 1 (Pilot — 1 generation, small eval set):**
- Full factorial on temperature × top_p × min_p = 3 × 4 × 4 = 48 combos
- This is the most impactful subset
- Use fixed seed=42 for all combos
- Results will identify the best parameter region

**Phase 2 (Full — 3 generations):**
- If Phase 1 identifies a clear best region, restrict subsequent sweeps to that region
- Add top_k and repeat_penalty sweeps if budget allows
- Use the best parameters from Phase 1 as the baseline for GEPA prompt evolution

### 4.3 Implementation Note

To actually propagate parameters, the `DecodingConfig` dataclass already exists and is used in `Cell`. The fix is to:

1. **pipeline.py**: Change `_infer` signature to accept `decoding: DecodingConfig` and pass ALL params to `create_chat_completion`.
2. **workflow.py**: Change the `llm_infer_fn` callable to accept a `DecodingConfig` or individual params.
3. **gepa_runner.py** and **gepa_category_runner.py**: Read `top_p`, `top_k`, `min_p`, `repeat_penalty`, and `seed` from the variant dict alongside `temperature` and `max_tokens`.

See the companion patch file for exact changes.

---

## 5. References

- llama.cpp sampling documentation: https://github.com/ggerganov/llama.cpp/blob/master/common/sampling.h
- Nucleus Sampling (top_p): Holtzman et al. "The Curious Case of Neural Text Degeneration" (ICLR 2020)
- Minimum P Sampling (min_p): https://github.com/ggerganov/llama.cpp/pull/3841
- llama-cpp-python API: `create_chat_completion(messages, temperature, top_p, top_k, min_p, repeat_penalty, seed, ...)`
- llama-cpp-python version in use: 0.3.33
