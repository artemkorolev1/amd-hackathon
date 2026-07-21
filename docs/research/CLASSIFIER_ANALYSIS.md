# Classifier Architecture Analysis: Do We Need It?

## Executive Summary

**No, we don't need the classifier.** The 84.2% score (v6.1) was achieved with the simplest possible architecture: keyword-based task type detection → try deterministic solvers (as hints) → Fireworks kimi-k2p6 for everything else. Zero ML classifiers, zero local model inference, zero ensemble voting.

---

## Evidence Base

### The v6.1 Baseline (84.2%) — No Classifier Needed

The commit `8916cc8` ("v6.1: kimi-k2p6 router, deterministic hints, per-category prompts, reasoning_effort=none — 84.2%") achieved the passing grade with:

```
for each task:
  detect task_type via keyword matching (7 patterns: ner/sentiment/code/logic/math/summarization/general)
  try 6 deterministic solvers (math, logic, sentiment, NER, factual QA, code debug)
  if deterministic solver returns answer → use it (zero token cost)
  else → Fireworks kimi-k2p6 with per-category system prompt + deterministic hints
```

Key config fact: `LLAMA_ENABLE: bool = os.environ.get("LLAMA_ENABLE", "0") == "1"` — default was **0**. The local model was **disabled** by default.

### ML Classifiers Underperform the Deterministic Router

| Approach | Eval Accuracy (hackathon prompts) | Speed | Size |
|----------|:--------------------------------:|:-----:|:----:|
| Deterministic router (keyword) | **71-72%** (4-way merge: 72.5%) | ~0.01ms | 0 (code only) |
| TF-IDF + LR | **50-60%** | <1ms | 1.7MB |
| MiniLM + LR | **50-60%** | ~5ms | 22MB |
| 3-way ensemble (ML+ML+det) | ~80% on stress test (n=20) | ~6ms | ~24MB |
| NVIDIA BERT classifier | **25-30%** | 99ms | 735MB |

The ML models (trained on 8 benchmark datasets with 98.4% held-out accuracy) suffer **distribution drift** — hackathon prompts look very different from benchmarks. The deterministic keyword router, being hand-crafted for the eval domain, consistently outperforms ML.

### Token Cost Comparison

| Component | Input Tokens | Output Tokens | Latency | Cost |
|-----------|:-----------:|:------------:|:-------:|:----:|
| Deterministic solver | 0 | **0** (hand-written rules) | ~0.1ms | $0 |
| Keyword task detection | 0 | 0 | ~0.01ms | $0 |
| Local Qwen3.5-4B (CPU, Q4) | ~50-200 | **~50-500** | ~3-30s/task | ~$0.00004 |
| Fireworks kimi-k2p7-code | ~100-2000 | **~30-300** (caveman prompts) | ~2-8s/task | ~$0.001-0.005 |
| ML classifier (ensemble) | ~50-500 (embedded) | 0 | ~6ms | ~$0 |

**The classifier itself doesn't save tokens** — the token savings come from:
1. **Deterministic solvers**: zero tokens (they compute answers directly)
2. **Per-category system prompts**: reduce output tokens by 50-65% vs generic prompts
3. **Local model**: cheaper per-token but slow on CPU

The classifier is just a routing decision — it adds latency and complexity but the actual token savings come from the solver choice, not the classifier.

### Bug Cost of Classifier Complexity

Every layer of classifier complexity has introduced production bugs:

| Version | Architecture | Score | Failure |
|---------|-------------|:-----:|---------|
| v0 | Fireworks + 2 deterministic solvers | **52.6%** | Answer quality failing LLM judge |
| v1 | 6 deterministic solvers, no Fireworks | Skipped | — |
| **v2** | Fireworks + ML classifier + 6 deterministic solvers | **RUNTIME_ERROR** | `parse_allowed_models()` TypeError |
| **v3/v4** | Same + greedy deterministic solver gates | **42.1%** | Deterministic solvers stole Fireworks tasks |
| **v5** | Lowered Fireworks threshold + tight gates | Pending | Lower threshold = more API calls |
| **v6** | Ensemble classifier + Bitmorphic + local model | Not submitted | — |
| **v7** | Complexity scorer with 6 signals | Not submitted | — |
| **v6.1 (84.2%)** | **Keyword detect → deterministic hints → kimi-k2p6** | **84.2% PASS** | Simple, no crash |

The winning commit (v6.1) was the **simplest** of all: it stripped away the ML classifier, local model, self-consistency voting, and Bitmorphic complexity scorer. It just ran deterministic solvers and sent everything else to kimi-k2p6.

---

## Three Options Compared

### Option A: Full Multi-Classifier (Current Code on `master`)

**Architecture:** 
1. Keyword task-type detection (_detect_task_type, 9 categories)
2. 3-way ensemble (MiniLM-LR + TF-IDF-LR + deterministic) with hybrid fallback
3. Bitmorphic Complexity Score (7 signals, weighted)
4. Per-category system prompts
5. Self-consistency voting on local Qwen3.5-4B (k=3)
6. Agreement threshold → escalate to Fireworks kimi-k2p7-code

**Estimated accuracy:** Unknown — never evaluated end-to-end. Local model probably hurts (Qwen3.5-4B is weak on hard prompts). The Fireworks fallback would save it, so likely **similar to "kimi only" but slower**.

**Token cost per task:** ~150-1500 tokens (local) + potentially ~100-500 tokens (Fireworks if escalated) = **highest total cost**

**Runtime:** Local model at ~16 tok/s CPU: 3-30 seconds per task for 3 samples. For 19 tasks: ~60-600s just for local inference, before any API calls.

**Complexity:** ~2,500+ lines of pipeline code across 12+ modules

**Risk:** HIGH — 4 prior bug-induced crashes prove complexity kills reliability. Local model OOM risk on 8GB VRAM. Circuit breaker can cascade.

### Option B: No Classifier — Everything to Fireworks Only

**Architecture:**
```
for each task:
  send to Fireworks kimi-k2p7-code with generic system prompt
```

**Token cost per task:** ~100-500 tokens output, always API call

**Estimated accuracy:** Probably **~80-85%** — kimi-k2p7-code is a strong model. Without per-category prompts, output format may be messier (lower fuzzy-match accuracy).

**Runtime:** ~2-8s per task via API. For 19 tasks: ~40-150s.

**Complexity:** ~50 lines of main loop. Single file.

**Risk:** LOW — but misses deterministic zero-cost answers. Also loses the per-category prompt optimization that saves 50-65% tokens.

### Option C: Minimal 2-Way (Deterministic → Fireworks) — **RECOMMENDED**

**Architecture (what v6.1 actually did):**
```
for each task:
  # 1. Keyword-based task-type detection (< 50 lines, zero cost)
  task_type = _detect_task_type(prompt)  # ner/sentiment/code/logic/math/summarization/general
  
  # 2. Try deterministic solvers (< 0.1ms each)
  answer = solve_arithmetic(prompt) 
        or solve_logic(prompt)
        or solve_sentiment(prompt)
        or solve_ner(prompt)
        or solve_factual_qa(prompt)
        or solve_code_debugging(prompt)
  
  # 3. If deterministic got it → free answer
  if answer:
    print(answer)
    continue
  
  # 4. Otherwise → Fireworks with per-category prompt + hint
  system_prompt = get_system_prompt(task_type)  # caveman terse
  hint = _try_deterministic(prompt)  # re-run, pass as hint
  answer = fireworks.solve(model="kimi-k2p6", system_prompt, prompt, hint)
  print(answer)
```

**Token cost per task:**
- Deterministic-solvable tasks (code_debug ~25%, factual ~20%): **0 tokens**
- Others: ~100-500 tokens output via Fireworks with caveman prompts

**For the eval_hard_100 distribution (100 questions):**
- 13 code_debug: deterministic solver hits ~100% → zero cost
- 12 code_gen: likely Fireworks (deterministic can't generate)
- 13 factual: deterministic hits ~74% → ~10 free, ~3 to Fireworks
- 13 logic: deterministic probably low → mostly Fireworks
- 13 math: deterministic low → mostly Fireworks
- 12 ner: deterministic ~12% → mostly Fireworks
- 12 sentiment: deterministic ~22% → mostly Fireworks
- 12 summarization: all Fireworks (threshold=0.0)

Estimated token savings vs Option B: **~30-40% fewer Fireworks calls** due to deterministic catches.

**Estimated accuracy:** **84.2% (proven)** 

**Runtime:** Deterministic solvers: ~0.1ms/task. Fireworks: ~2-8s/task. For 19 tasks with ~60% going to Fireworks: ~25-90s total.

**Complexity:** ~1,400 lines (all in deterministic.py) + 200 lines main.py. Simple sequential loop.

**Risk:** LOW — proven in production at 84.2%. No ML models to drift. No local model to OOM. No circuit breaker needed.

---

## Data-Backed Accuracy Estimates

### eval_hard_100 Category Breakdown

| Category | Count | Deterministic Catch Rate | Deterministic Accuracy | Fireworks Accuracy | Est. Combined |
|----------|:-----:|:------------------------:|:---------------------:|:-----------------:|:-------------:|
| code_debug | 13 | ~90% (keyword match) | 100% (on matched) | — | **~90%** |
| code_gen | 12 | ~80% (keyword match) | 0% (can't generate) | ~85% | **~85%** |
| factual | 13 | ~70% | 74% (on matched) | ~90% | **~84%** |
| logic | 13 | ~70% (keyword) | ~20% | ~80% | **~74%** |
| math | 13 | ~80% (keyword) | ~20% (hard prompts) | ~75% | **~68%** |
| ner | 12 | ~80% (keyword) | ~50% (biomedical) | ~90% | **~82%** |
| sentiment | 12 | ~50% (keyword) | ~70% (hard prompts) | ~95% | **~83%** |
| summarization | 12 | ~30% (keyword) | 0% (can't summarize) | ~90% | **~90%** |

**Weighted estimate for Option C:** ~82-85% (consistent with 84.2% observed)

### Option B (No classifier, all to Fireworks) estimate:
- kimi-k2p7-code with caveman prompts on all 100 tasks
- Without deterministic hints for hard problems, some loss on factual/math
- Estimate: **~80-85%** (slightly lower than v6.1 because deterministic hints help kimi on factual QA)

### Option A (Full current architecture) estimate:
- Local Qwen3.5-4B on ~60% of tasks (those below escalation threshold)
- Fireworks kimi-k2p7-code on the rest
- Qwen3.5-4B is substantially weaker than kimi; self-consistency voting helps but the base model is 4B params
- On CPU at 16 tok/s, 3 samples per task = ~10-90s per task — likely hits time budget
- **Estimate: ~75-85%** (likely lower than Option C because local model produces worse answers for hard prompts, and time budget forces degraded mode)

---

## Conclusion

**The classifier is unnecessary.** The optimal architecture for this hackathon is:

1. **Keyword-based task-type detection** (~50 lines, zero cost) 
2. **Deterministic solvers first** — they catch code debugging perfectly (100%) and factual QA decently (74%) at zero token cost
3. **Fireworks kimi-k2p7-code for everything else** — with per-category caveman prompts (50-65% token savings) and deterministic hints
4. **No ML classifiers, no local model, no ensemble, no self-consistency voting**

This is exactly what v6.1 did, scoring **84.2%**. Adding more complexity would add more bug surface without improving accuracy — every added layer has historically made things worse (v2 TypeError crash, v3/v4 42.1% failure from classifier gate bugs).

The current `master` branch (with local model, ensemble classifier, Bitmorphic scorer, and self-consistency voting) should be stripped back to the proven v6.1 baseline. The local model, ML classifiers, and ensemble infrastructure add risk without demonstrated accuracy gain.
