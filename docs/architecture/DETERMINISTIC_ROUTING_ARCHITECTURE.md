# Deterministic Routing Architecture Analysis

## 1. Pipeline Overview

The project has **three distinct classifier layers** that feed into a routing decision:

```
┌─────────────────────────────────────────────────────────────┐
│                    Layer 0: Input Prompt                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│  Layer 1: Category Detection                                 │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │ enhanced_classifier│  │ agent/router.py  │                │
│  │ (8-way, pure regex)│  │ (19-cat + bit-   │               │
│  │ standalone)        │  │  morphic fallback)│               │
│  └──────────────────┘  └────────┬─────────┘                 │
│         OR                       │                           │
│  ┌──────────────────┐           │                           │
│  │ collected_       │           │                           │
│  │ classifiers.py   │           │                           │
│  │ (8 classifiers   │           │                           │
│  │  from repos)     │           │                           │
│  └──────────────────┘           │                           │
└─────────────────────────────────┼───────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────┐
│  Layer 2: Complexity Scoring                                 │
│  ┌──────────────────────┐                                    │
│  │ agent/bitmorphic.py   │ 7-signal weighted scorer          │
│  │   → score (0-1)      │ Used as fallback in router.py     │
│  │   → difficulty bucket │ (SIMPLE/MODERATE/COMPLEX)        │
│  └──────────────────────┘                                    │
└─────────────────────────────────┬───────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────┐
│  Layer 3: Solver Selection                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Try 6 Deterministic Solvers (agent/solvers/         │   │
│  │  deterministic.py):                                  │   │
│  │  • solve_arithmetic()    – simple arithmetic, %      │   │
│  │  • solve_logic()          – syllogisms + puzzles     │   │
│  │  • solve_sentiment()      – keyword + negation       │   │
│  │  • solve_ner()            – diseases, caps, dates    │   │
│  │  • solve_factual_qa()     – trivia + overlap         │   │
│  │  • solve_code_debugging() – 10 bug patterns          │   │
│  │  (Each returns answer string or None)                │   │
│  └──────────────────────────────┬───────────────────────┘   │
│                                 │                            │
│  ┌──────────────────────────────▼───────────────────────┐   │
│  │  If solver returned answer → use it (zero tokens)    │   │
│  │  Else → Fireworks kimi-k2p6 (or local Qwen3.5-4B)   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 2. The Two-Stage Flow

The user-described "category first, then complexity" flow:

**Stage 1 — Category Detection:**
- Run the 19-category deterministic router (agent/router.py) OR the 8-way classifier (enhanced_classifier.py)
- Both use priority-ordered first-match-wins regex rules
- Output: category string

**Stage 2 — Complexity Scoring/Route Decision:**
- Run Bitmorphic complexity scorer (agent/bitmorphic.py) for a 0-1 score
- Or use category as a direct route hint
- Decision: local model vs Fireworks API based on complexity score threshold

**What actually won (v6.1 at 84.2%):**
```
for each task:
  1. Keyword task-type detection (7 patterns, ~50 lines)
  2. Try 6 deterministic solvers
  3. If solved → use answer (free)
  4. Else → Fireworks kimi-k2p6 with per-category prompt
```
No ML classifier, no local model, no Bitmorphic complexity scorer, no ensemble voting.

## 3. Classifier Inventory

### Primary Classifiers

| File | Categories | Mechanism | Accuracy | Latency |
|------|-----------|-----------|----------|---------|
| `agent/router.py` | 19 categories + 4-way | Priority rules + Bitmorphic fallback | 71.1% (19-cat) / 61.0% (4-way) | ~0.16ms |
| `enhanced_classifier.py` | 8 categories | First-match regex | ~71% (same base) | ~0.01ms |
| `classifiers/classifier_ml.py` | 8 categories | TF-IDF + LR | 53.7% | ~1.4ms |
| `classifiers/classifier_minilm.py` | 8 categories | MiniLM + LR | 59.6% | ~23ms |
| `agent/classifier_ensemble.py` | 8 + 19 (mapped) | Majority vote (3 classifiers) | 65.1% | ~10ms |
| `agent/hybrid_classifier.py` | 8 + 19 (mapped) | Ensemble-first, deterministic fallback | 68.4% | ~10ms |
| `agent/bitmorphic.py` | Complexity score (0-1) | 7-signal weighted | N/A (complexity only) | ~0.01ms |

### Per-Category Accuracy (Deterministic Router — from showdown report)

| Category | Accuracy | Count | Notes |
|----------|----------|-------|-------|
| code_debug | **100.0%** | 26/26 | Perfect — strong patterns |
| code_gen | **100.0%** | 26/26 | Perfect — explicit code keywords |
| math | **85.3%** | 29/34 | Good but steals some factual |
| ner | **65.2%** | 15/23 | Moderate — misses unfamiliar entities |
| factual | **58.8%** | 20/34 | Loses to math (steals 11 factuals) |
| sentiment | **61.5%** | 16/26 | Misses implicit/subtle sentiment |
| logic | **52.0%** | 13/25 | Many logic puzzles look like factual QA |
| summarization | **41.7%** | 10/24 | Poor — news text often misclassified as code |

### 4-Way Accuracy (from showdown report)

| 4-Way Category | Accuracy | Notes |
|----------------|----------|-------|
| code | **100%** | Perfect |
| reasoning | **89.2%** | Good |
| knowledge | **43.7%** | Poor — the dumping ground for everything |
| text | **45.2%** | Poor — sentiment/ner/summarization mix |

## 4. Category Mapping Chaos

Categories exist under **3 different naming schemes** in 5+ files with conversion tables everywhere:

### The 3 Systems:
1. **8-Way** (enhanced_classifier.py): `code_debug, code_gen, factual, logic, math, ner, sentiment, summarization`
2. **19-Way** (agent/router.py): `code_debugging, code_generation, math_arithmetic, math_reasoning, summarization, sentiment, named_entity_recognition, classification, logical_reasoning, translation, rewriting, extraction, data_formatting, factual_knowledge, creative_generation, analysis, instruction_following, question_answering, other_complex`
3. **ML 8-Way** (classifier_ml.py): `code_debugging, code_generation, factual_knowledge, logical_reasoning, math_reasoning, named_entity_recognition, sentiment_classification, text_summarisation`

### Conversion tables (each a mapping nightmare):
- `CATEGORY_MAP` (classifier_showdown.py)
- `ML_LABEL_MAP` (classifier_showdown.py)
- `EVAL_LABEL_MAP` (classifier_showdown.py)
- `CATEGORY_MAP_4WAY` (classifier_showdown.py)
- `ML_TO_PIPELINE` (classifier_ensemble.py)
- `DETERMINISTIC_TO_ML` (classifier_ensemble.py)
- `ROUTER_4WAY_MAP` (agent/router.py)
- `_to_4way` (collected_classifiers.py)

These tables disagree in subtle ways (e.g., `ner` maps to `knowledge` in 4-way in agent/router.py but to `text` in collected_classifiers.py).

## 5. Interdependencies Between Classifiers

```
agent/router.py ──────────────────────────────────────────────┐
  └── depends on agent/bitmorphic.py (fallback score)         │
  └── provides classify() → to agent/classifier_ensemble.py   │
  └── provides classify_4way() → standalone                   │
                                                              │
agent/classifier_ensemble.py ──────────────────────────────── │
  └── depends on classifiers/classifier_ml.py (TF-IDF + LR)   │
  └── depends on classifiers/classifier_minilm.py (MiniLM+LR) │
  └── depends on agent/router.py (deterministic)              │
  └── provides classify() → to agent/hybrid_classifier.py     │
  └── provides classify_4way() → to hybrid_classifier.py     │
                                                              │
agent/hybrid_classifier.py ────────────────────────────────── │
  └── depends on agent/classifier_ensemble.py                 │
  └── depends on agent/router.py                              │
                                                              │
agent/solvers/deterministic.py ────────────────────────────── │
  └── depends on agent/solvers/tools.py (calculator)          │
  └── standalone — called by main loop, not by classifiers    │
                                                              │
enhanced_classifier.py ────────────────────────────────────── │
  └── standalone — zero dependencies, pure stdlib regex       │
                                                              │
collected_classifiers.py ──────────────────────────────────── │
  └── standalone — 8 classifiers from winning repos           │
  └── not integrated into any routing pipeline                │
                                                              │
classifier_showdown.py ────────────────────────────────────── │
  └── wraps ALL classifiers + 4-way variants for comparison   │
  └── loads eval data from eval_*.json files                  │
                                                              │
build_ngram_profiles.py ───────────────────────────────────── │
  └── builds profiles from train_4way_split.json              │
  └── NO corresponding classifier uses these profiles yet     │
```

## 6. Gaps and Opportunities

### Gaps

1. **No unified category registry** — 3 naming schemes, 8+ mapping tables scattered across files, subtle disagreements between them.

2. **Summarization detection is poor** (41.7%) — news text with dates/ages looks like code or gets missed entirely.

3. **Factual→math bleed** — 11 factual prompts classified as math (stealing factual), likely because the prompts contain numbers.

4. **No summarization deterministic solver** — The solver suite has no `solve_summarization()`, yet summarization is 12% of the eval set.

5. **Bitmorphic misuse as category classifier** — When the priority rules miss a prompt, router.py falls through to Bitmorphic (a complexity scorer), mapping SIMPLE→factual, MODERATE→analysis, COMPLEX→other_complex. This is a lossy mapping of a secondary metric.

6. **No confidence scores from the deterministic router** — The base classifier returns a hard category string. The ensemble and hybrid layers try to derive confidence from vote counts, but the deterministic router itself has no uncertainty quantification.

7. **N-gram profiles exist but aren't used** — `build_ngram_profiles.py` creates log-probability profiles per 4-way category, but no classifier consumes them.

8. **Collected classifiers are unused** — The 8 winning-team classifiers in `collected_classifiers.py` are only used for the `__main__` demo; they're not part of any ensemble or consensus pipeline.

9. **The winning submission (v6.1) stripped all this** — The 84.2% pass was achieved with just keyword detection + 6 deterministic solvers + Fireworks. The current `master` branch has ~2,500+ lines of pipeline code that was never proven to improve accuracy.

### Opportunities

1. **Consolidate to unified 8-way or 4-way** — The 19-category system collapses to 4-way anyway for routing decisions. A single 8-way registry with clean mappings would eliminate the conversion-table sprawl.

2. **Add confidence output to deterministic router** — Count total pattern hits per category instead of first-match-wins; output a probability distribution.

3. **Add summarization solver** — Extractive summarization (first-sentence, keyword-density) would catch the ~58% of summarization prompts that today pass through to Fireworks.

4. **Fix factual→math gate** — The "Context:/Question:" check in enhanced_classifier.py prevents math from stealing factual, but the inverse (factual prompts with numbers going to math) still happens at 11/34 = 32% error rate.

5. **Build an n-gram Bayesian classifier** — The profiles from `build_ngram_profiles.py` are ready; a simple Naive Bayes classifier using these log-probs would be a zero-dependency addition.

6. **Tie collected classifiers into an ensemble** — 8 competing classifiers could vote on category; consensus routing via majority would leverage all 8 winning-team strategies.

7. **Route based on confidence bands** — Instead of hard "try deterministic → else Fireworks", use confidence tiers: high-confidence deterministic matches → answer immediately; medium-confidence → pass solver hints to Fireworks; low-confidence → full Fireworks with generic prompt.

8. **Document the winning v6.1 configuration** — The `CLASSIFIER_ANALYSIS.md` clearly shows the v6.1 baseline was simpler and better. The code should reflect that the complex ensemble/local-model path is experimental, not the recommended pipeline.
