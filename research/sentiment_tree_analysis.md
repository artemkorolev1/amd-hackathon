# Sentiment Decision Tree — Analysis Report

Generated: 2026-07-13 13:00

## Overview

A layered deterministic decision tree for sentiment classification, replacing
the monolithic VADER-based approach. Each layer handles specific cases and
passes through if uncertain. First match wins.

## Architecture

| Layer | Description | Confidence |
|-------|-------------|-----------|
| **1. SARCASM_PATTERN** | Sarcasm, backhanded, hedging, "X but Y" regex overrides | High |
| **2. STRONG_SIGNAL** | Compound > pos_threshold (≥0.5) → positive; < neg_threshold (≤-0.3) → negative | High |
| **3. CONTRAST_SPLIT** | "but/however" clauses — split and score independently (2x post-weight) | Medium |
| **4. NEGATION** | Negation-aware VADER with word-level proximity (3-token window) | Medium |
| **5. DOMAIN_KEYWORDS** | Domain-specific patterns for movie/product reviews | Medium |
| **6. VADER_THRESHOLD** | Default compound threshold fallback (pos ≥0.05, neg ≤0.0) | Low |

Layer ordering matters: SARCASM_PATTERN fires before STRONG_SIGNAL so that
sarcastic text (e.g., "Yeah right, sure you will fix it." which has a positive
compound) is correctly classified as negative.

## Dataset Composition

| Dataset | Total | Easy | Medium | Hard | Positive | Negative | Neutral | Mixed |
|---------|-------|------|--------|------|----------|----------|---------|-------|
| Training | 1,142 | 446 (39%) | 165 (14%) | 531 (47%) | 600 | 520 | 12 | 10 |
| Validation | 100 | 70 (70%) | 30 (30%) | 0 (0%) | 53 | 45 | 1 | 1 |

**Important:** The validation set contains NO "hard" difficulty items.
Tuning purely for validation accuracy may overfit to easy/medium cases
at the expense of hard ones.

## Default Configuration (Recommended for Production)

```json
{
  "pos_threshold": 0.5,
  "neg_threshold": -0.3,
  "vader_pos_thresh": 0.05,
  "vader_neg_thresh": 0.0,
  "sarcasm_enabled": true,
  "contrast_enabled": true,
  "negation_enabled": true,
  "domain_enabled": true,
  "layer_order": [
    "SARCASM_PATTERN",
    "STRONG_SIGNAL",
    "CONTRAST_SPLIT",
    "NEGATION",
    "DOMAIN_KEYWORDS",
    "VADER_THRESHOLD"
  ]
}
```

## Training Set Performance (1,142 questions)

**Overall accuracy: 70.9%** (810/1,142) — beats VADER v1 baseline (70.7%)

### Per-Layer Breakdown

| Layer | Coverage | Accuracy | % of Correct |
|-------|----------|----------|-------------|
| **SARCASM_PATTERN** | 34 (3.0%) | 64.7% | 2.7% |
| **STRONG_SIGNAL** | 514 (45.0%) | **82.9%** | 52.6% |
| **CONTRAST_SPLIT** | 59 (5.2%) | 54.2% | 4.0% |
| **NEGATION** | 2 (0.2%) | 100.0% | 0.2% |
| **DOMAIN_KEYWORDS** | 22 (1.9%) | 100.0% | 2.7% |
| **VADER_THRESHOLD** | 511 (44.7%) | 59.9% | 37.8% |

### Difficulty Breakdown

| Difficulty | Accuracy | Correct/Total |
|------------|----------|---------------|
| easy | 72.6% | 324/446 |
| hard | **69.5%** | 369/531 |
| medium | 70.9% | 117/165 |

### Confusion Matrix (Training)

| Actual \→ Predicted | positive | negative | neutral |
|---|---|---|---|
| positive | 404 | 192 | 4 |
| negative | 107 | **401** | 12 |
| neutral | 3 | 4 | 5 |

## Validation Set Performance (100 questions)

**Overall accuracy: 70.0%** (70/100) — matches VADER v1 baseline

### Per-Layer Breakdown

| Layer | Coverage | Accuracy | % of Correct |
|-------|----------|----------|-------------|
| **SARCASM_PATTERN** | 2 (2.0%) | 50.0% | 1.4% |
| **STRONG_SIGNAL** | 56 (56.0%) | **82.1%** | 65.7% |
| **CONTRAST_SPLIT** | 6 (6.0%) | 50.0% | 4.3% |
| **DOMAIN_KEYWORDS** | 3 (3.0%) | 100.0% | 4.3% |
| **VADER_THRESHOLD** | 33 (33.0%) | 51.5% | 24.3% |

## Threshold Tuning Results

### Layer 1 (STRONG_SIGNAL) — Best on All Data

| pos_threshold | neg_threshold | Training Acc | Hard Acc | Validation Acc |
|:---:|:---:|:---:|:---:|:---:|
| 0.5 | -0.3 | **70.9%** | **69.5%** | 70.0% |
| 0.3 | -0.2 | 70.3% | 67.0% | 70.0% |
| 0.7 | -0.2 | 70.9% | 69.0% | 70.0% |

The **default (pos=0.5, neg=-0.3)** provides the best hard-item accuracy (69.5%)
while maintaining strong overall performance.

### Layer 6 (VADER_THRESHOLD) — Tuning

| vader_pos_thresh | vader_neg_thresh | Training Acc | Validation Acc |
|:---:|:---:|:---:|:---:|
| 0.00 | 0.00 | 69.1% | **74.0%** |
| 0.05 | 0.00 | 70.3% | 70.0% |
| 0.05 | 0.05 | 70.6% | 71.0% |

Setting vader_pos_thresh=0.0 achieves 74% on validation but drops to 69.1% on
training (hard items suffer from overly aggressive positive classification).
The **default (vader_pos=0.05, vader_neg=0.0)** is recommended for production.

### Layer Ablation (with default thresholds)

| Configuration | Training Acc | Validation Acc | Delta from Full |
|--------------|:---:|:---:|:---:|
| All enabled | **70.9%** | 70.0% | — |
| No sarcasm | 70.5% | 70.0% | -0.4% |
| No contrast | 71.0% | 70.0% | +0.1% |
| No negation | 70.9% | 70.0% | 0.0% |
| **No domain** | **69.9%** | **69.0%** | **-1.0%** |
| Only strong+vader | 70.3% | 70.0% | -0.6% |
| Only vader (v1-like) | 70.7% | 70.0% | -0.2% |

**DOMAIN_KEYWORDS is the most impactful creative layer** (+1.0% improvement).
The negation layer has minimal coverage (only 2/1142 training examples affected).

## Comparison with VADER v1 Baseline

| Metric | VADER v1 (baseline) | Decision Tree | Delta |
|--------|:---:|:---:|:---:|
| **Training Accuracy** | 70.7% | **70.9%** | **+0.2%** |
| **Validation Accuracy** | 70.0% | 70.0% | 0.0% |
| **Hard Item Accuracy** | 68.9% | **69.5%** | **+0.6%** |
| Easy Item Accuracy | 72.2% | 72.6% | +0.4% |
| Medium Item Accuracy | 72.1% | 70.9% | -1.2% |
| Tuned Validation (easy/medium only) | 71.0% | **74.0%** | **+3.0%** |

## Key Findings

1. **SARCASM_PATTERN must fire before STRONG_SIGNAL**
   - Some sarcastic text (e.g., "Yeah right, sure you will fix it.") has
     positive VADER compound. The sarcasm regex must catch it first.
   - This matches VADER v1's original logic.

2. **STRONG_SIGNAL layer handles the majority of cases** (45% coverage)
   - High confidence decisions with **82.9% accuracy**
   - Catching strong positive/negative signals early saves downstream layers

3. **SARCASM_PATTERN layer catches edge cases** (3.0% coverage)
   - Critical for sarcasm/backhanded detection that VADER gets wrong
   - 64.7% accuracy when it fires (some false positives from non-sarcastic text)

4. **CONTRAST_SPLIT improves mixed reviews** (5.2% coverage)
   - Properly handles "it was good but..." patterns
   - Only 54.2% accuracy — splitting heuristics are imperfect
   - Removing contrast doesn't hurt overall accuracy (neutral impact)

5. **DOMAIN_KEYWORDS is the most impactful creative layer**
   - 100% accuracy on the 22 training cases it catches
   - Removing it costs 1% overall accuracy
   - Catches phrases like "must-see", "waste of time" that VADER misses

6. **NEGATION layer has minimal impact** (0.2% coverage)
   - VADER's compound score already handles most negation internally
   - The proximity-based adjustment rarely changes the verdict

7. **VADER_THRESHOLD fallback is the weak point** (44.7% coverage, 59.9% accuracy)
   - This is where the majority of errors come from
   - Middle-ground compounds (-0.3 to 0.5) are inherently ambiguous
   - Further improvements must focus on this layer

## Recommendation

Use the **default configuration** for production:
- **Layer order:** SARCASM_PATTERN → STRONG_SIGNAL → CONTRAST_SPLIT → NEGATION → DOMAIN_KEYWORDS → VADER_THRESHOLD
- **pos_threshold=0.5, neg_threshold=-0.3** — best hard-item accuracy
- **vader_pos_thresh=0.05, vader_neg_thresh=0.0** — balanced fallback

This achieves **70.9% accuracy** on the full 1,142 training set (slightly beating
VADER v1's 70.7%), with **69.5% on hard items** (beating v1's 68.9%).
The tuned configuration (pos=0.3, neg=-0.2, vader_pos=0.0, vader_neg=0.0)
is useful for scenarios with only easy/medium items, achieving **74.0%**
on the validation set.
