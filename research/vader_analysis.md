# VADER Analysis Report

Date: 2026-07-13 14:52

## 1. Variant Comparison (1142 training questions)

### v1 (basic VADER) — USING DEFAULTS
- Accuracy: 70.75% (808/1142)
  - easy: 73.54% (328/446)
  - hard: 67.42% (358/531)
  - medium: 73.94% (122/165)

### v2 (advanced) — CURRENT DEFAULT (_classify_sentiment = v2)
- Accuracy: 62.52% (714/1142) ← WORSE than v1
  - easy: 61.88% (276/446)
  - hard: 60.83% (323/531)
  - medium: 69.70% (115/165)

**Why v2 is worse**: The hedging detection (Phase 1 in v2) catches 17 items but incorrectly overrides 6 where v1 was correct. The `_RE_HEDGING` regex is too aggressive, matching phrases like "nothing outstanding about" and "good enough" which VADER correctly identified as positive/negative. Additionally, the negation-aware scoring and contrast splitting can wash out valid sentiment signals.

### domain_fallback only
- Accuracy: 6.92% (79/1142) — Only for compound=0 cases, expected low standalone

## 2. V1 Threshold Tuning

Default thresholds (POS=0.05, NEG=0.0) are already optimal.

| POS | NEG | Acc |
|-----|-----|-----|
| -0.10 | -0.20 | 67.43% |
| +0.00 | -0.20 | 66.02% |
| +0.00 | -0.10 | 67.43% |
| +0.05 | -0.20 | 57.27% |
| +0.05 | -0.10 | 58.67% |
| **+0.05** | **+0.00** | **70.75% ← BEST** |
| +0.10 | -0.20 | 54.55% |
| +0.10 | -0.10 | 55.95% |
| +0.10 | +0.00 | 68.04% |
| +0.10 | +0.05 | 68.30% |
| +0.20 | -0.20 | 52.54% |
| +0.20 | -0.10 | 53.94% |
| +0.20 | +0.00 | 66.02% |
| +0.20 | +0.05 | 66.29% |

## 3. V2 Threshold Tuning (with v2's logic, but custom thresholds)

Best v2: POS=0.1, NEG=0.05, acc=68.56%
Still worse than v1 default (70.75%).

## 4. Cross-Comparison (v1 vs v2 on train)

- Both right: 693
- v1 only right: 115
- v2 only right: 21
- Both wrong: 313

## 5. Compound Range Accuracy (v1)

| Range | Accuracy |
|-------|----------|
| compound < -0.3 | 81.70% |
| -0.3 <= compound < -0.05 | 70.00% |
| -0.05 <= compound <= 0.05 | 53.01% |
| 0.05 < compound <= 0.5 | 64.36% |
| compound > 0.5 | 83.23% |

## 6. MIXED Detection

- Only 10 mixed questions in training set
- MIXED detection at any bar (0.3, 0.4, 0.5) doesn't help
- Overall accuracy unchanged at 68.56% (v2 with best thresholds)
- MIXED disabled is the correct setting

## 7. Validation Set Performance

| Variant | Val Accuracy | Train Accuracy |
|---------|-------------|----------------|
| v1 (current) | 66.00% | 70.75% |
| v2 (advanced) | 66.00% | 62.52% |
| v1 best tuned | 66.00% | 70.75% |

## 8. Recommendations

1. **SOLVER**: Keep v1 (`_classify_sentiment_vader`). v2's hedging is too aggressive.
   - **NOTE**: The alias `_classify_sentiment = _classify_sentiment_v2` on line 1868 should be reverted to v1.
2. **THRESHOLDS**: Already optimal. Keep `_VADER_POS_THRESH = 0.05`, `_VADER_NEG_THRESH = 0.0`.
3. **MIXED**: Keep disabled (`_VADER_MIXED_ENABLED = False`).
4. **HYBRID**: Keep v1 solver. Current routing thresholds are appropriate.
5. **COVERAGE**: VADER handles ~70% correctly via compound threshold, 1.5% via patterns. The remaining ~28% need LLM help.
6. **NEUTRAL_MISS** (181 cases, 15.8%): VADER says neutral but answer isn't — these are the main targets for LLM improvement.
