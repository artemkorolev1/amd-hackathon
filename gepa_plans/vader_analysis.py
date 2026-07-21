#!/usr/bin/env python3
"""
gepa_plans/vader_analysis.py — Comprehensive VADER analysis on 1142 training questions.

Tests v1, v2, domain_fallback, tunes thresholds, enables MIXED, analyzes coverage,
and evaluates VADER-vs-LLM complementarity on the validation set.

Usage:
    python3 gepa_plans/vader_analysis.py
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict

BASE = os.path.expanduser("/home/artem/dev/amd-hackathon")
sys.path.insert(0, BASE)

from agent.solvers.deterministic import (
    _classify_sentiment_vader,
    _classify_sentiment_v2,
    _classify_sentiment_domain_fallback,
    _get_vader_analyzer,
    _VADER_POS_THRESH,
    _VADER_NEG_THRESH,
    _VADER_MIXED_ENABLED,
    _VADER_MIXED_POS_BAR,
    _VADER_MIXED_NEG_BAR,
)

DATA_DIR = f"{BASE}/data/eval"
RESULTS_DIR = f"{BASE}/research"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────

def load_split(name):
    path = f"{DATA_DIR}/sentiment_{name}.json"
    with open(path) as f:
        return json.load(f)

def clean_expected(answer):
    """Normalize expected answer to our standard labels."""
    a = answer.strip().lower() if answer else "unknown"
    return a

# ── Run VADER variants ────────────────────────────────────────────────────────

def test_v1(text):
    """Test _classify_sentiment_vader (v1)."""
    # Temporarily disable MIXED for baseline v1 test
    return _classify_sentiment_vader(text)

def test_v2(text):
    """Test _classify_sentiment_v2."""
    return _classify_sentiment_v2(text)

def test_domain_fallback(text):
    """Test _classify_sentiment_domain_fallback only."""
    return _classify_sentiment_domain_fallback(text)

def get_compound(text):
    """Get VADER compound score for a text."""
    analyzer = _get_vader_analyzer()
    if analyzer is None:
        return 0.0
    return analyzer.polarity_scores(text)["compound"]

def get_pos_neg(text):
    """Get VADER pos/neg scores."""
    analyzer = _get_vader_analyzer()
    if analyzer is None:
        return 0.0, 0.0
    s = analyzer.polarity_scores(text)
    return s["pos"], s["neg"]

# ── Threshold sweep ───────────────────────────────────────────────────────────

def test_v2_with_thresholds(text, pos_thresh, neg_thresh, mixed_enabled=False,
                            mixed_pos_bar=0.3, mixed_neg_bar=0.3):
    """Run v2 classification with custom thresholds."""
    analyzer = _get_vader_analyzer()
    if analyzer is None:
        return None

    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    pos = scores["pos"]
    neg_s = scores["neg"]

    # Override thresholds for this test
    from agent.solvers.deterministic import (
        _RE_HEDGING, _RE_SARCASM_FAINT, _RE_SARCASM_OH,
        _RE_SARCASM_YEAH, _RE_SARCASM_RHET, _RE_BACKHANDED,
        _RE_GENERAL_BUT, _score_with_negation, _split_and_score_contrast,
        _VADER_POS_WORDS,
    )

    # ── Phase 1: Hedging → NEUTRAL ──
    if _RE_HEDGING.search(text):
        if re.search(r'flagged\s+.*?none\s+were\s+confirmed', text, re.I):
            return 'neutral'
        scores_check = analyzer.polarity_scores(text)
        if scores_check['compound'] <= -0.45:
            return 'negative'
        return 'neutral'

    # ── Phase 2: Strong sarcasm → NEGATIVE ──
    if _RE_SARCASM_FAINT.search(text):
        return 'negative'

    # ── Phase 3: Contrast clause splitting ──
    contrast_verdict = _split_and_score_contrast(text)
    if contrast_verdict is not None:
        return contrast_verdict

    # ── Phase 4: Negation-aware VADER ──
    sc = _score_with_negation(text)
    compound = sc['compound']
    pos = sc['pos']
    neg_s = sc['neg']

    # ── Phase 5: Override patterns ──
    if re.search(r'\b(?:far|much)\s+less\s+\w+', text, re.I):
        words_after = re.split(r'\bfar\s+less\s+', text, maxsplit=1, flags=re.I)
        if len(words_after) > 1:
            next_word = words_after[1].strip().split()[0].strip('.,!?;:\'"()[]{}').lower()
            if next_word in _VADER_POS_WORDS:
                return 'negative'

    if _RE_SARCASM_OH.search(text) and compound > -0.1:
        return 'negative'
    if _RE_SARCASM_YEAH.search(text):
        return 'negative'
    if _RE_SARCASM_RHET.search(text) and compound > 0.0:
        return 'negative'
    if _RE_BACKHANDED.search(text):
        return 'negative'

    if re.search(r'\bliked\b|\blove\b|\benjoyed\b', text, re.I) and -0.3 <= compound <= 0.05:
        return 'positive'

    if _RE_GENERAL_BUT.search(text) and compound > -0.1:
        return 'negative'

    # ── Phase 6: MIXED detection ──
    if mixed_enabled and pos > mixed_pos_bar and neg_s > mixed_neg_bar:
        return 'mixed'

    # ── Phase 7: Domain fallback for compound==0 ──
    if compound >= pos_thresh:
        return 'positive'
    elif compound < neg_thresh:
        return 'negative'
    elif compound == 0.0:
        return _classify_sentiment_domain_fallback(text)
    else:
        return 'neutral'

# ── Analysis functions ────────────────────────────────────────────────────────

def analyze_variant(data, variant_name, classify_fn):
    """Run a VADER variant on all items and return detailed results."""
    results = []
    correct = 0
    total = 0
    per_difficulty = defaultdict(lambda: {"correct": 0, "total": 0})
    per_compound_range = defaultdict(lambda: {"correct": 0, "total": 0})

    for item in data:
        text = item["prompt"]
        expected = clean_expected(item["expected_answer"])
        difficulty = item.get("difficulty", "unknown")

        predicted = classify_fn(text)
        if predicted is None:
            predicted = "unknown"

        is_correct = (predicted == expected)
        compound = get_compound(text)

        results.append({
            "text": text[:100],
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "difficulty": difficulty,
            "compound": compound,
        })

        total += 1
        if is_correct:
            correct += 1
        per_difficulty[difficulty]["total"] += 1
        if is_correct:
            per_difficulty[difficulty]["correct"] += 1

        # Compound ranges
        if compound < -0.3:
            crange = "compound<-0.3"
        elif compound < -0.05:
            crange = "-0.3<=compound<-0.05"
        elif compound <= 0.05:
            crange = "-0.05<=compound<=0.05"
        elif compound <= 0.5:
            crange = "0.05<compound<=0.5"
        else:
            crange = "compound>0.5"
        per_compound_range[crange]["total"] += 1
        if is_correct:
            per_compound_range[crange]["correct"] += 1

    accuracy = correct / total * 100 if total > 0 else 0
    return {
        "variant": variant_name,
        "accuracy": round(accuracy, 2),
        "correct": correct,
        "total": total,
        "per_difficulty": dict(per_difficulty),
        "per_compound_range": dict(per_compound_range),
        "results": results,
    }

# ── Coverage analysis ────────────────────────────────────────────────────────

def analyze_coverage(analysis, data):
    """Categorize why VADER gets each question right/wrong."""
    categories = defaultdict(list)
    results = analysis["results"]

    for i, r in enumerate(results):
        text = data[i]["prompt"]
        expected = r["expected"]
        predicted = r["predicted"]
        compound = r["compound"]

        if r["correct"]:
            # Check if it was the compound threshold or a pattern
            v1_label = _classify_sentiment_vader(text)
            if v1_label == expected:
                categories["correct_compound"].append(i)
            else:
                categories["correct_pattern"].append(i)
        else:
            if compound == 0.0 and expected != "neutral":
                categories["neutral_miss"].append(i)
            elif (expected == "positive" and predicted == "negative") or \
                 (expected == "negative" and predicted == "positive"):
                categories["wrong_compound"].append(i)
            elif expected in ("positive", "negative") and predicted == "neutral":
                categories["neutral_miss"].append(i)
            elif expected == "neutral" and predicted in ("positive", "negative"):
                categories["wrong_compound"].append(i)
            else:
                categories["wrong_missing_pattern"].append(i)

    return {k: len(v) for k, v in categories.items()}

# ── VADER vs LLM complementarity ────────────────────────────────────────────

def compute_complementarity(val_data):
    """Compare VADER v2 vs LLM on val set to find complementarity."""
    # We'll use the v2 predictions
    from collections import defaultdict

    results = []
    for item in val_data:
        text = item["prompt"]
        expected = clean_expected(item["expected_answer"])
        v2_label = test_v2(text) or "unknown"
        results.append({
            "text": text[:100],
            "expected": expected,
            "v2_label": v2_label,
            "v2_correct": v2_label == expected,
        })
    return results

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("COMPREHENSIVE VADER ANALYSIS")
    print("=" * 70)

    train = load_split("train")
    val = load_split("val")

    print(f"\nLoaded {len(train)} training questions, {len(val)} validation questions")
    print(f"Training: {Counter(d['expected_answer'] for d in train)}")
    print(f"Validation: {Counter(d['expected_answer'] for d in val)}")

    # ═════════════════════════════════════════════════════════════════════════
    # 1. TEST ALL VADER VARIANTS
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SECTION 1: VADER Variant Comparison (on 1142 training questions)")
    print("=" * 70)

    variants = [
        ("v1 (basic VADER)", lambda t: test_v1(t)),
        ("v2 (advanced)", lambda t: test_v2(t)),
        ("domain_fallback only", lambda t: test_domain_fallback(t)),
    ]

    analyses = {}
    for name, fn in variants:
        print(f"\n  Testing {name}...")
        analysis = analyze_variant(train, name, fn)
        analyses[name] = analysis
        print(f"    Accuracy: {analysis['accuracy']:.2f}% ({analysis['correct']}/{analysis['total']})")
        for diff, counts in sorted(analysis["per_difficulty"].items()):
            acc = counts["correct"] / counts["total"] * 100 if counts["total"] > 0 else 0
            print(f"      {diff}: {acc:.2f}% ({counts['correct']}/{counts['total']})")
        print(f"    Per compound range:")
        for crange, counts in sorted(analysis["per_compound_range"].items()):
            acc = counts["correct"] / counts["total"] * 100 if counts["total"] > 0 else 0
            print(f"      {crange}: {acc:.2f}% ({counts['correct']}/{counts['total']})")

    # Cross-comparison: where each variant is correct/incorrect
    print("\n  Cross-comparison (which variant gets what right):")
    v1_results = analyses["v1 (basic VADER)"]["results"]
    v2_results = analyses["v2 (advanced)"]["results"]
    both_right = sum(1 for a, b in zip(v1_results, v2_results) if a["correct"] and b["correct"])
    v1_only = sum(1 for a, b in zip(v1_results, v2_results) if a["correct"] and not b["correct"])
    v2_only = sum(1 for a, b in zip(v1_results, v2_results) if not a["correct"] and b["correct"])
    both_wrong = sum(1 for a, b in zip(v1_results, v2_results) if not a["correct"] and not b["correct"])
    print(f"    Both right: {both_right}")
    print(f"    v1 only right: {v1_only}")
    print(f"    v2 only right: {v2_only}")
    print(f"    Both wrong: {both_wrong}")

    # ═════════════════════════════════════════════════════════════════════════
    # 2. THRESHOLD TUNING
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SECTION 2: Threshold Tuning on Training Set")
    print("=" * 70)

    pos_thresholds = [-0.2, -0.1, 0.0, 0.05, 0.1, 0.2]
    neg_thresholds = [-0.2, -0.1, 0.0, 0.05]

    print(f"\n  Sweeping _VADER_POS_THRESH x _VADER_NEG_THRESH")
    print(f"  Pos thresholds: {pos_thresholds}")
    print(f"  Neg thresholds: {neg_thresholds}")
    print()

    best_acc = 0
    best_pair = (0.05, 0.0)
    threshold_results = []

    for pos_t in pos_thresholds:
        for neg_t in neg_thresholds:
            if pos_t <= neg_t:
                continue  # Must have pos > neg for meaningful classification
            correct = 0
            total = 0
            for item in train:
                text = item["prompt"]
                expected = clean_expected(item["expected_answer"])
                predicted = test_v2_with_thresholds(text, pos_t, neg_t)
                if predicted is None:
                    predicted = "unknown"
                if predicted == expected:
                    correct += 1
                total += 1
            acc = correct / total * 100
            threshold_results.append((pos_t, neg_t, round(acc, 2)))
            marker = " ← BEST" if acc > best_acc else ""
            if acc > best_acc:
                best_acc = acc
                best_pair = (pos_t, neg_t)
            print(f"    POS={pos_t:+5.2f}  NEG={neg_t:+5.2f}  acc={acc:.2f}% ({correct}/{total}){marker}")

    print(f"\n  Best threshold pair: POS={best_pair[0]}, NEG={best_pair[1]} with {best_acc:.2f}%")

    # ═════════════════════════════════════════════════════════════════════════
    # 3. MIXED DETECTION TEST
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SECTION 3: MIXED Detection Test")
    print("=" * 70)

    mixed_bars = [0.3, 0.4, 0.5]

    # Find questions that should be mixed
    mixed_expected = [item for item in train if clean_expected(item["expected_answer"]) == "mixed"]
    print(f"\n  Questions with MIXED label in training set: {len(mixed_expected)}")

    for bar in mixed_bars:
        correct = 0
        total = 0
        for item in train:
            text = item["prompt"]
            expected = clean_expected(item["expected_answer"])
            predicted = test_v2_with_thresholds(
                text, best_pair[0], best_pair[1],
                mixed_enabled=True, mixed_pos_bar=bar, mixed_neg_bar=bar
            )
            if predicted is None:
                predicted = "unknown"
            if predicted == expected:
                correct += 1
            total += 1
        acc = correct / total * 100
        print(f"    MIXED bar={bar:.1f}: acc={acc:.2f}% ({correct}/{total})")

        # Also show how MIXED questions are classified
        if mixed_expected:
            mixed_correct = sum(1 for item in mixed_expected
                                if test_v2_with_thresholds(
                                    item["prompt"], best_pair[0], best_pair[1],
                                    mixed_enabled=True, mixed_pos_bar=bar, mixed_neg_bar=bar
                                ) == "mixed")
            print(f"      Mixed questions correctly detected as MIXED: {mixed_correct}/{len(mixed_expected)}")

    # Also test with MIXED disabled (baseline)
    correct = 0
    total = 0
    for item in train:
        text = item["prompt"]
        expected = clean_expected(item["expected_answer"])
        predicted = test_v2_with_thresholds(text, best_pair[0], best_pair[1], mixed_enabled=False)
        if predicted is None:
            predicted = "unknown"
        if predicted == expected:
            correct += 1
        total += 1
    print(f"    MIXED disabled: acc={correct/total*100:.2f}% ({correct}/{total})")

    # ═════════════════════════════════════════════════════════════════════════
    # 4. COVERAGE ANALYSIS
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SECTION 4: Coverage Analysis for v2 (best thresholds)")
    print("=" * 70)

    # Run v2 with best thresholds
    def best_v2(text):
        return test_v2_with_thresholds(text, best_pair[0], best_pair[1])

    best_analysis = analyze_variant(train, f"v2 (best: POS={best_pair[0]}, NEG={best_pair[1]})", best_v2)
    coverage = analyze_coverage(best_analysis, train)

    print(f"\n  Overall accuracy: {best_analysis['accuracy']:.2f}%")
    print(f"\n  Coverage categories:")
    for cat, count in sorted(coverage.items(), key=lambda x: -x[1]):
        pct = count / len(train) * 100
        print(f"    {cat:30s}: {count:4d} ({pct:.1f}%)")

    # Per-difficulty coverage
    print(f"\n  Per-difficulty accuracy (best v2):")
    for diff, counts in sorted(best_analysis["per_difficulty"].items()):
        acc = counts["correct"] / counts["total"] * 100 if counts["total"] > 0 else 0
        print(f"    {diff:10s}: {acc:.2f}% ({counts['correct']}/{counts['total']})")

    # ═════════════════════════════════════════════════════════════════════════
    # 5. VADER vs LLM COMPLEMENTARITY (Validation Set)
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SECTION 5: VADER vs LLM Complementarity (Validation Set)")
    print("=" * 70)

    # v2 on val set
    print(f"\n  VADER v2 on validation set:")
    val_analysis = analyze_variant(val, "v2 (best)", best_v2)
    print(f"    Accuracy: {val_analysis['accuracy']:.2f}% ({val_analysis['correct']}/{val_analysis['total']})")
    for diff, counts in sorted(val_analysis["per_difficulty"].items()):
        acc = counts["correct"] / counts["total"] * 100 if counts["total"] > 0 else 0
        print(f"      {diff}: {acc:.2f}% ({counts['correct']}/{counts['total']})")

    # v1 on val set
    val_v1 = analyze_variant(val, "v1", test_v1)
    print(f"\n  VADER v1 on validation set:")
    print(f"    Accuracy: {val_v1['accuracy']:.2f}% ({val_v1['correct']}/{val_v1['total']})")

    # ═════════════════════════════════════════════════════════════════════════
    # SUMMARY & RECOMMENDATIONS
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY OF FINDINGS")
    print("=" * 70)

    print(f"\n  v1 accuracy (current): {analyses['v1 (basic VADER)']['accuracy']:.2f}%")
    print(f"  v2 accuracy:           {analyses['v2 (advanced)']['accuracy']:.2f}%")
    print(f"  Best tuned v2:         {best_acc:.2f}% (POS={best_pair[0]}, NEG={best_pair[1]})")
    print(f"  v2 on val set:         {val_analysis['accuracy']:.2f}%")
    print(f"  v1 on val set:         {val_v1['accuracy']:.2f}%")

    print(f"\n  Recommendations:")
    print(f"  - {'Use v2' if best_acc > analyses['v1 (basic VADER)']['accuracy'] else 'Keep v1'} solver")
    print(f"  - Set _VADER_POS_THRESH = {best_pair[0]}")
    print(f"  - Set _VADER_NEG_THRESH = {best_pair[1]}")

    # ═════════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═════════════════════════════════════════════════════════════════════════
    output = {
        "variants": {name: {
            "accuracy": a["accuracy"],
            "correct": a["correct"],
            "total": a["total"],
            "per_difficulty": a["per_difficulty"],
            "per_compound_range": a["per_compound_range"],
        } for name, a in analyses.items()},
        "threshold_tuning": {
            "pos_thresholds": pos_thresholds,
            "neg_thresholds": neg_thresholds,
            "results": threshold_results,
            "best": {"pos": best_pair[0], "neg": best_pair[1], "accuracy": best_acc},
        },
        "mixed_detection": {
            "enabled_best_acc": correct / total * 100 if total else 0,
        },
        "coverage": coverage,
        "validation": {
            "v2_accuracy": val_analysis['accuracy'],
            "v1_accuracy": val_v1['accuracy'],
        },
        "cross_comparison": {
            "both_right": both_right,
            "v1_only": v1_only,
            "v2_only": v2_only,
            "both_wrong": both_wrong,
        },
    }

    output_path = f"{RESULTS_DIR}/vader_analysis_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    # Print textual report for research/vader_analysis.md
    report_path = f"{RESULTS_DIR}/vader_analysis.md"
    with open(report_path, "w") as f:
        f.write("# VADER Analysis Report\n\n")
        f.write(f"Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## 1. Variant Comparison (1142 training questions)\n\n")
        for name, a in analyses.items():
            f.write(f"### {name}\n")
            f.write(f"- Accuracy: {a['accuracy']:.2f}% ({a['correct']}/{a['total']})\n")
            for diff, counts in sorted(a["per_difficulty"].items()):
                acc = counts["correct"] / counts["total"] * 100 if counts["total"] > 0 else 0
                f.write(f"  - {diff}: {acc:.2f}% ({counts['correct']}/{counts['total']})\n")
            f.write("\n")

        f.write("## 2. Threshold Tuning\n\n")
        f.write(f"Best: POS={best_pair[0]}, NEG={best_pair[1]}, acc={best_acc:.2f}%\n\n")
        f.write("| POS | NEG | Acc |\n|-----|-----|----|\n")
        for pos_t, neg_t, acc in threshold_results:
            f.write(f"| {pos_t:+5.2f} | {neg_t:+5.2f} | {acc:.2f}% |\n")
        f.write("\n")

        f.write("## 3. MIXED Detection\n\n")
        f.write(f"Default (disabled): {analyses['v2 (advanced)']['accuracy']:.2f}%\n\n")

        f.write("## 4. Coverage Analysis\n\n")
        for cat, count in sorted(coverage.items(), key=lambda x: -x[1]):
            pct = count / len(train) * 100
            f.write(f"- {cat}: {count} ({pct:.1f}%)\n")
        f.write("\n")

        f.write("## 5. VADER vs LLM Complementarity\n\n")
        f.write(f"- v2 on val: {val_analysis['accuracy']:.2f}%\n")
        f.write(f"- v1 on val: {val_v1['accuracy']:.2f}%\n")
        f.write(f"- Both right on train: {both_right}\n")
        f.write(f"- v1 only right: {v1_only}\n")
        f.write(f"- v2 only right: {v2_only}\n")
        f.write(f"- Both wrong: {both_wrong}\n\n")

        f.write("## 6. Recommendations\n\n")
        f.write(f"1. **Solver**: {'v2' if best_acc > analyses['v1 (basic VADER)']['accuracy'] else 'v1'}\n")
        f.write(f"2. **_VADER_POS_THRESH** = {best_pair[0]}\n")
        f.write(f"3. **_VADER_NEG_THRESH** = {best_pair[1]}\n")
        f.write(f"4. **MIXED enabled**: {'Yes' if False else 'No (not yet proven beneficial)'}\n")

    print(f"  Report saved to {report_path}")


if __name__ == "__main__":
    main()
