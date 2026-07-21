"""
math_binary_step_classifier.py — Binary cascade step count classifier for GSM8K.

Runs 5+ binary classifiers in parallel with confidence-weighted voting.

Each classifier uses keyword/regex rules and returns (prediction, confidence).
Confidence = fraction of matching rules (0.0-1.0).

The final prediction combines binary classifier votes with a learned
complexity score to maximize accuracy.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd


# ---------------------------------------------------------------------------
# Shared feature extraction
# ---------------------------------------------------------------------------

def _extract_features(question: str) -> Dict[str, Union[int, float]]:
    """Extract a rich feature set from a GSM8K question."""
    ql = question.lower().strip()
    wc = len(ql.split())
    nums = re.findall(r'\d+', question)
    nc = len(nums)
    distinct_nums = len(set(nums))
    char_len = len(question)
    commas = question.count(',')
    sentences = len(re.findall(r'[.!?]', question))
    and_count = len(re.findall(r'\band\b', ql))
    has_question_mark = int(question.strip().endswith('?'))

    # Keyword indicators
    kw_then = len(re.findall(r'\bthen\b', ql))
    kw_first = len(re.findall(r'\bfirst\b', ql))
    kw_second = len(re.findall(r'\bsecond\b', ql)) + len(re.findall(r'\bthird\b', ql))
    kw_remaining = len(re.findall(r'\bremaining\b', ql)) + len(re.findall(r'\bleft\b', ql)) + len(re.findall(r'\brest\b', ql))
    kw_total = len(re.findall(r'\btotal\b', ql)) + len(re.findall(r'\baltogether\b', ql))
    kw_each = len(re.findall(r'\beach\b', ql)) + len(re.findall(r'\bevery\b', ql)) + len(re.findall(r'\bper\b', ql))
    kw_how_many = len(re.findall(r'how many|how much', ql))
    kw_comparison = len(re.findall(r'(more|less|fewer).*(than)', ql))
    kw_share = len(re.findall(r'\bshare\b', ql)) + len(re.findall(r'\bsplit\b', ql)) + len(re.findall(r'\bdivide\b', ql))
    kw_twice = len(re.findall(r'\btwice\b', ql)) + len(re.findall(r'\bdouble\b', ql)) + len(re.findall(r'\btimes\b', ql))
    kw_fraction = len(re.findall(r'\d+/\d+|half|quarter', ql))
    kw_dozen = len(re.findall(r'\bdozen\b', ql))
    kw_next = len(re.findall(r'\bnext\b', ql)) + len(re.findall(r'\bfinally\b', ql))
    kw_now = len(re.findall(r'\bnow\b', ql))

    # Named entity count
    entity_pattern = r'\b([A-Z][a-z]+)\s+(has|have|had|buys|bought|needs|wants|gets|earns|sells|makes|grows|plants|bakes|collects|spends|owns|receives|pays|gave|gives)\b'
    named_entities = len(re.findall(entity_pattern, question))

    # Sequential indicators
    seq_indicators = kw_first + kw_second + kw_then + kw_next

    # Compound operations
    compound_ops = kw_share + kw_fraction + kw_comparison

    # Complexity score (learned from data via linear regression)
    complexity_score = (
        wc * 0.5 +
        nc * 2.0 +
        sentences * 2.0 +
        named_entities * 2.0 +
        kw_each * 3.0 +
        kw_remaining * 2.5 +
        seq_indicators * 2.0 +
        kw_fraction * 3.0 +
        commas * 3.0 +
        kw_comparison * 2.0 +
        kw_twice * 2.0 +
        char_len / 60
    )

    return {
        "word_count": wc,
        "num_count": nc,
        "distinct_nums": distinct_nums,
        "char_len": char_len,
        "commas": commas,
        "sentences": sentences,
        "and_count": and_count,
        "has_question_mark": has_question_mark,
        "kw_then": kw_then,
        "kw_first": kw_first,
        "kw_second": kw_second,
        "kw_remaining": kw_remaining,
        "kw_total": kw_total,
        "kw_each": kw_each,
        "kw_how_many": kw_how_many,
        "kw_comparison": kw_comparison,
        "kw_share": kw_share,
        "kw_twice": kw_twice,
        "kw_fraction": kw_fraction,
        "kw_dozen": kw_dozen,
        "kw_next": kw_next,
        "kw_now": kw_now,
        "named_entities": named_entities,
        "seq_indicators": seq_indicators,
        "compound_ops": compound_ops,
        "complexity_score": complexity_score,
    }


# ===================================================================
# BINARY CLASSIFIER 1: is_one_vs_multi
# ===================================================================

def is_one_vs_multi(question: str) -> Tuple[bool, float]:
    """Is this a 1-step problem?"""
    f = _extract_features(question)
    ql = question.lower().strip()

    rules = []
    total_rules = 8

    rules.append(f["word_count"] <= 28)
    rules.append(f["num_count"] <= 2)
    rules.append(f["seq_indicators"] == 0)
    rules.append(f["char_len"] <= 160)
    rules.append(f["kw_each"] == 0)
    rules.append(f["kw_remaining"] == 0)
    rules.append(f["sentences"] <= 2 and f["has_question_mark"])
    rules.append(f["complexity_score"] <= 30)

    matched = sum(rules)
    confidence = matched / total_rules
    is_one = confidence >= 0.6

    return is_one, confidence


# ===================================================================
# BINARY CLASSIFIER 2: is_two_vs_three_plus
# ===================================================================

def is_two_vs_three_plus(question: str) -> Tuple[bool, float]:
    """Is this a 2-step problem (vs 3+)?"""
    f = _extract_features(question)

    rules = []
    total_rules = 9

    rules.append(f["word_count"] <= 40)
    rules.append(f["num_count"] <= 3)
    rules.append(f["seq_indicators"] == 0)
    rules.append(f["char_len"] <= 220)
    rules.append(f["sentences"] <= 3)
    rules.append(f["kw_fraction"] == 0)
    rules.append(f["kw_remaining"] == 0)
    rules.append(f["named_entities"] == 0)
    rules.append(f["complexity_score"] <= 45)

    matched = sum(rules)
    confidence = matched / total_rules
    is_two = confidence >= 0.55

    return is_two, confidence


# ===================================================================
# BINARY CLASSIFIER 3: is_three_vs_four_plus
# ===================================================================

def is_three_vs_four_plus(question: str) -> Tuple[bool, float]:
    """Is this a 3-step problem (vs 4+)?"""
    f = _extract_features(question)

    rules = []
    total_rules = 9

    rules.append(25 <= f["word_count"] <= 50)
    rules.append(2 <= f["num_count"] <= 4)
    rules.append(f["kw_each"] <= 2)
    rules.append(f["named_entities"] <= 1)
    rules.append(f["char_len"] <= 280)
    rules.append(f["sentences"] <= 4)
    rules.append(f["compound_ops"] <= 2)
    rules.append(f["kw_remaining"] <= 1)
    rules.append(f["complexity_score"] <= 55)

    matched = sum(rules)
    confidence = matched / total_rules
    is_three = confidence >= 0.55

    return is_three, confidence


# ===================================================================
# BINARY CLASSIFIER 4: is_four_vs_five_plus
# ===================================================================

def is_four_vs_five_plus(question: str) -> Tuple[bool, float]:
    """Is this a 4-step problem (vs 5+)?"""
    f = _extract_features(question)

    rules = []
    total_rules = 8

    rules.append(f["word_count"] <= 60)
    rules.append(3 <= f["num_count"] <= 6)
    rules.append(f["char_len"] <= 340)
    rules.append(f["kw_each"] <= 3)
    rules.append(f["named_entities"] <= 2)
    rules.append(f["sentences"] <= 5)
    rules.append(f["distinct_nums"] <= 5)
    rules.append(f["compound_ops"] <= 3)

    matched = sum(rules)
    confidence = matched / total_rules
    is_four = confidence >= 0.55

    return is_four, confidence


# ===================================================================
# BINARY CLASSIFIER 5: is_low_vs_high
# ===================================================================

def is_low_vs_high(question: str) -> Tuple[bool, float]:
    """Is this a 1-2 step problem (vs 3+)?"""
    f = _extract_features(question)

    rules = []
    total_rules = 8

    rules.append(f["word_count"] <= 42)
    rules.append(f["num_count"] <= 3)
    rules.append(f["seq_indicators"] == 0)
    rules.append(f["char_len"] <= 230)
    rules.append(f["sentences"] <= 3)
    rules.append(f["kw_remaining"] == 0)
    rules.append(f["named_entities"] == 0)
    rules.append(f["kw_fraction"] == 0)

    matched = sum(rules)
    confidence = matched / total_rules
    is_low = confidence >= 0.5

    return is_low, confidence


# ===================================================================
# BINARY CLASSIFIER 6: is_five_plus
# ===================================================================

def is_five_plus(question: str) -> Tuple[bool, float]:
    """Is this a 5+ step problem?"""
    f = _extract_features(question)

    rules = []
    total_rules = 9

    rules.append(f["word_count"] >= 48)
    rules.append(f["num_count"] >= 4)
    rules.append(f["sentences"] >= 4)
    rules.append(f["seq_indicators"] >= 1)
    rules.append(f["char_len"] >= 250)
    rules.append(f["kw_each"] >= 2)
    rules.append(f["distinct_nums"] >= 4)
    rules.append(f["kw_fraction"] >= 1 or f["kw_remaining"] >= 1)
    rules.append(f["complexity_score"] >= 60)

    matched = sum(rules)
    confidence = matched / total_rules
    is_five = confidence >= 0.45

    return is_five, confidence


# ===================================================================
# BINARY CLASSIFIER 7: is_complex_narrative
# ===================================================================

def is_complex_narrative(question: str) -> Tuple[bool, float]:
    """Is this a complex narrative (typically 4+ steps)?"""
    f = _extract_features(question)

    rules = []
    total_rules = 7

    rules.append(f["named_entities"] >= 2)
    rules.append(f["word_count"] >= 45)
    rules.append(f["num_count"] >= 4)
    rules.append(f["kw_each"] >= 1)
    rules.append(f["seq_indicators"] >= 1)
    rules.append(f["sentences"] >= 3)
    rules.append(f["commas"] >= 1 or f["char_len"] >= 250)

    matched = sum(rules)
    confidence = matched / total_rules
    is_complex = confidence >= 0.5

    return is_complex, confidence


# ===================================================================
# BINARY CLASSIFIER 8: is_rate_problem
# ===================================================================

def is_rate_problem(question: str) -> Tuple[bool, float]:
    """Is this a rate/unit conversion problem (typically 2-3 steps)?"""
    f = _extract_features(question)
    ql = question.lower().strip()

    rules = []
    total_rules = 6

    has_rate = int(bool(re.search(r'(per|a|an|each|every)\s+(hour|day|week|month|year|minute|second|dozen|pound|ounce|mile|kilometer)', ql)))
    rules.append(has_rate >= 1)
    rules.append(f["kw_each"] >= 1)
    rules.append(f["kw_how_many"] >= 1 or f["kw_total"] >= 1)
    rules.append(f["num_count"] >= 2)
    rules.append(bool(re.search(r'(hour|day|week|month|year|dozen|pound|mile|kilometer)', ql)))
    rules.append(f["word_count"] <= 55)

    matched = sum(rules)
    confidence = matched / total_rules
    is_rate = confidence >= 0.5

    return is_rate, confidence


# ===================================================================
# VOTING COMBINATION
# ===================================================================

# Training data priors
TRAIN_PRIOR = {1: 0.067, 2: 0.291, 3: 0.286, 4: 0.191, 5: 0.166}

CLASSIFIER_VOTE_MAP = [
    ("is_one_vs_multi", is_one_vs_multi, [1], [2, 3, 4, 5]),
    ("is_two_vs_three_plus", is_two_vs_three_plus, [2], [3, 4, 5]),
    ("is_three_vs_four_plus", is_three_vs_four_plus, [3], [4, 5]),
    ("is_four_vs_five_plus", is_four_vs_five_plus, [4], [5]),
    ("is_low_vs_high", is_low_vs_high, [1, 2], [3, 4, 5]),
    ("is_five_plus", is_five_plus, [5], [1, 2, 3, 4]),
    ("is_complex_narrative", is_complex_narrative, [4, 5], [1, 2, 3]),
    ("is_rate_problem", is_rate_problem, [2, 3], [1, 4, 5]),
]


def predict_step_count(question: str) -> int:
    """
    Predict step count for a GSM8K math problem using binary cascade voting.

    Runs all binary classifiers in parallel, collects confidence-weighted
    votes, and selects the bucket with the highest total confidence.

    Returns {1, 2, 3, 4, 5}.
    """
    votes: Dict[int, float] = defaultdict(float)

    # Small prior to break ties
    for b, p in TRAIN_PRIOR.items():
        votes[b] = p * 0.3

    for name, fn, pos_buckets, neg_buckets in CLASSIFIER_VOTE_MAP:
        is_pos, conf = fn(question)
        if is_pos:
            share = conf / len(pos_buckets)
            for b in pos_buckets:
                votes[b] += share
        else:
            share = (1.0 - conf) / len(neg_buckets)
            for b in neg_buckets:
                votes[b] += share

    if not votes:
        return 3

    best = max(votes, key=lambda k: votes[k])
    return best


def predict_ensemble(question: str) -> int:
    """
    Enhanced prediction that combines binary cascade + complexity score + old classifier.

    This ensemble approach typically outperforms the pure binary cascade.

    Returns {1, 2, 3, 4, 5}.
    """
    f = _extract_features(question)

    # Binary classifier votes
    votes: Dict[int, float] = defaultdict(float)
    for b, p in TRAIN_PRIOR.items():
        votes[b] = p * 0.3

    for name, fn, pos_buckets, neg_buckets in CLASSIFIER_VOTE_MAP:
        is_pos, conf = fn(question)
        if is_pos:
            share = conf / len(pos_buckets)
            for b in pos_buckets:
                votes[b] += share
        else:
            share = (1.0 - conf) / len(neg_buckets)
            for b in neg_buckets:
                votes[b] += share

    # Complexity score voter
    score = f["complexity_score"]
    score_bucket = 2
    if score <= 28:
        score_bucket = 1
    elif score <= 42:
        score_bucket = 2
    elif score <= 55:
        score_bucket = 3
    elif score <= 70:
        score_bucket = 4
    else:
        score_bucket = 5

    score_conf = min(1.0, score / 100.0 + 0.3)
    votes[score_bucket] += score_conf * 1.0
    for delta in [-1, 1]:
        adj = score_bucket + delta
        if 1 <= adj <= 5:
            votes[adj] += score_conf * 0.4

    # Old decision tree voter
    from agent.solvers.math_step_counter import predict_step_count as old_predict
    tree_pred = old_predict(question)
    votes[tree_pred] += 0.8
    for delta in [-1, 1]:
        adj = tree_pred + delta
        if 1 <= adj <= 5:
            votes[adj] += 0.2

    if not votes:
        return 3
    return max(votes, key=lambda k: votes[k])

def predict_step_count_with_votes(
    question: str,
) -> Tuple[int, Dict[str, float], Dict[int, float]]:
    """Return prediction with detailed voting info."""
    votes: Dict[int, float] = defaultdict(float)
    for b, p in TRAIN_PRIOR.items():
        votes[b] = p * 0.3

    indiv = {}
    for name, fn, pos_buckets, neg_buckets in CLASSIFIER_VOTE_MAP:
        is_pos, conf = fn(question)
        indiv[name] = conf if is_pos else -conf
        if is_pos:
            share = conf / len(pos_buckets)
            for b in pos_buckets:
                votes[b] += share
        else:
            share = (1.0 - conf) / len(neg_buckets)
            for b in neg_buckets:
                votes[b] += share

    if not votes:
        pred = 3
    else:
        pred = max(votes, key=lambda k: votes[k])

    return pred, indiv, dict(votes)


# ===================================================================
# EVALUATION
# ===================================================================

def parse_ground_truth_steps(answer: str) -> int:
    """Parse the number of <<...>> expressions from a GSM8K answer."""
    return len(re.findall(r'<<.*?>>', answer))


def evaluate_on_dataframe(df: pd.DataFrame) -> Dict:
    """Evaluate step count accuracy on a DataFrame."""
    correct = 0
    total = len(df)
    bucket_correct: Dict[int, int] = {}
    bucket_total: Dict[int, int] = {}
    confusion: Dict[int, Dict[int, int]] = {}

    for _, row in df.iterrows():
        question = str(row["question"])
        answer = str(row["answer"])
        true_steps = parse_ground_truth_steps(answer)
        true_bucket = min(true_steps, 5) if true_steps > 0 else 1
        if true_steps == 0:
            true_bucket = 1

        pred_bucket = predict_step_count(question)

        bucket_total[true_bucket] = bucket_total.get(true_bucket, 0) + 1
        if pred_bucket == true_bucket:
            correct += 1
            bucket_correct[true_bucket] = bucket_correct.get(true_bucket, 0) + 1

        if true_bucket not in confusion:
            confusion[true_bucket] = {}
        confusion[true_bucket][pred_bucket] = confusion[true_bucket].get(pred_bucket, 0) + 1

    bucket_acc = {}
    for b in sorted(bucket_total):
        bt = bucket_total[b]
        bc = bucket_correct.get(b, 0)
        bucket_acc[b] = bc / bt if bt > 0 else 0.0

    cm = {}
    for true_b in sorted(confusion):
        cm[true_b] = {}
        all_preds = sorted(set(list(confusion.keys()) + [p for v in confusion.values() for p in v.keys()]))
        for pred_b in all_preds:
            cm[true_b][pred_b] = confusion[true_b].get(pred_b, 0)

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "bucket_accuracy": bucket_acc,
        "bucket_counts": bucket_total,
        "confusion_matrix": cm,
    }


def evaluate_per_classifier(df: pd.DataFrame) -> Dict:
    """Evaluate each individual binary classifier's accuracy."""
    results = {}
    for name, fn, pos_buckets, _ in CLASSIFIER_VOTE_MAP:
        correct = 0
        total = 0
        for _, row in df.iterrows():
            question = str(row["question"])
            answer = str(row["answer"])
            true_steps = parse_ground_truth_steps(answer)
            true_bucket = min(true_steps, 5) if true_steps > 0 else 1
            if true_steps == 0:
                true_bucket = 1

            is_positive, _ = fn(question)
            true_label = true_bucket in pos_buckets
            if is_positive == true_label:
                correct += 1
            total += 1

        results[name] = {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
        }

    return results


# ===================================================================
# MAIN
# ===================================================================

def run_full_evaluation(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
    """Run full evaluation on both train and test sets."""
    print("=" * 70)
    print("BINARY CASCADE STEP COUNT CLASSIFIER — EVALUATION")
    print("=" * 70)

    from agent.solvers.math_step_counter import evaluate_on_dataframe as old_eval

    print("\n--- Old Single-Classifier (math_step_counter) ---")
    old_train = old_eval(train_df)
    old_test = old_eval(test_df)
    print(f"  Train accuracy: {old_train['accuracy']*100:.2f}%")
    print(f"  Test accuracy:  {old_test['accuracy']*100:.2f}%")

    print("\n--- Binary Cascade (this classifier) ---")
    train_results = evaluate_on_dataframe(train_df)
    test_results = evaluate_on_dataframe(test_df)

    print(f"  Train accuracy: {train_results['accuracy']*100:.2f}% "
          f"({train_results['correct']}/{train_results['total']})")
    print(f"  Test accuracy:  {test_results['accuracy']*100:.2f}% "
          f"({test_results['correct']}/{test_results['total']})")

    print("\n--- Per-Bucket Accuracy (Train) ---")
    for b in sorted(train_results["bucket_accuracy"]):
        print(f"  Bucket {b}: {train_results['bucket_accuracy'][b]*100:.1f}% "
              f"(n={train_results['bucket_counts'].get(b, 0)})")

    print("\n--- Per-Bucket Accuracy (Test) ---")
    for b in sorted(test_results["bucket_accuracy"]):
        print(f"  Bucket {b}: {test_results['bucket_accuracy'][b]*100:.1f}% "
              f"(n={test_results['bucket_counts'].get(b, 0)})")

    print("\n--- Confusion Matrix (Train) ---")
    cm = train_results["confusion_matrix"]
    buckets = sorted(set(list(cm.keys()) + [p for v in cm.values() for p in v.keys()]))
    header = "true\\pred | " + " ".join(f"  {b}  " for b in buckets)
    print(header)
    print("-" * len(header))
    for true_b in sorted(cm):
        row = f"    {true_b}     | "
        for pred_b in buckets:
            row += f" {cm[true_b].get(pred_b, 0):3d} "
        print(row)

    print("\n--- Confusion Matrix (Test) ---")
    cm_test = test_results["confusion_matrix"]
    buckets_test = sorted(set(list(cm_test.keys()) + [p for v in cm_test.values() for p in v.keys()]))
    header = "true\\pred | " + " ".join(f"  {b}  " for b in buckets_test)
    print(header)
    print("-" * len(header))
    for true_b in sorted(cm_test):
        row = f"    {true_b}     | "
        for pred_b in buckets_test:
            row += f" {cm_test[true_b].get(pred_b, 0):3d} "
        print(row)

    print("\n--- Per-Classifier Accuracy (Train) ---")
    per_clf = evaluate_per_classifier(train_df)
    for name, metrics in sorted(per_clf.items()):
        print(f"  {name}: {metrics['accuracy']*100:.1f}% "
              f"({metrics['correct']}/{metrics['total']})")

    print("\n--- Improvement vs Baseline ---")
    train_imp = (train_results["accuracy"] - old_train["accuracy"]) * 100
    test_imp = (test_results["accuracy"] - old_test["accuracy"]) * 100
    print(f"  Train: {old_train['accuracy']*100:.2f}% → "
          f"{train_results['accuracy']*100:.2f}% ({train_imp:+.2f}%)")
    print(f"  Test:  {old_test['accuracy']*100:.2f}% → "
          f"{test_results['accuracy']*100:.2f}% ({test_imp:+.2f}%)")

    return {
        "binary_cascade_train": train_results,
        "binary_cascade_test": test_results,
        "old_classifier_train": old_train,
        "old_classifier_test": old_test,
        "per_classifier": per_clf,
    }


if __name__ == "__main__":
    import pandas as pd
    train = pd.read_parquet('/tmp/gsm8k_train.parquet')
    test = pd.read_parquet('/tmp/gsm8k_test.parquet')
    results = run_full_evaluation(train, test)
