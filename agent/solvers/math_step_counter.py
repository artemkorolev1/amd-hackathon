"""
math_step_counter.py — Deterministic step count classifier for GSM8K.

Predicts how many computational steps a math word problem needs.
All heuristics are keyword-based (no ML). Uses rules extracted from
a decision tree trained on 7,473 GSM8K training examples (max_depth=4).

Step buckets: 1, 2, 3, 4, 5+ (where 5+ is the catch-all for >=5 steps).
"""

import re
from typing import Dict, Optional, Union


def _extract_features(question: str) -> Dict[str, Union[int, float]]:
    """Extract feature vector from a GSM8K question."""
    ql = question.lower()
    wc = len(ql.split())
    nums = re.findall(r'\d+', question)
    num_count = len(nums)
    distinct_nums = len(set(nums))
    char_len = len(question)
    commas = question.count(',')
    sentences = len(re.findall(r'[.!?]', question))
    and_count = len(re.findall(r'\band\b', ql))

    # Keyword counts
    kw_each = ql.count('each') + ql.count('every')
    kw_per = ql.count(' per ')
    kw_remaining = ql.count('remaining') + ql.count(' left ') + ql.count(' rest ')
    kw_first_then = ql.count('first') + ql.count(' then ') + ql.count(' after ')
    kw_half = ql.count('half')
    kw_twice = ql.count('twice') + ql.count('double')
    kw_total = ql.count('total') + ql.count('altogether') + ql.count(' together ')
    kw_more = ql.count(' more ') + ql.count(' fewer ') + ql.count(' less ')
    kw_how_many = ql.count('how many') + ql.count('how much')
    kw_times = ql.count(' times ')
    kw_share = ql.count('share') + ql.count('split') + ql.count('divide')
    kw_average = ql.count('average')
    kw_dozen = ql.count('dozen')
    kw_difference = ql.count('difference')

    # Composite features
    has_fraction = int(bool(re.search(r'\d+/\d+|half|quarter', ql)))
    has_unit = int(bool(re.search(r'(hour|day|week|month|year|minute|dozen|pound)', ql)))
    has_comparison = int(bool(re.search(r'(more|less|fewer).*(than)', ql)))
    has_multiple_rate = int(bool(re.search(r'(per|each|a ).*(hour|day|week|month|year|dozen)', ql)))
    numbers_with_context = len(re.findall(r'\d+.*?(each|per|every|times|dozen)', ql))

    # Scoring features
    multi_step_score = (
        kw_each * 1.0 + kw_per * 1.0 + kw_remaining * 1.0 + kw_first_then * 1.0
        + kw_half * 0.5 + kw_twice * 0.5 + kw_total * 0.5 + kw_times * 0.5
    )

    return {
        "num_count": num_count,
        "distinct_nums": distinct_nums,
        "word_count": wc,
        "char_len": char_len,
        "commas": commas,
        "sentences": sentences,
        "and_count": and_count,
        "has_fraction": has_fraction,
        "has_unit": has_unit,
        "has_comparison": has_comparison,
        "has_multiple_rate": has_multiple_rate,
        "numbers_with_context": numbers_with_context,
        "kw_each": kw_each,
        "kw_per": kw_per,
        "kw_remaining": kw_remaining,
        "kw_first_then": kw_first_then,
        "kw_half": kw_half,
        "kw_twice": kw_twice,
        "kw_total": kw_total,
        "kw_more": kw_more,
        "kw_how_many": kw_how_many,
        "kw_times": kw_times,
        "kw_share": kw_share,
        "kw_average": kw_average,
        "kw_dozen": kw_dozen,
        "kw_difference": kw_difference,
        "multi_step_score": multi_step_score,
    }


def predict_step_count(question: str) -> int:
    """
    Predict the number of computational steps needed for a math word problem.

    Returns a step count in {1, 2, 3, 4, 5} where 5 means "5 or more".
    Uses decision rules extracted from a trained decision tree (max_depth=4).
    """
    feats = _extract_features(question)
    wc = feats["word_count"]
    nc = feats["num_count"]
    char_len = feats["char_len"]
    has_frac = feats["has_fraction"]
    commas = feats["commas"]
    kw_each = feats["kw_each"]
    kw_half = feats["kw_half"]
    kw_twice = feats["kw_twice"]

    if wc <= 40:
        if nc <= 3:
            if wc <= 31:
                return 2
            else:  # wc > 31
                if has_frac:
                    return 3
                return 2
        else:  # nc > 3
            if nc <= 4:
                if commas <= 2:
                    return 3
                else:
                    return 2
            else:  # nc > 4
                if kw_each <= 1:
                    return 3
                return 4
    else:  # wc > 40
        if nc <= 4:
            if char_len <= 321:
                if kw_half:
                    return 4
                return 3
            else:  # char_len > 321
                if kw_twice:
                    return 5
                return 3
        else:  # nc > 4
            return 5


def get_features(question: str) -> Dict[str, Union[int, float]]:
    """Public API: get feature vector for a question."""
    return _extract_features(question)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def parse_ground_truth_steps(answer: str) -> int:
    """Parse the number of <<...>> expressions from a GSM8K answer."""
    return len(re.findall(r'<<.*?>>', answer))


def evaluate_on_dataframe(df) -> Dict:
    """
    Evaluate step counter accuracy on a DataFrame with 'question' and 'answer' columns.
    Returns accuracy metrics.
    """
    correct = 0
    total = len(df)
    bucket_correct = {}
    bucket_total = {}

    for _, row in df.iterrows():
        question = row["question"]
        answer = row["answer"]
        true_steps = parse_ground_truth_steps(answer)
        true_bucket = min(true_steps, 5) if true_steps > 0 else 1
        if true_steps == 0:
            true_bucket = 1

        pred_bucket = predict_step_count(question)

        bucket_total[true_bucket] = bucket_total.get(true_bucket, 0) + 1
        if pred_bucket == true_bucket:
            correct += 1
            bucket_correct[true_bucket] = bucket_correct.get(true_bucket, 0) + 1

    bucket_acc = {}
    for b in sorted(bucket_total):
        bt = bucket_total[b]
        bc = bucket_correct.get(b, 0)
        bucket_acc[b] = bc / bt if bt > 0 else 0.0

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "bucket_accuracy": bucket_acc,
        "bucket_counts": bucket_total,
    }
