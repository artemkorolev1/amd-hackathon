#!/usr/bin/env python3
"""Tune v1 VADER thresholds and analyze v2 hedging issues."""
import json, sys, os
sys.path.insert(0, os.path.expanduser("/home/artem/dev/amd-hackathon"))
from agent.solvers.deterministic import (
    _classify_sentiment_vader, _get_vader_analyzer,
    _RE_HEDGING, _RE_SARCASM_OH, _RE_SARCASM_YEAH,
    _RE_SARCASM_RHET, _RE_BACKHANDED, _RE_GENERAL_BUT,
)
from collections import Counter

with open("data/eval/sentiment_train.json") as f:
    train = json.load(f)

analyzer = _get_vader_analyzer()

# Check v2 hedging false positives
hedging_matches = []
for item in train:
    text = item["prompt"]
    expected = item["expected_answer"].strip().lower()
    if _RE_HEDGING.search(text):
        v1_label = _classify_sentiment_vader(text)
        hedging_matches.append((text[:120], expected, v1_label))

print(f"Hedging pattern matches: {len(hedging_matches)}")
wrong_hedge = [(t, e, v1) for t, e, v1 in hedging_matches if e != 'neutral' and v1 == e]
print(f"Hedging affecting correct v1 predictions: {len(wrong_hedge)}")
for t, e, v1 in wrong_hedge[:15]:
    print(f"  [{e}] {t}")

# Tune v1 thresholds
print("\n=== V1 THRESHOLD TUNING ===")
pos_ths = [-0.2, -0.1, 0.0, 0.05, 0.1, 0.2]
neg_ths = [-0.2, -0.1, 0.0, 0.05]

best_acc = 0
best_pair = (0.05, 0.0)
results = []
for pos_t in pos_ths:
    for neg_t in neg_ths:
        if pos_t <= neg_t:
            continue
        correct = 0
        for item in train:
            text = item["prompt"]
            expected = item["expected_answer"].strip().lower()
            scores = analyzer.polarity_scores(text)
            compound = scores["compound"]

            # Pattern overrides (from v1)
            if _RE_SARCASM_OH.search(text) and compound > -0.1:
                predicted = "negative"
            elif _RE_SARCASM_YEAH.search(text):
                predicted = "negative"
            elif _RE_SARCASM_RHET.search(text) and compound > 0.0:
                predicted = "negative"
            elif _RE_BACKHANDED.search(text):
                predicted = "negative"
            elif _RE_GENERAL_BUT.search(text) and compound > -0.1:
                predicted = "negative"
            elif compound >= pos_t:
                predicted = "positive"
            elif compound <= neg_t:
                predicted = "negative"
            else:
                predicted = "neutral"

            if predicted == expected:
                correct += 1

        acc = correct / len(train) * 100
        results.append((pos_t, neg_t, round(acc, 2)))
        marker = " ← BEST" if acc > best_acc else ""
        if acc > best_acc:
            best_acc = acc
            best_pair = (pos_t, neg_t)
        print(f"  POS={pos_t:+5.2f} NEG={neg_t:+5.2f} acc={acc:.2f}%{marker}")

print(f"\nBest v1: POS={best_pair[0]}, NEG={best_pair[1]}, acc={best_acc:.2f}%")
