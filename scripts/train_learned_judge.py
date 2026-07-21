"""Train a learned judge classifier on extracted eval data.
Features: text + structural + category-specific. Model: RandomForest + TF-IDF."""

import json
import os
import re
import sys
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

HERE = "/home/artem/dev/amd-hackathon"
DATA_PATH = os.path.join(HERE, "data", "judge_training.jsonl")
MODEL_PATH = os.path.join(HERE, "models", "learned_judge.pkl")
VECTORIZER_PATH = os.path.join(HERE, "models", "judge_vectorizer.pkl")


# ── Feature extractors ──────────────────────────────────────────────────


class StructuralFeatures(BaseEstimator, TransformerMixin):
    """Extract hand-crafted structural features from answer/prompt pairs."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X is list of dicts with keys: prompt, answer, expected, category
        features = []
        for rec in X:
            prompt = rec.get("prompt", "")
            answer = rec.get("answer", "")
            expected = rec.get("expected", "")
            category = str(rec.get("category", "")).lower()

            f = []

            # Length features
            f.append(len(answer))
            f.append(len(prompt))
            f.append(len(expected))
            f.append(len(answer.split()))
            f.append(len(prompt.split()))
            f.append(len(expected.split()))

            # Ratio features
            f.append(len(answer) / max(len(expected), 1))
            f.append(len(answer.split()) / max(len(expected.split()), 1))

            # Content features
            f.append(1.0 if "```" in answer else 0.0)  # has code fence
            f.append(answer.count("\n"))  # line count
            f.append(len(set(re.findall(r"\w+", answer.lower()))))  # unique words

            # Ends with punctuation
            f.append(1.0 if answer.strip() and answer.strip()[-1] in ".!?" else 0.0)

            # Numeric content
            nums_answer = len(re.findall(r"\d+(?:\.\d+)?", answer))
            nums_expected = len(re.findall(r"\d+(?:\.\d+)?", expected))
            f.append(nums_answer)
            f.append(nums_expected)
            f.append(nums_answer - nums_expected)

            # Category-specific features
            if "code" in category or category in ("code_gen", "code_debug"):
                f.append(1.0 if re.search(r"\bdef |\bclass |\breturn\b|\bimport ", answer) else 0.0)
                f.append(1.0 if re.search(r"\bdef |\bclass ", expected) else 0.0)
            else:
                f.append(0.0)
                f.append(0.0)

            if "math" in category:
                f.append(1.0 if re.search(r"[-+]?\d*\.?\d+", answer) else 0.0)
                f.append(1.0 if re.search(r"\b(solve|calculate|compute|equation)\b", prompt.lower()) else 0.0)
            else:
                f.append(0.0)
                f.append(0.0)

            if "ner" in category or "entity" in category:
                f.append(1.0 if re.search(r"[A-Z]{2,}:", answer) else 0.0)  # TYPE: format
            else:
                f.append(0.0)

            if "sentiment" in category:
                has_sentiment = re.search(r"\b(positive|negative|neutral|mixed)\b", answer.lower())
                f.append(1.0 if has_sentiment else 0.0)
            else:
                f.append(0.0)

            if "summar" in category:
                f.append(min(len(answer.split()) / 100, 1.0))  # length score
            else:
                f.append(0.0)

            # Overlap with expected answer
            ans_words = set(re.findall(r"[a-zA-Z]{3,}", answer.lower()))
            exp_words = set(re.findall(r"[a-zA-Z]{3,}", expected.lower()))
            if exp_words:
                overlap = len(ans_words & exp_words) / len(exp_words)
            else:
                overlap = 0.0
            f.append(overlap)

            # Exact match (lowercase)
            f.append(1.0 if answer.strip().lower() == expected.strip().lower() else 0.0)

            # Substring match
            f.append(1.0 if expected.strip().lower() in answer.strip().lower() else 0.0)

            features.append(f)

        return np.array(features)


class TextFeatures(BaseEstimator, TransformerMixin):
    """TF-IDF features from answer + prompt text."""

    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )

    def fit(self, X, y=None):
        texts = [
            f"{rec.get('category', '')} {rec.get('answer', '')} {rec.get('prompt', '')[:200]}"
            for rec in X
        ]
        self.tfidf.fit(texts)
        return self

    def transform(self, X):
        texts = [
            f"{rec.get('category', '')} {rec.get('answer', '')} {rec.get('prompt', '')[:200]}"
            for rec in X
        ]
        return self.tfidf.transform(texts).toarray()


# ── Load data ────────────────────────────────────────────────────────────

print(f"Loading data from {DATA_PATH}...")
records = []
with open(DATA_PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"Loaded {len(records)} records")

# Filter to records with both answer and expected
valid = [r for r in records if r.get("answer") and r.get("expected")]
print(f"With answer + expected: {len(valid)}")

# Labels
y = np.array([1 if r["correct"] else 0 for r in valid])

# Quick stats
print(f"Positive (correct): {y.sum()}")
print(f"Negative (incorrect): {len(y) - y.sum()}")

# ── Train/test split ─────────────────────────────────────────────────────

train_recs, test_recs, y_train, y_test = train_test_split(
    valid, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(train_recs)} | Test: {len(test_recs)}")

# ── Build feature union pipeline ─────────────────────────────────────────

# Structural features (hand-crafted)
struct = StructuralFeatures()
X_train_struct = struct.fit_transform(train_recs)
X_test_struct = struct.transform(test_recs)

# Text features (TF-IDF on answer+prompt)
txt = TextFeatures()
txt.fit(train_recs)
X_train_txt = txt.transform(train_recs)
X_test_txt = txt.transform(test_recs)

# Combined
X_train = np.hstack([X_train_struct, X_train_txt])
X_test = np.hstack([X_test_struct, X_test_txt])

print(f"Feature dimensions: structural={X_train_struct.shape[1]}, text={X_train_txt.shape[1]}, total={X_train.shape[1]}")

# ── Train classifier ─────────────────────────────────────────────────────

print("\nTraining RandomForest...")
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train, y_train)

# ── Evaluate ─────────────────────────────────────────────────────────────

train_pred = clf.predict(X_train)
test_pred = clf.predict(X_test)
train_prob = clf.predict_proba(X_train)[:, 1]
test_prob = clf.predict_proba(X_test)[:, 1]

print(f"\nTrain accuracy: {np.mean(train_pred == y_train):.3f}")
print(f"Test accuracy:  {np.mean(test_pred == y_test):.3f}")
print(f"Train AUC:      {roc_auc_score(y_train, train_prob):.3f}")
print(f"Test AUC:       {roc_auc_score(y_test, test_prob):.3f}")

print(f"\nClassification Report (Test):")
print(classification_report(y_test, test_pred, target_names=["incorrect", "correct"]))

print(f"Confusion Matrix (Test):")
cm = confusion_matrix(y_test, test_pred)
print(cm)

# ── Feature importance (structural features only) ────────────────────────

n_struct = X_train_struct.shape[1]
print(f"\nTop-10 structural feature importances:")
importances = clf.feature_importances_[:n_struct]
struct_names = [
    "ans_len", "prompt_len", "exp_len", "ans_words", "prompt_words", "exp_words",
    "len_ratio", "word_ratio", "has_code_fence", "newlines", "unique_words",
    "ends_punct", "nums_answer", "nums_expected", "nums_diff",
    "code_has_def", "code_has_exp_def",
    "math_has_num", "math_has_keywords",
    "ner_has_format",
    "sent_has_sentiment",
    "summ_length_score",
    "word_overlap", "exact_match", "substr_match",
]
for i, name in enumerate(struct_names):
    print(f"  {name:<20} {importances[i]:.4f}")

# ── Save model ───────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(clf, f)
with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump({"struct": struct, "txt": txt, "struct_names": struct_names}, f)

print(f"\nModel saved to {MODEL_PATH}")

# ── Threshold calibration ───────────────────────────────────────────────

# Find optimal threshold for F1
from sklearn.metrics import precision_recall_curve, f1_score
precisions, recalls, thresholds = precision_recall_curve(y_test, test_prob)

best_f1 = 0
best_thresh = 0.5
for thresh in np.arange(0.3, 0.9, 0.05):
    pred = (test_prob >= thresh).astype(int)
    f1 = f1_score(y_test, pred)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f"\nOptimal threshold: {best_thresh:.2f} (F1={best_f1:.3f})")

# Per-category breakdown
print(f"\nPer-category test accuracy:")
cat_test_pred = {}
for rec, prob, true in zip(test_recs, test_prob, y_test):
    cat = str(rec.get("category", "unknown")).lower()
    if cat not in cat_test_pred:
        cat_test_pred[cat] = {"correct": 0, "total": 0, "correct_prob_sum": 0.0}
    cat_test_pred[cat]["total"] += 1
    cat_test_pred[cat]["correct_prob_sum"] += prob
    if (prob >= best_thresh) == bool(true):
        cat_test_pred[cat]["correct"] += 1

for cat in sorted(cat_test_pred.keys()):
    v = cat_test_pred[cat]
    acc = v["correct"] / v["total"] * 100 if v["total"] > 0 else 0
    avg_prob = v["correct_prob_sum"] / v["total"]
    print(f"  {cat:<20} {v['total']:>4} items  acc={acc:>5.1f}%  avg_prob={avg_prob:.3f}")
