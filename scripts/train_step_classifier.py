#!/usr/bin/env python3
"""
train_step_classifier.py — Build and evaluate ML-based step-count classifiers
for GSM8K math problems. Compares logistic regression, decision tree, random
forest, and XGBoost against keyword-classifier baselines.

Usage:
    PYTHONPATH=/home/artem/dev/amd-hackathon python scripts/train_step_classifier.py
"""

import json
import re
import warnings
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
TRAIN_PATH = "/tmp/gsm8k_train.parquet"
TEST_PATH = "/tmp/gsm8k_test.parquet"
REPORT_DIR = Path("/home/artem/dev/amd-hackathon/gepa_plans")
MODEL_DIR = Path("/home/artem/dev/amd-hackathon/agent/solvers/models")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Ground-truth step count extraction
# ---------------------------------------------------------------------------
def extract_step_count(answer: str) -> int:
    """Count <<...>> markers in a GSM8K answer. Minimum 1 (a problem always
    has at least one step). But some answers have 0 markers — keep as-is."""
    markers = re.findall(r"<<.*?>>", answer)
    return len(markers)


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
VERB_SET = {
    "has", "have", "had", "buys", "bought", "buy", "sells", "sold", "sell",
    "gives", "gave", "give", "spends", "spent", "spend", "costs", "cost",
    "earns", "earned", "earn", "makes", "made", "make", "pays", "paid", "pay",
    "walks", "walked", "walk", "runs", "ran", "run", "drives", "drove", "drive",
    "collects", "collected", "collect", "receives", "received", "receive",
}

COMPARISON_KEYWORDS = {
    "more than", "less than", "fewer than", "times as many",
    "times as much", "as many as",
}

RATIO_KEYWORDS = {
    "ratio", "fraction", "percentage", "percent", "half", "quarter",
    "third", "fourth",
}

SEQUENCE_WORDS = {
    "then", "next", "after", "finally", "first", "second", "third",
    "last", "before", "previously",
}

QUESTION_PHRASES = {
    "how many", "how much", "what is", "what are", "what was",
    "find the", "calculate", "determine",
}

ENTITY_PATTERN = re.compile(r"\b\w+'s\b")

NUMERIC_PATTERN = re.compile(r"\b\d+(?:[,.]\d+)*\b")

COLLECTIVE_KEYWORDS = {
    "each", "per", "every", "total", "combined", "altogether", "together",
}


def engineer_features(question: str) -> dict:
    """Build a feature dict from a single question string."""
    q = question
    ql = q.lower()
    words = q.split()
    sentences = re.split(r"[.!?]+", q)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Numeric values
    numbers_str = NUMERIC_PATTERN.findall(q)
    numbers = []
    for n_str in numbers_str:
        try:
            numbers.append(float(n_str.replace(",", "")))
        except ValueError:
            pass

    # Counts
    word_count = len(words)
    char_count = len(q)
    sentence_count = len(sentences)
    num_count = len(numbers)
    unique_num_count = len(set(numbers))

    # Verb count
    verb_count = sum(1 for w in words if w.lower().rstrip("s") in VERB_SET or
                     w.lower() in VERB_SET)

    # Comparison keywords
    comp_count = sum(1 for kw in COMPARISON_KEYWORDS if kw in ql)

    # Ratio/fraction/percent indicators
    ratio_count = sum(1 for kw in RATIO_KEYWORDS if kw in ql)

    # Sequence words
    seq_count = sum(1 for kw in SEQUENCE_WORDS if kw in ql)

    # Question phrases
    qphrase_count = sum(1 for kw in QUESTION_PHRASES if kw in ql)

    # Entity count (possessive nouns)
    entity_count = len(ENTITY_PATTERN.findall(q))

    # Average number magnitude
    avg_magnitude = np.mean(numbers) if numbers else 0.0

    # Commas, semicolons
    comma_count = q.count(",")
    semicolon_count = q.count(";")

    # Collective keywords
    collective_count = sum(1 for kw in COLLECTIVE_KEYWORDS if kw in ql)

    # Has specific individual keywords (binary features)
    has_each = int("each" in ql.split())
    has_per = int(" per " in f" {ql} ")
    has_every = int("every" in ql.split())
    has_total = int("total" in ql.split())
    has_combined = int("combined" in ql)
    has_altogether = int("altogether" in ql)

    # Number of digits (distinct from numeric tokens)
    digit_count = sum(c.isdigit() for c in q)

    # Presence of common math operators in text
    has_plus = int(" plus " in f" {ql} ")
    has_minus_text = int(" minus " in f" {ql} ")
    has_times = int(" times " in f" {ql} ")

    return {
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
        "num_count": num_count,
        "unique_num_count": unique_num_count,
        "verb_count": verb_count,
        "comp_count": comp_count,
        "ratio_count": ratio_count,
        "seq_count": seq_count,
        "qphrase_count": qphrase_count,
        "entity_count": entity_count,
        "avg_magnitude": avg_magnitude,
        "comma_count": comma_count,
        "semicolon_count": semicolon_count,
        "collective_count": collective_count,
        "has_each": has_each,
        "has_per": has_per,
        "has_every": has_every,
        "has_total": has_total,
        "has_combined": has_combined,
        "has_altogether": has_altogether,
        "digit_count": digit_count,
        "has_plus": has_plus,
        "has_minus_text": has_minus_text,
        "has_times_text": has_times,
    }


FEATURE_NAMES = list(engineer_features("test 42").keys())


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Create feature matrix from DataFrame with 'question' column."""
    rows = []
    for _, row in df.iterrows():
        feats = engineer_features(row["question"])
        rows.append([feats[k] for k in FEATURE_NAMES])
    return np.array(rows)


# ---------------------------------------------------------------------------
# 4. Cascade classifiers
# ---------------------------------------------------------------------------
class BinaryCascadeClassifier(BaseEstimator, ClassifierMixin):
    """Predict step count via a cascade of binary classifiers:
    Stage 1: 1 step vs 2+ steps
    Stage 2: among 2+, 2 steps vs 3+ steps
    Stage 3: among 3+, 3 steps vs 4+ steps
    etc.

    Uses DecisionTree as the base for each stage.
    """

    def __init__(self, min_samples_leaf=5, max_depth=6):
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.stages = []

    def fit(self, X, y):
        unique_labels = sorted(np.unique(y))
        self.classes_ = np.array(unique_labels)
        self.stages = []
        # Build cascade: at each level separate the smallest label from the rest
        for i, label in enumerate(unique_labels[:-1]):
            y_bin = np.where(y <= label, 0, 1)  # 0 = this label, 1 = higher
            clf = DecisionTreeClassifier(
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth,
                random_state=42,
            )
            clf.fit(X, y_bin)
            self.stages.append((label, clf))
        return self

    def predict(self, X):
        preds = []
        for x in X:
            x = x.reshape(1, -1)
            # Start from highest label
            label = self.classes_[-1]
            for stage_label, clf in self.stages:
                p = clf.predict(x)[0]
                if p == 0:
                    label = stage_label
                    break
            preds.append(label)
        return np.array(preds)

    def predict_proba(self, X):
        # Fallback — not critical for our eval
        preds = self.predict(X)
        n_classes = len(self.classes_)
        probs = np.zeros((len(X), n_classes))
        for i, p in enumerate(preds):
            idx = np.where(self.classes_ == p)[0][0]
            probs[i, idx] = 1.0
        return probs


# ---------------------------------------------------------------------------
# 5. Main evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Per-class metrics
    unique_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
    p, r, f1, s = precision_recall_fscore_support(
        y_test, y_pred, labels=unique_labels, zero_division=0
    )

    per_class = {}
    for i, label in enumerate(unique_labels):
        per_class[int(label)] = {
            "precision": round(p[i], 4),
            "recall": round(r[i], 4),
            "f1": round(f1[i], 4),
            "support": int(s[i]),
        }

    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

    return {
        "model": model_name,
        "accuracy": round(acc, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": [int(l) for l in unique_labels],
    }


def print_results(results: list, X_test, y_test, baselines: dict):
    print("=" * 80)
    print("GSM8K Step Count Classification — Model Comparison Report")
    print("=" * 80)

    print("\n--- Data Summary ---")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples:  {len(X_test)}")
    print(f"  Features:      {len(FEATURE_NAMES)}")
    print(f"  Step range:    {int(y_train.min())}–{int(y_train.max())}")
    print(f"  Feature names: {FEATURE_NAMES}")

    print("\n--- Baselines ---")
    for name, acc in baselines.items():
        print(f"  {name:40s}  Accuracy: {acc:.4f}")

    print("\n--- Model Accuracy ---")
    for r_ in results:
        print(f"  {r_['model']:40s}  Accuracy: {r_['accuracy']:.4f}")

    print("\n--- Per-class Metrics ---\n")
    for r_ in results:
        print(f"  [{r_['model']}] (Accuracy: {r_['accuracy']:.4f})")
        print(f"  {'Steps':>6s}  {'Prec':>8s}  {'Recall':>8s}  {'F1':>8s}  {'Support':>8s}")
        for label in sorted(r_["per_class"].keys()):
            m = r_["per_class"][label]
            print(f"  {label:>6d}  {m['precision']:>8.4f}  {m['recall']:>8.4f}  {m['f1']:>8.4f}  {m['support']:>8d}")
        print()

    print("\n--- Confusion Matrices ---\n")
    for r_ in results:
        print(f"  [{r_['model']}]")
        labels = r_["labels"]
        cm = np.array(r_["confusion_matrix"])
        header = "      " + "".join(f"{l:>6d}" for l in labels)
        print(f"  {header}")
        for i, l in enumerate(labels):
            row = f"  {l:>4d} " + "".join(f"{cm[i, j]:>6d}" for j in range(len(labels)))
            print(row)
        print()

    print("=" * 80)
    print("Models saved to:", MODEL_DIR)
    print("Report saved to:", REPORT_DIR / "step_classifier_report.md")


def save_report(results: list, baselines: dict, best_model_name: str):
    lines = []
    lines.append("# GSM8K Step Count Classifier — Evaluation Report")
    lines.append("")
    lines.append(f"Generated by `scripts/train_step_classifier.py`")
    lines.append("")
    lines.append("## Data Summary")
    lines.append(f"- Train samples: {len(X_train)}")
    lines.append(f"- Test samples:  {len(X_test)}")
    lines.append(f"- Features:      {len(FEATURE_NAMES)}")
    lines.append(f"- Step range:    {int(y_train.min())}–{int(y_train.max())}")
    lines.append(f"- Features: `{'`, `'.join(FEATURE_NAMES)}`")
    lines.append("")
    lines.append("## Baselines")
    for name, acc in baselines.items():
        lines.append(f"- **{name}**: {acc:.4f}")
    lines.append("")
    lines.append("## Model Accuracy")
    lines.append("")
    lines.append("| Model | Accuracy | vs Baseline |")
    lines.append("|-------|----------|-------------|")
    highest_acc = max(r["accuracy"] for r in results)
    best_baseline = max(baselines.values())
    for r_ in sorted(results, key=lambda x: -x["accuracy"]):
        vs_base = f"+{r_['accuracy'] - best_baseline:.4f}" if r_['accuracy'] > best_baseline else f"{r_['accuracy'] - best_baseline:.4f}"
        marker = " ★" if r_["model"] == best_model_name else ""
        lines.append(f"| {r_['model']:<40s} | {r_['accuracy']:.4f}{marker} | {vs_base} |")
    lines.append("")
    lines.append("## Per-class Metrics")
    lines.append("")
    for r_ in results:
        lines.append(f"### {r_['model']} (Accuracy: {r_['accuracy']:.4f})")
        lines.append("")
        lines.append("| Steps | Precision | Recall | F1 | Support |")
        lines.append("|-------|-----------|--------|----|---------|")
        for label in sorted(r_["per_class"].keys()):
            m = r_["per_class"][label]
            lines.append(f"| {label} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['support']} |")
        lines.append("")
    lines.append("## Confusion Matrices")
    lines.append("")
    for r_ in results:
        lines.append(f"### {r_['model']}")
        lines.append("")
        labels = r_["labels"]
        cm = np.array(r_["confusion_matrix"])
        header = "| | " + " | ".join(f"Pred {l}" for l in labels) + " |"
        sep = "|---" + "|---" * len(labels) + "|"
        lines.append(header)
        lines.append(sep)
        for i, l in enumerate(labels):
            row = f"| True {l} | " + " | ".join(str(cm[i, j]) for j in range(len(labels))) + " |"
            lines.append(row)
        lines.append("")
    lines.append("## Feature Names")
    lines.append(f"`{'`, `'.join(FEATURE_NAMES)}`")
    lines.append("")
    report = "\n".join(lines)
    report_path = REPORT_DIR / "step_classifier_report.md"
    report_path.write_text(report)
    print(f"Report saved to {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading GSM8K data...")
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    # Extract ground-truth step counts
    print("Extracting step counts...")
    y_train = np.array(train_df["answer"].apply(extract_step_count))
    y_test = np.array(test_df["answer"].apply(extract_step_count))

    print(f"Train: {len(y_train)} samples, {y_train.min()}–{y_train.max()} steps")
    print(f"Test:  {len(y_test)} samples, {y_test.min()}–{y_test.max()} steps")
    print(f"Train class distribution: {pd.Series(y_train).value_counts().sort_index().to_dict()}")
    print(f"Test class distribution:  {pd.Series(y_test).value_counts().sort_index().to_dict()}")

    # Engineer features
    print("\nEngineering features...")
    X_train = feature_matrix(train_df)
    X_test = feature_matrix(test_df)

    print(f"Feature matrix shape: train={X_train.shape}, test={X_test.shape}")

    # Baselines
    print("\nComputing baselines...")
    mode_train = pd.Series(y_train).mode()[0]
    # Weighted random baseline based on train distribution
    train_dist = pd.Series(y_train).value_counts(normalize=True)

    baselines = {}
    # Baseline 1: always predict mode
    y_pred_mode = np.full_like(y_test, mode_train)
    baselines["Always-predict-mode (most common)"] = accuracy_score(y_test, y_pred_mode)

    # Baseline 2: always predict 2 (common default)
    y_pred_2 = np.full_like(y_test, 2)
    baselines["Always-predict-2"] = accuracy_score(y_test, y_pred_2)

    # Baseline 3: always predict 3 (another common)
    y_pred_3 = np.full_like(y_test, 3)
    baselines["Always-predict-3"] = accuracy_score(y_test, y_pred_3)

    # Baseline 4: weighted random sampling from train distribution
    rng = np.random.RandomState(42)
    y_pred_random = rng.choice(train_dist.index, size=len(y_test), p=train_dist.values)
    baselines["Weighted-random (train dist)"] = accuracy_score(y_test, y_pred_random)

    print(f"  Always-predict-mode ({mode_train}): {baselines['Always-predict-mode (most common)']:.4f}")
    print(f"  Always-predict-2:            {baselines['Always-predict-2']:.4f}")
    print(f"  Always-predict-3:            {baselines['Always-predict-3']:.4f}")
    print(f"  Weighted-random:             {baselines['Weighted-random (train dist)']:.4f}")

    # -----------------------------------------------------------------------
    # Train classifiers
    # -----------------------------------------------------------------------
    results = []
    best_acc = -1
    best_model = None
    best_name = ""

    # ---- Logistic Regression ----
    print("\n--- Logistic Regression ---")
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=2000,
            C=1.0,
            solver="lbfgs",
            random_state=42,
        )),
    ])
    lr_pipe.fit(X_train, y_train)
    lr_result = evaluate_model(lr_pipe, X_test, y_test, "Logistic Regression (multinomial)")
    results.append(lr_result)
    print(f"  Accuracy: {lr_result['accuracy']:.4f}")
    if lr_result["accuracy"] > best_acc:
        best_acc = lr_result["accuracy"]
        best_model = lr_pipe
        best_name = "Logistic Regression (multinomial)"

    # ---- Decision Tree ----
    print("\n--- Decision Tree ---")
    dt = DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=10,
        min_samples_split=20,
        random_state=42,
    )
    dt.fit(X_train, y_train)
    dt_result = evaluate_model(dt, X_test, y_test, "Decision Tree (pruned)")
    results.append(dt_result)
    print(f"  Accuracy: {dt_result['accuracy']:.4f}")
    if dt_result["accuracy"] > best_acc:
        best_acc = dt_result["accuracy"]
        best_model = dt
        best_name = "Decision Tree (pruned)"

    # ---- Random Forest ----
    print("\n--- Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_result = evaluate_model(rf, X_test, y_test, "Random Forest (100 trees)")
    results.append(rf_result)
    print(f"  Accuracy: {rf_result['accuracy']:.4f}")
    if rf_result["accuracy"] > best_acc:
        best_acc = rf_result["accuracy"]
        best_model = rf
        best_name = "Random Forest (100 trees)"

    # ---- XGBoost ----
    xgb_available = False
    xgb_result = None
    print("\n--- XGBoost ---")
    try:
        import xgboost as xgb
        xgb_available = True
    except ImportError:
        print("  XGBoost not available, trying to install...")
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "xgboost", "--break-system-packages", "-q"],
            capture_output=True,
        )
        try:
            import xgboost as xgb
            xgb_available = True
        except ImportError:
            print("  XGBoost still not available after install attempt")

    if xgb_available:
        xgb_clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss",
            use_label_encoder=False,
        )
        xgb_clf.fit(X_train, y_train)
        xgb_result = evaluate_model(xgb_clf, X_test, y_test, "XGBoost")
        results.append(xgb_result)
        print(f"  Accuracy: {xgb_result['accuracy']:.4f}")
        if xgb_result["accuracy"] > best_acc:
            best_acc = xgb_result["accuracy"]
            best_model = xgb_clf
            best_name = "XGBoost"
    else:
        print("  Skipped — not available")

    # ---- Binary Cascade Classifier ----
    print("\n--- Binary Cascade Classifier ---")
    cascade = BinaryCascadeClassifier(min_samples_leaf=5, max_depth=6)
    cascade.fit(X_train, y_train)
    cascade_result = evaluate_model(cascade, X_test, y_test, "Binary Cascade (DT-based)")
    results.append(cascade_result)
    print(f"  Accuracy: {cascade_result['accuracy']:.4f}")

    # ---- Evaluate with 5-fold cross-validation on training set ----
    print("\n--- 5-Fold Cross-Validation (on train set) ---")
    for clf, name in [
        (lr_pipe, "Logistic Regression"),
        (dt, "Decision Tree"),
        (rf, "Random Forest"),
    ]:
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
        print(f"  {name:30s}  Mean: {cv_scores.mean():.4f}  Std: {cv_scores.std():.4f}")

    # ---- Feature importance (Random Forest) ----
    print("\n--- Random Forest Feature Importances ---")
    rf_importances = sorted(
        zip(FEATURE_NAMES, rf.feature_importances_),
        key=lambda x: -x[1],
    )
    for name, imp in rf_importances[:15]:
        print(f"  {name:25s}  {imp:.4f}")

    # ---- Save best model ----
    if best_model is not None:
        import joblib
        model_path = MODEL_DIR / "step_count_model.joblib"
        joblib.dump(best_model, model_path)
        print(f"\nBest model saved: {model_path} ({best_name}, acc={best_acc:.4f})")

    # ---- Print full results ----
    print_results(results, X_test, y_test, baselines)

    # ---- Save report ----
    save_report(results, baselines, best_name)
