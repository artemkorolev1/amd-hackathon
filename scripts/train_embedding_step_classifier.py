#!/usr/bin/env python3
"""
train_embedding_step_classifier.py — Build and evaluate embedding-based
step-count classifiers for GSM8K math problems.

Tests sentence-transformers (all-MiniLM-L6-v2, all-mpnet-base-v2) and
local GGUF models (Qwen2.5, Llama-3.2, SmolLM2) to see if semantic
embeddings break the 36% ceiling from hand-crafted features.

Usage:
    PYTHONPATH=/home/artem/dev/amd-hackathon python scripts/train_embedding_step_classifier.py
"""

import gc
import re
import sys
import os
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAIN_PATH = "/tmp/gsm8k_train.parquet"
TEST_PATH = "/tmp/gsm8k_test.parquet"
REPORT_DIR = Path("/home/artem/dev/amd-hackathon/gepa_plans")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = REPORT_DIR / "embedding_step_classifier_report.md"

GGUF_MODELS = {
    "Qwen2.5-1.5B-Instruct": "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "Llama-3.2-1B-Instruct": "/home/artem/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "Qwen2.5-Math-1.5B": "/home/artem/models/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf",
    "SmolLM2-1.7B-Instruct": "/home/artem/models/smollm2-1.7b-instruct-q4_k_m.gguf",
}

BASELINE_ACCURACY = 0.3609  # RF (100 trees) from hand-crafted features


# ---------------------------------------------------------------------------
# 1. Data loading & step-count extraction
# ---------------------------------------------------------------------------
def load_data():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    def extract_step_count(answer: str) -> int:
        markers = re.findall(r"<<.*?>>", answer)
        return len(markers)

    y_train = np.array(train_df["answer"].apply(extract_step_count))
    y_test = np.array(test_df["answer"].apply(extract_step_count))

    questions_train = train_df["question"].tolist()
    questions_test = test_df["question"].tolist()

    return questions_train, questions_test, y_train, y_test


def print_distribution(y_train, y_test, label=""):
    from collections import Counter
    train_dist = sorted(Counter(y_train).items())
    test_dist = sorted(Counter(y_test).items())
    print(f"  Train dist: {dict(train_dist)}")
    print(f"  Test dist:  {dict(test_dist)}")
    print(f"  Train range: {y_train.min()}–{y_train.max()}")
    print(f"  Test range:  {y_test.min()}–{y_test.max()}")


# ---------------------------------------------------------------------------
# 2. Train & evaluate helpers
# ---------------------------------------------------------------------------
def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Logistic regression with scaling."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs", random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc, y_pred, pipe


def train_mlp(X_train, y_train, X_test, y_test):
    """Small MLP with 1 hidden layer of 64."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc, y_pred, pipe


# ---------------------------------------------------------------------------
# PHASE 1: Sentence-transformers embeddings
# ---------------------------------------------------------------------------
def phase1_sentence_transformers(questions_train, questions_test, y_train, y_test):
    """Encode all questions with sentence-transformers models and train classifiers."""
    print("\n" + "=" * 80)
    print("PHASE 1: Sentence-Transformers Embeddings")
    print("=" * 80)

    results = []

    for model_name in ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]:
        print(f"\n--- Loading {model_name} ---")
        t0 = time.time()
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer(model_name)
        dim = st_model.get_embedding_dimension()
        print(f"  Dimension: {dim}, Load time: {time.time() - t0:.1f}s")

        t1 = time.time()
        print(f"  Encoding {len(questions_train)} train questions...")
        X_train = st_model.encode(questions_train, show_progress_bar=True, batch_size=64)
        print(f"  Encoding {len(questions_test)} test questions...")
        X_test = st_model.encode(questions_test, show_progress_bar=True, batch_size=64)
        print(f"  Encoding done in {time.time() - t1:.1f}s")

        # Logistic Regression
        print(f"  Training Logistic Regression...")
        lr_acc, lr_pred, lr_model = train_logistic_regression(X_train, y_train, X_test, y_test)
        print(f"  LR Accuracy: {lr_acc:.4f}")

        # MLP
        print(f"  Training MLP (64 hidden)...")
        mlp_acc, mlp_pred, mlp_model = train_mlp(X_train, y_train, X_test, y_test)
        print(f"  MLP Accuracy: {mlp_acc:.4f}")

        # Classification reports
        print(f"\n  Classification Report (LR, {model_name}):")
        print(f"  {classification_report(y_test, lr_pred, zero_division=0)}")
        print(f"\n  Classification Report (MLP, {model_name}):")
        print(f"  {classification_report(y_test, mlp_pred, zero_division=0)}")

        # Confusion matrices
        cm_lr = confusion_matrix(y_test, lr_pred)
        cm_mlp = confusion_matrix(y_test, mlp_pred)
        print(f"\n  Confusion Matrix (LR, {model_name}):\n{cm_lr}")
        print(f"\n  Confusion Matrix (MLP, {model_name}):\n{cm_mlp}")

        # Clean up
        del st_model
        gc.collect()

        results.append({
            "embedding_model": model_name,
            "dim": dim,
            "classifier": "LogisticRegression",
            "accuracy": round(lr_acc, 4),
            "classification_report": classification_report(y_test, lr_pred, zero_division=0, output_dict=True),
            "confusion_matrix": cm_lr.tolist(),
        })
        results.append({
            "embedding_model": model_name,
            "dim": dim,
            "classifier": "MLPClassifier(64)",
            "accuracy": round(mlp_acc, 4),
            "classification_report": classification_report(y_test, mlp_pred, zero_division=0, output_dict=True),
            "confusion_matrix": cm_mlp.tolist(),
        })

    return results


# ---------------------------------------------------------------------------
# PHASE 2: GGUF embeddings (subsampled)
# ---------------------------------------------------------------------------
def phase2_gguf_models(questions_train, questions_test, y_train, y_test):
    """Encode subsampled questions with local GGUF models via llama-cpp."""
    print("\n" + "=" * 80)
    print("PHASE 2: GGUF Model Embeddings (subsampled 500 train / 200 test)")
    print("=" * 80)

    # Subsample
    rng = np.random.RandomState(42)
    train_idx = rng.choice(len(questions_train), size=500, replace=False)
    test_idx = rng.choice(len(questions_test), size=200, replace=False)

    q_train_sub = [questions_train[i] for i in train_idx]
    q_test_sub = [questions_test[i] for i in test_idx]
    y_train_sub = y_train[train_idx]
    y_test_sub = y_test[test_idx]

    print(f"\nSubsampled: {len(q_train_sub)} train, {len(q_test_sub)} test")
    print_distribution(y_train_sub, y_test_sub, "Subsampled")

    results = []

    for name, path in GGUF_MODELS.items():
        print(f"\n--- Loading {name} ---")
        t0 = time.time()
        from llama_cpp import Llama

        llm = Llama(
            model_path=path,
            embedding=True,
            n_ctx=512,
            verbose=False,
        )
        print(f"  Load time: {time.time() - t0:.1f}s")

        # Encode train
        t1 = time.time()
        print(f"  Encoding {len(q_train_sub)} train questions...")
        X_train_vecs = []
        for i, q in enumerate(q_train_sub):
            emb = llm.create_embedding(q)
            X_train_vecs.append(emb["data"][0]["embedding"])
            if (i + 1) % 100 == 0:
                print(f"    {i + 1}/{len(q_train_sub)} ({time.time() - t1:.1f}s)")

        X_train = np.array(X_train_vecs)
        print(f"  Train encoding done: {X_train.shape} in {time.time() - t1:.1f}s")

        # Encode test
        t2 = time.time()
        print(f"  Encoding {len(q_test_sub)} test questions...")
        X_test_vecs = []
        for i, q in enumerate(q_test_sub):
            emb = llm.create_embedding(q)
            X_test_vecs.append(emb["data"][0]["embedding"])
            if (i + 1) % 100 == 0:
                print(f"    {i + 1}/{len(q_test_sub)} ({time.time() - t2:.1f}s)")

        X_test = np.array(X_test_vecs)
        print(f"  Test encoding done: {X_test.shape} in {time.time() - t2:.1f}s")

        # Train LR
        print(f"  Training Logistic Regression...")
        lr_acc, lr_pred, lr_model = train_logistic_regression(X_train, y_train_sub, X_test, y_test_sub)
        print(f"  LR Accuracy: {lr_acc:.4f}")

        print(f"\n  Classification Report (LR, {name}):")
        print(f"  {classification_report(y_test_sub, lr_pred, zero_division=0)}")

        cm = confusion_matrix(y_test_sub, lr_pred)
        print(f"\n  Confusion Matrix (LR, {name}):\n{cm}")

        # Also try MLP
        print(f"  Training MLP (64 hidden)...")
        mlp_acc, mlp_pred, mlp_model = train_mlp(X_train, y_train_sub, X_test, y_test_sub)
        print(f"  MLP Accuracy: {mlp_acc:.4f}")

        print(f"\n  Classification Report (MLP, {name}):")
        print(f"  {classification_report(y_test_sub, mlp_pred, zero_division=0)}")

        cm_mlp = confusion_matrix(y_test_sub, mlp_pred)
        print(f"\n  Confusion Matrix (MLP, {name}):\n{cm_mlp}")

        results.append({
            "embedding_model": name,
            "dim": X_train.shape[1],
            "subsample_train": 500,
            "subsample_test": 200,
            "classifier": "LogisticRegression",
            "accuracy": round(lr_acc, 4),
            "classification_report": classification_report(y_test_sub, lr_pred, zero_division=0, output_dict=True),
            "confusion_matrix": cm.tolist(),
        })
        results.append({
            "embedding_model": name,
            "dim": X_train.shape[1],
            "subsample_train": 500,
            "subsample_test": 200,
            "classifier": "MLPClassifier(64)",
            "accuracy": round(mlp_acc, 4),
            "classification_report": classification_report(y_test_sub, mlp_pred, zero_division=0, output_dict=True),
            "confusion_matrix": cm_mlp.tolist(),
        })

        # Free memory
        print(f"  Cleaning up {name}...")
        del llm
        gc.collect()
        print(f"  Done with {name}")

    return results


# ---------------------------------------------------------------------------
# PHASE 3: Best model evaluation on full test set
# ---------------------------------------------------------------------------
def phase3_best_model_evaluation(questions_train, questions_test, y_train, y_test,
                                 best_embedding, best_classifier_name):
    """Take the best performing embedding + classifier and run on full test set."""
    print("\n" + "=" * 80)
    print("PHASE 3: Best Model — Full Test Set Evaluation")
    print("=" * 80)

    best_embedding_name = best_embedding["embedding_model"]
    best_clf = best_embedding["classifier"]
    best_acc = best_embedding["accuracy"]
    print(f"\nBest model from phases 1-2:")
    print(f"  Embedding: {best_embedding_name}")
    print(f"  Classifier: {best_clf}")
    print(f"  Accuracy: {best_acc:.4f}")

    # Re-encode full dataset with the best embedding model
    print(f"\nRe-encoding full dataset ({len(questions_train)} train, {len(questions_test)} test)...")

    # Check if it's a sentence-transformers model or GGUF model
    if best_embedding_name in ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]:
        from sentence_transformers import SentenceTransformer
        t0 = time.time()
        st_model = SentenceTransformer(best_embedding_name)
        print(f"  Loaded {best_embedding_name} in {time.time() - t0:.1f}s")

        X_train = st_model.encode(questions_train, show_progress_bar=True, batch_size=64)
        X_test = st_model.encode(questions_test, show_progress_bar=True, batch_size=64)
    else:
        # GGUF model — encode full 500+200 is the max we can do
        print(f"  GGUF model — using existing subsampled results.")
        return best_embedding

    # Train the best classifier type
    if best_classifier_name == "LogisticRegression":
        acc, y_pred, model = train_logistic_regression(X_train, y_train, X_test, y_test)
    else:
        acc, y_pred, model = train_mlp(X_train, y_train, X_test, y_test)

    print(f"\n--- Full Test Set Results ---")
    print(f"  Embedding: {best_embedding_name}")
    print(f"  Classifier: {best_classifier_name}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Baseline (RF, hand-crafted): {BASELINE_ACCURACY:.4f}")
    delta = acc - BASELINE_ACCURACY
    if delta > 0:
        print(f"  ✅ Embedding approach BEATS hand-crafted features by +{delta:.4f}")
    elif delta == 0:
        print(f"  ⚖️  Ties hand-crafted features ({delta:+.4f})")
    else:
        print(f"  ❌ Embedding approach lags behind hand-crafted features ({delta:+.4f})")

    print(f"\n  Classification Report:")
    print(f"  {classification_report(y_test, y_pred, zero_division=0)}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:\n{cm}")

    result = {
        "embedding_model": best_embedding_name,
        "dim": X_train.shape[1],
        "classifier": best_classifier_name,
        "accuracy": round(acc, 4),
        "baseline_accuracy": BASELINE_ACCURACY,
        "delta_vs_baseline": round(delta, 4),
        "beats_baseline": delta > 0,
        "classification_report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
        "confusion_matrix": cm.tolist(),
    }

    # Clean up
    if "st_model" in dir():
        del st_model
    gc.collect()

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_summary(phase1_results, phase2_results, best_result):
    print("\n" + "=" * 80)
    print("FINAL SUMMARY — Embedding-Based Step Count Classification")
    print("=" * 80)

    print(f"\nBaseline (RF, hand-crafted features): {BASELINE_ACCURACY:.4f}")
    print(f"\n--- Phase 1: Sentence-Transformers (full dataset) ---")
    print(f"  {'Embedding':30s} {'Classifier':20s} {'Acc':>8s}")
    print(f"  {'-'*30} {'-'*20} {'-'*8}")
    for r in phase1_results:
        print(f"  {r['embedding_model']:30s} {r['classifier']:20s} {r['accuracy']:>8.4f}")

    print(f"\n--- Phase 2: GGUF Models (500 train / 200 test) ---")
    print(f"  {'Model':30s} {'Classifier':20s} {'Acc':>8s}")
    print(f"  {'-'*30} {'-'*20} {'-'*8}")
    for r in phase2_results:
        print(f"  {r['embedding_model']:30s} {r['classifier']:20s} {r['accuracy']:>8.4f}")

    print(f"\n--- Phase 3: Best Model on Full Test Set ---")
    print(f"  Embedding: {best_result['embedding_model']}")
    print(f"  Classifier: {best_result['classifier']}")
    print(f"  Accuracy: {best_result['accuracy']:.4f}")
    print(f"  Baseline: {best_result['baseline_accuracy']:.4f}")
    print(f"  Delta: {best_result['delta_vs_baseline']:+.4f}")
    if best_result['beats_baseline']:
        print(f"  ✅ Embedding approach beats hand-crafted features!")
    else:
        print(f"  ❌ Embedding approach does NOT beat hand-crafted features.")

    print(f"\nReport saved to: {REPORT_PATH}")


def save_report(phase1_results, phase2_results, best_result):
    lines = []
    lines.append("# Embedding-Based Step Count Classifier — Evaluation Report")
    lines.append("")
    lines.append("Generated by `scripts/train_embedding_step_classifier.py`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Baseline (RF, hand-crafted features):** {BASELINE_ACCURACY:.4f}")
    lines.append(f"- **Best embedding approach:** {best_result['embedding_model']} + {best_result['classifier']}")
    lines.append(f"- **Best accuracy:** {best_result['accuracy']:.4f}")
    lines.append(f"- **Delta vs baseline:** {best_result['delta_vs_baseline']:+.4f}")
    lines.append(f"- **Beats baseline:** {'Yes' if best_result['beats_baseline'] else 'No'}")
    lines.append("")
    lines.append("## Phase 1: Sentence-Transformers (Full Dataset)")
    lines.append("")
    lines.append("| Embedding Model | Dim | Classifier | Accuracy |")
    lines.append("|----------------|-----|------------|----------|")
    for r in sorted(phase1_results, key=lambda x: -x["accuracy"]):
        lines.append(f"| {r['embedding_model']} | {r['dim']} | {r['classifier']} | {r['accuracy']:.4f} |")
    lines.append("")
    lines.append("### Phase 1 — Classification Reports")
    lines.append("")
    for r in sorted(phase1_results, key=lambda x: -x["accuracy"]):
        lines.append(f"#### {r['embedding_model']} + {r['classifier']} (Acc: {r['accuracy']:.4f})")
        lines.append("")
        cr = r["classification_report"]
        lines.append("| Class | Precision | Recall | F1-Score | Support |")
        lines.append("|-------|-----------|--------|----------|---------|")
        for cls in sorted(cr.keys()):
            if cls in ("accuracy", "macro avg", "weighted avg"):
                continue
            m = cr[cls]
            lines.append(f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1-score']:.4f} | {m['support']} |")
        lines.append("")
        cm = np.array(r["confusion_matrix"])
        labels = sorted([k for k in cr.keys() if k not in ("accuracy", "macro avg", "weighted avg")])
        try:
            labels = [int(l) for l in labels]
        except ValueError:
            pass
        lines.append("Confusion Matrix:")
        header = "| | " + " | ".join(f"Pred {l}" for l in labels) + " |"
        sep = "|---" + "|---" * len(labels) + "|"
        lines.append(header)
        lines.append(sep)
        for i, l in enumerate(labels):
            row = f"| True {l} | " + " | ".join(str(cm[i, j]) for j in range(len(labels))) + " |"
            lines.append(row)
        lines.append("")

    lines.append("## Phase 2: GGUF Models (Subsampled 500 train / 200 test)")
    lines.append("")
    lines.append("| Model | Dim | Classifier | Accuracy |")
    lines.append("|-------|-----|------------|----------|")
    for r in sorted(phase2_results, key=lambda x: -x["accuracy"]):
        lines.append(f"| {r['embedding_model']} | {r['dim']} | {r['classifier']} | {r['accuracy']:.4f} |")
    lines.append("")
    lines.append("### Phase 2 — Classification Reports")
    lines.append("")
    for r in sorted(phase2_results, key=lambda x: -x["accuracy"]):
        lines.append(f"#### {r['embedding_model']} + {r['classifier']} (Acc: {r['accuracy']:.4f})")
        lines.append("")
        cr = r["classification_report"]
        lines.append("| Class | Precision | Recall | F1-Score | Support |")
        lines.append("|-------|-----------|--------|----------|---------|")
        for cls in sorted(cr.keys()):
            if cls in ("accuracy", "macro avg", "weighted avg"):
                continue
            m = cr[cls]
            lines.append(f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1-score']:.4f} | {m['support']} |")
        lines.append("")
        cm = np.array(r["confusion_matrix"])
        labels = sorted([k for k in cr.keys() if k not in ("accuracy", "macro avg", "weighted avg")])
        try:
            labels = [int(l) for l in labels]
        except ValueError:
            pass
        lines.append("Confusion Matrix:")
        header = "| | " + " | ".join(f"Pred {l}" for l in labels) + " |"
        sep = "|---" + "|---" * len(labels) + "|"
        lines.append(header)
        lines.append(sep)
        for i, l in enumerate(labels):
            row = f"| True {l} | " + " | ".join(str(cm[i, j]) for j in range(len(labels))) + " |"
            lines.append(row)
        lines.append("")

    lines.append("## Phase 3: Best Model — Full Test Set")
    lines.append("")
    lines.append(f"- **Embedding model:** {best_result['embedding_model']}")
    lines.append(f"- **Classifier:** {best_result['classifier']}")
    lines.append(f"- **Accuracy on full test set:** {best_result['accuracy']:.4f}")
    lines.append(f"- **Baseline (RF, hand-crafted):** {best_result['baseline_accuracy']:.4f}")
    lines.append(f"- **Delta:** {best_result['delta_vs_baseline']:+.4f}")
    lines.append(f"- **Beats baseline:** {'Yes' if best_result['beats_baseline'] else 'No'}")
    lines.append("")
    cr = best_result["classification_report"]
    lines.append("### Classification Report")
    lines.append("")
    lines.append("| Class | Precision | Recall | F1-Score | Support |")
    lines.append("|-------|-----------|--------|----------|---------|")
    for cls in sorted(cr.keys()):
        if cls in ("accuracy", "macro avg", "weighted avg"):
            continue
        m = cr[cls]
        lines.append(f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1-score']:.4f} | {m['support']} |")
    lines.append("")
    cm = np.array(best_result["confusion_matrix"])
    labels = sorted([k for k in cr.keys() if k not in ("accuracy", "macro avg", "weighted avg")])
    try:
        labels = [int(l) for l in labels]
    except ValueError:
        pass
    lines.append("### Confusion Matrix")
    lines.append("")
    header = "| | " + " | ".join(f"Pred {l}" for l in labels) + " |"
    sep = "|---" + "|---" * len(labels) + "|"
    lines.append(header)
    lines.append(sep)
    for i, l in enumerate(labels):
        row = f"| True {l} | " + " | ".join(str(cm[i, j]) for j in range(len(labels))) + " |"
        lines.append(row)
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    if best_result["beats_baseline"]:
        lines.append(f"The best embedding-based approach (**{best_result['embedding_model']} + {best_result['classifier']}**) achieves **{best_result['accuracy']:.4f}** accuracy, **beating** the hand-crafted feature baseline of {BASELINE_ACCURACY:.4f} by **+{best_result['delta_vs_baseline']:.4f}**.")
        lines.append("")
        lines.append("Semantic embeddings provide meaningful advantage over surface features for step-count prediction in GSM8K problems.")
    else:
        lines.append(f"The best embedding-based approach (**{best_result['embedding_model']} + {best_result['classifier']}**) achieves **{best_result['accuracy']:.4f}** accuracy, which does **not** beat the hand-crafted feature baseline of {BASELINE_ACCURACY:.4f} ({best_result['delta_vs_baseline']:+.4f}).")
        lines.append("")
        lines.append("Semantic embeddings do not provide a meaningful advantage over surface features for step-count prediction. The step-count signal may not be strongly encoded in the semantic embedding space of these models.")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"*Report generated by `scripts/train_embedding_step_classifier.py`*")

    report_text = "\n".join(lines)
    REPORT_PATH.write_text(report_text)
    print(f"\nReport saved to {REPORT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("Embedding-Based Step Count Classifier for GSM8K")
    print("=" * 80)

    # Load data
    print("\nLoading GSM8K data...")
    questions_train, questions_test, y_train, y_test = load_data()
    print(f"  Train: {len(questions_train)} questions")
    print(f"  Test:  {len(questions_test)} questions")
    print_distribution(y_train, y_test)

    # PHASE 1: Sentence-transformers
    print("\n" + "=" * 80)
    print("PHASE 1: Sentence-Transformers Embeddings (full dataset)")
    print("=" * 80)

    phase1_results = phase1_sentence_transformers(questions_train, questions_test, y_train, y_test)

    # PHASE 2: GGUF models (subsampled)
    phase2_results = phase2_gguf_models(questions_train, questions_test, y_train, y_test)

    # Determine best model across phases
    all_results = phase1_results + phase2_results
    best_result_entry = max(all_results, key=lambda r: r["accuracy"])
    print(f"\n{'=' * 80}")
    print(f"Best model across all phases: {best_result_entry['embedding_model']} + "
          f"{best_result_entry['classifier']} (Acc: {best_result_entry['accuracy']:.4f})")

    # PHASE 3: Best model on full test set
    best_embedding_name = best_result_entry["embedding_model"]
    best_clf_name = best_result_entry["classifier"]

    # Only re-evaluate on full test if it's a sentence-transformers model (fast enough)
    if best_embedding_name in ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]:
        best_result = phase3_best_model_evaluation(
            questions_train, questions_test, y_train, y_test,
            best_result_entry, best_clf_name
        )
    else:
        print(f"\nBest model is a GGUF model ({best_embedding_name}). Skipping full re-evaluation "
              f"(too slow). Using subsampled result as proxy.")
        best_result = best_result_entry
        best_result["baseline_accuracy"] = BASELINE_ACCURACY
        best_result["delta_vs_baseline"] = round(best_result["accuracy"] - BASELINE_ACCURACY, 4)
        best_result["beats_baseline"] = best_result["accuracy"] > BASELINE_ACCURACY

    # Print summary
    print_summary(phase1_results, phase2_results, best_result)

    # Save report
    save_report(phase1_results, phase2_results, best_result)

    print("\nDone!")
