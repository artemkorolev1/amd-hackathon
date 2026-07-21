"""Retrain learned judge using feature extractors from agent/judge.py for pickle compat."""
import json, os, pickle, sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

HERE = "/home/artem/dev/amd-hackathon"
sys.path.insert(0, HERE)
from agent.judge import _StructuralFeatureExtractor, _TextFeatureExtractor

DATA_PATH = os.path.join(HERE, "data", "judge_training.jsonl")
MODEL_PATH = os.path.join(HERE, "models", "learned_judge.pkl")
VEC_PATH = os.path.join(HERE, "models", "judge_vectorizer.pkl")

print("Loading data...")
records = []
with open(DATA_PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

valid = [r for r in records if r.get("answer") and r.get("expected")]
print(f"With answer + expected: {len(valid)}")

y = np.array([1 if r["correct"] else 0 for r in valid])
print(f"Positive: {y.sum()} | Negative: {len(y) - y.sum()}")

train_recs, test_recs, y_train, y_test = train_test_split(
    valid, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(train_recs)} | Test: {len(test_recs)}")

struct = _StructuralFeatureExtractor()
txt = _TextFeatureExtractor()

struct.fit(train_recs)
txt.fit(train_recs)

X_train = np.hstack([struct.transform(train_recs), txt.transform(train_recs)])
X_test = np.hstack([struct.transform(test_recs), txt.transform(test_recs)])
print(f"Features: {X_train.shape[1]}")

clf = RandomForestClassifier(
    n_estimators=200, max_depth=20, min_samples_leaf=5,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
clf.fit(X_train, y_train)

print(f"\nTrain acc: {np.mean(clf.predict(X_train) == y_train):.3f}")
print(f"Test acc:  {np.mean(clf.predict(X_test) == y_test):.3f}")
print(f"Test AUC:  {roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]):.3f}")
print(f"\n{classification_report(y_test, clf.predict(X_test), target_names=['incorrect','correct'])}")
print(confusion_matrix(y_test, clf.predict(X_test)))

with open(MODEL_PATH, "wb") as f:
    pickle.dump(clf, f)
with open(VEC_PATH, "wb") as f:
    pickle.dump({"struct": struct, "txt": txt}, f)
print(f"Saved to {MODEL_PATH}")
