"""
New complexity classifier — MiniLM-L6-v2 + LogisticRegression.

Replaces the old heuristic stage3 scorers with a trained model
that predicts whether a prompt is "complex" or "simple".

Uses a SentenceTransformer to embed the prompt, then passes the
embedding through a LogisticRegression model from scikit-learn.

Model files live at:
  /home/artem/dev/amd-hackathon-shared/classifiers/best_complexity_model/

Lazy-loads the models on first call (cacheable singleton).
"""

import json
import logging
import os
import pickle
import threading

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.environ.get(
    "COMPLEXITY_MODEL_DIR",
    os.path.join(_BASE, "data", "training", "complexity", "best_complexity_model"),
)

# ── Lazy singleton state ──────────────────────────────────────────────
_MODEL = None
_META = None
_EMBEDDER = None
_VECTORIZER = None  # Only used if feature_type == "tfidf"
_LOCK = threading.Lock()


def _load():
    """Lazy-load all model assets once (thread-safe)."""
    global _MODEL, _META, _EMBEDDER, _VECTORIZER
    if _MODEL is not None:
        return

    with _LOCK:
        if _MODEL is not None:
            return

        meta_path = os.path.join(_MODELS_DIR, "metadata.json")
        model_path = os.path.join(_MODELS_DIR, "model.pkl")

        if not os.path.exists(meta_path) or not os.path.exists(model_path):
            logger.warning(
                "Complexity model not found at %s — using fallback (0.5)",
                _MODELS_DIR,
            )
            _MODEL = "unavailable"
            _META = {"feature_type": "fallback"}
            return

        with open(meta_path) as f:
            _META = json.load(f)

        logger.info(
            "Loading complexity model: %s (type=%s)",
            _META.get("name", "unknown"),
            _META.get("feature_type", "unknown"),
        )

        _MODEL = pickle.load(open(model_path, "rb"))

        if _META.get("feature_type") == "tfidf":
            vec_path = os.path.join(_MODELS_DIR, "vectorizer.pkl")
            _VECTORIZER = pickle.load(open(vec_path, "rb"))
        else:
            from sentence_transformers import SentenceTransformer
            embedder_name = _META.get("embedder", "all-MiniLM-L6-v2")
            _EMBEDDER = SentenceTransformer(embedder_name)

        logger.info("Complexity model loaded")


def score(prompt: str) -> float:
    """Predict complexity score (0.0 = simple, 1.0 = complex).

    This is a TASK-AGNOSTIC scorer — unlike the old stage3 scorers,
    this model was trained on diverse prompts and doesn't need a
    category parameter.

    Returns:
        float: Probability of being 'complex' (class 1).
    """
    _load()

    if _MODEL == "unavailable":
        logger.warning("Complexity model unavailable — returning 0.5")
        return 0.5

    texts = [prompt]

    if _VECTORIZER is not None:
        # TF-IDF feature type
        X = _VECTORIZER.transform(texts)
        probs = _MODEL.predict_proba(X)[:, 1]
    elif _EMBEDDER is not None:
        # Sentence embedding feature type
        import numpy as np
        embs = _EMBEDDER.encode(texts, show_progress_bar=False)
        # Ensure 2D array
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        probs = _MODEL.predict_proba(embs)[:, 1]
    else:
        logger.warning("Unknown feature_type — returning 0.5")
        return 0.5

    return float(probs[0])


def describe(prompt: str) -> dict:
    """Get full diagnostic for a prompt."""
    _load()
    s = score(prompt)
    return {
        "complexity_score": s,
        "prediction": "complex" if s > 0.5 else "simple",
        "model": _META.get("name", "unknown") if isinstance(_META, dict) else "unknown",
        "feature_type": _META.get("feature_type", "unknown") if isinstance(_META, dict) else "unknown",
    }


if __name__ == "__main__":
    import sys
    for arg in sys.argv[1:]:
        s = score(arg)
        print(f"  {s:.4f} | {arg[:80]}")
