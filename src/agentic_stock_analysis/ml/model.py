from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import joblib
from sklearn.ensemble import RandomForestClassifier

# Artifact path: src/agentic_stock_analysis/ml/artifacts/stock_model.pkl
_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = _ARTIFACT_DIR / "stock_model.pkl"

# In-process cache (per uvicorn worker)
_MODEL = None


def train_model(df, features: List[str], model_path: Path = MODEL_PATH):
    """
    Train a RandomForestClassifier on the given dataframe and feature columns.
    """
    X = df[features]
    y = df["Target"]
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y)
    joblib.dump(model, str(model_path))
    return model


def load_model(model_path=MODEL_PATH):
    """
    Load a trained model from disk.
    """
    return joblib.load(str(model_path))


def get_model():
    """
    Do NOT train here.
    Training should be handled by ensure_model_trained() at startup.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    _MODEL = load_model(MODEL_PATH)
    return _MODEL
