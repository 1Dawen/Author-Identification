"""Input checks, splits and common classification metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from . import LABELS


def validate_frame(frame: pd.DataFrame, *, labeled: bool) -> pd.DataFrame:
    """Return normalized Kaggle-style rows, rejecting missing IDs and empty text."""
    required = {"id", "text", "author"} if labeled else {"id", "text"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Dataset is empty")
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError("IDs must be unique and non-null")
    clean = frame.copy()
    if clean["text"].isna().any() or not clean["text"].map(lambda s: isinstance(s, str) and bool(s.strip())).all():
        raise ValueError("All text entries must be nonempty strings")
    if labeled and not clean["author"].isin(LABELS).all():
        raise ValueError("Labels must be EAP, HPL or MWS")
    return clean


def holdout_split(frame: pd.DataFrame, *, test_size: float = 0.2, seed: int = 40):
    frame = validate_frame(frame, labeled=True)
    counts = frame["author"].value_counts()
    if len(counts) != len(LABELS) or counts.min() < 2:
        raise ValueError("Stratified validation requires at least two samples per author")
    training, validation = train_test_split(
        frame, test_size=test_size, stratify=frame["author"], random_state=seed
    )
    return training.reset_index(drop=True), validation.reset_index(drop=True)


def metrics_for(y_true, probabilities) -> dict[str, float]:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape != (len(y_true), len(LABELS)):
        raise ValueError("Expected one probability vector of length three per observation")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Probabilities must be nonnegative, finite and sum to one")
    # XGBoost can return float32 rows that differ from 1 by ~1e-7.
    # Normalize to float64 before sklearn's stricter probability checks.
    p = p / p.sum(axis=1, keepdims=True)
    true = np.asarray(y_true)
    return {"accuracy": float(accuracy_score(true, np.asarray(LABELS)[np.argmax(p, axis=1)])),
            "log_loss": float(log_loss(true, p, labels=LABELS))}


def submission_frame(ids, probabilities) -> pd.DataFrame:
    p = np.asarray(probabilities, dtype=float)
    metrics_for([LABELS[0]] * len(ids), p)  # Check dimensions and probability validity.
    p = p / p.sum(axis=1, keepdims=True)
    return pd.DataFrame({"id": list(ids), **{label: p[:, i] for i, label in enumerate(LABELS)}})
