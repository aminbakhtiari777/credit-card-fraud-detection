"""Dataset schema and feature validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

PCA_FEATURES = [f"V{index}" for index in range(1, 29)]
FEATURES = ["Time", *PCA_FEATURES, "Amount"]
TARGET = "Class"


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return numeric features and a validated binary fraud target."""
    required = set(FEATURES + [TARGET])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    clean = frame.copy()
    clean[FEATURES] = clean[FEATURES].apply(pd.to_numeric, errors="coerce")
    labels = pd.to_numeric(clean[TARGET], errors="coerce")
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError("Class must contain only binary labels 0 and 1")
    if np.isinf(clean[FEATURES].to_numpy()).any():
        raise ValueError("Feature values must be finite")
    return clean[FEATURES], labels.astype(int)
