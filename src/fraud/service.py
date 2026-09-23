"""Versioned artifact loading and fraud prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class FraudService:
    def __init__(self, model: Any, threshold: float, metadata: dict | None = None) -> None:
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1")
        self.model = model
        self.threshold = float(threshold)
        self.metadata = metadata or {}

    @classmethod
    def from_path(cls, path: str | Path) -> "FraudService":
        artifact_path = Path(path)
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
        artifact = joblib.load(artifact_path)
        required = {"model", "threshold"}
        if not isinstance(artifact, dict) or not required.issubset(artifact):
            raise ValueError("Invalid fraud model artifact")
        return cls(artifact["model"], artifact["threshold"], artifact.get("metadata"))

    def predict(self, record: dict[str, Any]) -> dict[str, float | bool]:
        probability = float(self.model.predict_proba(pd.DataFrame([record]))[0, 1])
        return {
            "fraud_probability": round(probability, 6),
            "is_fraud": probability >= self.threshold,
            "threshold": round(self.threshold, 6),
        }
