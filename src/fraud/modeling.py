"""Leakage-safe model training, threshold selection, and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .features import FEATURES


@dataclass(frozen=True)
class FraudMetrics:
    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float
    false_positive_rate: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    threshold: float


def build_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2_000,
            random_state=random_state,
        )),
    ])


def select_threshold(
    labels: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    minimum_recall: float = 0.80,
) -> float:
    """Choose the highest-precision threshold meeting the recall target."""
    if not 0 < minimum_recall <= 1:
        raise ValueError("minimum_recall must be in (0, 1]")
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if thresholds.size == 0:
        return 0.5
    eligible = np.flatnonzero(recall[:-1] >= minimum_recall)
    if eligible.size == 0:
        return float(thresholds[int(np.argmax(recall[:-1]))])
    best = eligible[int(np.argmax(precision[:-1][eligible]))]
    return float(np.clip(thresholds[best], 1e-6, 1 - 1e-6))


def evaluate(
    model: Pipeline,
    features: pd.DataFrame,
    labels: pd.Series,
    threshold: float,
) -> FraudMetrics:
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return FraudMetrics(
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        f1=float(f1_score(labels, predictions, zero_division=0)),
        pr_auc=float(average_precision_score(labels, probabilities)),
        roc_auc=float(roc_auc_score(labels, probabilities)),
        false_positive_rate=float(fp / (fp + tn)) if fp + tn else 0.0,
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
        threshold=float(threshold),
    )


def global_feature_effects(model: Pipeline, limit: int = 10) -> dict[str, list[dict[str, float | str]]]:
    """Expose global coefficient direction; not a transaction-level explanation."""
    coefficients = model.named_steps["classifier"].coef_[0]
    ranked = sorted(zip(FEATURES, coefficients), key=lambda item: item[1])
    negative = [{"feature": name, "coefficient": round(float(value), 6)}
                for name, value in ranked[:limit]]
    positive = [{"feature": name, "coefficient": round(float(value), 6)}
                for name, value in reversed(ranked[-limit:])]
    return {"lower_risk": negative, "higher_risk": positive}


def metrics_dict(metrics: FraudMetrics) -> dict[str, float | int]:
    return asdict(metrics)
