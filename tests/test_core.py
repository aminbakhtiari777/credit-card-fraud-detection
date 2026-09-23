from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud.features import FEATURES, split_features_target
from fraud.modeling import build_pipeline, evaluate, global_feature_effects, select_threshold
from fraud.service import FraudService


def sample_frame(rows: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(rows, len(FEATURES))), columns=FEATURES)
    frame["Time"] = np.arange(rows) * 30
    frame["Amount"] = np.abs(frame["Amount"] * 90)
    frame["Class"] = 0
    fraud_rows = np.arange(0, rows, 10)
    frame.loc[fraud_rows, "Class"] = 1
    frame.loc[fraud_rows, "V14"] -= 6
    frame.loc[fraud_rows, "V17"] -= 5
    frame.loc[fraud_rows, "V4"] += 5
    frame.loc[0, "Amount"] = np.nan
    return frame


class FraudCoreTests(unittest.TestCase):
    def test_dataset_schema_and_binary_target(self) -> None:
        features, labels = split_features_target(sample_frame())
        self.assertEqual(list(features.columns), FEATURES)
        self.assertEqual(set(labels.unique()), {0, 1})

    def test_missing_feature_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            split_features_target(sample_frame().drop(columns=["V28"]))

    def test_threshold_meets_recall_target(self) -> None:
        labels = np.array([0, 0, 0, 1, 1])
        probabilities = np.array([0.05, 0.10, 0.60, 0.70, 0.90])
        threshold = select_threshold(labels, probabilities, minimum_recall=1.0)
        predictions = probabilities >= threshold
        self.assertEqual(predictions[labels == 1].sum(), 2)

    def test_pipeline_evaluation_and_explanation(self) -> None:
        features, labels = split_features_target(sample_frame())
        model = build_pipeline()
        model.fit(features, labels)
        probabilities = model.predict_proba(features)[:, 1]
        threshold = select_threshold(labels, probabilities)
        metrics = evaluate(model, features, labels, threshold)
        effects = global_feature_effects(model, limit=3)
        self.assertGreater(metrics.pr_auc, 0.5)
        self.assertEqual(len(effects["higher_risk"]), 3)

    def test_versioned_artifact_round_trip(self) -> None:
        features, labels = split_features_target(sample_frame())
        model = build_pipeline()
        model.fit(features, labels)
        artifact = {"model": model, "threshold": 0.4, "metadata": {"version": "test"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.joblib"
            joblib.dump(artifact, path)
            service = FraudService.from_path(path)
            prediction = service.predict(features.iloc[1].to_dict())
        self.assertEqual(service.metadata["version"], "test")
        self.assertGreaterEqual(prediction["fraud_probability"], 0)
        self.assertLessEqual(prediction["fraud_probability"], 1)


if __name__ == "__main__":
    unittest.main()
