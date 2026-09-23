"""Train, tune, evaluate, and persist the fraud-detection artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud.features import split_features_target  # noqa: E402
from fraud.modeling import (  # noqa: E402
    build_pipeline,
    evaluate,
    global_feature_effects,
    metrics_dict,
    select_threshold,
)

DEFAULT_DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data-file", type=Path)
    source.add_argument("--data-url", default=DEFAULT_DATA_URL)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--minimum-recall", type=float, default=0.80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.data_file if args.data_file else args.data_url)
    features, labels = split_features_target(frame)

    x_development, x_test, y_development, y_test = train_test_split(
        features, labels, test_size=0.20, random_state=42, stratify=labels
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_development,
        y_development,
        test_size=0.20,
        random_state=42,
        stratify=y_development,
    )

    model = build_pipeline()
    model.fit(x_train, y_train)
    validation_probabilities = model.predict_proba(x_validation)[:, 1]
    threshold = select_threshold(
        y_validation, validation_probabilities, minimum_recall=args.minimum_recall
    )

    model.fit(x_development, y_development)
    metrics = evaluate(model, x_test, y_test, threshold)
    explanations = global_feature_effects(model)
    report = {
        "metrics": metrics_dict(metrics),
        "global_feature_effects": explanations,
        "selection": {
            "minimum_validation_recall": args.minimum_recall,
            "threshold_selected_on": "validation",
            "evaluation_set": "untouched_test",
        },
    }

    artifact = {
        "model": model,
        "threshold": threshold,
        "metadata": {
            "version": "1.0.0",
            "model_type": "class-balanced logistic regression",
            **report,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output_dir / "fraud_artifact.joblib")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
