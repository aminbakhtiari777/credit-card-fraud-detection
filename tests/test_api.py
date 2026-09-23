from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@unittest.skipUnless(
    importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx"),
    "FastAPI development dependencies are not installed",
)
class ApiTests(unittest.TestCase):
    def test_health_model_info_and_prediction(self) -> None:
        from fastapi.testclient import TestClient
        from fraud.api import create_app

        class FakeService:
            threshold = 0.42
            metadata = {"version": "test"}

            def predict(self, record):
                self.last_record = record
                return {
                    "fraud_probability": 0.81,
                    "is_fraud": True,
                    "threshold": self.threshold,
                }

        payload = {"Time": 1.0, "Amount": 10.0}
        payload.update({f"V{index}": 0.0 for index in range(1, 29)})
        with TestClient(create_app(FakeService())) as client:
            self.assertTrue(client.get("/health").json()["model_loaded"])
            self.assertEqual(client.get("/model-info").json()["metadata"]["version"], "test")
            response = client.post("/predict", json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["is_fraud"])
