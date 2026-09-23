"""FastAPI application for fraud-risk inference."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from .schemas import FraudPrediction, TransactionFeatures
from .service import FraudService

DEFAULT_MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/fraud_artifact.joblib"))


def create_app(service: FraudService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if application.state.fraud_service is None:
            try:
                application.state.fraud_service = FraudService.from_path(DEFAULT_MODEL_PATH)
            except (FileNotFoundError, ValueError) as exc:
                application.state.model_error = str(exc)
        yield

    application = FastAPI(
        title="Credit Card Fraud Detection API",
        version="1.0.0",
        description="Returns fraud probability and a thresholded analyst-priority decision.",
        lifespan=lifespan,
    )
    application.state.fraud_service = service
    application.state.model_error = None

    @application.get("/health")
    def health(request: Request) -> dict[str, str | bool | None]:
        loaded = request.app.state.fraud_service is not None
        return {
            "status": "ok" if loaded else "degraded",
            "model_loaded": loaded,
            "detail": request.app.state.model_error,
        }

    @application.get("/model-info")
    def model_info(request: Request) -> dict:
        fraud_service = request.app.state.fraud_service
        if fraud_service is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        return {
            "threshold": fraud_service.threshold,
            "metadata": fraud_service.metadata,
        }

    @application.post("/predict", response_model=FraudPrediction)
    def predict(payload: TransactionFeatures, request: Request) -> dict[str, float | bool]:
        fraud_service = request.app.state.fraud_service
        if fraud_service is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        return fraud_service.predict(payload.model_dump())

    return application


app = create_app()
