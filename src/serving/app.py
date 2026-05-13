"""
FastAPI serving application for the drift mitigation pipeline.

Loads the latest champion model from the MLflow Model Registry and
exposes a /predict endpoint along with health and metadata endpoints.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field, field_validator
from starlette.responses import Response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics (RF-08)
# ---------------------------------------------------------------------------

PREDICTION_COUNTER = Counter(
    "model_predictions_total",
    "Total number of prediction requests",
    ["status"],
)
PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Prediction endpoint latency in seconds",
)
PREDICTION_SCORE = Histogram(
    "model_prediction_score",
    "Distribution of predicted probability scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "random-forest-classifier")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
FEATURE_NAMES = [f"feature_{i}" for i in range(10)]


# ---------------------------------------------------------------------------
# Model loader (singleton)
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Singleton wrapper around a loaded MLflow model."""

    _model: Any = None
    _model_version: str | None = None
    _loaded_at: float | None = None

    @classmethod
    def load(cls) -> None:
        """Load (or reload) the champion model from the MLflow registry."""
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        logger.info("Loading model from '%s' …", model_uri)
        try:
            cls._model = mlflow.sklearn.load_model(model_uri)
            client = mlflow.MlflowClient()
            version_info = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
            cls._model_version = version_info.version
            cls._loaded_at = time.time()
            logger.info("Model version %s loaded successfully.", cls._model_version)
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            raise RuntimeError(f"Model load error: {exc}") from exc

    @classmethod
    def get(cls) -> Any:
        """Return the loaded model or raise if not yet loaded."""
        if cls._model is None:
            raise RuntimeError("Model not loaded. Call ModelRegistry.load() first.")
        return cls._model

    @classmethod
    def metadata(cls) -> dict[str, Any]:
        """Return metadata about the currently loaded model."""
        return {
            "model_name": MODEL_NAME,
            "model_alias": MODEL_ALIAS,
            "model_version": cls._model_version,
            "loaded_at": cls._loaded_at,
        }


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Load the model on startup; release resources on shutdown."""
    logger.info("Application startup — loading model …")
    ModelRegistry.load()
    yield
    logger.info("Application shutdown.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Drift Mitigation Pipeline — Prediction Service",
    description="Serves predictions from the champion model registered in MLflow.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Schema for a single prediction request."""

    features: list[float] = Field(
        ...,
        min_length=10,
        max_length=10,
        description="Exactly 10 numeric feature values.",
        examples=[[0.5, -1.2, 0.8, 1.1, -0.3, 0.9, 0.2, -0.7, 1.4, 0.0]],
    )

    @field_validator("features")
    @classmethod
    def validate_feature_length(cls, v: list[float]) -> list[float]:
        if len(v) != 10:
            raise ValueError(f"Expected 10 features, got {len(v)}.")
        return v


class BatchPredictRequest(BaseModel):
    """Schema for a batch prediction request."""

    instances: list[PredictRequest] = Field(
        ...,
        min_length=1,
        description="List of prediction instances.",
    )


class PredictResponse(BaseModel):
    """Schema for a prediction response."""

    prediction: int
    probability_class_0: float
    probability_class_1: float
    model_version: str | None


class BatchPredictResponse(BaseModel):
    """Schema for a batch prediction response."""

    predictions: list[PredictResponse]
    count: int
    model_version: str | None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Operations"])
async def health_check() -> dict[str, str]:
    """Liveness probe — confirms the service is running."""
    return {"status": "healthy"}


@app.get("/ready", tags=["Operations"])
async def readiness_check() -> dict[str, Any]:
    """Readiness probe — confirms the model is loaded and ready."""
    try:
        ModelRegistry.get()
        return {"status": "ready", **ModelRegistry.metadata()}
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@app.get("/model/info", tags=["Model"])
async def model_info() -> dict[str, Any]:
    """Return metadata about the currently loaded model."""
    return ModelRegistry.metadata()


@app.post("/model/reload", tags=["Model"])
async def reload_model() -> dict[str, str]:
    """Force a model reload from the MLflow registry (e.g. after retraining)."""
    try:
        ModelRegistry.load()
        return {"status": "reloaded", "model_version": ModelRegistry.metadata()["model_version"]}
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest) -> PredictResponse:
    """Generate a single class prediction with probability scores.

    Args:
        request: A PredictRequest containing 10 feature values.

    Returns:
        PredictResponse with predicted class and probabilities.
    """
    model = ModelRegistry.get()
    start = time.perf_counter()

    try:
        X = pd.DataFrame([request.features], columns=FEATURE_NAMES)

        path = "data/production/production_dataset.csv"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_header = not os.path.exists(path)
        X.to_csv(path, mode='a', header=write_header, index=False)

        proba = model.predict_proba(X)[0]
        pred = int(np.argmax(proba))

        elapsed = time.perf_counter() - start
        PREDICTION_LATENCY.observe(elapsed)
        PREDICTION_COUNTER.labels(status="success").inc()
        PREDICTION_SCORE.observe(float(proba[1]))

        logger.debug(
            "Prediction: class=%d  proba=[%.4f, %.4f]  latency=%.4fs",
            pred, proba[0], proba[1], elapsed,
        )

        return PredictResponse(
            prediction=pred,
            probability_class_0=round(float(proba[0]), 6),
            probability_class_1=round(float(proba[1]), 6),
            model_version=ModelRegistry.metadata()["model_version"],
        )
    except Exception as exc:
        PREDICTION_COUNTER.labels(status="error").inc()
        logger.error("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
async def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Generate predictions for multiple instances in a single call.

    Args:
        request: A BatchPredictRequest containing a list of instances.

    Returns:
        BatchPredictResponse with all predictions.
    """
    model = ModelRegistry.get()
    start = time.perf_counter()

    try:
        rows = [inst.features for inst in request.instances]
        X = pd.DataFrame(rows, columns=FEATURE_NAMES)

        path = "data/production/production_dataset.csv"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_header = not os.path.exists(path)
        X.to_csv(path, mode='a', header=write_header, index=False)

        probas = model.predict_proba(X)
        preds = np.argmax(probas, axis=1)

        elapsed = time.perf_counter() - start
        PREDICTION_LATENCY.observe(elapsed)
        PREDICTION_COUNTER.labels(status="success").inc(len(rows))

        responses = [
            PredictResponse(
                prediction=int(preds[i]),
                probability_class_0=round(float(probas[i][0]), 6),
                probability_class_1=round(float(probas[i][1]), 6),
                model_version=ModelRegistry.metadata()["model_version"],
            )
            for i in range(len(rows))
        ]
        return BatchPredictResponse(
            predictions=responses,
            count=len(responses),
            model_version=ModelRegistry.metadata()["model_version"],
        )
    except Exception as exc:
        PREDICTION_COUNTER.labels(status="error").inc()
        logger.error("Batch prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/metrics", tags=["Operations"])
async def metrics() -> Response:
    """Expose Prometheus-format metrics for scraping (RF-08)."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
