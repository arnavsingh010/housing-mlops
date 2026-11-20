"""FastAPI service: `uvicorn housing_mlops.api:app`.

Endpoints
    GET  /health         liveness + whether a model is loaded
    GET  /model-info     metadata + test metrics of the loaded model
    POST /predict        one house  -> one price
    POST /predict/batch  many houses -> many prices
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from housing_mlops import __version__
from housing_mlops.config import load_config
from housing_mlops.data import FEATURES
from housing_mlops.predict import load_model

logger = logging.getLogger("housing_mlops.api")


class HouseFeatures(BaseModel):
    """One California block group. Ranges mirror the validation in data.py."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41,
                "AveRooms": 6.98,
                "AveBedrms": 1.02,
                "Population": 322,
                "AveOccup": 2.56,
                "Latitude": 37.88,
                "Longitude": -122.23,
            }
        }
    )

    MedInc: float = Field(..., ge=0, le=50, description="Median income (tens of thousands $)")
    HouseAge: float = Field(..., ge=0, le=100, description="Median house age (years)")
    AveRooms: float = Field(..., gt=0, description="Average rooms per household")
    AveBedrms: float = Field(..., gt=0, description="Average bedrooms per household")
    Population: float = Field(..., ge=0, description="Block group population")
    AveOccup: float = Field(..., gt=0, description="Average occupants per household")
    Latitude: float = Field(..., ge=30, le=45)
    Longitude: float = Field(..., ge=-130, le=-110)


class Prediction(BaseModel):
    predicted_value_usd: float


class BatchRequest(BaseModel):
    houses: list[HouseFeatures] = Field(..., min_length=1, max_length=1000)


class BatchPrediction(BaseModel):
    predictions: list[Prediction]


def _resolve_model_path(model_path: str | Path | None) -> Path:
    if model_path:
        return Path(model_path)
    if os.environ.get("MODEL_PATH"):
        return Path(os.environ["MODEL_PATH"])
    return Path(load_config()["artifacts"]["model_path"])


def create_app(model_path: str | Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        path = _resolve_model_path(model_path)
        app.state.bundle = None
        try:
            app.state.bundle = load_model(path)  # loaded ONCE, not per request
            logger.info("Loaded model from %s", path)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
        yield

    app = FastAPI(
        title="Housing price API",
        version=__version__,
        description="Predict California median house value from block-group features.",
        lifespan=lifespan,
    )

    def get_bundle(request: Request) -> dict[str, Any]:
        bundle = request.app.state.bundle
        if bundle is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Train it first.")
        return bundle

    def run(bundle: dict[str, Any], houses: list[HouseFeatures]) -> list[Prediction]:
        df = pd.DataFrame([h.model_dump() for h in houses])[FEATURES]
        preds = bundle["pipeline"].predict(df) * 100_000  # target is in $100k units
        return [Prediction(predicted_value_usd=round(float(p), 0)) for p in preds]

    @app.get("/")
    def root() -> dict[str, str]:
        return {"service": "housing-mlops", "docs": "/docs", "health": "/health"}

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        return {"status": "ok", "model_loaded": request.app.state.bundle is not None}

    @app.get("/model-info")
    def model_info(request: Request) -> dict[str, Any]:
        bundle = get_bundle(request)
        info: dict[str, Any] = {"metadata": bundle["metadata"]}
        metrics_path = Path(load_config()["artifacts"]["metrics_path"])
        if metrics_path.exists():
            info["metrics"] = json.loads(metrics_path.read_text()).get("metrics")
        return info

    @app.post("/predict", response_model=Prediction)
    def predict(house: HouseFeatures, request: Request) -> Prediction:
        return run(get_bundle(request), [house])[0]

    @app.post("/predict/batch", response_model=BatchPrediction)
    def predict_batch(body: BatchRequest, request: Request) -> BatchPrediction:
        return BatchPrediction(predictions=run(get_bundle(request), body.houses))

    return app


app = create_app()
