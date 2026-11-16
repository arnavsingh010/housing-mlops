"""Training + evaluation."""

from __future__ import annotations

import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from housing_mlops.data import FEATURES, TARGET, load_train_test, validate_data
from housing_mlops.features import build_pipeline

logger = logging.getLogger(__name__)


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run the full training pipeline. Returns the metrics dict and saves artifacts."""
    train_df, test_df = load_train_test(cfg)
    validate_data(train_df)
    validate_data(test_df)

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    pipeline = build_pipeline(cfg)
    logger.info("Training %s on %d rows", cfg["model"]["type"], len(X_train))
    pipeline.fit(X_train, y_train)

    metrics = compute_metrics(y_test, pipeline.predict(X_test))
    logger.info("Test metrics: %s", {k: round(v, 4) for k, v in metrics.items()})

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_type": cfg["model"]["type"],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": FEATURES,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    result = {"metrics": metrics, "metadata": metadata}

    model_path = Path(cfg["artifacts"]["model_path"])
    metrics_path = Path(cfg["artifacts"]["metrics_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({"pipeline": pipeline, "metadata": metadata}, model_path)
    metrics_path.write_text(json.dumps(result, indent=2))
    logger.info("Saved model -> %s, metrics -> %s", model_path, metrics_path)
    return result


def passes_quality_gate(metrics: dict[str, float], cfg: dict[str, Any]) -> bool:
    return metrics["r2"] >= cfg["quality_gate"]["min_r2"]
