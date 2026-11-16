"""Inference helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from housing_mlops.data import FEATURES, validate_data


def load_model(model_path: str | Path) -> dict[str, Any]:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"No model at {model_path}. Run `housing-mlops train` first.")
    return joblib.load(model_path)


def predict_df(df: pd.DataFrame, model_path: str | Path) -> pd.DataFrame:
    """Return a copy of `df` with a `PredictedValue` column (in USD)."""
    validate_data(df, require_target=False)
    bundle = load_model(model_path)
    preds = bundle["pipeline"].predict(df[FEATURES])
    out = df.copy()
    out["PredictedValue"] = (preds * 100_000).round(0)  # target is in $100k units
    return out
