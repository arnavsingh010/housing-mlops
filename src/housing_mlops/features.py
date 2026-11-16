"""Feature engineering + the full sklearn preprocessing/model pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from housing_mlops.data import FEATURES


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns. Pure function: same input -> same output."""
    X = X[FEATURES].copy()
    X["BedroomRatio"] = X["AveBedrms"] / X["AveRooms"].replace(0, np.nan)
    X["PeoplePerRoom"] = X["AveOccup"] / X["AveRooms"].replace(0, np.nan)
    X["LogPopulation"] = np.log1p(X["Population"].clip(lower=0))
    return X


_MODELS = {
    "hist_gb": HistGradientBoostingRegressor,
    "random_forest": RandomForestRegressor,
    "ridge": Ridge,
}


def build_pipeline(cfg: dict[str, Any]) -> Pipeline:
    model_type = cfg["model"]["type"]
    if model_type not in _MODELS:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from {list(_MODELS)}")
    params = cfg["model"]["params"].get(model_type, {})
    return Pipeline(
        [
            ("features", FunctionTransformer(add_features)),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", _MODELS[model_type](**params)),
        ]
    )
