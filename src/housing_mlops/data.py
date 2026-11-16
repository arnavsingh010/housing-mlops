"""Data loading + validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
TARGET = "MedHouseVal"  # median house value, in $100k

# Sanity ranges used by validate_data
_RANGES = {
    "MedInc": (0, 50),
    "HouseAge": (0, 100),
    "Latitude": (30, 45),
    "Longitude": (-130, -110),
}


def make_synthetic(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """California-like synthetic data. Used offline and in tests."""
    rng = np.random.default_rng(seed)
    med_inc = np.clip(rng.lognormal(1.2, 0.5, n), 0.5, 15)
    house_age = rng.integers(1, 53, n).astype(float)
    ave_rooms = np.clip(rng.normal(5.4, 1.5, n), 2, 12)
    ave_bedrms = np.clip(ave_rooms * rng.normal(0.2, 0.03, n), 0.5, 5)
    population = np.clip(rng.lognormal(7.1, 0.6, n), 50, 20000)
    ave_occup = np.clip(rng.normal(3.0, 0.6, n), 1, 8)
    lat = rng.uniform(32.5, 42.0, n)
    lon = rng.uniform(-124.3, -114.3, n)
    coast_bonus = np.exp(-((lon + 121.0) ** 2) / 8.0)
    y = (
        0.45 * med_inc
        + 0.05 * ave_rooms
        - 0.004 * house_age
        - 0.08 * (ave_occup - 3)
        + 0.9 * coast_bonus
        + rng.normal(0, 0.25, n)
    )
    y = np.clip(y, 0.15, 5.0)
    return pd.DataFrame(
        {
            "MedInc": med_inc,
            "HouseAge": house_age,
            "AveRooms": ave_rooms,
            "AveBedrms": ave_bedrms,
            "Population": population,
            "AveOccup": ave_occup,
            "Latitude": lat,
            "Longitude": lon,
            TARGET: y,
        }
    )


def _fetch_california() -> pd.DataFrame:
    from sklearn.datasets import fetch_california_housing

    return fetch_california_housing(as_frame=True).frame


def load_data(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load data from cache -> California housing download -> synthetic fallback."""
    data_cfg = cfg["data"]
    cache = Path(data_cfg["cache_path"])

    if data_cfg["source"] == "synthetic":
        logger.info("Using synthetic data")
        return make_synthetic()

    if cache.exists():
        logger.info("Loading cached data from %s", cache)
        return pd.read_csv(cache)

    try:
        logger.info("Downloading California housing dataset...")
        df = _fetch_california()
    except Exception as exc:  # network issues etc.
        logger.warning("Download failed (%s). Falling back to synthetic data.", exc)
        return make_synthetic()

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    logger.info("Cached dataset at %s (%d rows)", cache, len(df))
    return df


def validate_data(df: pd.DataFrame, require_target: bool = True) -> None:
    """Raise ValueError if the data does not look the way the pipeline expects."""
    required = FEATURES + ([TARGET] if require_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("DataFrame is empty")
    if require_target and df[TARGET].isna().any():
        raise ValueError("Target column contains missing values")
    for col, (lo, hi) in _RANGES.items():
        bad = df[col].dropna()
        bad = bad[(bad < lo) | (bad > hi)]
        if len(bad):
            raise ValueError(f"{col} has {len(bad)} values outside [{lo}, {hi}]")
