"""Data loading + validation."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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


def _download_raw(url: str) -> pd.DataFrame:
    """Download the raw California housing CSV (1990 census block groups)."""
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 (fixed https URL from config)
        return pd.read_csv(resp)


def convert_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw census table -> the schema the model uses (same columns as sklearn's dataset).

    Rows with a missing `total_bedrooms` (207 of 20,640, ~1%) are dropped so that train and
    test files are clean; the pipeline's imputer still protects inference-time inputs.
    """
    raw = raw.dropna(subset=["total_bedrooms"])
    households = raw["households"].replace(0, np.nan)
    return pd.DataFrame(
        {
            "MedInc": raw["median_income"],
            "HouseAge": raw["housing_median_age"],
            "AveRooms": raw["total_rooms"] / households,
            "AveBedrms": raw["total_bedrooms"] / households,
            "Population": raw["population"],
            "AveOccup": raw["population"] / households,
            "Latitude": raw["latitude"],
            "Longitude": raw["longitude"],
            TARGET: raw["median_house_value"] / 100_000,
        }
    )


def split_data(df: pd.DataFrame, data_cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproducible train/test split."""
    train_df, test_df = train_test_split(
        df, test_size=data_cfg["test_size"], random_state=data_cfg["random_state"]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def prepare_data(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download -> convert -> validate -> split -> write train.csv and test.csv."""
    data_cfg = cfg["data"]
    logger.info("Downloading raw dataset from %s", data_cfg["raw_url"])
    df = convert_raw(_download_raw(data_cfg["raw_url"]))
    validate_data(df)
    train_df, test_df = split_data(df, data_cfg)

    for df_part, path in ((train_df, data_cfg["train_path"]), (test_df, data_cfg["test_path"])):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df_part.to_csv(path, index=False)
        logger.info("Wrote %s (%d rows)", path, len(df_part))
    return train_df, test_df


def load_train_test(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df): files on disk -> download+split -> synthetic fallback."""
    data_cfg = cfg["data"]

    if data_cfg["source"] == "synthetic":
        logger.info("Using synthetic data")
        return split_data(make_synthetic(), data_cfg)

    train_path, test_path = Path(data_cfg["train_path"]), Path(data_cfg["test_path"])
    if train_path.exists() and test_path.exists():
        logger.info("Loading %s and %s", train_path, test_path)
        return pd.read_csv(train_path), pd.read_csv(test_path)

    try:
        return prepare_data(cfg)
    except Exception as exc:  # network issues etc.
        logger.warning("Could not prepare real data (%s). Falling back to synthetic data.", exc)
        return split_data(make_synthetic(), data_cfg)


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
