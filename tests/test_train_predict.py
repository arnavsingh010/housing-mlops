import json
from pathlib import Path

import pandas as pd
import pytest

from housing_mlops.data import FEATURES, make_synthetic
from housing_mlops.predict import predict_df
from housing_mlops.train import passes_quality_gate, train


def test_train_creates_artifacts_and_reasonable_metrics(cfg):
    result = train(cfg)
    assert Path(cfg["artifacts"]["model_path"]).exists()
    saved = json.loads(Path(cfg["artifacts"]["metrics_path"]).read_text())
    assert saved["metrics"].keys() == {"rmse", "mae", "r2"}
    assert result["metrics"]["r2"] > 0.3
    assert passes_quality_gate(result["metrics"], cfg)


def test_quality_gate_fails_when_threshold_too_high(cfg):
    cfg["quality_gate"]["min_r2"] = 0.9999
    assert not passes_quality_gate(train(cfg)["metrics"], cfg)


def test_predict_adds_column_in_dollars(cfg):
    train(cfg)
    df = make_synthetic(5)[FEATURES]
    out = predict_df(df, cfg["artifacts"]["model_path"])
    assert len(out) == 5
    assert (out["PredictedValue"] > 1_000).all()  # dollars, not $100k units


def test_predict_without_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        predict_df(make_synthetic(2)[FEATURES], tmp_path / "missing.joblib")


def test_predict_rejects_bad_input(cfg):
    train(cfg)
    bad = pd.DataFrame({"MedInc": [1.0]})
    with pytest.raises(ValueError):
        predict_df(bad, cfg["artifacts"]["model_path"])
