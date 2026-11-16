import numpy as np
import pytest

from housing_mlops.data import FEATURES, make_synthetic
from housing_mlops.features import add_features, build_pipeline


def test_add_features_adds_columns_and_keeps_rows():
    X = make_synthetic(40)[FEATURES]
    out = add_features(X)
    assert len(out) == len(X)
    assert {"BedroomRatio", "PeoplePerRoom", "LogPopulation"} <= set(out.columns)
    assert not np.isinf(out.to_numpy()).any()


def test_add_features_does_not_mutate_input():
    X = make_synthetic(10)[FEATURES]
    before = X.copy()
    add_features(X)
    assert X.equals(before)


def test_pipeline_handles_missing_values(cfg):
    df = make_synthetic(200)
    X, y = df[FEATURES].copy(), df["MedHouseVal"]
    X.loc[X.index[:10], "AveRooms"] = np.nan
    pipe = build_pipeline(cfg).fit(X, y)
    assert not np.isnan(pipe.predict(X)).any()


@pytest.mark.parametrize("model_type", ["ridge", "random_forest", "hist_gb"])
def test_all_model_types_build(cfg, model_type):
    cfg["model"]["type"] = model_type
    assert build_pipeline(cfg).steps[-1][0] == "model"


def test_unknown_model_raises(cfg):
    cfg["model"]["type"] = "nope"
    with pytest.raises(ValueError, match="Unknown model type"):
        build_pipeline(cfg)
