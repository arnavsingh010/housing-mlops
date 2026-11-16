import pytest

from housing_mlops.data import FEATURES, TARGET, make_synthetic, validate_data


def test_synthetic_shape_and_columns():
    df = make_synthetic(n=100)
    assert len(df) == 100
    assert set(FEATURES + [TARGET]) <= set(df.columns)


def test_synthetic_is_deterministic():
    assert make_synthetic(50, seed=1).equals(make_synthetic(50, seed=1))


def test_validate_ok():
    validate_data(make_synthetic(50))


def test_validate_missing_column():
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_data(make_synthetic(50).drop(columns=["MedInc"]))


def test_validate_missing_target():
    df = make_synthetic(50)
    df.loc[0, TARGET] = None
    with pytest.raises(ValueError, match="Target"):
        validate_data(df)


def test_validate_out_of_range():
    df = make_synthetic(50)
    df.loc[0, "Latitude"] = 99
    with pytest.raises(ValueError, match="Latitude"):
        validate_data(df)


def test_validate_empty():
    with pytest.raises(ValueError):
        validate_data(make_synthetic(10).iloc[0:0])
