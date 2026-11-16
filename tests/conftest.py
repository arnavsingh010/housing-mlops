import pytest

from housing_mlops.config import load_config


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """Config using synthetic data and writing artifacts to a temp dir."""
    monkeypatch.chdir(_repo_root())
    return load_config(
        overrides={
            "data": {"source": "synthetic", "cache_path": str(tmp_path / "h.csv")},
            "model": {"type": "ridge"},  # fast
            "artifacts": {
                "model_path": str(tmp_path / "model.joblib"),
                "metrics_path": str(tmp_path / "metrics.json"),
            },
            "quality_gate": {"min_r2": 0.3},
        }
    )


def _repo_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[1]
