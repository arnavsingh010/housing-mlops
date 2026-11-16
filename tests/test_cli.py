import json

import yaml

from housing_mlops.cli import main
from housing_mlops.data import FEATURES, make_synthetic


def _write_cfg(cfg, tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


def test_cli_train_then_predict_json_and_csv(cfg, tmp_path, capsys):
    cfg_path = _write_cfg(cfg, tmp_path)
    assert main(["--config", cfg_path, "train"]) == 0

    record = make_synthetic(1)[FEATURES].iloc[0].to_dict()
    assert main(["--config", cfg_path, "predict", "--json", json.dumps(record)]) == 0
    assert "PredictedValue" in capsys.readouterr().out

    csv_in, csv_out = tmp_path / "in.csv", tmp_path / "out.csv"
    make_synthetic(4)[FEATURES].to_csv(csv_in, index=False)
    assert (
        main(["--config", cfg_path, "predict", "--input", str(csv_in), "--output", str(csv_out)])
        == 0
    )
    assert csv_out.exists()


def test_cli_evaluate(cfg, tmp_path):
    cfg_path = _write_cfg(cfg, tmp_path)
    main(["--config", cfg_path, "train"])
    assert main(["--config", cfg_path, "evaluate"]) == 0


def test_cli_train_fails_quality_gate(cfg, tmp_path):
    cfg["quality_gate"]["min_r2"] = 0.9999
    assert main(["--config", _write_cfg(cfg, tmp_path), "train"]) == 1


def test_cli_missing_config_returns_error(tmp_path):
    assert main(["--config", str(tmp_path / "nope.yaml"), "train"]) == 2


def test_cli_predict_without_model_returns_error(cfg, tmp_path):
    assert main(["--config", _write_cfg(cfg, tmp_path), "predict", "--sample", "3"]) == 2
