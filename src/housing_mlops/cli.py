"""Command line interface: `housing-mlops train|predict|evaluate`."""

from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

from housing_mlops import __version__
from housing_mlops.config import load_config
from housing_mlops.data import FEATURES, TARGET, load_train_test, prepare_data, validate_data
from housing_mlops.predict import load_model, predict_df
from housing_mlops.train import compute_metrics, passes_quality_gate, train

logger = logging.getLogger("housing_mlops")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="housing-mlops", description="Housing price ML pipeline")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--config", default=None, help="Path to YAML config")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare", help="Download data and write train.csv / test.csv")
    sub.add_parser("train", help="Train on train.csv, evaluate on test.csv, save artifacts")
    sub.add_parser("evaluate", help="Re-evaluate saved model on test.csv")

    pr = sub.add_parser("predict", help="Predict prices")
    src = pr.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="CSV with feature columns")
    src.add_argument("--json", help="Single JSON record, e.g. '{\"MedInc\": 5, ...}'")
    src.add_argument("--sample", type=int, metavar="N", help="Predict N rows from the dataset")
    pr.add_argument("--output", help="Write predictions to this CSV")
    return p


def _cmd_train(cfg) -> int:
    result = train(cfg)
    print(json.dumps(result["metrics"], indent=2))
    if not passes_quality_gate(result["metrics"], cfg):
        logger.error("Quality gate FAILED: r2 < %s", cfg["quality_gate"]["min_r2"])
        return 1
    print("Quality gate passed")
    return 0


def _cmd_prepare(cfg) -> int:
    train_df, test_df = prepare_data(cfg)
    print(f"train: {len(train_df)} rows -> {cfg['data']['train_path']}")
    print(f"test:  {len(test_df)} rows -> {cfg['data']['test_path']}")
    return 0


def _cmd_evaluate(cfg) -> int:
    _, test_df = load_train_test(cfg)
    validate_data(test_df)
    bundle = load_model(cfg["artifacts"]["model_path"])
    metrics = compute_metrics(test_df[TARGET], bundle["pipeline"].predict(test_df[FEATURES]))
    print(json.dumps(metrics, indent=2))
    return 0 if passes_quality_gate(metrics, cfg) else 1


def _cmd_predict(cfg, args) -> int:
    if args.input:
        df = pd.read_csv(args.input)
    elif args.json:
        df = pd.DataFrame([json.loads(args.json)])
    else:
        df = load_train_test(cfg)[1].sample(args.sample, random_state=0)  # rows from test set

    out = predict_df(df, cfg["artifacts"]["model_path"])
    if args.output:
        out.to_csv(args.output, index=False)
        print(f"Wrote {len(out)} predictions to {args.output}")
    else:
        cols = FEATURES[:3] + ["PredictedValue"]
        print(out[cols].to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        if args.command == "prepare":
            return _cmd_prepare(cfg)
        if args.command == "train":
            return _cmd_train(cfg)
        if args.command == "evaluate":
            return _cmd_evaluate(cfg)
        return _cmd_predict(cfg, args)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
