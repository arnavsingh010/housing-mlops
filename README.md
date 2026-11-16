# housing-mlops

A small but complete MLOps project: download a public housing dataset, validate and preprocess it,
train a model, enforce a quality gate, and serve predictions from the terminal.

```
data (California housing) -> validate -> feature engineering -> impute/scale -> model
                                                         |-> metrics.json + model.joblib
                                                         |-> quality gate (CI fails if R2 too low)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install        # or: pip install -e ".[dev]"
make hooks          # git hooks (ruff on commit, pytest on push)

housing-mlops train
housing-mlops predict --sample 5
housing-mlops predict --json '{"MedInc":8.3,"HouseAge":41,"AveRooms":6.9,"AveBedrms":1.02,"Population":322,"AveOccup":2.5,"Latitude":37.88,"Longitude":-122.23}'
housing-mlops predict --input my_houses.csv --output preds.csv
housing-mlops evaluate
```

If you are offline, the data step falls back to a synthetic California-like dataset so everything still runs.

## Layout

| Path | Purpose |
|---|---|
| `configs/config.yaml` | All knobs: data source, model type + params, artifact paths, quality gate |
| `src/housing_mlops/data.py` | Load (cache -> download -> synthetic fallback) and validate data |
| `src/housing_mlops/features.py` | Feature engineering + sklearn `Pipeline` (impute, scale, model) |
| `src/housing_mlops/train.py` | Train, compute RMSE/MAE/R2, save model + metrics + metadata |
| `src/housing_mlops/predict.py` | Inference with input validation |
| `src/housing_mlops/cli.py` | `housing-mlops train / evaluate / predict` |
| `tests/` | pytest suite (offline, uses synthetic data) |
| `.pre-commit-config.yaml` | Hooks: whitespace, yaml/toml, large files, private keys, ruff, pytest on push |
| `.github/workflows/ci.yml` | Lint + tests (py 3.10-3.12) + coverage gate + train smoke test |
| `.github/workflows/cd.yml` | On tag `v*`: test, train, build and push Docker image to GHCR, attach model to release |
| `Dockerfile` | Self-contained image with a trained model |

## Common tasks

```bash
make lint   make format   make test   make train   make docker
```

Switch model: set `model.type` in the config to `hist_gb`, `random_forest` or `ridge`.

## Releasing

```bash
git tag v0.1.0 && git push --tags
```

## Ideas for next steps

DVC for data versioning, MLflow for experiment tracking, a FastAPI serving layer,
drift monitoring, scheduled retraining workflow.
