# Contributions Guide

This repository (`evbuddy`) is the ML/DVC core of the EVBuddy ecosystem.

## Scope

Use this repo for:

- data ingestion and preprocessing
- feature engineering and transformation
- model training and evaluation
- DVC pipeline and reproducibility changes
- CI checks related to the ML pipeline

Use sibling repos for app/API changes:

- `evbuddy-backend` for FastAPI inference service
- `evbuddy-frontend` for React web app
- `evbuddy-android` for Flutter app

## Development Setup

```bash
poetry env use python3.12
poetry install --with dev
poetry run dvc pull
```

## Branch and Commit Workflow

1. Create a branch from `main`.
2. Make focused commits (one topic per commit).
3. If a stage output changes, regenerate lock/artifacts consistently.
4. Push branch and open a PR.

## Required Local Checks Before PR

```bash
poetry run pytest -v
poetry run dvc repro
```

If you changed only specific stages, run targeted repro:

```bash
poetry run dvc repro <stage-name>
```

## Training Modes

- Main trainer: `src/models/train_models.py` (Dask + XGBoost)
- Baseline trainer: `src/models/train_models_pandas.py` (pandas)

Run baseline manually:

```bash
poetry run python -m src.models.train_models_pandas
```

## Testing

Run all tests:

```bash
poetry run pytest -v
```

Run model quality gate:

```bash
poetry run pytest -v tests/quality/test_metrics_thresholds.py
```

## DVC Stages and Artifacts

Current stage sequence from `dvc.yaml`:

1. `json_to_csv`
2. `derive_opening_hours`
3. `concat_locations`
4. `extract_stations`
5. `extract_ports`
6. `build_timeseries`
7. `feature_encoding`
8. `feature_selection`
9. `visualisation`
10. `feature_transform`
11. `train_models`

Primary outputs:

- processed dense dataset under `data/processed`
- horizon models:
  - `models/xgb_occupied_h10m.json`
  - `models/xgb_occupied_h20m.json`
  - `models/xgb_occupied_h30m.json`
- horizon metrics:
  - `models/metrics_h10m.json`
  - `models/metrics_h20m.json`
  - `models/metrics_h30m.json`

Use these commands for reproducible stage execution:

```bash
poetry run dvc repro
poetry run dvc repro <stage-name>
```

If you intentionally changed tracked outputs:

```bash
poetry run dvc push
```

## MLflow Tracking

Training logs can be sent to MLflow via `MLFLOW_TRACKING_URI`.

Typical local endpoint:

- `http://127.0.0.1:5000`

Environment variables:

- `MLFLOW_TRACKING_URI`
- `MLFLOW_EXPERIMENT_NAME` (default: `evbuddy-train-models`)

Current operational expectation:

- tracking endpoint is reachable only through a private network path, not as a public open endpoint.

## DVC Rules

- Do not edit `dvc.lock` manually.
- If pipeline definitions changed, run `dvc repro` to regenerate `dvc.lock`.
- If artifacts changed intentionally, run:

```bash
poetry run dvc push
```

## Troubleshooting

If `dvc.yaml` and `dvc.lock` diverge:

```bash
poetry run dvc repro <stage-name>
poetry run dvc push
git add dvc.yaml dvc.lock
git commit -m "sync dvc lock"
```

If Dask workers hit memory limits:

```bash
EV_BUDDY_DASK_N_WORKERS=1 EV_BUDDY_DASK_THREADS_PER_WORKER=1 poetry run dvc repro train_models
```

## PR Checklist

- [ ] Code is focused and minimal for the requested change.
- [ ] Tests pass locally.
- [ ] `dvc.lock` is in sync with `dvc.yaml` (if relevant).
- [ ] Metrics/model changes are intentional and explained in PR description.
- [ ] README/docs updated when behavior or workflow changed.

## Notes on Reproducibility

- Prefer deterministic settings where possible.
- Document any known nondeterminism (e.g., training variance due to parallelism).
- When reporting model changes, include before/after metrics diffs.
