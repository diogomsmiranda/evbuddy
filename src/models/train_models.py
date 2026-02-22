from __future__ import annotations

import json
import math
import os
import random
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import (
    FRESHNESS_COLUMNS,
    HORIZON_MINUTES,
    MODEL_STATIC_FEATURE_COLUMNS,
    MODELS_DIR,
    PROCESSED_DENSE_10MIN_CSV,
    PROCESSED_DENSE_10MIN_PARQUET,
    TIME_COLUMNS,
)

try:
    import dask.dataframe as dd
    import xgboost as xgb
    import xgboost.dask as xgb_dask
    from dask.distributed import Client
except ModuleNotFoundError:
    dd = None
    xgb = None
    xgb_dask = None
    Client = None

try:
    import mlflow
except ModuleNotFoundError:
    mlflow = None

RANDOM_SEED = 42
GRID_MINUTES = 10
STALE_CAP_MINUTES = 30
USE_SCALE_POS_WEIGHT = False
CLASSIFICATION_THRESHOLD = 0.5

CPU_COUNT = os.cpu_count() or 1

DEFAULT_DASK_N_WORKERS = 3
DEFAULT_DASK_THREADS_PER_WORKER = max(1, CPU_COUNT // DEFAULT_DASK_N_WORKERS)

FEATURE_COLUMNS = [
    *MODEL_STATIC_FEATURE_COLUMNS,
    *FRESHNESS_COLUMNS,
    *TIME_COLUMNS,
]

MODELS_DIR = (
    MODELS_DIR / "with_scale_pos_weight" if USE_SCALE_POS_WEIGHT else MODELS_DIR
)


def resolve_positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}={raw!r}. Use a positive integer.") from exc

    if value > 0:
        return value

    raise ValueError(f"Invalid {name}={value}. Use a positive integer.")


DASK_N_WORKERS = resolve_positive_int_env(
    "EV_BUDDY_DASK_N_WORKERS", DEFAULT_DASK_N_WORKERS
)
DASK_THREADS_PER_WORKER = resolve_positive_int_env(
    "EV_BUDDY_DASK_THREADS_PER_WORKER",
    DEFAULT_DASK_THREADS_PER_WORKER,
)


def configure_mlflow() -> bool:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if not tracking_uri:
        return False

    if mlflow is None:
        print(
            "TRAIN_MODELS: MLFLOW_TRACKING_URI is set but mlflow is not installed. "
            "Skipping MLflow tracking."
        )
        return False

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "evbuddy-train-models")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(
        f"TRAIN_MODELS: MLflow enabled (uri={tracking_uri}, experiment={experiment_name})"
    )
    return True


def log_mlflow_run(
    requested_horizon: int,
    metrics: dict[str, object],
    model_path: Any,
    metrics_path: Any,
) -> None:
    if mlflow is None:
        return

    run_name = f"train_models_h{requested_horizon}m"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "pipeline": "evbuddy",
                "stage": "train_models",
                "trainer": "xgboost_dask",
            }
        )

        for key in (
            "requested_horizon_minutes",
            "effective_horizon_minutes",
            "rows_horizon",
            "rows_train",
            "rows_valid",
            "dask_n_workers",
            "dask_threads_per_worker",
            "use_scale_pos_weight",
        ):
            if key in metrics and metrics[key] is not None:
                mlflow.log_param(key, metrics[key])

        for key in (
            "auc",
            "logloss",
            "brier",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "positive_rate_train",
            "positive_rate_valid",
        ):
            value = metrics.get(key)
            if value is not None:
                mlflow.log_metric(key, float(value))

        artifact_dir = f"h{requested_horizon}m"
        mlflow.log_artifact(str(model_path), artifact_path=artifact_dir)
        mlflow.log_artifact(str(metrics_path), artifact_path=artifact_dir)


def required_columns() -> list[str]:
    return list(
        dict.fromkeys(
            ["st_id", "ts", "age_minutes", "available_ports", *FEATURE_COLUMNS]
        )
    )


def load_dense_dataset() -> Any:
    columns = required_columns()

    if PROCESSED_DENSE_10MIN_PARQUET.exists():
        ddf = dd.read_parquet(
            PROCESSED_DENSE_10MIN_PARQUET, columns=columns, split_row_groups=True
        )
        input_path = PROCESSED_DENSE_10MIN_PARQUET
    elif PROCESSED_DENSE_10MIN_CSV.exists():
        ddf = dd.read_csv(
            PROCESSED_DENSE_10MIN_CSV,
            usecols=columns,
            assume_missing=True,
        )
        input_path = PROCESSED_DENSE_10MIN_CSV
    else:
        raise FileNotFoundError(
            "Processed dense dataset not found. Run `python -m src.features.feature_transform` first."
        )

    ddf["ts"] = dd.to_datetime(ddf["ts"], errors="coerce")
    ddf = ddf.dropna(subset=["ts"])

    missing = sorted(set(columns).difference(ddf.columns))
    if missing:
        raise KeyError(f"Missing required columns in dense dataset: {missing}")

    print(
        "TRAIN_MODELS: "
        f"Loaded {input_path} with {ddf.npartitions} partitions "
        f"(workers={DASK_N_WORKERS}, threads_per_worker={DASK_THREADS_PER_WORKER})"
    )
    return ddf


def build_horizon_dataset(ddf: Any, requested_horizon_minutes: int) -> tuple[Any, int]:
    offset_steps = math.ceil(requested_horizon_minutes / GRID_MINUTES)
    effective_horizon_minutes = offset_steps * GRID_MINUTES

    working = ddf.loc[:, required_columns()]

    def add_label_columns(pdf: Any) -> Any:
        sorted_pdf = pdf.sort_values(["st_id", "ts"])
        shifted = sorted_pdf.groupby("st_id", sort=False)["available_ports"].shift(
            -offset_steps
        )
        sorted_pdf["available_ports_at_label"] = shifted
        eligible_mask = (
            sorted_pdf["age_minutes"].le(STALE_CAP_MINUTES)
            & sorted_pdf["available_ports"].notna()
            & sorted_pdf["available_ports_at_label"].notna()
        )
        eligible_pdf = sorted_pdf.loc[eligible_mask].assign(
            target=lambda df: (df["available_ports_at_label"] == 0).astype("int8")
        )
        return eligible_pdf

    meta = working._meta.assign(
        available_ports_at_label=np.float32(),
        target=np.int8(),
    )
    horizon_ddf = working.map_partitions(add_label_columns, meta=meta)
    return horizon_ddf, effective_horizon_minutes


def split_train_valid(horizon_ddf: Any) -> tuple[Any, Any]:
    ts_series = horizon_ddf["ts"]
    ts_ns = ts_series.astype("int64")
    split_ns = ts_ns.quantile(0.8).compute()
    if split_ns is None:
        raise ValueError("Could not compute train/validation split timestamp.")
    split_ns = int(split_ns)

    train_ddf = horizon_ddf[ts_ns < split_ns]
    valid_ddf = horizon_ddf[ts_ns >= split_ns]

    rows_train = int(train_ddf.shape[0].compute())
    rows_valid = int(valid_ddf.shape[0].compute())
    if rows_train == 0 or rows_valid == 0:
        raise ValueError("Train/validation split produced an empty partition.")

    return train_ddf, valid_ddf


def train_single_horizon(
    client: Any, ddf: Any, requested_horizon_minutes: int
) -> tuple[Any, dict[str, object]]:
    horizon_ddf, effective_horizon_minutes = build_horizon_dataset(
        ddf,
        requested_horizon_minutes=requested_horizon_minutes,
    )

    rows_horizon = int(horizon_ddf.shape[0].compute())
    if rows_horizon == 0:
        raise ValueError(
            f"No eligible rows found for requested horizon {requested_horizon_minutes}m."
        )

    train_ddf, valid_ddf = split_train_valid(horizon_ddf)

    X_train = train_ddf[FEATURE_COLUMNS].astype("float32")
    X_valid = valid_ddf[FEATURE_COLUMNS].astype("float32")
    y_train = train_ddf["target"].astype("int8")
    y_valid = valid_ddf["target"].astype("int8")

    rows_train = int(train_ddf.shape[0].compute())
    rows_valid = int(valid_ddf.shape[0].compute())
    positive_train = int(y_train.sum().compute())
    negative_train = rows_train - positive_train

    if positive_train == 0 or negative_train == 0:
        raise ValueError(
            "Training data has only one class for horizon "
            f"{requested_horizon_minutes}m (pos={positive_train}, neg={negative_train})."
        )

    scale_pos_weight = negative_train / positive_train if USE_SCALE_POS_WEIGHT else 1.0

    dtrain = xgb_dask.DaskDMatrix(
        client, X_train, y_train, feature_names=FEATURE_COLUMNS
    )
    dvalid = xgb_dask.DaskDMatrix(
        client, X_valid, y_valid, feature_names=FEATURE_COLUMNS
    )

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.05,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "seed": RANDOM_SEED,
        "nthread": DASK_THREADS_PER_WORKER,
        "tree_method": "hist",
        "scale_pos_weight": scale_pos_weight,
    }

    training_output = xgb_dask.train(
        client,
        params,
        dtrain,
        num_boost_round=800,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    booster = training_output["booster"]

    best_iteration = getattr(booster, "best_iteration", None)
    if best_iteration is not None and best_iteration >= 0:
        valid_prob_da = xgb_dask.predict(
            client,
            booster,
            dvalid,
            iteration_range=(0, best_iteration + 1),
        )
    else:
        valid_prob_da = xgb_dask.predict(client, booster, dvalid)

    y_valid_np = y_valid.compute().to_numpy(dtype=np.int8, copy=False)
    valid_prob = valid_prob_da.compute()
    valid_pred = (valid_prob >= CLASSIFICATION_THRESHOLD).astype(int)

    auc = None
    if np.unique(y_valid_np).size > 1:
        auc = float(roc_auc_score(y_valid_np, valid_prob))

    metrics: dict[str, object] = {
        "requested_horizon_minutes": int(requested_horizon_minutes),
        "effective_horizon_minutes": int(effective_horizon_minutes),
        "rows_horizon": rows_horizon,
        "rows_train": rows_train,
        "rows_valid": rows_valid,
        "positive_rate_train": float(positive_train / rows_train),
        "positive_rate_valid": float(np.mean(y_valid_np)),
        "n_stations_train": int(train_ddf["st_id"].nunique().compute()),
        "n_stations_valid": int(valid_ddf["st_id"].nunique().compute()),
        "use_scale_pos_weight": USE_SCALE_POS_WEIGHT,
        "scale_pos_weight_value": float(scale_pos_weight),
        "feature_list": FEATURE_COLUMNS,
        "auc": auc,
        "logloss": float(log_loss(y_valid_np, valid_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_valid_np, valid_prob)),
        "classification_threshold": float(CLASSIFICATION_THRESHOLD),
        "best_iteration": (
            int(best_iteration)
            if best_iteration is not None and best_iteration >= 0
            else None
        ),
        "dask_n_workers": int(DASK_N_WORKERS),
        "dask_threads_per_worker": int(DASK_THREADS_PER_WORKER),
        "accuracy": float(accuracy_score(y_valid_np, valid_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_valid_np, valid_pred)),
        "precision": float(precision_score(y_valid_np, valid_pred, zero_division=0)),
        "recall": float(recall_score(y_valid_np, valid_pred, zero_division=0)),
        "f1": float(f1_score(y_valid_np, valid_pred, zero_division=0)),
    }

    return booster, metrics


def main() -> None:
    if dd is None or xgb is None or xgb_dask is None or Client is None:
        raise ModuleNotFoundError(
            "Dask training dependencies are not installed. "
            "Install with `poetry add --group dev dask distributed` and retry."
        )

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    mlflow_enabled = configure_mlflow()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(
        "TRAIN_MODELS: "
        f"workers={DASK_N_WORKERS}, threads_per_worker={DASK_THREADS_PER_WORKER}, "
    )

    client = Client(
        n_workers=DASK_N_WORKERS,
        threads_per_worker=DASK_THREADS_PER_WORKER,
        processes=True,
        dashboard_address=None,
    )

    try:
        ddf = load_dense_dataset()

        for requested_horizon in HORIZON_MINUTES:
            booster, metrics = train_single_horizon(client, ddf, requested_horizon)

            model_path = MODELS_DIR / f"xgb_occupied_h{requested_horizon}m.json"
            metrics_path = MODELS_DIR / f"metrics_h{requested_horizon}m.json"

            booster.save_model(model_path)
            with metrics_path.open("w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2)
            if mlflow_enabled:
                log_mlflow_run(requested_horizon, metrics, model_path, metrics_path)

            print(
                "TRAIN_MODELS: "
                f"h={requested_horizon}m (effective={metrics['effective_horizon_minutes']}m), "
                f"train={metrics['rows_train']:,}, valid={metrics['rows_valid']:,}, "
                f"auc={metrics['auc']}, logloss={metrics['logloss']:.6f}, "
                f"brier={metrics['brier']:.6f}"
            )
            print(f"TRAIN_MODELS: Wrote {model_path}")
            print(f"TRAIN_MODELS: Wrote {metrics_path}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
