from __future__ import annotations

import json
import math
import random

import numpy as np
import pandas as pd
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
    STATE_COLUMNS,
    TIME_COLUMNS,
)

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:
    XGBClassifier = None

RANDOM_SEED = 42
GRID_MINUTES = 10
STALE_CAP_MINUTES = 30
USE_SCALE_POS_WEIGHT = False
CLASSIFICATION_THRESHOLD = 0.5

FEATURE_COLUMNS = [
    *MODEL_STATIC_FEATURE_COLUMNS,
    *FRESHNESS_COLUMNS,
    *TIME_COLUMNS,
]

MODELS_DIR = (
    MODELS_DIR / "with_scale_pos_weight" if USE_SCALE_POS_WEIGHT else MODELS_DIR
)


def load_dense_dataset() -> pd.DataFrame:
    if PROCESSED_DENSE_10MIN_PARQUET.exists():
        try:
            df = pd.read_parquet(PROCESSED_DENSE_10MIN_PARQUET)
            input_path = PROCESSED_DENSE_10MIN_PARQUET
        except (ImportError, ModuleNotFoundError, ValueError) as exc:
            if not PROCESSED_DENSE_10MIN_CSV.exists():
                raise FileNotFoundError(
                    "Parquet output exists but could not be read and CSV fallback is missing. "
                    f"Parquet path: {PROCESSED_DENSE_10MIN_PARQUET}"
                ) from exc
            df = pd.read_csv(PROCESSED_DENSE_10MIN_CSV)
            input_path = PROCESSED_DENSE_10MIN_CSV
    elif PROCESSED_DENSE_10MIN_CSV.exists():
        df = pd.read_csv(PROCESSED_DENSE_10MIN_CSV)
        input_path = PROCESSED_DENSE_10MIN_CSV
    else:
        raise FileNotFoundError(
            "Processed dense dataset not found. Run `python -m src.features.feature_transform` first."
        )

    parsed_ts = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    invalid_count = int(parsed_ts.isna().sum())
    if invalid_count > 0:
        raise ValueError(f"Found {invalid_count} invalid timestamps in ts column.")

    df["ts"] = parsed_ts
    df = df.sort_values(["st_id", "ts"]).reset_index(drop=True)

    print(f"TRAIN_MODELS: Loaded {input_path} ({len(df):,} rows)")
    return df


def validate_columns(df: pd.DataFrame) -> None:
    required_columns = {
        "st_id",
        "st_location_id",
        "ts",
        *FEATURE_COLUMNS,
        *STATE_COLUMNS,
    }
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise KeyError(f"Missing required columns in dense dataset: {missing}")


def build_horizon_dataset(
    df: pd.DataFrame,
    requested_horizon_minutes: int,
) -> tuple[pd.DataFrame, int]:
    offset_steps = math.ceil(requested_horizon_minutes / GRID_MINUTES)
    effective_horizon_minutes = offset_steps * GRID_MINUTES

    horizon_df = df.copy()
    horizon_df["available_ports_at_label"] = horizon_df.groupby("st_id", sort=False)[
        "available_ports"
    ].shift(-offset_steps)

    eligible_mask = (
        horizon_df["age_minutes"].le(STALE_CAP_MINUTES)
        & horizon_df["available_ports"].notna()
        & horizon_df["available_ports_at_label"].notna()
    )
    horizon_df = horizon_df.loc[eligible_mask].copy()
    horizon_df["target"] = (horizon_df["available_ports_at_label"] == 0).astype(int)

    return horizon_df, effective_horizon_minutes


def split_train_valid(horizon_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_ts = pd.Index(horizon_df["ts"].dropna().sort_values().unique())
    if len(unique_ts) < 2:
        raise ValueError(
            "Not enough unique timestamps to create train/validation split."
        )

    split_idx = int(len(unique_ts) * 0.8)
    split_idx = max(1, min(split_idx, len(unique_ts) - 1))

    train_ts = unique_ts[:split_idx]
    valid_ts = unique_ts[split_idx:]

    train_df = horizon_df[horizon_df["ts"].isin(train_ts)].copy()
    valid_df = horizon_df[horizon_df["ts"].isin(valid_ts)].copy()

    if train_df.empty or valid_df.empty:
        raise ValueError("Train/validation split produced an empty partition.")

    return train_df, valid_df


def train_single_horizon(
    df: pd.DataFrame,
    requested_horizon_minutes: int,
) -> tuple[XGBClassifier, dict[str, object]]:
    horizon_df, effective_horizon_minutes = build_horizon_dataset(
        df,
        requested_horizon_minutes=requested_horizon_minutes,
    )

    if horizon_df.empty:
        raise ValueError(
            f"No eligible rows found for requested horizon {requested_horizon_minutes}m."
        )

    train_df, valid_df = split_train_valid(horizon_df)

    X_train = (
        train_df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").astype(float)
    )
    X_valid = (
        valid_df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").astype(float)
    )
    y_train = train_df["target"].astype(int)
    y_valid = valid_df["target"].astype(int)

    positive_train = int(y_train.sum())
    negative_train = int((y_train == 0).sum())
    if positive_train == 0 or negative_train == 0:
        raise ValueError(
            "Training data has only one class for horizon "
            f"{requested_horizon_minutes}m (pos={positive_train}, neg={negative_train})."
        )

    scale_pos_weight = negative_train / positive_train if USE_SCALE_POS_WEIGHT else 1.0

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=30,
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    valid_prob = model.predict_proba(X_valid)[:, 1]
    valid_pred = (valid_prob >= CLASSIFICATION_THRESHOLD).astype(int)

    auc = None
    if y_valid.nunique() > 1:
        auc = float(roc_auc_score(y_valid, valid_prob))

    metrics: dict[str, object] = {
        "requested_horizon_minutes": int(requested_horizon_minutes),
        "effective_horizon_minutes": int(effective_horizon_minutes),
        "rows_train": int(len(train_df)),
        "rows_valid": int(len(valid_df)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_valid": float(y_valid.mean()),
        "n_stations_train": int(train_df["st_id"].nunique()),
        "n_stations_valid": int(valid_df["st_id"].nunique()),
        "use_scale_pos_weight": USE_SCALE_POS_WEIGHT,
        "scale_pos_weight_value": float(scale_pos_weight),
        "feature_list": FEATURE_COLUMNS,
        "auc": auc,
        "logloss": float(log_loss(y_valid, valid_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_valid, valid_prob)),
        "classification_threshold": float(CLASSIFICATION_THRESHOLD),
        "accuracy": float(accuracy_score(y_valid, valid_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_valid, valid_pred)),
        "precision": float(precision_score(y_valid, valid_pred, zero_division=0)),
        "recall": float(recall_score(y_valid, valid_pred, zero_division=0)),
        "f1": float(f1_score(y_valid, valid_pred, zero_division=0)),
    }

    return model, metrics


def main() -> None:
    if XGBClassifier is None:
        raise ModuleNotFoundError(
            "xgboost is not installed in this environment. "
            "Install dependencies with `poetry install` and retry."
        )

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    dense_df = load_dense_dataset()
    validate_columns(dense_df)

    for requested_horizon in HORIZON_MINUTES:
        model, metrics = train_single_horizon(dense_df, requested_horizon)

        model_path = MODELS_DIR / f"xgb_occupied_h{requested_horizon}m.json"
        metrics_path = MODELS_DIR / f"metrics_h{requested_horizon}m.json"

        model.save_model(model_path)
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        print(
            "TRAIN_MODELS: "
            f"h={requested_horizon}m (effective={metrics['effective_horizon_minutes']}m), "
            f"train={metrics['rows_train']:,}, valid={metrics['rows_valid']:,}, "
            f"auc={metrics['auc']}, logloss={metrics['logloss']:.6f}, brier={metrics['brier']:.6f}"
        )
        print(f"TRAIN_MODELS: Wrote {model_path}")
        print(f"TRAIN_MODELS: Wrote {metrics_path}")


if __name__ == "__main__":
    main()
