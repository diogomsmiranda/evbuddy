from __future__ import annotations

import pandas as pd
import pytest

from src.models.train_models_pandas import (
    FEATURE_COLUMNS,
    build_horizon_dataset,
    split_train_valid,
    train_single_horizon,
    validate_columns,
)
from src.utils import STATE_COLUMNS


def test_build_horizon_dataset_computes_effective_horizon_and_target() -> None:
    df = pd.DataFrame(
        {
            "st_id": [1, 1, 1, 2, 2, 2],
            "ts": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T00:10:00Z",
                    "2025-01-01T00:20:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T00:10:00Z",
                    "2025-01-01T00:20:00Z",
                ],
                utc=True,
            ),
            "age_minutes": [5, 5, 5, 5, 5, 5],
            "available_ports": [1, 0, 1, 0, 1, 0],
        }
    )

    horizon_df, effective = build_horizon_dataset(df, requested_horizon_minutes=15)
    assert effective == 20
    assert "available_ports_at_label" in horizon_df.columns
    assert "target" in horizon_df.columns
    assert len(horizon_df) > 0
    assert set(horizon_df["target"].unique()).issubset({0, 1})


def test_split_train_valid_requires_multiple_timestamps() -> None:
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2025-01-01T00:00:00Z"] * 3, utc=True),
            "st_id": [1, 1, 1],
        }
    )
    with pytest.raises(ValueError):
        split_train_valid(df)


def test_split_train_valid_produces_non_empty_partitions() -> None:
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T00:10:00Z",
                    "2025-01-01T00:20:00Z",
                    "2025-01-01T00:30:00Z",
                    "2025-01-01T00:40:00Z",
                ],
                utc=True,
            ),
            "st_id": [1, 1, 1, 1, 1],
        }
    )

    train_df, valid_df = split_train_valid(df)
    assert len(train_df) > 0
    assert len(valid_df) > 0
    assert train_df["ts"].max() < valid_df["ts"].min()


def test_validate_columns_raises_on_missing_required_fields() -> None:
    df = pd.DataFrame(
        {
            "st_id": [1],
            "ts": pd.to_datetime(["2025-01-01T00:00:00Z"], utc=True),
            # st_location_id and many required feature/state columns intentionally missing
        }
    )
    with pytest.raises(KeyError):
        validate_columns(df)


def test_train_single_horizon_raises_when_training_has_one_class() -> None:
    ts = pd.date_range("2025-01-01T00:00:00Z", periods=12, freq="10min", tz="UTC")
    base = {
        "st_id": [1] * len(ts),
        "st_location_id": [10] * len(ts),
        "ts": ts,
        "available_ports": [1] * len(ts),  # always non-zero -> target stays single-class
        "unavailable_ports": [0] * len(ts),
        "unavailable_rate": [0.0] * len(ts),
        "available_ac_ports": [1] * len(ts),
        "available_dc_ports": [0] * len(ts),
    }
    for col in FEATURE_COLUMNS:
        if col in base:
            continue
        base[col] = [0.0] * len(ts)
    for col in STATE_COLUMNS:
        if col in base:
            continue
        base[col] = [0.0] * len(ts)

    df = pd.DataFrame(base)
    with pytest.raises(ValueError, match="only one class"):
        train_single_horizon(df, requested_horizon_minutes=10)
