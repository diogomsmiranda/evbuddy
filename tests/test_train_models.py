from __future__ import annotations

import pandas as pd
import pytest

from src.models.train_models import (
    FEATURE_COLUMNS,
    build_horizon_dataset,
    required_columns,
    split_train_valid,
)

dd = pytest.importorskip("dask.dataframe")

def _base_frame() -> pd.DataFrame:
    ts = pd.to_datetime(
        [
            "2025-01-01 00:00:00",
            "2025-01-01 00:10:00",
            "2025-01-01 00:20:00",
            "2025-01-01 00:30:00",
            "2025-01-01 00:40:00",
            "2025-01-01 00:50:00",
        ]
    )
    base: dict[str, list[object]] = {
        "st_id": [1, 1, 1, 2, 2, 2],
        "ts": ts.tolist(),
        "age_minutes": [5, 5, 5, 5, 5, 5],
        "available_ports": [1, 0, 1, 0, 1, 0],
    }
    for col in FEATURE_COLUMNS:
        if col in base:
            continue
        base[col] = [0.0] * len(ts)
    return pd.DataFrame(base)


def test_required_columns_are_unique_and_complete() -> None:
    cols = required_columns()
    assert len(cols) == len(set(cols))
    assert "st_id" in cols
    assert "ts" in cols
    assert "age_minutes" in cols
    assert "available_ports" in cols


def test_build_horizon_dataset_dask_computes_labels() -> None:
    pdf = _base_frame()
    ddf = dd.from_pandas(pdf, npartitions=2)

    horizon_ddf, effective = build_horizon_dataset(ddf, requested_horizon_minutes=15)
    horizon_pdf = horizon_ddf.compute()

    assert effective == 20
    assert "available_ports_at_label" in horizon_pdf.columns
    assert "target" in horizon_pdf.columns
    assert len(horizon_pdf) > 0
    assert set(horizon_pdf["target"].unique()).issubset({0, 1})


def test_split_train_valid_dask_non_empty() -> None:
    pdf = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2025-01-01 00:00:00",
                    "2025-01-01 00:10:00",
                    "2025-01-01 00:20:00",
                    "2025-01-01 00:30:00",
                    "2025-01-01 00:40:00",
                    "2025-01-01 00:50:00",
                ],
            ),
            "st_id": [1, 1, 1, 1, 1, 1],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    ddf = dd.from_pandas(pdf, npartitions=3)

    train_ddf, valid_ddf = split_train_valid(ddf)
    train_pdf = train_ddf.compute()
    valid_pdf = valid_ddf.compute()

    assert not train_pdf.empty
    assert not valid_pdf.empty
    assert train_pdf["ts"].max() <= valid_pdf["ts"].min()


def test_split_train_valid_dask_raises_when_split_empty() -> None:
    pdf = pd.DataFrame(
        {
            "st_id": [1, 1, 1, 1],
            "ts": pd.to_datetime(["2025-01-01 00:00:00"] * 4),
            "target": [0, 1, 0, 1],
        }
    )
    ddf = dd.from_pandas(pdf, npartitions=2)

    with pytest.raises(ValueError, match="empty partition"):
        split_train_valid(ddf)
