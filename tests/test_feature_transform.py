from __future__ import annotations

import pandas as pd

from src.features.feature_transform import (
    build_station_dense_grid,
    parse_and_bucket_snapshots,
)
from src.utils import FRESHNESS_COLUMNS, STATE_COLUMNS, STATIC_COLUMNS, TIME_COLUMNS


def test_parse_and_bucket_snapshots_deduplicates_by_station_and_bucket() -> None:
    df = pd.DataFrame(
        {
            "st_id": [1, 1, 1],
            "snapshot_ts": [
                "2025-01-01T00:01:00Z",
                "2025-01-01T00:08:00Z",
                "2025-01-01T00:11:00Z",
            ],
            "available_ports": [1, 2, 3],
        }
    )

    bucketed = parse_and_bucket_snapshots(df)

    assert len(bucketed) == 2
    assert bucketed.iloc[0]["available_ports"] == 2  # keep last in 00:00 bucket
    assert bucketed.iloc[1]["available_ports"] == 3
    assert "ts" in bucketed.columns


def test_build_station_dense_grid_adds_freshness_and_time_features() -> None:
    base_static = {col: 1 for col in STATIC_COLUMNS}

    station_df = pd.DataFrame(
        [
            {
                "st_id": 7,
                "ts": pd.Timestamp("2025-01-01T00:00:00Z"),
                **base_static,
                "available_ports": 1,
                "unavailable_ports": 0,
                "unavailable_rate": 0.0,
                "available_ac_ports": 1,
                "available_dc_ports": 0,
            },
            {
                "st_id": 7,
                "ts": pd.Timestamp("2025-01-01T00:50:00Z"),
                **base_static,
                "available_ports": 0,
                "unavailable_ports": 1,
                "unavailable_rate": 1.0,
                "available_ac_ports": 0,
                "available_dc_ports": 0,
            },
        ]
    )

    dense = build_station_dense_grid(station_df)

    for col in [*FRESHNESS_COLUMNS, *TIME_COLUMNS]:
        assert col in dense.columns

    stale_rows = dense[dense["is_stale"] == 1]
    assert not stale_rows.empty
    assert stale_rows["age_minutes"].min() > 30
    assert stale_rows[STATE_COLUMNS].isna().all().all()
