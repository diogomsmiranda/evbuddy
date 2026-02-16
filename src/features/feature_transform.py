from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import (
    FRESHNESS_COLUMNS,
    INTERIM_TIMESERIES_SELECTED_CSV,
    PROCESSED_DENSE_10MIN_CSV,
    PROCESSED_DENSE_10MIN_PARQUET,
    PROCESSED_DIR,
    STATE_COLUMNS,
    STATIC_COLUMNS,
    TIME_COLUMNS,
)

FREQUENCY_MINUTES = 10
STALE_CAP_MINUTES = 30


def validate_columns(df: pd.DataFrame) -> None:
    required_columns = {"snapshot_ts", "st_id", *STATE_COLUMNS, *STATIC_COLUMNS}
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def representative_static_value(series: pd.Series) -> object:
    non_null = series.dropna()
    if non_null.empty:
        return np.nan

    mode_values = non_null.mode(dropna=True)
    if not mode_values.empty:
        return mode_values.iloc[0]
    return non_null.iloc[0]


def parse_and_bucket_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    parsed_ts = pd.to_datetime(df["snapshot_ts"], errors="coerce", utc=True)
    invalid_count = int(parsed_ts.isna().sum())
    if invalid_count > 0:
        raise ValueError(f"Found {invalid_count} invalid timestamps in snapshot_ts.")

    working_df = df.copy()
    working_df["snapshot_ts"] = parsed_ts.dt.tz_convert("Europe/Madrid")  # sanity
    working_df = working_df.sort_values(["st_id", "snapshot_ts"]).reset_index(
        drop=True
    )  # sanity

    working_df["ts"] = working_df["snapshot_ts"].dt.floor(f"{FREQUENCY_MINUTES}min")
    working_df = working_df.sort_values(["st_id", "ts", "snapshot_ts"]).drop_duplicates(
        subset=["st_id", "ts"],
        keep="last",
    )
    working_df = working_df.sort_values(["st_id", "ts"]).reset_index(drop=True)
    return working_df


def build_station_dense_grid(station_df: pd.DataFrame) -> pd.DataFrame:
    station_id = station_df["st_id"].iloc[0]
    station_static = {
        col: representative_static_value(station_df[col]) for col in STATIC_COLUMNS
    }

    observed_df = station_df.set_index("ts")[STATE_COLUMNS].sort_index()
    observed_index = observed_df.index
    dense_index = pd.date_range(
        start=observed_index.min(),
        end=observed_index.max(),
        freq=f"{FREQUENCY_MINUTES}min",
        tz=observed_index.tz,
    )

    dense_df = observed_df.reindex(dense_index)
    dense_df.index.name = "ts"
    dense_df["st_id"] = station_id
    dense_df["is_observed"] = dense_df.index.isin(observed_index).astype(int)

    for col, value in station_static.items():
        dense_df[col] = value

    dense_df[STATE_COLUMNS] = dense_df[STATE_COLUMNS].ffill()

    observed_mask = dense_df["is_observed"] == 1
    observed_ts = dense_df.index.to_series(index=dense_df.index).where(observed_mask)
    last_observed_ts = observed_ts.ffill()

    age_minutes = (
        (dense_df.index.to_series(index=dense_df.index) - last_observed_ts)
        .dt.total_seconds()
        .div(60.0)
    )
    dense_df["age_minutes"] = age_minutes
    dense_df["is_stale"] = (dense_df["age_minutes"] > STALE_CAP_MINUTES).astype(int)

    stale_mask = dense_df["is_stale"] == 1
    dense_df.loc[stale_mask, STATE_COLUMNS] = np.nan

    dense_df["hour"] = dense_df.index.hour
    dense_df["dayofweek"] = dense_df.index.dayofweek
    dense_df["sin_hour"] = np.sin(2 * np.pi * dense_df["hour"] / 24.0)
    dense_df["cos_hour"] = np.cos(2 * np.pi * dense_df["hour"] / 24.0)
    dense_df["sin_dow"] = np.sin(2 * np.pi * dense_df["dayofweek"] / 7.0)
    dense_df["cos_dow"] = np.cos(2 * np.pi * dense_df["dayofweek"] / 7.0)

    dense_df = dense_df.reset_index()
    return dense_df


def build_dense_dataset(bucketed_df: pd.DataFrame) -> pd.DataFrame:
    station_frames: list[pd.DataFrame] = []
    for _, station_df in bucketed_df.groupby("st_id", sort=False):
        station_frames.append(build_station_dense_grid(station_df))

    if not station_frames:
        raise ValueError("No station data available to build the dense grid.")

    dense_df = pd.concat(station_frames, ignore_index=True)
    dense_df = dense_df.sort_values(["st_id", "ts"]).reset_index(drop=True)

    output_columns = [
        "st_id",
        "ts",
        *STATE_COLUMNS,
        *STATIC_COLUMNS,
        *FRESHNESS_COLUMNS,
        *TIME_COLUMNS,
    ]
    dense_df = dense_df[output_columns]
    return dense_df


def write_processed_dataset(df: pd.DataFrame) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(PROCESSED_DENSE_10MIN_PARQUET, index=False)
        print(f"FEATURE_TRANSFORM: Wrote {PROCESSED_DENSE_10MIN_PARQUET}")
        return PROCESSED_DENSE_10MIN_PARQUET
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        df.to_csv(PROCESSED_DENSE_10MIN_CSV, index=False)
        print(
            "FEATURE_TRANSFORM: Parquet unavailable, wrote "
            f"{PROCESSED_DENSE_10MIN_CSV} ({exc})"
        )
        return PROCESSED_DENSE_10MIN_CSV


def main() -> None:
    if not INTERIM_TIMESERIES_SELECTED_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found at {INTERIM_TIMESERIES_SELECTED_CSV}"
        )

    raw_df = pd.read_csv(INTERIM_TIMESERIES_SELECTED_CSV)
    validate_columns(raw_df)

    bucketed_df = parse_and_bucket_snapshots(raw_df)
    dense_df = build_dense_dataset(bucketed_df)
    output_path = write_processed_dataset(dense_df)

    print(
        "FEATURE_TRANSFORM: "
        f"rows={len(dense_df):,}, stations={dense_df['st_id'].nunique():,}, "
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()
