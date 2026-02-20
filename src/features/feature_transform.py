from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

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
OUTPUT_COLUMNS = [
    "st_id",
    "ts",
    *STATE_COLUMNS,
    *STATIC_COLUMNS,
    *FRESHNESS_COLUMNS,
    *TIME_COLUMNS,
]
ENABLE_FUNCTION_MEMORY_PROFILE = (
    os.getenv("EV_BUDDY_FUNCTION_MEMORY_PROFILE", "0") == "1"
)

try:
    from memory_profiler import memory_usage as _memory_usage
    from memory_profiler import profile as _line_profile
except ModuleNotFoundError:
    _memory_usage = None
    _line_profile = None

memory_usage = _memory_usage


def profile(func: Callable[..., Any]) -> Callable[..., Any]:
    if not ENABLE_FUNCTION_MEMORY_PROFILE or _line_profile is None:
        return func
    return _line_profile(func)


def run_profiled_step(
    step_name: str,
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    if not ENABLE_FUNCTION_MEMORY_PROFILE or memory_usage is None:
        return fn(*args, **kwargs)

    mem_trace, result = memory_usage(
        (fn, args, kwargs),
        interval=0.1,
        retval=True,
        include_children=True,
    )
    if mem_trace:
        mem_min = min(mem_trace)
        mem_max = max(mem_trace)
        print(
            "FEATURE_TRANSFORM_MEM: "
            f"{step_name} min={mem_min:.1f} MiB max={mem_max:.1f} MiB delta={mem_max - mem_min:.1f} MiB"
        )
    return result


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


def format_month_year(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m")


@profile
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


@profile
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


@profile
def write_processed_dataset_streaming(
    bucketed_df: pd.DataFrame,
) -> tuple[Path, int, int]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    total_stations = int(bucketed_df["st_id"].nunique())
    if total_stations == 0:
        raise ValueError("No station data available to build the dense grid.")

    writer = None
    total_rows = 0
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        if PROCESSED_DENSE_10MIN_PARQUET.exists():
            PROCESSED_DENSE_10MIN_PARQUET.unlink()

        for idx, (_, station_df) in enumerate(
            bucketed_df.groupby("st_id", sort=False),
            start=1,
        ):
            station_id = station_df["st_id"].iloc[0]
            station_min_ts = station_df["ts"].min()
            station_max_ts = station_df["ts"].max()
            print(
                "FEATURE_TRANSFORM: "
                f"station {station_id} ({idx}/{total_stations}) "
                f"{format_month_year(station_min_ts)} -> {format_month_year(station_max_ts)}"
            )

            station_dense_df = build_station_dense_grid(station_df).loc[
                :, OUTPUT_COLUMNS
            ]
            table = pa.Table.from_pandas(station_dense_df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    PROCESSED_DENSE_10MIN_PARQUET, table.schema, compression="snappy"
                )
            writer.write_table(table)
            total_rows += len(station_dense_df)

        if writer is None:
            raise ValueError("No station data was written to Parquet.")

        if PROCESSED_DENSE_10MIN_CSV.exists():
            PROCESSED_DENSE_10MIN_CSV.unlink()
        print(f"FEATURE_TRANSFORM: Wrote {PROCESSED_DENSE_10MIN_PARQUET}")
        return PROCESSED_DENSE_10MIN_PARQUET, total_rows, total_stations
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        if writer is not None:
            writer.close()
            writer = None
        if PROCESSED_DENSE_10MIN_PARQUET.exists():
            PROCESSED_DENSE_10MIN_PARQUET.unlink()
        if PROCESSED_DENSE_10MIN_CSV.exists():
            PROCESSED_DENSE_10MIN_CSV.unlink()

        total_rows = 0
        header = True
        for idx, (_, station_df) in enumerate(
            bucketed_df.groupby("st_id", sort=False),
            start=1,
        ):
            station_id = station_df["st_id"].iloc[0]
            station_min_ts = station_df["ts"].min()
            station_max_ts = station_df["ts"].max()
            print(
                "FEATURE_TRANSFORM: "
                f"station {station_id} ({idx}/{total_stations}) "
                f"{format_month_year(station_min_ts)} -> {format_month_year(station_max_ts)}"
            )

            station_dense_df = build_station_dense_grid(station_df).loc[
                :, OUTPUT_COLUMNS
            ]
            mode = "w" if header else "a"
            station_dense_df.to_csv(
                PROCESSED_DENSE_10MIN_CSV,
                mode=mode,
                header=header,
                index=False,
            )
            total_rows += len(station_dense_df)
            header = False

        print(
            "FEATURE_TRANSFORM: Parquet unavailable, wrote "
            f"{PROCESSED_DENSE_10MIN_CSV} ({exc})"
        )
        return PROCESSED_DENSE_10MIN_CSV, total_rows, total_stations
    finally:
        if writer is not None:
            writer.close()


def main() -> None:
    if ENABLE_FUNCTION_MEMORY_PROFILE and memory_usage is None:
        print(
            "FEATURE_TRANSFORM_MEM: memory_profiler is not installed. "
            "Run `poetry install --with dev` to enable function-level memory profiling."
        )

    if not INTERIM_TIMESERIES_SELECTED_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found at {INTERIM_TIMESERIES_SELECTED_CSV}"
        )

    raw_df = pd.read_csv(INTERIM_TIMESERIES_SELECTED_CSV)
    validate_columns(raw_df)

    bucketed_df = run_profiled_step(
        "parse_and_bucket_snapshots",
        parse_and_bucket_snapshots,
        raw_df,
    )
    output_path, total_rows, total_stations = run_profiled_step(
        "write_processed_dataset_streaming",
        write_processed_dataset_streaming,
        bucketed_df,
    )
    del bucketed_df
    del raw_df

    print(
        "FEATURE_TRANSFORM: "
        f"rows={total_rows:,}, stations={total_stations:,}, "
        f"output={output_path}"
    )

    return None


if __name__ == "__main__":
    main()
