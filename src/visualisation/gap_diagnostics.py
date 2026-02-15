from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from src.utils import INTERIM_TIMESERIES_SELECTED_CSV

INPUT_FILE = INTERIM_TIMESERIES_SELECTED_CSV

@dataclass
class GapStats:
    n_rows: int
    n_stations: int
    n_locations: Optional[int]
    timestamps_per_year: pd.Series
    rows_per_station_desc: pd.Series
    gap_hours_desc: pd.Series
    segment_length_desc: pd.Series
    n_segments_ge_10: int
    max_segment_len: int


def _parse_csv_list_of_ints(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def load_df(path: str, time_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if time_col not in df.columns:
        raise ValueError(
            f"Timestamp column '{time_col}' not found. Available: {list(df.columns)}"
        )

    # Parse timestamps; keep tz-aware if possible
    ts = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    bad = ts.isna().sum()
    if bad:
        raise ValueError(
            f"{bad} rows have invalid timestamps in '{time_col}'. Fix/clean and re-run."
        )
    df[time_col] = ts
    return df


def compute_gap_stats(
    df: pd.DataFrame,
    time_col: str = "snapshot_ts",
    station_col: str = "st_id",
    location_col: Optional[str] = "st_location_id",
    max_gap_hours_for_segment: float = 6.0,
) -> GapStats:
    if station_col not in df.columns:
        raise ValueError(
            f"Station column '{station_col}' not found. Available: {list(df.columns)}"
        )

    # Basic counts
    n_rows = len(df)
    n_stations = df[station_col].nunique(dropna=True)
    n_locations = (
        df[location_col].nunique(dropna=True)
        if (location_col and location_col in df.columns)
        else None
    )
    timestamps_per_year = (
        df[time_col].dt.year.value_counts().sort_index().rename("timestamps")
    )

    # Rows per station
    rows_per_station = df.groupby(station_col, dropna=True).size()
    rows_per_station_desc = rows_per_station.describe(
        percentiles=[0.25, 0.5, 0.75]
    ).rename("rows_per_station")

    # Gaps within station
    d = df[[station_col, time_col]].sort_values([station_col, time_col]).copy()
    d["gap_hours"] = d.groupby(station_col)[time_col].diff().dt.total_seconds() / 3600.0
    gaps = d["gap_hours"].dropna()

    gap_hours_desc = gaps.describe(percentiles=[0.25, 0.5, 0.75]).rename("gap_hours")

    # Segment lengths: consecutive points where gap <= threshold belong to same segment
    # New segment starts when gap is NaN (first point) or gap > threshold.
    thr = float(max_gap_hours_for_segment)
    d["new_segment"] = (d["gap_hours"].isna()) | (d["gap_hours"] > thr)
    d["segment_id"] = d.groupby(station_col)["new_segment"].cumsum()

    seg_lengths = d.groupby([station_col, "segment_id"]).size()
    segment_length_desc = seg_lengths.describe(percentiles=[0.25, 0.5, 0.75]).rename(
        "segment_length"
    )

    n_segments_ge_10 = int((seg_lengths >= 10).sum())
    max_segment_len = int(seg_lengths.max()) if len(seg_lengths) else 0

    return GapStats(
        n_rows=n_rows,
        n_stations=n_stations,
        n_locations=n_locations,
        timestamps_per_year=timestamps_per_year,
        rows_per_station_desc=rows_per_station_desc,
        gap_hours_desc=gap_hours_desc,
        segment_length_desc=segment_length_desc,
        n_segments_ge_10=n_segments_ge_10,
        max_segment_len=max_segment_len,
    )


def label_coverage(
    df: pd.DataFrame,
    time_col: str = "snapshot_ts",
    station_col: str = "st_id",
    travel_times_min: Iterable[int] = (5, 10, 15, 30, 60),
    tolerance_min: int = 2,
) -> pd.DataFrame:
    """
    Computes what fraction of rows have an observed record near (t + travel_time) within ±tolerance.

    Method:
      For each station, we sort timestamps and, for each row timestamp t, look for an observed
      timestamp within [t+dt - tol, t+dt + tol]. Uses numpy searchsorted for efficiency.

    Returns a dataframe with columns:
      travel_time_min, tolerance_min, matched_rows, total_rows, match_rate
    """
    if station_col not in df.columns or time_col not in df.columns:
        raise ValueError("Missing required columns for label_coverage.")

    dt_list = list(travel_times_min)
    tol = pd.Timedelta(minutes=int(tolerance_min))

    d = df[[station_col, time_col]].sort_values([station_col, time_col]).copy()
    results = []

    # group by station for per-station searchsorted
    for dt_min in dt_list:
        dt = pd.Timedelta(minutes=int(dt_min))
        matched_total = 0
        total = 0

        for _, g in d.groupby(station_col):
            ts = g[time_col].to_numpy(dtype="datetime64[ns]")
            if len(ts) == 0:
                continue

            total += len(ts)
            # target windows
            target = ts + np.timedelta64(int(dt.total_seconds() * 1_000_000_000), "ns")
            lo = target - np.timedelta64(int(tol.total_seconds() * 1_000_000_000), "ns")
            hi = target + np.timedelta64(int(tol.total_seconds() * 1_000_000_000), "ns")

            # Find first index >= lo; if that index timestamp <= hi, it's a match
            idx = np.searchsorted(ts, lo, side="left")
            in_bounds = idx < len(ts)
            ok = np.zeros_like(in_bounds, dtype=bool)
            if in_bounds.any():
                ok[in_bounds] = ts[idx[in_bounds]] <= hi[in_bounds]
            matched_total += int(ok.sum())

        results.append(
            {
                "travel_time_min": int(dt_min),
                "tolerance_min": int(tolerance_min),
                "matched_rows": int(matched_total),
                "total_rows": int(total),
                "match_rate": (matched_total / total) if total else 0.0,
            }
        )

    return pd.DataFrame(results).sort_values("travel_time_min").reset_index(drop=True)


def print_gap_report(stats: GapStats) -> None:
    print("\n=== Basic shape ===")
    print(f"Rows: {stats.n_rows:,}")
    print(f"Unique stations: {stats.n_stations:,}")
    if stats.n_locations is not None:
        print(f"Unique locations: {stats.n_locations:,}")

    print("\n=== Timestamps per year ===")
    print(stats.timestamps_per_year.to_string())

    print("\n=== Rows per station (describe) ===")
    print(stats.rows_per_station_desc.to_string())

    print(
        "\n=== Gap (hours) between consecutive snapshots within station (describe) ==="
    )
    print(stats.gap_hours_desc.to_string())

    print("\n=== Continuous segment lengths (gap <= threshold) (describe) ===")
    print(stats.segment_length_desc.to_string())
    print(f"Segments with length >= 10: {stats.n_segments_ge_10:,}")
    print(f"Max segment length: {stats.max_segment_len:,}")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv-path",
        default=str(INPUT_FILE),
        help="Path to CSV file (defaults to selected timeseries)",
    )
    ap.add_argument("--time-col", default="snapshot_ts", help="Timestamp column name")
    ap.add_argument("--station-col", default="st_id", help="Station ID column name")
    ap.add_argument(
        "--location-col", default="st_location_id", help="Location ID column name"
    )
    ap.add_argument(
        "--gap-hours",
        type=float,
        default=6.0,
        help="Max gap (hours) to consider a segment continuous",
    )
    ap.add_argument(
        "--label-times-min",
        default="5,10,15,30,60",
        help="Comma-separated travel times (minutes) to check label coverage",
    )
    ap.add_argument(
        "--label-tolerance-min",
        type=int,
        default=2,
        help="Tolerance window (minutes) for label coverage",
    )
    ap.add_argument(
        "--output-csv",
        default="",
        help="Optional path to write label coverage table as CSV",
    )
    args = ap.parse_args(argv)

    df = load_df(args.csv_path, args.time_col)

    # Gap stats
    stats = compute_gap_stats(
        df,
        time_col=args.time_col,
        station_col=args.station_col,
        location_col=args.location_col,
        max_gap_hours_for_segment=args.gap_hours,
    )
    print_gap_report(stats)

    # Optional: occupied rate quick check if columns exist
    if "available_ports" in df.columns:
        occupied = (df["available_ports"] == 0).mean()
        print("\n=== Quick label proxy (available_ports == 0) ===")
        print(f"Occupied rate (raw snapshots): {occupied:.4f}")

    # Label coverage
    travel_times = _parse_csv_list_of_ints(args.label_times_min)
    cov = label_coverage(
        df,
        time_col=args.time_col,
        station_col=args.station_col,
        travel_times_min=travel_times,
        tolerance_min=args.label_tolerance_min,
    )
    print("\n=== Label coverage (exists an observed snapshot near t + travel_time) ===")
    print(cov.to_string(index=False, formatters={"match_rate": "{:.6f}".format}))

    if args.output_csv:
        cov.to_csv(args.output_csv, index=False)
        print(f"\nWrote label coverage table to: {args.output_csv}")


if __name__ == "__main__":
    main()
