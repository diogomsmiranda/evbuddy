from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import INTERIM_TIMESERIES_SELECTED_CSV

OUTPUT_DIR = Path("reports/figures/sparsity")


def plot_records_per_day(df: pd.DataFrame, output_path: Path) -> None:
    if "snapshot_ts" not in df.columns:
        raise KeyError("Column 'snapshot_ts' not found.")

    ts = pd.to_datetime(df["snapshot_ts"], errors="coerce", utc=True)
    if ts.isna().all():
        raise ValueError("snapshot_ts could not be parsed as datetime.")

    counts = ts.dropna().dt.floor("D").value_counts().sort_index()
    if counts.empty:
        raise ValueError("No valid snapshot_ts values to plot.")

    full_index = pd.date_range(
        start=counts.index.min(), end=counts.index.max(), freq="D"
    )
    counts = counts.reindex(full_index, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(counts.index, counts.values, color="#2f6f8f", width=0.9)
    ax.set_title("Records per day")
    ax.set_xlabel("Day")
    ax.set_ylabel("Nr records")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_path}")


def plot_daily_coverage(df: pd.DataFrame, output_path: Path) -> None:
    if "snapshot_ts" not in df.columns:
        raise KeyError("Column 'snapshot_ts' not found.")

    ts = pd.to_datetime(df["snapshot_ts"], errors="coerce", utc=True)
    if ts.isna().all():
        raise ValueError("snapshot_ts could not be parsed as datetime.")

    counts = ts.dropna().dt.floor("D").value_counts().sort_index()
    if counts.empty:
        raise ValueError("No valid snapshot_ts values to plot.")

    full_index = pd.date_range(
        start=counts.index.min(), end=counts.index.max(), freq="D"
    )
    counts = counts.reindex(full_index, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.bar(counts.index, counts.values, color="#2f6f8f", width=1.0)
    ax.set_title("Records per day (full range)")
    ax.set_xlabel("Day")
    ax.set_ylabel("Nr records")
    ax.set_ylim(bottom=0)
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INTERIM_TIMESERIES_SELECTED_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found at {INTERIM_TIMESERIES_SELECTED_CSV}"
        )

    df = pd.read_csv(INTERIM_TIMESERIES_SELECTED_CSV)

    plot_records_per_day(df, OUTPUT_DIR / "records_per_day.png")
    plot_daily_coverage(df, OUTPUT_DIR / "records_per_day_full.png")


if __name__ == "__main__":
    main()
