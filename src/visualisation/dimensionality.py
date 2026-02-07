from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import (
    INTERIM_LOCATIONS_CSV,
    INTERIM_PORTS_CSV,
    INTERIM_STATIONS_CSV,
    INTERIM_TIMESERIES_ENCODED_CSV,
    RAW_OPENDATA_CSV_DIR,
)

INTERIM_DATASETS = {
    "locations": INTERIM_LOCATIONS_CSV,
    "stations": INTERIM_STATIONS_CSV,
    "ports": INTERIM_PORTS_CSV,
    "timeseries": INTERIM_TIMESERIES_ENCODED_CSV,
}
OUTPUT_RAW_DIR = Path("reports/figures/dimensionality/raw")
OUTPUT_INTERIM_DIR = Path("reports/figures/dimensionality/interim")


def plot_records_vs_variables(df: pd.DataFrame, title: str, output_path: Path) -> None:
    values = {"nr records": df.shape[0], "nr variables": df.shape[1]}
    fig, ax = plt.subplots(figsize=(4, 2))
    labels = list(values.keys())
    heights = list(values.values())
    ax.bar(labels, heights, color="#2f6f8f")
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.set_ylim(bottom=0)
    ax.set_yticks(sorted(set([0, *heights])))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_path}")


def plot_missing_values(df: pd.DataFrame, title: str, output_path: Path) -> None:
    missing = df.isna().sum()
    if missing.empty:
        print(f"No columns to plot for {title}")
        return

    labels = missing.index.tolist()
    values = missing.values.tolist()
    width = max(8, min(0.35 * len(labels), 24))
    fig, ax = plt.subplots(figsize=(width, 5))
    bars = ax.bar(labels, values, color="#2f6f8f")
    ax.set_title(title)
    ax.set_xlabel("Variables")
    ax.set_ylabel("Nr missing values")
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="x", labelrotation=45)
    ax.bar_label(bars, labels=[str(v) for v in values], padding=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_path}")


def run_raw() -> None:
    OUTPUT_RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(RAW_OPENDATA_CSV_DIR.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {RAW_OPENDATA_CSV_DIR}")

    for path in paths:
        df = pd.read_csv(path)
        file_tag = path.stem
        title_prefix = "_".join(file_tag.split("_")[:2]) or file_tag
        plot_records_vs_variables(
            df,
            title=f"{title_prefix}: records vs variables",
            output_path=OUTPUT_RAW_DIR / f"{file_tag}_records_variables.png",
        )
        plot_missing_values(
            df,
            title=f"{file_tag}: missing values per variable",
            output_path=OUTPUT_RAW_DIR / f"{file_tag}_missing_values.png",
        )


def run_interim(entity: str) -> None:
    input_file = INTERIM_DATASETS[entity]
    output_dir = OUTPUT_INTERIM_DIR / entity
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_file.exists():
        raise FileNotFoundError(f"Interim dataset not found at {input_file}")

    df = pd.read_csv(input_file)
    file_tag = input_file.stem
    title_prefix = "_".join(file_tag.split("_")[:2]) or file_tag
    plot_records_vs_variables(
        df,
        title=f"{title_prefix}: records vs variables",
        output_path=output_dir / f"{file_tag}_records_variables.png",
    )
    plot_missing_values(
        df,
        title=f"{file_tag}: missing values per variable",
        output_path=output_dir / f"{file_tag}_missing_values.png",
    )


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    run_raw_flag = "raw" in args
    entity = next((arg for arg in args if arg in INTERIM_DATASETS), "stations")

    run_interim(entity)
    if run_raw_flag:
        print("Running dimensionality analysis on raw data.")
        run_raw()


if __name__ == "__main__":
    main()
