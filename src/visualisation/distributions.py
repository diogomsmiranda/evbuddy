from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import INTERIM_LOCATIONS_CSV, INTERIM_PORTS_CSV, INTERIM_STATIONS_CSV

DATASETS = {
    "locations": {
        "input": INTERIM_LOCATIONS_CSV,
        "output": Path("reports/figures/distributions/interim/locations"),
        "exclude": {"loc_stations"},
    },
    "stations": {
        "input": INTERIM_STATIONS_CSV,
        "output": Path("reports/figures/distributions/interim/stations"),
        "exclude": {"st_ports"},
    },
    "ports": {
        "input": INTERIM_PORTS_CSV,
        "output": Path("reports/figures/distributions/interim/ports"),
        "exclude": set(),
    },
}


def plot_value_counts(series: pd.Series, title: str, output_path: Path) -> None:
    counts = (
        series.astype("string").fillna("<NA>").value_counts(dropna=False).sort_index()
    )
    # print counts for the column "last_updated" if title == "last_updated"
    if "last_updated" in title:
        print(f"Value counts for column '{title}':\n{counts}\n")

    labels = counts.index.tolist()
    values = counts.values.tolist()
    width = max(8, min(0.5 * len(labels), 24))
    fig, ax = plt.subplots(figsize=(width, 6))
    bars = ax.bar(labels, values, color="#2f6f8f")
    ax.bar_label(
        bars, labels=[str(v) for v in values], padding=2, fontsize=8
    )  # only ran with "ports" because of bar density
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_path}")


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    entity = args[0] if args and args[0] in DATASETS else "stations"
    config = DATASETS[entity]
    input_file: Path = config["input"]
    output_dir: Path = config["output"]
    exclude_columns: set[str] = config["exclude"]

    if not input_file.exists():
        raise FileNotFoundError(f"Dataset not found at {input_file}")

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_file)

    for column in df.columns:
        if column in exclude_columns:
            continue
        safe_name = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in column
        )
        output_path = output_dir / f"{safe_name}.png"
        print(f"Plotting distribution for column '{column}'...")
        plot_value_counts(df[column], title=column, output_path=output_path)


if __name__ == "__main__":
    main()
