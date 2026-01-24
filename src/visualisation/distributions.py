from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import INTERIM_LOCATIONS_CSV, INTERIM_PORTS_CSV, INTERIM_STATIONS_CSV  # noqa: E402

DATASETS = {
    "locations": {
        "input": INTERIM_LOCATIONS_CSV,
        "output": Path("reports/figures/distributions/interim/locations"),
        "exclude": {"stations"},
    },
    "stations": {
        "input": INTERIM_STATIONS_CSV,
        "output": Path("reports/figures/distributions/interim/stations"),
        "exclude": {"ports"},
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

    labels = counts.index.tolist()
    values = counts.values.tolist()
    width = max(8, min(0.5 * len(labels), 24))
    fig, ax = plt.subplots(figsize=(width, 6))
    ax.bar(labels, values, color="#2f6f8f")
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
