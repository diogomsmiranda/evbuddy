from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

INPUT_FILE = Path("data/interim/stations_extracted.csv")
OUTPUT_DIR = Path("reports/figures/distributions/interim")
EXCLUDE_COLUMNS = {"ports"}


def plot_value_counts(series: pd.Series, title: str, output_path: Path) -> None:
    counts = (
        series.astype("string")
        .fillna("<NA>")
        .value_counts(dropna=False)
        .sort_index()
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


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found at {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)

    for column in df.columns:
        if column in EXCLUDE_COLUMNS:
            continue
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in column)
        output_path = OUTPUT_DIR / f"{safe_name}.png"
        print(f"Plotting distribution for column '{column}'...")
        plot_value_counts(df[column], title=column, output_path=output_path)


if __name__ == "__main__":
    main()
