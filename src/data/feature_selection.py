from __future__ import annotations

from pathlib import Path

from sklearn.feature_selection import VarianceThreshold
import pandas as pd

from src.utils import INTERIM_TIMESERIES_CSV

INPUT_FILE = INTERIM_TIMESERIES_CSV.parent / "stations_timeseries_encoded.csv"
OUTPUT_FILE = INTERIM_TIMESERIES_CSV.parent / "stations_timeseries_selected.csv"


def feature_selection_variance_threshold(
    input: Path, output: Path, threshold: float = 0.0
) -> None:
    if not input.exists():
        raise FileNotFoundError(f"Dataset not found at {input}")

    df = pd.read_csv(input)
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    numeric_cols = numeric_df.columns[numeric_df.notna().all()]
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(numeric_df[numeric_cols])

    selected_mask = selector.get_support()
    selected_columns = numeric_cols[selected_mask]
    dropped_columns = numeric_cols[~selected_mask]

    print("FEATURE_SELECTION: Dropped features:")
    if len(dropped_columns) == 0:
        print("None")
    else:
        for col in dropped_columns:
            print(col)

    selected_df = df[selected_columns]
    selected_df.to_csv(output, index=False)
    print(f"FEATURE_SELECTION: Selected features saved to {output}")


def main(argv: list[str] | None = None) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    feature_selection_variance_threshold(INPUT_FILE, OUTPUT_FILE)


if __name__ == "__main__":
    main()
