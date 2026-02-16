from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import INTERIM_TIMESERIES_CSV

INPUT_FILE = INTERIM_TIMESERIES_CSV
OUTPUT_FILE = INTERIM_TIMESERIES_CSV.parent / "stations_timeseries_encoded.csv"


def encode_categorical_features(input: Path, output: Path) -> None:
    if not input.exists():
        raise FileNotFoundError(f"Dataset not found at {input}")

    df = pd.read_csv(input, parse_dates=["snapshot_ts"])
    if "snapshot_ts" in df.columns:
        df = df.set_index("snapshot_ts")
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    if len(categorical_cols) == 0:
        encoded_df = df.copy()
    else:
        encoded_df = pd.get_dummies(df, columns=categorical_cols, dtype=int)

    if encoded_df.index.name == "snapshot_ts":
        encoded_df = encoded_df.reset_index()

    encoded_df.to_csv(output, index=False)
    print(f"FEATURE_ENCODING: Encoded features saved to {output}")


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    encode_categorical_features(INPUT_FILE, OUTPUT_FILE)


if __name__ == "__main__":
    main()
