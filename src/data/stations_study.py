from __future__ import annotations

import pandas as pd

from src.utils import INTERIM_STATIONS_CSV


def main() -> None:
    path = INTERIM_STATIONS_CSV
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)
    dup_mask = df.duplicated()
    dup_count = dup_mask.sum()
    unique_ids = df["st_id"].nunique()
    total_rows = len(df)

    print(f"Stations file: {path}")
    print(f"Total rows: {total_rows}")
    print(f"Unique st_id: {unique_ids}")
    print(f"stations.csv entries duplicated: {dup_count}")
    if dup_count:
        print(df[dup_mask].to_string(index=False))


if __name__ == "__main__":
    main()
