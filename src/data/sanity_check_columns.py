from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/opendata_datasets_csv")
INTERIM_DIR = Path("data/interim/opendata_datasets_csv")


def load_columns(path: Path) -> list[str]:
    df = pd.read_csv(path, nrows=0)
    return list(df.columns)


def main() -> None:
    raw_paths = sorted(
        p
        for p in RAW_DIR.glob("*.csv")
        if p.stem.startswith("2022_") or p.stem.startswith("2023_")
    )
    if not raw_paths:
        raise FileNotFoundError(f"No 2022/2023 CSV files found in {RAW_DIR}")

    reference_columns = load_columns(raw_paths[0])
    reference_set = set(reference_columns)
    mismatches = 0

    interim_paths = sorted(INTERIM_DIR.glob("*.csv"))
    if not interim_paths:
        raise FileNotFoundError(f"No interim CSV files found in {INTERIM_DIR}")

    for path in interim_paths:
        columns = load_columns(path)
        column_set = set(columns)

        missing = sorted(reference_set - column_set)
        extra = sorted(column_set - reference_set)

        if missing or extra:
            mismatches += 1
            print(f"{path.name}:")
            if missing:
                print(f"  Missing columns ({len(missing)}): {missing}")
            if extra:
                print(f"  Extra columns ({len(extra)}): {extra}")
        else:
            print(f"{path.name}: OK")

    if mismatches:
        print(f"Found {mismatches} file(s) with column mismatches.")
    else:
        print("All interim files match the 2022/2023 raw column set.")


if __name__ == "__main__":
    main()
