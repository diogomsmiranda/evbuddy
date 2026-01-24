from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import (
    INTERIM_DIR,
    INTERIM_LOCATIONS_CSV,
    INTERIM_OH_OPENDATA_CSV_DIR,
    RAW_OPENDATA_CSV_DIR,
)


def get_reference_columns(raw_paths: list[Path]) -> list[str]:
    return pd.read_csv(raw_paths[0], nrows=0).columns.tolist()


def load_with_schema_check(path: Path, reference_columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in reference_columns if col not in df.columns]
    extra = [col for col in df.columns if col not in reference_columns]
    if missing or extra:
        parts = [f"Schema mismatch in {path}."]
        if missing:
            parts.append(f"Missing columns ({len(missing)}): {missing}")
        if extra:
            parts.append(f"Extra columns ({len(extra)}): {extra}")
        raise ValueError(" ".join(parts))
    print(f"Loaded {path} ({len(df)} rows)")
    return df[reference_columns]


def main() -> None:
    raw_paths = sorted(
        p
        for p in RAW_OPENDATA_CSV_DIR.glob("*.csv")
        if p.stem.startswith("2022_") or p.stem.startswith("2023_")
    )
    if not raw_paths:
        raise FileNotFoundError(
            f"No 2022/2023 CSV files found in {RAW_OPENDATA_CSV_DIR}"
        )

    interim_paths = sorted(INTERIM_OH_OPENDATA_CSV_DIR.glob("*.csv"))
    if not interim_paths:
        raise FileNotFoundError(
            f"No interim CSV files found in {INTERIM_OH_OPENDATA_CSV_DIR}"
        )

    reference_columns = get_reference_columns(raw_paths)
    dataframes = [load_with_schema_check(path, reference_columns) for path in raw_paths]
    dataframes.extend(
        load_with_schema_check(path, reference_columns) for path in interim_paths
    )

    final_df = pd.concat(dataframes, ignore_index=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERIM_LOCATIONS_CSV
    final_df.to_csv(output_path, index=False)
    print(f"Wrote {output_path} ({len(final_df)} rows)")


if __name__ == "__main__":
    main()
