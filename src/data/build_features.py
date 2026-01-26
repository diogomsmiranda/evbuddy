from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


from src.utils import (
    INTERIM_DIR,
    INTERIM_LOCATIONS_CSV,
    INTERIM_PORTS_CSV,
    INTERIM_STATIONS_CSV,
)

OUTPUT_DIR = INTERIM_DIR / "features"

SKIP_NORMALIZATION = {
    "loc_id",
    "st_id",
    "port_id",
    "st_location_id",
    "port_station_id",
    "port_station_location_id",
    "loc_coordinates_latitude",
    "loc_coordinates_longitude",
    "loc_last_updated",
    "port_last_updated",
    "loc_stations",
    "st_ports",
    "st_notes",
}


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    "Normalize categorical features by lowercasing and replacing spaces with underscores."
    for col in df.select_dtypes(include=["str"]).columns:
        if col not in SKIP_NORMALIZATION:
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .str.lower()
                .str.replace(r"\s+", "_", regex=True)
            ).astype("category")
    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].astype(int)
    return df


def turn_numeric(input: Path, output: Path) -> None:
    "Uses dummyfication to convert categorical features to numeric."
    if not input.exists():
        raise FileNotFoundError(f"Dataset not found at {input}")
    df = pd.read_csv(input)
    normalized_df = normalize_cols(df)
    print(f"Normalized DataFrame dimensions: {normalized_df.shape}")
    categorical_cols = normalized_df.select_dtypes(include=["str", "category"]).columns
    print(f"Categorical columns to be dummyfied: {list(categorical_cols)}")
    numeric_df = pd.get_dummies(normalized_df, columns=categorical_cols, dtype=int)
    numeric_df.to_csv(output, index=False)
    print(f"FEATURES: Numeric features saved to {output}")


def main(argv: list[str] | None = None) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    args = argv if argv is not None else sys.argv[1:]
    if not args or args[0] not in {"stations", "ports", "locations", "all"}:
        raise SystemExit(
            "Usage: python build_features.py [stations|ports|locations|all]"
        )

    match args[0]:
        case "stations":
            input_path = INTERIM_STATIONS_CSV
            output_path = OUTPUT_DIR / "stations.csv"
        case "ports":
            input_path = INTERIM_PORTS_CSV
            output_path = OUTPUT_DIR / "ports.csv"
        case "locations":
            input_path = INTERIM_LOCATIONS_CSV
            output_path = OUTPUT_DIR / "locations.csv"
        case _:
            input_path = None
            output_path = None

    if args[0] == "all":
        turn_numeric(INTERIM_STATIONS_CSV, OUTPUT_DIR / "stations.csv")
        turn_numeric(INTERIM_PORTS_CSV, OUTPUT_DIR / "ports.csv")
        turn_numeric(INTERIM_LOCATIONS_CSV, OUTPUT_DIR / "locations.csv")
    else:
        turn_numeric(input_path, output_path)


if __name__ == "__main__":
    main()
