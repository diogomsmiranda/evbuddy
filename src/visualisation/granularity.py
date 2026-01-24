from __future__ import annotations

import pandas as pd
import json

from src.utils import RAW_OPENDATA_CSV_DIR, INTERIM_OH_OPENDATA_CSV_DIR


def derive_opening_hours_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Derive opening hours related variables from existing columns."""
    columns = [
        "opening_hours_weekday_begin",
        "opening_hours_weekday_end",
        "opening_hours_hour_begin",
        "opening_hours_hour_end",
    ]

    if "opening_hours" not in df.columns:
        return None

    def parse_opening_hours(value: object) -> dict:
        if pd.isna(value):
            return {}
        if isinstance(value, str):
            value = json.loads(value)
        if isinstance(value, list):
            return value[0] if value else {}
        if isinstance(value, dict):
            return value
        return {}

    opening = pd.json_normalize(df["opening_hours"].map(parse_opening_hours)).reindex(
        columns=["weekday_begin", "weekday_end", "hour_begin", "hour_end"]
    )
    opening.columns = columns
    df[columns] = opening
    df = df.drop(columns=["opening_hours"])
    return df


def main() -> None:
    INTERIM_OH_OPENDATA_CSV_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(RAW_OPENDATA_CSV_DIR.glob("*.csv"))
    for path in paths:
        df = pd.read_csv(path)
        file_tag = path.stem
        result = derive_opening_hours_variables(df)
        if result is not None:
            output_path = (
                INTERIM_OH_OPENDATA_CSV_DIR / f"{file_tag}_with_opening_hours.csv"
            )
            result.to_csv(output_path, index=False)
            print(f"Wrote {output_path} ({len(result)} rows)")


if __name__ == "__main__":
    main()
