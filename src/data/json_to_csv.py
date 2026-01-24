from __future__ import annotations

from pathlib import Path

import pandas as pd
import json

from src.utils import RAW_OPENDATA_CSV_DIR, RAW_OPENDATA_JSON_DIR


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict) and "locations" in data:
        records = data["locations"]
    elif isinstance(data, list):
        records = data
    else:
        records = [data]

    if not isinstance(records, list):
        raise ValueError(f"Unexpected JSON structure in {path}")

    return records


def normalize_records(records: list[dict]) -> pd.DataFrame:
    df = pd.json_normalize(records, sep="_")

    for column in df.columns:
        if df[column].map(lambda value: isinstance(value, (list, dict))).any():
            df[column] = df[column].map(
                # json-encode nested structures for json.loads use later
                lambda value: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict))
                else value
            )

    return df


def convert_all(
    input_dir: Path = RAW_OPENDATA_JSON_DIR, output_dir: Path = RAW_OPENDATA_CSV_DIR
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(input_dir.glob("*.json")):
        records = load_records(path)
        df = normalize_records(records)
        output_path = output_dir / f"{path.stem}.csv"
        df.to_csv(output_path, index=False)
        print(f"Wrote {output_path} ({len(df)} rows)")


if __name__ == "__main__":
    convert_all()
