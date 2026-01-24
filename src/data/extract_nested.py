from __future__ import annotations

import json
import pandas as pd

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import (  # noqa: E402
    INTERIM_DIR,
    INTERIM_LOCATIONS_CSV,
    INTERIM_PORTS_CSV,
    INTERIM_STATIONS_CSV,
)

OUTPUT_DIR = INTERIM_DIR
INPUT_PATHS = {
    "locations": INTERIM_LOCATIONS_CSV,
}
OUTPUT_PATHS = {
    "stations": INTERIM_STATIONS_CSV,
    "ports": INTERIM_PORTS_CSV,
}


def parse_json_value(value: object) -> list | dict | None:
    if pd.isna(value):
        print("PARSE_JSON: Value is NaN")
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            print("PARSE_JSON: Failed to decode JSON from string")
            return None
    return value


def first_dict(value: object) -> dict:
    parsed = parse_json_value(value)
    if isinstance(parsed, list) and parsed:
        return parsed[0] if isinstance(parsed[0], dict) else {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def extract_nested(
    input: Path, to_nest: str, output: Path, parent: str, parent_columns: list[str]
) -> None:
    if not input.exists():
        raise FileNotFoundError(f"Dataset not found at {input}")

    df = pd.read_csv(input)
    records: list[dict] = []
    for _, row in df.iterrows():
        nested_value = parse_json_value(row.get(to_nest))
        if not isinstance(nested_value, list):
            print(f"EXTRACT: Invalid data type: {type(nested_value)}")
            continue

        parent_data = row[parent_columns].to_dict()
        parent_data = {f"{parent}_{key}": value for key, value in parent_data.items()}

        for item in nested_value:
            if not isinstance(item, dict):
                print(f"EXTRACT: Skipping invalid item in {to_nest}")
                continue
            item.pop("coordinates", None)
            if to_nest == "ports":
                status_data = first_dict(item.get("port_status"))
                if "status" in status_data:
                    item["status"] = status_data["status"]
                item.pop("port_status", None)

                auth_data = first_dict(item.get("authentications"))
                if "authentication_id" in auth_data:
                    item["authentication_id"] = auth_data["authentication_id"]
                if "payment_required" in auth_data:
                    item["payment_required"] = auth_data["payment_required"]
                item.pop("authentications", None)
            records.append({**item, **parent_data})
    nested_df = pd.json_normalize(records)
    for column in nested_df.columns:
        if nested_df[column].map(lambda value: isinstance(value, (list, dict))).any():
            nested_df[column] = nested_df[column].map(
                # json-encode nested structures for json.loads use later
                lambda value: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict))
                else value
            )
    nested_df.to_csv(output, index=False)
    print(f"Wrote {output} ({len(nested_df)} rows)")


def main(argv: list[str] | None = None) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    args = argv if argv is not None else sys.argv[1:]
    if not args or args[0] not in {"stations", "ports"}:
        raise SystemExit("Usage: python extract_nested.py [stations|ports]")

    if args[0] == "stations":
        extract_nested(
            input=INPUT_PATHS["locations"],
            to_nest="stations",
            output=OUTPUT_PATHS["stations"],
            parent="location",
            parent_columns=["id"],
        )
    else:
        extract_nested(
            input=OUTPUT_PATHS["stations"],
            to_nest="ports",
            output=OUTPUT_PATHS["ports"],
            parent="station",
            parent_columns=["id", "location_id"],
        )


if __name__ == "__main__":
    main()
