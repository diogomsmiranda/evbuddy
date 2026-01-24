from __future__ import annotations
from pathlib import Path
import pandas as pd
import json

INPUT_FILE = Path("data/interim/locations.csv")
OUTPUT_DIR = Path("data/interim")

def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found at {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)
    location_columns = ["id"] # what location columns to keep
    records: list[dict] = []
    for _, row in df.iterrows():
        stations_value = row.get("stations")
        if pd.isna(stations_value):
            continue
        
        stations_data = json.loads(stations_value) if isinstance(stations_value, str) else stations_value

        location_data = row[location_columns].to_dict()
        
        #change columns of location_data keys to 'location_' to avoid conflict with station name
        location_data = {f"location_{key}": value for key, value in location_data.items()}
        
        for station in stations_data:
            station.pop("coordinates", None)
            records.append({**station, **location_data})


    stations_df = pd.json_normalize(records)
    # print all the columns that have been extracted
    print("Extracted columns:")
    for col in stations_df.columns:
        print(f" - {col}")

    output_path = OUTPUT_DIR / "stations.csv"
    stations_df.to_csv(output_path, index=False)
    print(f"Wrote {output_path} ({len(stations_df)} rows)")

if __name__ == "__main__":
    main()
