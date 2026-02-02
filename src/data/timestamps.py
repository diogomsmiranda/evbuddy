from __future__ import annotations

import pandas as pd

from src.utils import INTERIM_TIMESERIES_CSV


INPUT_FILE = INTERIM_TIMESERIES_CSV
OUTPUT_FILE = INTERIM_TIMESERIES_CSV


def main() -> None:
    df = pd.read_csv(INPUT_FILE)

    if "loc_last_updated_parsed" not in df.columns:
        raise KeyError("Column 'loc_last_updated_parsed' not found.")
    if "loc_last_updated" not in df.columns:
        raise KeyError("Column 'loc_last_updated' not found.")

    df["loc_last_updated"] = df["loc_last_updated_parsed"]
    df = df.drop(columns=["loc_last_updated_parsed"])

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"TIMESTAMPS: Updated {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
