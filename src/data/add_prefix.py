# write a script which adds a prefix to all columns in a csv file
from pathlib import Path
import pandas as pd

from src.utils import (
    INTERIM_LOCATIONS_CSV,
    INTERIM_STATIONS_CSV,
    INTERIM_PORTS_CSV,
)


def add_prefix_to_csv_columns(input_csv: Path, output_csv: Path, prefix: str) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV file not found at {input_csv}")

    df = pd.read_csv(input_csv)

    df = df.add_prefix(prefix)

    df.to_csv(output_csv, index=False)
    print(f"Columns prefixed with '{prefix}' and saved to {output_csv}")


def main() -> None:
    add_prefix_to_csv_columns(
        input_csv=INTERIM_LOCATIONS_CSV, output_csv=INTERIM_LOCATIONS_CSV, prefix="loc_"
    )
    add_prefix_to_csv_columns(
        input_csv=INTERIM_STATIONS_CSV, output_csv=INTERIM_STATIONS_CSV, prefix="st_"
    )
    add_prefix_to_csv_columns(
        input_csv=INTERIM_PORTS_CSV, output_csv=INTERIM_PORTS_CSV, prefix="port_"
    )


if __name__ == "__main__":
    main()
