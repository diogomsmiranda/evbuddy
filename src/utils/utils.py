from __future__ import annotations

from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"

RAW_OPENDATA_CSV_DIR = RAW_DIR / "opendata_datasets_csv"
RAW_OPENDATA_JSON_DIR = RAW_DIR / "opendata_datasets_json"

INTERIM_OH_OPENDATA_CSV_DIR = INTERIM_DIR / "oh_opendata_datasets_csv"
INTERIM_LOCATIONS_CSV = INTERIM_DIR / "locations.csv"
INTERIM_STATIONS_CSV = INTERIM_DIR / "stations.csv"
INTERIM_PORTS_CSV = INTERIM_DIR / "ports.csv"

INTERIM_TIMESERIES_CSV = INTERIM_DIR / "features/stations_timeseries.csv"
