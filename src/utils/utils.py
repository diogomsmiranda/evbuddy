from __future__ import annotations

from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path("models")

RAW_OPENDATA_CSV_DIR = RAW_DIR / "opendata_datasets_csv"
RAW_OPENDATA_JSON_DIR = RAW_DIR / "opendata_datasets_json"

INTERIM_OH_OPENDATA_CSV_DIR = INTERIM_DIR / "oh_opendata_datasets_csv"
INTERIM_LOCATIONS_CSV = INTERIM_DIR / "locations.csv"
INTERIM_STATIONS_CSV = INTERIM_DIR / "stations.csv"
INTERIM_PORTS_CSV = INTERIM_DIR / "ports.csv"

INTERIM_TIMESERIES_CSV = INTERIM_DIR / "features/stations_timeseries.csv"
INTERIM_TIMESERIES_ENCODED_CSV = (
    INTERIM_DIR / "features/stations_timeseries_encoded.csv"
)

INTERIM_TIMESERIES_SELECTED_CSV = (
    INTERIM_DIR / "features/stations_timeseries_selected.csv"
)

PROCESSED_DENSE_10MIN_PARQUET = PROCESSED_DIR / "dense_10min.parquet"
PROCESSED_DENSE_10MIN_CSV = PROCESSED_DIR / "dense_10min.csv"

HORIZON_MINUTES = [5, 10, 15, 20, 25, 30]

STATE_COLUMNS = [
    "available_ports",
    "unavailable_ports",
    "unavailable_rate",
    "available_ac_ports",
    "available_dc_ports",
]

STATIC_COLUMNS = [
    "st_location_id",
    "total_ports",
    "total_ac_ports",
    "total_dc_ports",
    "is_ac_available",
    "is_dc_available",
    "has_dc_fast",
    "has_ccs",
    "has_chademo",
    "has_type2",
    "has_wall_outlet",
    "loc_onstreet_location",
    "loc_coordinates_latitude",
    "loc_coordinates_longitude",
    "loc_address_postal_code",
]

MODEL_STATIC_FEATURE_COLUMNS = [
    col for col in STATIC_COLUMNS if col != "st_location_id"
]

FRESHNESS_COLUMNS = ["is_observed", "is_stale", "age_minutes"]
TIME_COLUMNS = ["hour", "dayofweek", "sin_hour", "cos_hour", "sin_dow", "cos_dow"]
