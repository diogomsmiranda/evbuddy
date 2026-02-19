from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
import pytest
from pandera import Check

from src.utils import INTERIM_TIMESERIES_SELECTED_CSV


TIMESERIES_SCHEMA = pa.DataFrameSchema(
    {
        "snapshot_ts": pa.Column(str, nullable=False),
        "st_id": pa.Column(int, nullable=False),
        "st_location_id": pa.Column(int, nullable=False),
        "total_ports": pa.Column(int, checks=Check.ge(0), nullable=False),
        "available_ports": pa.Column(int, checks=Check.ge(0), nullable=False),
        "unavailable_ports": pa.Column(int, checks=Check.ge(0), nullable=False),
        "unavailable_rate": pa.Column(float, checks=Check.in_range(0.0, 1.0), nullable=False),
    },
    strict=False,
    coerce=True,
)


def test_selected_timeseries_schema_and_constraints() -> None:
    if not INTERIM_TIMESERIES_SELECTED_CSV.exists():
        pytest.skip(f"Missing dataset: {INTERIM_TIMESERIES_SELECTED_CSV}")

    df = pd.read_csv(INTERIM_TIMESERIES_SELECTED_CSV)
    validated = TIMESERIES_SCHEMA.validate(df)

    assert not validated.empty
    assert (validated["available_ports"] <= validated["total_ports"]).all()
    assert (validated["unavailable_ports"] <= validated["total_ports"]).all()
