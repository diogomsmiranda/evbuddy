from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data import extract_nested as mod


def test_extract_nested_stations_flattens_and_renames(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "locations.csv"
    output_path = tmp_path / "stations.csv"

    loc_stations = [
        {
            "id": 101,
            "ports": [{"id": 1}],
            "notes": "station notes",
        }
    ]
    df = pd.DataFrame(
        {
            "loc_id": [9001],
            "loc_stations": [json.dumps(loc_stations)],
        }
    )
    df.to_csv(input_path, index=False)

    monkeypatch.setitem(mod.OUTPUT_PATHS, "stations", output_path)

    mod.extract_nested(
        input=input_path,
        to_nest="loc_stations",
        output=output_path,
        parent_columns=["loc_id"],
    )

    result = pd.read_csv(output_path)
    assert "st_id" in result.columns
    assert "st_location_id" in result.columns
    assert int(result.loc[0, "st_id"]) == 101
    assert int(result.loc[0, "st_location_id"]) == 9001


def test_extract_nested_ports_extracts_status_and_auth(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "stations.csv"
    output_path = tmp_path / "ports.csv"

    st_ports = [
        {
            "id": 5001,
            "connector_type": "TYPE_2",
            "power_kw": 22,
            "port_status": [{"status": "AVAILABLE"}],
            "authentications": [
                {"authentication_id": "APP", "payment_required": False}
            ],
        }
    ]
    df = pd.DataFrame(
        {
            "st_id": [101],
            "st_location_id": [9001],
            "st_ports": [json.dumps(st_ports)],
        }
    )
    df.to_csv(input_path, index=False)

    monkeypatch.setitem(mod.OUTPUT_PATHS, "ports", output_path)

    mod.extract_nested(
        input=input_path,
        to_nest="st_ports",
        output=output_path,
        parent_columns=["st_id", "st_location_id"],
    )

    result = pd.read_csv(output_path)
    assert "port_id" in result.columns
    assert "port_station_id" in result.columns
    assert "port_station_location_id" in result.columns
    assert "port_status" in result.columns
    assert "port_authentication_id" in result.columns
    assert "port_payment_required" in result.columns

    assert int(result.loc[0, "port_id"]) == 5001
    assert int(result.loc[0, "port_station_id"]) == 101
    assert int(result.loc[0, "port_station_location_id"]) == 9001
    assert result.loc[0, "port_status"] == "AVAILABLE"
    assert result.loc[0, "port_authentication_id"] == "APP"
