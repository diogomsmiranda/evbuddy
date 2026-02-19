from __future__ import annotations

import pandas as pd

from src.data.build_features import classify_dc_fast, classify_dc_fast_vectorized


def test_classify_dc_fast_by_power() -> None:
    row = pd.Series(
        {
            "port_connector_type": "TYPE_2",
            "port_charging_mechanism": "SOCKET",
            "port_power_kw": 50,
        }
    )
    assert classify_dc_fast(row)


def test_classify_dc_fast_by_connector() -> None:
    row = pd.Series(
        {
            "port_connector_type": "CHADEMO",
            "port_charging_mechanism": "SOCKET",
            "port_power_kw": 11,
        }
    )
    assert classify_dc_fast(row)


def test_classify_dc_fast_vectorized() -> None:
    df = pd.DataFrame(
        [
            {
                "port_power_kw": 11,
                "port_connector_type": "TYPE_2",
                "port_charging_mechanism": "SOCKET",
            },
            {
                "port_power_kw": 50,
                "port_connector_type": "TYPE_2",
                "port_charging_mechanism": "SOCKET",
            },
            {
                "port_power_kw": 7,
                "port_connector_type": "CCS_TYPE_2",
                "port_charging_mechanism": "SOCKET",
            },
            {
                "port_power_kw": 7,
                "port_connector_type": "TYPE_2",
                "port_charging_mechanism": "CABLE",
            },
        ]
    )
    result = classify_dc_fast_vectorized(df)
    assert result.tolist() == [False, True, True, True]
