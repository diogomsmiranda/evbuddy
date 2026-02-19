from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from src.features.feature_selection import feature_selection_variance_threshold


SELECTED_SCHEMA = pa.DataFrameSchema(
    {
        "snapshot_ts": pa.Column(str, nullable=False),
        "signal": pa.Column(int, nullable=False),
    },
    strict=True,
)


def test_variance_threshold_keeps_only_variable_numeric_columns(
    tmp_path: Path,
) -> None:
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"

    df = pd.DataFrame(
        {
            "snapshot_ts": [
                "2025-01-01T00:00:00Z",
                "2025-01-01T00:10:00Z",
                "2025-01-01T00:20:00Z",
            ],
            "signal": [1, 2, 3],
            "constant": [7, 7, 7],
            "text_col": ["a", "b", "c"],
            "with_na": [1, None, 3],
        }
    )
    df.to_csv(input_file, index=False)

    feature_selection_variance_threshold(input_file, output_file)

    result = pd.read_csv(output_file)
    assert "snapshot_ts" in result.columns
    assert "signal" in result.columns
    assert "constant" not in result.columns
    assert "text_col" not in result.columns
    assert "with_na" not in result.columns

    validated = SELECTED_SCHEMA.validate(result)
    assert validated.shape[0] == 3
