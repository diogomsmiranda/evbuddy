from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.visualisation import dimensionality, distributions, sparsity


def _write_small_timeseries_csv(path: Path) -> None:
    pd.DataFrame(
        {
            "snapshot_ts": [
                "2025-01-01T00:00:00Z",
                "2025-01-01T00:10:00Z",
                "2025-01-02T00:00:00Z",
            ],
            "st_id": [1, 1, 2],
            "st_location_id": [10, 10, 20],
            "available_ports": [1, 0, 2],
            "total_ports": [2, 2, 2],
            "unavailable_rate": [0.5, 1.0, 0.0],
            "has_dc_fast": [0, 0, 1],
        }
    ).to_csv(path, index=False)


def test_sparsity_main_writes_expected_pngs(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "timeseries_selected.csv"
    out_dir = tmp_path / "sparsity"
    _write_small_timeseries_csv(input_path)

    monkeypatch.setattr(sparsity, "INTERIM_TIMESERIES_SELECTED_CSV", input_path)
    monkeypatch.setattr(sparsity, "OUTPUT_DIR", out_dir)

    sparsity.main()

    assert (out_dir / "records_per_day.png").exists()
    assert (out_dir / "records_per_day_full.png").exists()


def test_distributions_main_writes_expected_pngs(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "timeseries_selected.csv"
    out_dir = tmp_path / "distributions"
    _write_small_timeseries_csv(input_path)

    monkeypatch.setitem(
        distributions.DATASETS,
        "timeseries",
        {"input": input_path, "output": out_dir, "exclude": set()},
    )

    distributions.main(["timeseries"])

    assert (out_dir / "snapshot_ts.png").exists()
    assert (out_dir / "available_ports.png").exists()


def test_dimensionality_main_writes_expected_pngs(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "timeseries_encoded.csv"
    out_root = tmp_path / "dimensionality"
    _write_small_timeseries_csv(input_path)

    monkeypatch.setitem(dimensionality.INTERIM_DATASETS, "timeseries", input_path)
    monkeypatch.setattr(dimensionality, "OUTPUT_INTERIM_DIR", out_root)

    dimensionality.main(["timeseries"])

    entity_out = out_root / "timeseries"
    assert (entity_out / "timeseries_encoded_records_variables.png").exists()
    assert (entity_out / "timeseries_encoded_missing_values.png").exists()
