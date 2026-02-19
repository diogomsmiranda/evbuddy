from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

MIN_AUC = 0.50
MAX_LOGLOSS = 1.50
REQUIRED_KEYS = {
    "requested_horizon_minutes",
    "effective_horizon_minutes",
    "auc",
    "logloss",
    "brier",
}


def _metric_files(root: Path) -> list[Path]:
    return sorted(root.glob("metrics_h*m.json"))


def test_metrics_files_have_required_keys_and_thresholds() -> None:
    files = _metric_files(Path("models"))
    require_metrics = os.getenv("REQUIRE_METRICS_FILES", "0") == "1"
    if not files:
        if require_metrics:
            pytest.fail("No metrics_h*m.json files found under models/.")
        pytest.skip("No metrics_h*m.json files found under models/.")

    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        missing = REQUIRED_KEYS.difference(data)
        assert not missing, f"{path} missing keys: {sorted(missing)}"

        auc = data.get("auc")
        if auc is not None:
            assert float(auc) >= MIN_AUC, f"{path} auc below threshold: {auc}"

        assert float(data["logloss"]) <= MAX_LOGLOSS, (
            f"{path} logloss above threshold: {data['logloss']}"
        )
        assert 0.0 <= float(data["brier"]) <= 1.0, (
            f"{path} brier out of range: {data['brier']}"
        )
