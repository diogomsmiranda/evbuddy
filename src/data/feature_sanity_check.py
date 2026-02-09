from __future__ import annotations

import pandas as pd

from src.utils import INTERIM_TIMESERIES_SELECTED_CSV


def main() -> None:
    df = pd.read_csv(INTERIM_TIMESERIES_SELECTED_CSV)
    print("FEATURE_SANITY_CHECK: Selected features:")
    for col in df.columns:
        print(col)
    print(f"FEATURE_SANITY_CHECK: Total features: {len(df.columns)}")


if __name__ == "__main__":
    main()
