from __future__ import annotations

import pandas as pd

from src.utils import INTERIM_TIMESERIES_CSV


def main() -> None:
    df = pd.read_csv(INTERIM_TIMESERIES_CSV)

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    if len(categorical_cols) == 0:
        print("SANITY_CHECK: No categorical features found.")
        return

    print("SANITY_CHECK: Categorical features:")
    for col in categorical_cols:
        print(col)


if __name__ == "__main__":
    main()
