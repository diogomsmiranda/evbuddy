from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.utils import INTERIM_PORTS_CSV

OUTPUT_DIR = Path("reports/figures")


def analyze_ports(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("port_station_id")
    results = []

    for station_id, group in grouped:
        connector_unique = group["port_connector_type"].nunique()
        power_unique = group["port_power_kw"].nunique()
        mech_unique = group["port_charging_mechanism"].nunique()

        results.append(
            {
                "station_id": station_id,
                "connector_types_unique": connector_unique,
                "power_kw_unique": power_unique,
                "charging_mechanism_unique": mech_unique,
            }
        )

    return pd.DataFrame(results)


def combo_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    combos = (
        df.groupby(["port_connector_type", "port_power_kw", "port_charging_mechanism"])[
            "port_id"
        ]
        .nunique()
        .reset_index()
        .rename(columns={"port_id": "count"})
        .sort_values("count", ascending=False)
    )
    return combos


def combo_frequencies_by_station(df: pd.DataFrame) -> pd.DataFrame:
    combos = (
        df.groupby(
            [
                "port_station_id",
                "port_connector_type",
                "port_power_kw",
                "port_charging_mechanism",
            ]
        )["port_id"]
        .nunique()
        .reset_index()
        .rename(columns={"port_id": "count"})
    )
    return combos


def ports_with_inconsistent_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["port_last_updated_parsed"] = pd.to_datetime(
        df.get("port_last_updated"), errors="coerce", utc=True
    )

    records = []
    for port_id, group in df.groupby("port_id"):
        unique_combos = group[
            ["port_connector_type", "port_power_kw", "port_charging_mechanism"]
        ].drop_duplicates()
        if len(unique_combos) > 1:
            sorted_group = group.sort_values("port_last_updated_parsed")
            combos = list(
                zip(
                    sorted_group["port_connector_type"],
                    sorted_group["port_power_kw"],
                    sorted_group["port_charging_mechanism"],
                    sorted_group.get(
                        "port_last_updated",
                        pd.Series(index=sorted_group.index, dtype=object),
                    ),
                    sorted_group.get(
                        "port_status",
                        pd.Series(index=sorted_group.index, dtype=object),
                    ),
                )
            )
            records.append({"port_id": port_id, "timeline": combos})
    return pd.DataFrame(records)


def main() -> None:
    input_path = INTERIM_PORTS_CSV
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found at {input_path}")

    df = pd.read_csv(input_path)

    dup_mask = df.duplicated()
    dup_count = dup_mask.sum()
    print(f"ports.csv entries duplicated: {dup_count}")
    if dup_count:
        print(df[dup_mask].to_string(index=False))

    summary = analyze_ports(df)

    total = len(summary)
    connectors_vary = summary["connector_types_unique"].gt(1).sum()
    power_vary = summary["power_kw_unique"].gt(1).sum()
    mech_vary = summary["charging_mechanism_unique"].gt(1).sum()

    print(f"Stations analyzed: {total}")
    print(f"Stations with varying connector types: {connectors_vary}")
    print(f"Stations with varying power_kw: {power_vary}")
    print(f"Stations with varying charging_mechanism: {mech_vary}")

    # Show detail for stations that vary
    varying = summary[
        summary["connector_types_unique"].gt(1)
        | summary["power_kw_unique"].gt(1)
        | summary["charging_mechanism_unique"].gt(1)
    ]
    if not varying.empty:
        print("\nStations with variation:")
        print(varying.sort_values("station_id").to_string(index=False))
    else:
        print("\nNo variation found within stations.")

    print("\nMost common connector/power/mechanism combinations:")
    combos = combo_frequencies(df)
    print(combos.to_string(index=False))

    print("\nAll distinct connector/power/mechanism combinations:")
    print(
        combos[
            ["port_connector_type", "port_power_kw", "port_charging_mechanism"]
        ].to_string(index=False)
    )

    # Correlation across the categorical fields via one-hot encoding
    cat_df = df[
        ["port_connector_type", "port_power_kw", "port_charging_mechanism"]
    ].copy()
    cat_df["port_power_kw"] = cat_df["port_power_kw"].astype("category")
    dummies = pd.get_dummies(cat_df, dtype=int)
    corr_matrix = dummies.corr()
    print("\nCorrelation matrix (one-hot encoded connector/power/mechanism):")
    print(corr_matrix)

    # Plot correlation matrix using seaborn heatmap
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        annot=False,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation: connector / power_kw / charging_mechanism")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    corr_path = OUTPUT_DIR / "ports_correlation.png"
    plt.savefig(corr_path, dpi=150)
    plt.close()
    print(f"Saved correlation heatmap to {corr_path}")

    combos_by_station = combo_frequencies_by_station(df)
    multi_combo_stations = combos_by_station.groupby("port_station_id").filter(
        lambda g: len(g) > 1
    )
    if not multi_combo_stations.empty:
        print("\nStations with multiple connector/power/mechanism combos:")
        multi_combo_sorted = multi_combo_stations.sort_values(
            ["port_station_id", "count"], ascending=[True, False]
        )
        port_counts = df.groupby("port_station_id")["port_id"].nunique()
        for station_id, group in multi_combo_sorted.groupby("port_station_id"):
            total_ports = port_counts.get(station_id, 0)
            print(f"Station {station_id} (ports: {total_ports}):")
            print(group.to_string(index=False))
    else:
        print("\nAll stations have a single connector/power/mechanism combo.")

    inconsistent_ports = ports_with_inconsistent_history(df)
    if not inconsistent_ports.empty:
        print("\nPorts with changing connector/power/mechanism over time:")
        for _, row in inconsistent_ports.iterrows():
            print(f"port_id {row['port_id']}:")
            for connector, power, mech, ts, status in row["timeline"]:
                ts_val = ts if pd.notna(ts) else "<no timestamp>"
                print(f"  {ts_val}: {connector} / {power} / {mech} / status={status}")
    else:
        print("\nNo ports changed connector/power/mechanism over time.")


if __name__ == "__main__":
    main()
