from __future__ import annotations

import pandas as pd

from src.utils import (
    INTERIM_DIR,
    INTERIM_LOCATIONS_CSV,
    INTERIM_PORTS_CSV,
    INTERIM_STATIONS_CSV,
)

OUTPUT_DIR = INTERIM_DIR / "features"
OUTPUT_FILE = OUTPUT_DIR / "stations_timeseries.csv"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    loc_df = pd.read_csv(INTERIM_LOCATIONS_CSV)
    st_df = pd.read_csv(INTERIM_STATIONS_CSV)
    ports_df = pd.read_csv(INTERIM_PORTS_CSV)
    return loc_df, st_df, ports_df


def classify_dc_fast(row: pd.Series) -> bool:
    connector = row.get("port_connector_type")
    mech = row.get("port_charging_mechanism")
    try:
        power = float(row.get("port_power_kw"))
    except (TypeError, ValueError):
        power = 0.0

    dc_connectors = {"CCS_TYPE_2", "CHADEMO"}
    if power > 22:
        return True
    if connector in dc_connectors:
        return True
    if mech == "CABLE":
        return True
    return False


def preprocess_locations(loc_df: pd.DataFrame) -> pd.DataFrame:
    loc_df["loc_last_updated_parsed"] = pd.to_datetime(
        loc_df["loc_last_updated"], errors="coerce", utc=True
    )

    loc_df = (
        loc_df.sort_values(["loc_id", "loc_last_updated_parsed"]).drop_duplicates(
            subset=["loc_id", "loc_last_updated"], keep="last"
        )  # sanity check
    )
    return loc_df


def preprocess_ports(ports_df: pd.DataFrame) -> pd.DataFrame:
    ports_df = ports_df.copy()
    ports_df["port_last_updated_parsed"] = pd.to_datetime(
        ports_df.get("port_last_updated"), errors="coerce", utc=True
    )
    ports_df = ports_df.dropna(
        subset=["port_last_updated_parsed"]
    )  # none timestamps dropped
    ports_df = (
        ports_df.sort_values(["port_id", "port_last_updated_parsed"])
        .drop_duplicates(subset=["port_id", "port_last_updated"], keep="last")
        .reset_index(drop=True)
    )

    if "port_notes" in ports_df.columns:
        ports_df = ports_df[ports_df["port_notes"] != "MOTORCYCLE_ONLY"].reset_index(
            drop=True
        )
    return ports_df


def statuses_over_time(station_ports: pd.DataFrame, timestamps) -> pd.DataFrame:
    if station_ports.empty:
        return pd.DataFrame(index=timestamps)
    status_table = (
        station_ports.pivot_table(
            index="port_last_updated_parsed",
            columns="port_id",
            values="port_status",
            aggfunc="last",
        )
        .reindex(timestamps)
        .ffill()
    )
    return status_table


def classify_dc_fast_vectorized(df: pd.DataFrame) -> pd.Series:
    power = pd.to_numeric(df.get("port_power_kw"), errors="coerce").fillna(0.0)
    connector = df.get("port_connector_type")
    mech = df.get("port_charging_mechanism")
    dc_connectors = {"CCS_TYPE_2", "CHADEMO"}
    return (power > 22) | connector.isin(dc_connectors) | (mech == "CABLE")


def build_station_timeseries(
    loc_df: pd.DataFrame, st_df: pd.DataFrame, ports_df: pd.DataFrame
) -> pd.DataFrame:
    loc_df = preprocess_locations(loc_df)
    ports_df = preprocess_ports(ports_df)
    # no need to preprocess stations
    loc_first = (
        loc_df.sort_values(["loc_id", "loc_last_updated_parsed"])
        .groupby("loc_id", as_index=False)
        .head(1)
    )
    loc_by_id = loc_first.set_index("loc_id").to_dict(orient="index")
    ports_by_station = dict(tuple(ports_df.groupby("port_station_id")))

    records: list[dict[str, object]] = []
    for st_id, st_group in st_df.groupby("st_id"):
        st_row = st_group.iloc[0].to_dict()
        loc_id = st_row.get("st_location_id")
        # get the first matching loc row (same behavior as prior)
        loc_row = loc_by_id.get(loc_id, {})

        station_ports = ports_by_station.get(st_id)
        if station_ports is None or station_ports.empty:
            continue

        station_ports = station_ports.sort_values("port_last_updated_parsed")
        ts_index = station_ports["port_last_updated_parsed"].dropna().unique()
        ts_index = pd.Index(ts_index).sort_values()
        status_ffill = statuses_over_time(station_ports, ts_index)
        total_ports_const = status_ffill.shape[1]

        port_attr_cols = [
            "port_connector_type",
            "port_charging_mechanism",
            "port_power_kw",
        ]
        latest_attrs = (
            station_ports.set_index("port_last_updated_parsed")
            .groupby("port_id")[port_attr_cols]
            .apply(lambda g: g.reindex(ts_index, method="ffill"))
        )
        latest_attrs.index.names = ["port_id", "snapshot_ts"]

        for ts in ts_index:
            status_row = status_ffill.loc[ts]
            available_ports = (status_row == "AVAILABLE").sum()
            total_ports = total_ports_const
            unavailable_ports = total_ports - available_ports
            unavailable_rate = round(
                unavailable_ports / total_ports if total_ports else 0, 2
            )

            latest_per_port = latest_attrs.xs(ts, level="snapshot_ts")
            dc_flags = classify_dc_fast_vectorized(latest_per_port)
            number_of_dc_ports = int(dc_flags.sum())
            number_of_ac_ports = total_ports - number_of_dc_ports
            has_dc_fast = int(number_of_dc_ports > 0)

            has_ccs = int(
                (latest_per_port["port_connector_type"] == "CCS_TYPE_2").any()
            )
            has_chademo = int(
                (latest_per_port["port_connector_type"] == "CHADEMO").any()
            )
            has_type2 = int(
                latest_per_port["port_connector_type"]
                .isin(["TYPE_2", "MENNEKES"])
                .any()
            )
            has_wall_outlet = int(
                (latest_per_port["port_connector_type"] == "WALL_OUTLET").any()
            )

            # Available AC/DC counts using latest per-port state and status at ts
            dc_flags = dc_flags.reindex(status_row.index, fill_value=False)
            available_ac_ports = ((status_row == "AVAILABLE") & (~dc_flags)).sum()
            available_dc_ports = ((status_row == "AVAILABLE") & (dc_flags)).sum()
            is_ac_available = int(available_ac_ports > 0)
            is_dc_available = int(available_dc_ports > 0)

            record = {
                "snapshot_ts": ts.isoformat(),
                "total_ports": total_ports_const,
                "available_ports": available_ports,
                "unavailable_ports": unavailable_ports,
                "unavailable_rate": unavailable_rate,
                "total_ac_ports": number_of_ac_ports,
                "total_dc_ports": number_of_dc_ports,
                "available_ac_ports": available_ac_ports,
                "available_dc_ports": available_dc_ports,
                "is_ac_available": is_ac_available,
                "is_dc_available": is_dc_available,
                "has_dc_fast": has_dc_fast,
                "has_ccs": has_ccs,
                "has_chademo": has_chademo,
                "has_type2": has_type2,
                "has_wall_outlet": has_wall_outlet,
            }

            record.update(st_row)
            record.update(loc_row)

            record.pop("st_ports", None)
            record.pop("loc_stations", None)

            records.append(record)

        print(f"Built timeseries for station {st_id} ({len(ts_index)} timestamps)")

    result_df = pd.DataFrame(records)
    if not result_df.empty:
        result_df = result_df.sort_values(["st_id", "snapshot_ts"]).reset_index(
            drop=True
        )
    return result_df


def main(argv: list[str] | None = None) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loc_df, st_df, ports_df = load_inputs()
    ts_df = build_station_timeseries(loc_df, st_df, ports_df)
    ts_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote {OUTPUT_FILE} ({len(ts_df)} rows)")


if __name__ == "__main__":
    main()
