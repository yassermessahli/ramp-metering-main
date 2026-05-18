# evaluation/parsers.py

import re
import xml.etree.ElementTree as ET

import pandas as pd


def get_route_type(trip_id):
    """Categorizes trips based on ID prefix."""
    if "main" in trip_id:
        return "Mainline"
    if "on_ramp" in trip_id:
        return "On-Ramp"
    if "off_ramp" in trip_id:
        return "Off-Ramp"
    return "Other"


def parse_tripinfo_for_episode_stats(tripinfo_path):
    """Parses SUMO tripinfo XML to compute episode statistics."""
    try:
        tree = ET.parse(tripinfo_path)
        root = tree.getroot()
    except (FileNotFoundError, ET.ParseError):
        print(f"\nWarning: Could not parse tripinfo file at {tripinfo_path}")
        return {}

    trip_data = []
    for trip in root.findall("tripinfo"):
        if trip.get("duration"):
            trip_attrs = trip.attrib
            trip_attrs["route_type"] = get_route_type(trip_attrs["id"])

            emissions_element = trip.find("emissions")
            if emissions_element is not None:
                trip_attrs.update(emissions_element.attrib)

            for key, val in trip_attrs.items():
                try:
                    trip_attrs[key] = float(val)
                except (ValueError, TypeError):
                    continue

            trip_attrs["vaporized"] = 1 if "vaporized" in trip.keys() else 0
            trip_data.append(trip_attrs)

    if not trip_data:
        return {
            "total_throughput": 0,
            "total_travel_time": 0,
            "avg_travel_time": 0,
            "median_travel_time": 0,
            "std_dev_travel_time": 0,
            "total_time_loss": 0,
            "avg_time_loss": 0,
            "median_time_loss": 0,
            "std_dev_time_loss": 0,
            "sum_of_squared_time_loss": 0,
            "total_waiting_time": 0,
            "avg_waiting_time": 0,
            "num_teleported_tripinfo": 0,
            "total_co2_mg": 0,
            "total_fuel_ml": 0,
            "total_nox_mg": 0,
        }

    df = pd.DataFrame(trip_data)
    df.fillna(0, inplace=True)
    df["timeLoss_sq"] = df["timeLoss"] ** 2

    overall_stats = {
        "total_throughput": len(df),
        "total_travel_time": df["duration"].sum(),
        "avg_travel_time": df["duration"].mean(),
        "median_travel_time": df["duration"].median(),
        "std_dev_travel_time": df["duration"].std(),
        "total_time_loss": df["timeLoss"].sum(),
        "avg_time_loss": df["timeLoss"].mean(),
        "median_time_loss": df["timeLoss"].median(),
        "std_dev_time_loss": df["timeLoss"].std(),
        "sum_of_squared_time_loss": df["timeLoss_sq"].sum(),
        "total_waiting_time": df["waitingTime"].sum(),
        "avg_waiting_time": df["waitingTime"].mean(),
        "num_teleported_tripinfo": df["vaporized"].sum(),
        "total_co2_mg": df["CO2_abs"].sum(),
        "total_fuel_ml": df["fuel_abs"].sum(),
        "total_nox_mg": df["NOx_abs"].sum(),
    }

    expected_routes = ["Mainline", "On-Ramp", "Off-Ramp"]
    df["route_type"] = pd.Categorical(
        df["route_type"], categories=expected_routes, ordered=True
    )

    route_stats = (
        df.groupby("route_type", observed=False)
        .agg(
            avg_time_loss=("timeLoss", "mean"),
            avg_travel_time=("duration", "mean"),
            throughput=("id", "count"),
        )
        .unstack()
    )

    route_stats.index = [f"{col[1]}_{col[0]}" for col in route_stats.index]
    route_stats_dict = route_stats.to_dict()

    return {**overall_stats, **route_stats_dict}


def parse_sumo_log(log_path):
    """Parses SUMO log file for demand and insertion metrics."""
    try:
        with open(log_path) as f:
            content = f.read()
    except FileNotFoundError:
        return {}

    inserted_loaded_match = re.search(
        r"Vehicles:\s*\n\s*Inserted:\s*(\d+)\s*\(Loaded:\s*(\d+)\)", content
    )
    emergency_stops_match = re.search(r"Emergency Stops:\s*(\d+)", content)
    demand_inserted = int(inserted_loaded_match.group(1)) if inserted_loaded_match else 0
    demand_loaded = int(inserted_loaded_match.group(2)) if inserted_loaded_match else 0

    return {
        "demand_loaded": demand_loaded,
        "demand_inserted": demand_inserted,
        "service_rate": demand_inserted / demand_loaded if demand_loaded > 0 else 0,
        "num_emergency_stops": int(emergency_stops_match.group(1))
        if emergency_stops_match
        else 0,
    }


def parse_framework_log(log_path, spillback_threshold=20):
    """Parses framework CSV log for detector metrics and spillback analysis."""
    try:
        df = pd.read_csv(log_path)
        if df.empty:
            return {}
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"\nWarning: Framework log not found at {log_path}")
        return {}

    avg_metric_cols = [
        "mainline_flow_upstream_v/h",
        "mainline_occ_upstream_percent",
        "mainline_speed_upstream_km/h",
        "mainline_flow_mergeArea_v/h",
        "mainline_occ_mergeArea_percent",
        "mainline_speed_mergeArea_km/h",
        "mainline_flow_downstream_v/h",
        "mainline_occ_downstream_percent",
        "mainline_speed_downstream_km/h",
        "ramp_queue_veh",
    ]

    avg_metrics = {
        f"avg_{col}": df[col].mean() for col in avg_metric_cols if col in df.columns
    }

    total_spillback_time = 0
    if "ramp_queue_veh" in df.columns and "sim_time" in df.columns:
        spillback_df = df[df["ramp_queue_veh"] > spillback_threshold]

        if not spillback_df.empty:
            time_per_step = df["sim_time"].diff().median()
            if pd.notna(time_per_step) and time_per_step > 0:
                total_spillback_time = len(spillback_df) * time_per_step

    avg_metrics["total_spillback_time_sec"] = total_spillback_time

    return avg_metrics
