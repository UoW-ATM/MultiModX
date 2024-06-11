from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import h3


def compute_costs(df_itineraries: pd.DataFrame, network_paths_config):
    df_itineraries_with_costs = (
        df_itineraries
        .pipe(calculate_distance, network_paths_config)
        .pipe(calculate_CO2_cost_per_itinerary)
        .pipe(calculate_price_cost_per_itinerary)
        .pipe(calculate_travel_time_cost_per_itinerary)
    )

    return df_itineraries_with_costs


def calculate_distance(
    df_itineraries: pd.DataFrame, network_paths_config,
):
    airports = pd.read_csv(
        Path(network_paths_config['network_path']) /
        network_paths_config['air_network']['airports_static']
    )
    train_stations = pd.read_csv(
        Path(network_paths_config['network_path']) /
        network_paths_config['rail_network']['gtfs'] / 'stops.txt'
    )

    code_to_lat = airports.set_index("icao_id")["lat"].to_dict()
    code_to_lon = airports.set_index("icao_id")["lon"].to_dict()
    code_to_lat.update(train_stations.astype(
        str).set_index("stop_id")["stop_lat"])
    code_to_lon.update(train_stations.astype(
        str).set_index("stop_id")["stop_lon"])

    if "nservices" in df_itineraries.columns:
        max_connections = df_itineraries["nservices"].max()
    else:
        max_connections = 2

    for service in range(max_connections):
        df_itineraries[f"origin_{service}_latitude"] = df_itineraries[f"origin_{
            service}"].map(code_to_lat).astype(float)
        df_itineraries[f"origin_{service}_longitude"] = df_itineraries[f"origin_{
            service}"].map(code_to_lon).astype(float)
        df_itineraries[f"destination_{service}_latitude"] = df_itineraries[f"destination_{
            service}"].map(code_to_lat).astype(float)
        df_itineraries[f"destination_{service}_longitude"] = df_itineraries[f"destination_{
            service}"].map(code_to_lon).astype(float)

        df_itineraries[f"distance_{service}"] = df_itineraries.apply(
            lambda row: h3.point_dist(
                (row[f'origin_{service}_latitude'],
                 row[f'origin_{service}_longitude']),
                (row[f'destination_{service}_latitude'],
                 row[f'destination_{service}_longitude'])
            ), axis=1
        )

    return df_itineraries


def calculate_CO2_cost_per_itinerary(df_itineraries: pd.DataFrame):
    cost_per_km_rail = 33
    cost_per_km_plane = 200
    df_itineraries["CO2_total_cost"] = 0

    if "nservices" in df_itineraries.columns:
        max_connections = df_itineraries["nservices"].max()
    else:
        max_connections = 2

    for service_number in range(max_connections):
        df_itineraries[f"CO2_service_{service_number}"] = df_itineraries.apply(
            lambda row: cost_per_km_rail*row[f"distance_{service_number}"]
            if row[f"mode_{service_number}"] == 'rail' else cost_per_km_plane*row[f"distance_{service_number}"],
            axis=1)
        df_itineraries["CO2_total_cost"] += df_itineraries[
            f"CO2_service_{service_number}"
        ].fillna(0)

    df_itineraries["CO2_total_cost"] = df_itineraries["CO2_total_cost"] / 100

    return df_itineraries


def calculate_price_cost_per_itinerary(df_itineraries: pd.DataFrame):
    df_itineraries["price_cost"] = np.random.choice(
        np.arange(10, 1000+0.5*1, 1), size=len(df_itineraries)
    ).astype(int)

    return df_itineraries


def calculate_travel_time_cost_per_itinerary(df_itineraries: pd.DataFrame):
    cost_of_time = {
        "connecting_time": 2,
        "access_time": 2,
        "egress_time": 2,
        "waiting_time": 1,
        "travel_time": 0.5
    }

    df_itineraries["travel_time_cost"] = (
        df_itineraries["access_time"] * cost_of_time["access_time"] +
        df_itineraries["egress_time"] * cost_of_time["egress_time"]
    )

    if "nservices" in df_itineraries.columns:
        max_connections = df_itineraries["nservices"].max()
    else:
        max_connections = 2

    for service_number in range(max_connections):
        if service_number != max_connections - 1:
            df_itineraries["travel_time_cost"] += (
                df_itineraries[f"connecting_time_{service_number}_{service_number + 1}"].fillna(0) *
                cost_of_time["connecting_time"] +
                df_itineraries[f"waiting_time_{service_number}_{service_number + 1}"].fillna(0) *
                cost_of_time["waiting_time"]
            )
        df_itineraries["travel_time_cost"] += (
            df_itineraries[f"travel_time_{service_number}"].fillna(0) *
            cost_of_time["travel_time"]
        )

    return df_itineraries
