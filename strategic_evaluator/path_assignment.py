from typing import Dict
import numpy as np
import pandas as pd
from scipy.stats import zscore


def assign_passenger_to_itinerary(
    df_itineraries: pd.DataFrame, network_paths_config_demand,
):
    archetypes_sensitivities = pd.read_csv(
        network_paths_config_demand['sensitivities'])
    print(archetypes_sensitivities)
    print(archetypes_sensitivities.columns)
    df_itineraries = calculate_total_cost_per_archetype(
        df_itineraries, archetypes_sensitivities
    )
    df_itineraries = calculate_share_on_each_path(
        df_itineraries, archetypes_sensitivities
    )

    return df_itineraries


def calculate_total_cost_per_archetype(df_itineraries: pd.DataFrame, sensitivities: Dict):
    df_itineraries["calibration_constant"] = 0.6
    for _, row in sensitivities.iterrows():
        archetype_name = row["archetype_name"]
        df_itineraries[f"total_cost_{archetype_name}"] = (
            df_itineraries["calibration_constant"] +
            row["CO2_sensitivity"] * df_itineraries["CO2_total_cost"] +
            row["price_sensitivity"] * df_itineraries["price_cost"] +
            row["travel_time_sensitivity"] *
            df_itineraries["total_travel_time"]
        )
        df_itineraries[f"normalized_cost_{archetype_name}"] = zscore(
            df_itineraries[f"total_cost_{archetype_name}"])
    return df_itineraries


def calculate_share_on_each_path(
    df_itineraries: pd.DataFrame, sensitivities: pd.DataFrame
):
    for archetype in sensitivities["archetype_name"]:
        df_itineraries[f"share_of_{archetype}_on_that_path"] = 0
        for od_pair, data in df_itineraries.groupby(["origin", "destination"]):
            total_costs_of_all_options = sum(
                np.exp(data[f"normalized_cost_{archetype}"]))
            df_itineraries.loc[data.index, f"share_of_{archetype}_on_that_path"] = (
                np.exp(data[f"normalized_cost_{
                       archetype}"]) / total_costs_of_all_options
            ).astype(float)

    return df_itineraries
