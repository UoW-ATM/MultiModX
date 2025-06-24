import os
import geopandas as gpd
import pandas as pd
import re
from config import DATA_FOLDER, MAPS_FOLDER, VARIABLES, RAIL_FOLDER, INFRASTRUCTURE_FOLDER


def get_case_study_folders():
    return [f for f in os.listdir(DATA_FOLDER) if os.path.isdir(os.path.join(DATA_FOLDER, f))]

def get_components_cs(folders):
    components = {
        "cs": set(),
        "pp": set(),
        "nd": set(),
        "so": set()
    }

    pattern = r"processed_cs(\d+)\.pp(\d+)\.nd(\d+)\.so(\d+\.\d+)"

    for folder in folders:
        match = re.match(pattern, folder)
        if match:
            cs, pp, nd, so = match.groups()
            components["cs"].add(cs)
            components["pp"].add(pp)
            components["nd"].add(nd)
            components["so"].add(so)

    # Sort values for UI
    for key in components:
        components[key] = sorted(components[key])

    return components

def load_variable_options():
    variable_options = [
        {"label": label, "value": label}  # use label as value
        for label in VARIABLES.keys()
    ]
    return variable_options

def load_nuts3_geodata():
    nuts_gdf = gpd.read_file(MAPS_FOLDER)
    nuts_gdf = nuts_gdf[(nuts_gdf["LEVL_CODE"] == 3) & (nuts_gdf['NUTS_ID'].str.startswith("ES"))]  # NUTS3 only
    return nuts_gdf

def load_rail_stops():
    rail_stops = pd.read_csv(RAIL_FOLDER + '/stops.txt', dtype={'stop_id': str})
    return rail_stops

def load_airports():
    airports = pd.read_csv(INFRASTRUCTURE_FOLDER + '/airports_info' +'/airports_coordinates_v1.1.csv')
    return airports


def read_pax_itineraries(case_study, subfolder="", file=""):
    csv_path = os.path.join(DATA_FOLDER, case_study, subfolder, file)
    df_id = None
    if os.path.isfile(csv_path):
        df_it = pd.read_csv(csv_path, usecols=['origin', 'destination', 'path',
                                               'type', 'fare',
                                               'total_time', 'total_waiting_time',
                                               'access_time', 'egress_time',
                                               'd2i_time', 'i2d_time', 'pax'])

    return df_it

def read_pax_paths(case_study, subfolder="", files=""):
    csv_path = os.path.join(DATA_FOLDER, case_study, subfolder, files['pax_assigned_to_paths'])
    csv_poss_it_cluster = os.path.join(DATA_FOLDER, case_study, subfolder, files['possible_it_clustered'])
    df_p = None
    if os.path.isfile(csv_path) and os.path.isfile(csv_poss_it_cluster):
        df_p = pd.read_csv(csv_path, usecols=['origin', 'destination', 'journey_type',
                                              'alternative_id',
                                               'total_travel_time', 'total_cost',
                                               'total_emissions', 'total_waiting_time',
                                               'nservices', 'num_pax'])


        df_pp = pd.read_csv(csv_poss_it_cluster, usecols=['alternative_id', 'path', 'access_time', 'egress_time',
                                                          'd2i_time', 'i2d_time']).drop_duplicates()
        df_p = df_p.merge(df_pp, on=['alternative_id'])

    return df_p

