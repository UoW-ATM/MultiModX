from pathlib import Path
import pandas as pd
import numpy as np
from itertools import combinations
import multiprocessing as mp
import gc
import ast
import time
import logging
import sys
import shutil
import os

from joblib import Parallel, delayed

from strategic_evaluator.mobility_network import Service, NetworkLayer, Network
from strategic_evaluator.mobility_network_particularities import (mct_air_network, fastest_air_time_heuristic,
                                                                  initialise_air_network, mct_rail_network,
                                                                  services_from_after_function_rail,
                                                                  initialise_rail_network,
                                                                  fastest_rail_time_heuristic,
                                                                  fastest_precomputed_distance_time_heuristic)

from libs.gtfs import get_stop_times_on_date, add_date_and_handle_overflow
from libs.emissions_costs_computation import (compute_emissions_pax_short_mid_flights, compute_costs_air,
                                              compute_emissions_rail, compute_costs_rail)
from libs.time_converstions import  convert_to_utc_vectorized
from libs.passenger_assigner.passenger_assigner import assign_passengers_options_solver, fill_flights_too_empy

logger = logging.getLogger(__name__)


def create_region_access_dict(df_ra_air):
    regions_access_air = {}
    for i, r in df_ra_air.iterrows():
        if r.region not in regions_access_air.keys():
            regions_access_air[r.region] = []
        dict_info_station = None
        for st in regions_access_air[r.region]:
            if st['station'] == r.station:
                # Same station as before
                dict_info_station = st
                break
        if dict_info_station is None:
            dict_info_station = {'station': r.station}
            regions_access_air[r.region].append(dict_info_station)

        if r.access_type in dict_info_station.keys():
            dict_info_access = dict_info_station[r.access_type]
        else:
            dict_info_access = {}
            dict_info_station[r.access_type] = dict_info_access

        #dict_info_access[r.pax_type] = r.avg_time
        dict_info_access[r.pax_type] = {'total_time':r.total_time,'avg_time':r.avg_time}

    return regions_access_air


def create_air_layer(df_fs, df_as, df_mct, mct_default=None, df_ra_air=None, keep_only_fastest_service=0,
                     dict_dist_origin_destination=None,
                     heuristic_precomputed_distance=None):

    # Regions access -- Create dictionary
    regions_access_air = None
    if df_ra_air is not None:
        regions_access_air = create_region_access_dict(df_ra_air)

    #  MCTs between air services
    dict_mct_std = dict(zip(df_mct['icao_id'], df_mct['standard']))
    dict_mct_international = dict(zip(df_mct['icao_id'], df_mct['international']))
    dict_mct_domestic = dict(zip(df_mct['icao_id'], df_mct['domestic']))
    dict_mct = {'std': dict_mct_std, 'int': dict_mct_international, 'dom': dict_mct_domestic}
    if mct_default is not None:
        dict_mct['avg_default'] = mct_default
    else:
        # Average of standard if nothing provided
        dict_mct['avg_default'] = round(df_mct['standard'].mean())

    # Rename sobt and sibt
    df_fs.rename(columns={'sobt': 'departure_time', 'sibt': 'arrival_time'}, inplace=True)

    if keep_only_fastest_service > 0:
        df_fs['duration'] = None
        df_fs['duration'] = df_fs.apply(lambda x: x.arrival_time - x.departure_time, axis=1)

    if keep_only_fastest_service == 1:
        # Keep only the fastest service considering alliance
        df_fs = df_fs.groupby(['origin', 'destination', 'alliance']).apply(lambda x: x.sort_values('duration').iloc[0]).copy()
        df_fs = df_fs.reset_index(drop=True)

    if keep_only_fastest_service == 2:
        # Keep only one fastest regardless of airline/alliance
        df_fs = df_fs.groupby(['origin', 'destination']).apply(lambda x: x.sort_values('duration').iloc[0]).copy()
        df_fs = df_fs.reset_index(drop=True)

    # Create Services for flights (one per flight in the dataframe)
    df_fs['service'] = df_fs.apply(lambda x: Service(x.service_id, x.origin, x.destination,
                                                     x.departure_time, x.arrival_time, x.cost,
                                                     x.provider, x.alliance, x.emissions,
                                                     gcdistance=x.gcdistance,
                                                     country_origin=x.country_origin,
                                                     country_destination=x.country_destination,
                                                     mode='air',
                                                     #coordinates_origin=(x.lat_orig, x.lon_orig),
                                                     #coordinates_destination=(x.lat_dest, x.lon_dest)
                                                     ), axis=1)

    # Use precomputed distance vs time function or default air
    if heuristic_precomputed_distance is not None:
        heuristic_func = fastest_precomputed_distance_time_heuristic
    else:
        heuristic_func = fastest_air_time_heuristic

    # Create network
    anl = NetworkLayer('air',
                       df_fs[['service_id', 'origin', 'destination', 'departure_time', 'arrival_time', 'cost',
                              'provider', 'alliance', 'service']].copy(),
                       dict_mct, regions_access=regions_access_air,
                       custom_mct_func=mct_air_network,
                       custom_heuristic_func=heuristic_func,
                       custom_initialisation=initialise_air_network,
                       keep_only_fastest_service=keep_only_fastest_service,
                       nodes_coordinates=df_as.copy().rename({'icao_id': 'node'}, axis=1),
                       dict_dist_origin_destination=dict_dist_origin_destination,
                       heuristic_precomputed_distance=heuristic_precomputed_distance
                       )

    return anl


def create_rail_layer(df_rail, from_gtfs=True, date_considered='20240101', df_stops_considered=None, df_ra_rail=None,
                      keep_only_fastest_service=0,
                      df_mct=None,
                      mct_default=None,
                      df_stops=None,
                      heuristic_precomputed_distance=None):

    if from_gtfs:
        # Process GTFS to create services
        if type(date_considered) is str:
            date_considered = pd.to_datetime(date_considered, format='%Y%m%d')
        rail_services_df = pre_process_rail_gtfs_to_services(df_rail, date_considered, df_stops)
    else:
        rail_services_df = df_rail

    # MCT between rail if provided
    dict_mct_rail = {}
    avg_transfer_time = None

    if df_mct is not None:
        #  MCTs between air services
        dict_mct_rail = dict(zip(df_mct['stop_id'], df_mct['default transfer time']))
    if mct_default is not None:
        avg_transfer_time = mct_default
    elif df_mct is not None:
        avg_transfer_time = round(df_mct['default transfer time'].mean())

    dict_mct = {'std': dict_mct_rail, 'avg_default': avg_transfer_time}

    # Regions access -- Create dictionary
    regions_access_rail = None
    if df_ra_rail is not None:
        regions_access_rail = create_region_access_dict(df_ra_rail)

    if df_stops_considered is not None:
        rail_services_df = rail_services_df[(rail_services_df.origin.isin(df_stops_considered.stop_id)) &
                                            (rail_services_df.destination.isin(df_stops_considered.stop_id))].copy()

    # Create Service objects
    if df_stops is not None:
        df_stops = df_stops.copy().rename({'stop_id': 'node', 'stop_lat': 'lat', 'stop_lon': 'lon'},
                                          axis=1)[['node', 'lat', 'lon']]
        df_stops['node'] = df_stops['node'].astype(str)

        rail_services_df = rail_services_df.merge(df_stops, left_on=['origin'], right_on=['node'], how='left')
        rail_services_df.rename(columns={'lat': 'lat_orig', 'lon': 'lon_orig'}, inplace=True)
        rail_services_df.drop(columns='node', inplace=True)
        rail_services_df = rail_services_df.merge(df_stops, left_on=['destination'], right_on=['node'], how='left')
        rail_services_df.rename(columns={'lat': 'lat_dest', 'lon': 'lon_dest'}, inplace=True)
        rail_services_df.drop(columns='node', inplace=True)

    if 'gcdistance' not in rail_services_df.columns:
        dict_dist_origin_destination = create_dict_distance_origin_destination(rail_services_df[['origin',
                                                                                                 'destination',
                                                                                                 'lat_orig',
                                                                                                 'lon_orig',
                                                                                                 'lat_dest',
                                                                                                 'lon_dest']].drop_duplicates())

        rail_services_df['gcdistance'] = rail_services_df.apply(lambda x: dict_dist_origin_destination[x['origin'],
                                                                                            x['destination']], axis=1)

    rail_services_df['service'] = rail_services_df.apply(lambda x:
                                                         Service(x.service_id, x.origin, x.destination,
                                                                 x.departure_time, x.arrival_time,
                                                                 x.cost, x.provider, x.alliance, x.emissions, x.seats,
                                                                 gcdistance=x.gcdistance,
                                                                 country_origin=x.country,
                                                                 country_destination=x.country,
                                                                 mode='rail',
                                                                 #coordinates_origin=(x.lat_orig, x.lon_orig),
                                                                 #coordinates_destination=(x.lat_dest, x.lon_dest)
                                                                 ),
                                                         axis=1)


    # Keep only fastest services if requested to do so
    if keep_only_fastest_service == 1:
        rail_services_df['duration'] = rail_services_df['arrival_time'] - rail_services_df['departure_time']
        # Group by 'origin' and 'destination', then keep only the first row of each group
        rail_services_df = rail_services_df.groupby(['origin', 'destination', 'alliance']).apply(lambda x:
                                                                                     x.sort_values('duration').iloc[0])
        rail_services_df = rail_services_df.reset_index(drop=True)
    elif keep_only_fastest_service == 2:
        rail_services_df['duration'] = rail_services_df['arrival_time'] - rail_services_df['departure_time']
        # Group by 'origin' and 'destination', then keep only the first row of each group
        rail_services_df = rail_services_df.groupby(['origin', 'destination']).apply(lambda x:
                                                                                     x.sort_values('duration').iloc[0])
        rail_services_df = rail_services_df.reset_index(drop=True)


    # Use precomputed distance vs time function or default air
    if heuristic_precomputed_distance is not None:
        heuristic_func = fastest_precomputed_distance_time_heuristic
    else:
        heuristic_func = fastest_rail_time_heuristic

    nl_rail = NetworkLayer('rail', rail_services_df,
                           dict_mct=dict_mct,
                           regions_access=regions_access_rail,
                           custom_initialisation=initialise_rail_network,
                           custom_mct_func=mct_rail_network,
                           custom_services_from_after_func=services_from_after_function_rail,
                           nodes_coordinates=df_stops,
                           heuristic_precomputed_distance=heuristic_precomputed_distance,
                           custom_heuristic_func=heuristic_func)

    return nl_rail


def apply_ban_policy(network_definition_config, flight_ban_policy, pre_processed_version=0):
    # We have a flight ban with relates to rail operations, so compute which o-d pairs need to be 'removed'
    # from the flights
    df_od_banned = compute_od_pairs_banned(network_definition_config, flight_ban_policy,
                                           pre_processed_version)

    # Now we know which OD pairs we are banning, save it!

    # Read flight_schedules_proc_ and save it again without the banned od pairs
    path_network = network_definition_config['network_path']
    processed_folder = network_definition_config['processed_folder']
    fflights = 'flight_schedules_proc_' + str(pre_processed_version) + '.csv'
    df_fs = pd.read_csv(Path(path_network) / processed_folder / fflights)

    # merge to flag banned flights
    df_merged = df_fs.merge(df_od_banned, on=["origin", "destination"], how="left", indicator=True)

    # Split into banned and allowed
    df_fs_banned = df_merged[df_merged["_merge"] == "both"].drop(columns=["_merge"])
    df_fs_allowed = df_merged[df_merged["_merge"] == "left_only"].drop(columns=["_merge"])

    # Save flights kept
    fflights = 'flight_schedules_proc_' + str(pre_processed_version) + '.csv'
    df_fs_allowed.to_csv(Path(path_network) / processed_folder / fflights, index=False)
    # Save flights banned
    fflights = 'flight_schedules_proc_' + str(pre_processed_version) + '_banned.csv'
    df_fs_banned.to_csv(Path(path_network) / processed_folder / fflights, index=False)
    # Save od pair banned
    fodbanned = 'origin_destination_banned_' + str(pre_processed_version) + '.csv'
    df_od_banned.to_csv(Path(path_network) / processed_folder / fodbanned, index=False)


def preprocess_input(network_definition_config, pre_processed_version=0, policy_package=None):
    if pre_processed_version == 0:
        # Read from raw data and create proc_#.csv files for air and/or rail
        if 'air_network' in network_definition_config.keys():
            preprocess_air_layer(network_definition_config['network_path'], network_definition_config['air_network'],
                                 network_definition_config['processed_folder'],
                                 pre_processed_version=pre_processed_version)

        if 'rail_network' in network_definition_config.keys():
            pre_process_rail_layer(network_definition_config['network_path'], network_definition_config['rail_network'],
                                   network_definition_config['processed_folder'],
                                   pre_processed_version=pre_processed_version)

        if ((policy_package is not None) and
                (('flight_ban' in policy_package.keys())
                 and ('rail_network' in network_definition_config.keys())
                 and ('air_network' in network_definition_config.keys()))):
            apply_ban_policy(network_definition_config, policy_package['flight_ban'],pre_processed_version)

    else:
        # We don't preprocess but just copy the preprocessed values
        if 'air_network' in network_definition_config:
            # We have air network, copy the processed file
            flight_schedule_pre_proc_path = (Path(network_definition_config['network_path']) /
                                             network_definition_config['pre_processed_input_folder'] /
                                             ('flight_schedules_proc_' + str(pre_processed_version) + '.csv'))

            flight_schedule_pre_proc_dest = (Path(network_definition_config['network_path']) /
                                             network_definition_config['processed_folder'] /
                                             ('flight_schedules_proc_' + str(pre_processed_version) + '.csv'))

            shutil.copy2(flight_schedule_pre_proc_path, flight_schedule_pre_proc_dest)

        if 'rail_network' in network_definition_config:
            # We have rail network, copy the processed file, it might need some preprocessing
            rnd = network_definition_config['rail_network'][0]  # TODO now working only with one rail network

            if rnd['create_rail_layer_from'] == 'services':
                # All is done then just copy the rail_timetable_proc_#.csv and we're done
                rail_services_pre_proc_path = (Path(network_definition_config['network_path']) /
                                                 network_definition_config['pre_processed_input_folder'] /
                                                 ('rail_timetable_proc_' + str(pre_processed_version) + '.csv'))

                rail_services_pre_proc_dest = (Path(network_definition_config['network_path']) /
                                                 network_definition_config['processed_folder'] /
                                                 ('rail_timetable_proc_' + str(pre_processed_version) + '.csv'))

                shutil.copy2(rail_services_pre_proc_path, rail_services_pre_proc_dest)

            else:
                # We have them in gtfs format... might need some processing
                def copy_if_exist(orig_path, destination_path):
                    if os.path.exists(orig_path):
                        shutil.copy2(orig_path, destination_path)

                # Check if rail_timetable_all_gtfs_#.csv exist and if sc copy it
                rail_gtfs_all_pre_proc_path = (Path(network_definition_config['network_path']) /
                                               network_definition_config['pre_processed_input_folder'] /
                                               ('rail_timetable_all_gtfs_' + str(pre_processed_version) + '.csv'))

                rail_gtfs_all_pre_proc_dest = (Path(network_definition_config['network_path']) /
                                               network_definition_config['processed_folder'] /
                                               ('rail_timetable_all_gtfs_' + str(pre_processed_version) + '.csv'))

                copy_if_exist(rail_gtfs_all_pre_proc_path, rail_gtfs_all_pre_proc_dest)

                # Check if rail_timetable_proc_gtfs_#.csv exist and if sc copy it
                rail_gtfs_proc_pre_path = (Path(network_definition_config['network_path']) /
                                               network_definition_config['pre_processed_input_folder'] /
                                               ('rail_timetable_proc_gtfs_' + str(pre_processed_version) + '.csv'))

                rail_gtfs_proc_pre_dest = (Path(network_definition_config['network_path']) /
                                               network_definition_config['processed_folder'] /
                                               ('rail_timetable_proc_gtfs_' + str(pre_processed_version) + '.csv'))

                copy_if_exist(rail_gtfs_proc_pre_path, rail_gtfs_proc_pre_dest)

                if not os.path.exists(rail_gtfs_proc_pre_dest):
                    # We don't have the filtered version
                    df_stops_considered = pd.read_csv(Path(network_definition_config['network_path']) /
                                                      rnd['rail_stations_considered'], dtype={'stop_id': str})
                    df_gtfs_all_pre = pd.read_csv(rail_gtfs_all_pre_proc_dest, dtype={'stop_id': str})
                    df_gtfs_proc_pre = df_gtfs_all_pre[df_gtfs_all_pre['stop_id'].isin(df_stops_considered['stop_id'])]
                    df_gtfs_proc_pre.to_csv(rail_gtfs_proc_pre_dest, index=False)



def compute_cost_emissions_air(df_fs):
    # print(df_fs[df_fs.gcdistance==0])
    # TODO: emissions for long-haul flights
    df_fs['emissions'] = df_fs.apply(lambda row:
                                     compute_emissions_pax_short_mid_flights(row['gcdistance'], row['seats'])
                                     if pd.isnull(row['emissions']) else row['emissions'], axis=1)

    df_fs['emissions'] = pd.to_numeric(df_fs['emissions'], errors='coerce')

    df_fs['cost'] = df_fs.apply(lambda row: compute_costs_air(row['gcdistance'])
                                            if pd.isnull(row['cost']) else row['cost'], axis=1)

    return df_fs


def compute_cost_emissions_rail(df_rs):
    df_rs['emissions'] = df_rs.apply(lambda row:
                                     compute_emissions_rail(row['gcdistance'], row['country'])
                                     if pd.isnull(row['emissions']) else row['emissions'], axis=1)

    df_rs['cost'] = df_rs.apply(lambda row: compute_costs_rail(row['gcdistance'], row['country'])
                                            if pd.isnull(row['cost']) else row['cost'], axis=1)

    return df_rs


def compute_od_pairs_banned(network_definition_config, flight_ban_def, pre_processed_version):
    from libs.uow_tool_belt.general_tools import haversine

    path_network = network_definition_config['network_path']
    processed_folder = network_definition_config['processed_folder']
    fflights = 'flight_schedules_proc_' + str(pre_processed_version) + '.csv'
    df_fs = pd.read_csv(Path(path_network) / processed_folder / fflights)
    df_fs = df_fs[['origin', 'destination', 'gcdistance']]
    df_fs = df_fs.groupby(["origin", "destination"], as_index=False).agg({"gcdistance": "min"})
    ftrains = 'rail_timetable_proc_' + str(pre_processed_version) + '.csv'
    df_t = pd.read_csv(Path(path_network) / processed_folder / ftrains)

    def get_od_flights_from_rail(df_fs, df_t):
        # Get all the o-d pairs in the df_fs which have a train in the df_t file

        # Keep only one representative for origin - destination
        df_t = df_t[['origin', 'destination',
                     'lat_orig', 'lon_orig',
                     'lat_dest', 'lon_dest']].drop_duplicates()

        df_stops = pd.concat([df_t[['origin', 'lat_orig', 'lon_orig']].rename({'origin': 'stop_id',
                                                         'lat_orig': 'lat',
                                                         'lon_orig': 'lon'},
                                      axis=1),
                              df_t[['destination', 'lat_dest', 'lon_dest']].rename({'destination': 'stop_id',
                                                         'lat_dest': 'lat',
                                                         'lon_dest': 'lon'},
                                                        axis=1)]).drop_duplicates()

        df_airports = pd.concat([df_fs[['origin']].rename({'origin': 'airport_id'}, axis=1),
                                 df_fs[['destination']].rename({'destination': 'airport_id'}, axis=1)]).drop_duplicates()


        # [0] as in the toml we can have a list of air networks
        df_airports_st = pd.read_csv(Path(path_network) / network_definition_config['air_network'][0]['airports_static'])
        df_airports = df_airports.merge(df_airports_st, left_on='airport_id', right_on='icao_id')

        # Threshold to decide rail station and airport serve same origin/destination
        threshold_km = flight_ban_def.get('distance_rail_serves_airport', 30)

        # Compute all pairwise distances between stops and airports
        df_stops["key"] = 1  # Temporary key for cross join
        df_airports["key"] = 1

        df_merged = df_stops.merge(df_airports, on="key").drop(columns=["key"])  # Cross join

        # Compute the GCD using the haversine function
        df_merged["distance_km"] = df_merged.apply(
            lambda row: haversine(row["lon_x"], row["lat_x"], row["lon_y"], row["lat_y"]), axis=1)

        # Filter based on threshold
        df_nearby = df_merged[df_merged["distance_km"] <= threshold_km]

        # Select relevant columns to have relationship between rail stations and airport
        df_nearby = df_nearby[["stop_id", "airport_id", "distance_km"]].reset_index(drop=True)

        # Filter the origin destination paris from df_fs for which a train exists in the already filtered df_t
        # Merge df_fs (airports) with df_nearby (to find train stations serving the airports)
        df_fs_origin_stations = df_fs.merge(df_nearby, left_on="origin", right_on="airport_id") \
            .rename(columns={"stop_id": "train_origin"}) \
            .drop(columns=["airport_id", "distance_km"])

        df_fs_dest_stations = df_fs.merge(df_nearby, left_on="destination", right_on="airport_id") \
            .rename(columns={"stop_id": "train_destination"}) \
            .drop(columns=["airport_id", "distance_km"])

        # Now, merge the two new dataframes to get all train stations corresponding to each airport-airport route
        df_possible_routes = df_fs_origin_stations.merge(df_fs_dest_stations, on=["origin", "destination"])

        # Find valid train routes that match the possible routes
        df_valid_routes = df_possible_routes.merge(df_t, left_on=["train_origin", "train_destination"],
                                                   right_on=["origin", "destination"]) \
            .drop(columns=["train_origin", "train_destination"])

        # The final list of origin-destination airport pairs that have train connections
        df_final_filtered_fs = df_valid_routes[["origin_x", "destination_x"]].drop_duplicates().rename({'origin_x': 'origin',
                                                                                                        'destination_x': 'destination'}, axis=1)

        df_fs = df_final_filtered_fs
        # Ensure that if a given o-d pair is in the list the return d-o is also banned
        # Create a set of existing (origin, destination) pairs
        existing_pairs = set(zip(df_fs["origin"], df_fs["destination"]))

        # Find missing reverse pairs
        missing_pairs = {(dest, orig) for orig, dest in existing_pairs if (dest, orig) not in existing_pairs}

        # Create a DataFrame with missing pairs
        df_missing = pd.DataFrame(missing_pairs, columns=["origin", "destination"])

        # Append missing pairs to the original DataFrame
        df_fs = pd.concat([df_fs, df_missing], ignore_index=True)

        return df_fs


    if flight_ban_def['ban_type'] == 'distance':
        # Remove all flights shorter or equal to the distance, easy!
        # Need to read the trains first
        # Need to filter the trains that go between airports (areas)

        # Keep only flights which could be impacted by the ban
        df_fs = df_fs[df_fs['gcdistance'] <= flight_ban_def['ban_length']]

        # Filter the flights which appear in the rails that are kept
        df_fs = get_od_flights_from_rail(df_fs, df_t)

    else:
        # Remove all flights which have a train faster han a given time...
        # Need to read the trains first
        # Need to filter the trains that go between airports (areas)
        # Then keep the ones which are faster than the threshold

        # Keep only the trains within the flight ban
        df_t['departure_time'] = pd.to_datetime(df_t['departure_time'])
        df_t['arrival_time'] = pd.to_datetime(df_t['arrival_time'])
        df_t['time'] = (df_t['arrival_time']-df_t['departure_time']).dt.total_seconds()/60
        df_t = df_t[df_t.time <= flight_ban_def['ban_length']]

        # Filter the flights which appear in the rails that are kept
        df_fs = get_od_flights_from_rail(df_fs, df_t)

    return df_fs[['origin', 'destination']]


def preprocess_air_layer(path_network, air_networks, processed_folder, pre_processed_version=0):
    df_fss = []
    for air_network in air_networks:
        df_fs = pd.read_csv(Path(path_network) / air_network['flight_schedules'], keep_default_na=False)
        df_fs.replace('', None, inplace=True)

        if 'gcdistance' not in df_fs.columns:
            # Read airport data (needed for the heuristic to compute GCD and
            # estimate time needed from current node to destination)
            df_as = pd.read_csv(Path(path_network) / air_network['airports_static'])

            df_fs = df_fs.merge(df_as[['icao_id', 'lat', 'lon']], left_on='origin', right_on='icao_id')
            df_fs = df_fs.merge(df_as[['icao_id', 'lat', 'lon']], left_on='destination', right_on='icao_id',
                                suffixes=('_orig', '_dest'))
            df_fs.drop(columns=['icao_id_orig', 'icao_id_dest'], inplace=True)

            dict_dist_origin_destination = create_dict_distance_origin_destination(df_fs[['origin', 'destination',
                                                                                          'lat_orig', 'lon_orig',
                                                                                          'lat_dest',
                                                                                          'lon_dest']].drop_duplicates())

            df_fs['gcdistance'] = df_fs.apply(lambda x: dict_dist_origin_destination[x['origin'], x['destination']],
                                              axis=1)

        if 'provider' not in df_fs.columns:
            df_fs['provider'] = None
        if 'alliance' not in df_fs.columns:
            df_fs['alliance'] = None
        if 'cost' not in df_fs.columns:
            df_fs['cost'] = None
        if 'seats' not in df_fs.columns:
            df_fs['seats'] = 180  # TODO hard coded number of seats
        if 'emissions' not in df_fs.columns:
            df_fs['emissions'] = None

        df_fs['alliance'] = df_fs['alliance'].fillna(df_fs['provider'])

        if 'alliances' in air_network.keys():
            # Replace alliance for the alliances defined in the alliances.csv file
            df_alliances = pd.read_csv(Path(path_network) / air_network['alliances'])
            df_fs = df_fs.merge(df_alliances, how='left', on='provider', suffixes=("", "_alliance"))
            df_fs.loc[~df_fs.alliance_alliance.isna(), 'alliance'] = df_fs.loc[~df_fs.alliance_alliance.isna()]['alliance_alliance']
            df_fs = df_fs.drop(['alliance_alliance'], axis=1)

        df_fss += [df_fs]

    df_fs = pd.concat(df_fss, ignore_index=True)

    df_fs = compute_cost_emissions_air(df_fs)

    fflights = 'flight_schedules_proc_' + str(pre_processed_version) + '.csv'
    df_fs.to_csv(Path(path_network) / processed_folder / fflights, index=False)


def pre_process_rail_layer(path_network, rail_networks, processed_folder, pre_processed_version=0):
    df_rss = []
    df_stop_times_alls = []
    df_stop_timess = []

    for rail_network in rail_networks:
        # TODO: filter by parent stations
        df_stop_times = pd.read_csv(Path(path_network) / rail_network['gtfs'] / 'stop_times.txt',
                                    dtype={'stop_id': str})

        date_rail = rail_network['date_rail']  # '20230503'

        if date_rail != 'None':
            # Filter rail trips that operate on that day

            df_trips = pd.read_csv(Path(path_network) / rail_network['gtfs'] / 'trips.txt')
            df_calendar = pd.read_csv(Path(path_network) / rail_network['gtfs'] / 'calendar.txt')
            df_calendar_dates = pd.read_csv(Path(path_network) / rail_network['gtfs'] / 'calendar_dates.txt')
            df_calendar.columns = list(df_calendar.columns.str.strip())
            df_calendar_dates.columns = list(df_calendar_dates.columns.str.strip())
            df_calendar['start_date'] = pd.to_datetime(df_calendar['start_date'], format='%Y%m%d')
            df_calendar['end_date'] = pd.to_datetime(df_calendar['end_date'], format='%Y%m%d')
            df_calendar_dates['date'] = pd.to_datetime(df_calendar_dates['date'], format='%Y%m%d')

            date_rail = pd.to_datetime(date_rail, format='%Y%m%d')

            df_stop_times = get_stop_times_on_date(date_rail, df_calendar, df_calendar_dates, df_trips, df_stop_times)

        df_agency = pd.read_csv(Path(path_network) / rail_network['gtfs'] / 'agency.txt')
        df_stops = pd.read_csv(Path(path_network) / rail_network['gtfs'] / 'stops.txt', dtype={'stop_id': str})

        date_to_set_rail = rail_network['date_to_set_rail']
        date_to_set_rail = pd.to_datetime(date_to_set_rail, format='%Y%m%d')

        # Note that country is set for both stops when in reality the trip could be accross countries...
        # TODO improve country identification of stops in rail, for now just got from the toml file
        country = rail_network['country']

        # TODO: cost, emissions...
        # TODO: agency linked with services
        rail_provider = df_agency.iloc[0].agency_name
        rail_alliance = df_agency.iloc[0].agency_name
        if 'provider' not in df_stop_times.columns:
            df_stop_times['provider'] = rail_provider
        if 'alliance' not in df_stop_times.columns:
            df_stop_times['alliance'] = rail_alliance

        df_stop_times['country'] = country
        df_stop_times_alls += [df_stop_times]

        # Filter GTFS to keep only rail stations considered
        if 'rail_stations_considered' in rail_network.keys():
            df_stops_considered = pd.read_csv(Path(path_network) / rail_network['rail_stations_considered'],
                                              dtype={'stop_id': str})
            df_stop_times = df_stop_times[df_stop_times.stop_id.isin(df_stops_considered.stop_id)].copy()

        df_stop_timess += [df_stop_times]

        # Keep processing to translate GTFS form to 'Services' form
        df_rs = pre_process_rail_gtfs_to_services(df_stop_times, date_to_set_rail, df_stops)

        df_rss += [df_rs]

    df_rs = pd.concat(df_rss, ignore_index=True)
    df_rs = compute_cost_emissions_rail(df_rs)

    # Save rail services
    frail = 'rail_timetable_proc_' + str(pre_processed_version) + '.csv'
    df_rs.to_csv((Path(path_network) / processed_folder) / frail, index=False)

    # Save information in GTFS form
    df_stop_times = pd.concat(df_stop_timess, ignore_index=True)
    frail = 'rail_timetable_proc_gtfs_' + str(pre_processed_version) + '.csv'
    df_stop_times.to_csv(Path(path_network) / processed_folder / frail, index=False)

    # Save GTFS with all rail stations for the day
    df_stop_times = pd.concat(df_stop_times_alls, ignore_index=True)
    frail = 'rail_timetable_all_gtfs_' + str(pre_processed_version) + '.csv'
    df_stop_times.to_csv(Path(path_network) / processed_folder / frail, index=False)


def pre_process_rail_gtfs_to_services(df_stop_times, date_rail, df_stops=None):
    n_stops_trip = df_stop_times.groupby('trip_id').count().reset_index()

    # Keep trips with at least two stops of interest in the trip
    df_stop_times = df_stop_times[
        df_stop_times.trip_id.isin(n_stops_trip[n_stops_trip.arrival_time > 1].trip_id)].copy()

    # Create rail services as all possible segments in the trips
    rail_services = []

    df_sorted = df_stop_times.sort_values(by=['trip_id', 'stop_sequence'])

    df_sorted['arrival_time'] = df_sorted['arrival_time'].apply(lambda x: add_date_and_handle_overflow(x, date_rail))
    df_sorted['departure_time'] = df_sorted['departure_time'].apply(lambda x: add_date_and_handle_overflow(x,
                                                                                                           date_rail))

    grouped = df_sorted.groupby('trip_id')
    rail_cost = None
    rail_seats = 295  # TODO hard coded number of seats now (average Spain)
    rail_emissions = None

    for trip_id, group_df in grouped:
        stop_ids = group_df['stop_id'].tolist()
        stop_sequences = group_df['stop_sequence'].tolist()
        for i, j in combinations(range(len(stop_ids)), 2):
            service_id = f"{trip_id}_{stop_sequences[i]}_{stop_sequences[j]}"
            origin = str(stop_ids[i])
            destination = str(stop_ids[j])
            departure_time = group_df.iloc[i]['departure_time']
            arrival_time = group_df.iloc[j]['arrival_time']
            rail_provider = group_df.iloc[i]['provider']
            rail_alliance = group_df.iloc[i]['alliance']
            country = None
            if 'country' in group_df.iloc[i]:
                country = group_df.iloc[i]['country']

            service_df_row = (
                service_id, str(origin), str(destination), departure_time, arrival_time, rail_provider, rail_alliance,
                rail_cost, rail_seats, rail_emissions, country)

            rail_services.append(service_df_row)

    columns_interest = ['service_id', 'origin', 'destination', 'departure_time', 'arrival_time',
                        'provider', 'alliance', 'cost', 'seats', 'emissions', 'country']

    rail_services_df = pd.DataFrame(rail_services, columns=columns_interest)

    if df_stops is not None:
        df_stops = df_stops.copy().rename({'stop_id': 'node', 'stop_lat': 'lat', 'stop_lon': 'lon'},
                                          axis=1)[['node', 'lat', 'lon']]
        df_stops['node'] = df_stops['node'].astype(str)

        rail_services_df = rail_services_df.merge(df_stops, left_on=['origin'], right_on=['node'], how='left')
        rail_services_df.rename(columns={'lat': 'lat_orig', 'lon': 'lon_orig'}, inplace=True)
        rail_services_df.drop(columns='node', inplace=True)
        rail_services_df = rail_services_df.merge(df_stops, left_on=['destination'], right_on=['node'], how='left')
        rail_services_df.rename(columns={'lat': 'lat_dest', 'lon': 'lon_dest'}, inplace=True)
        rail_services_df.drop(columns='node', inplace=True)

    if 'gcdistance' not in rail_services_df.columns:
        dict_dist_origin_destination = create_dict_distance_origin_destination(rail_services_df[['origin',
                                                                                                 'destination',
                                                                                                 'lat_orig',
                                                                                                 'lon_orig',
                                                                                                 'lat_dest',
                                                                                                 'lon_dest']].drop_duplicates())

        rail_services_df['gcdistance'] = rail_services_df.apply(lambda x: dict_dist_origin_destination[x['origin'],
        x['destination']], axis=1)

    return rail_services_df


def create_dict_distance_origin_destination(origin_destination_df):
    from libs.uow_tool_belt.general_tools import haversine

    dict_dist_origin_destination = {}

    for odr in origin_destination_df[['origin', 'destination', 'lat_orig', 'lon_orig',
                                     'lat_dest', 'lon_dest']].drop_duplicates().iterrows():
            lat_orig = odr[1].lat_orig
            lon_orig = odr[1].lon_orig
            lat_dest = odr[1].lat_dest
            lon_dest = odr[1].lon_dest
            dict_dist_origin_destination[(odr[1].origin, odr[1].destination)] = haversine(lon_orig, lat_orig,
                                                                                          lon_dest, lat_dest)
    return dict_dist_origin_destination


def create_network(path_network_dict, compute_simplified=False, allow_mixed_operators=True,
                   heuristics_precomputed=None,
                   pre_processed_version=0,
                   policy_package=None):

    if policy_package is None:
        policy_package = {}

    df_regions_access = None
    df_transitions = None
    layers = []
    network = None

    df_airport_processes = None
    df_rail_station_processes = None
    if 'processing_time' in path_network_dict.keys():
        # Read processing time at infrastructure if provided
        df_processing_times_airl = []
        df_processing_times_raill = []
        default_k2g = []
        default_g2k = []
        default_k2p = []
        default_p2k = []

        for pt in path_network_dict['processing_time']:
            # Read processing times airports and rails
            if pt.get('airport_processes') is not None:
                df_airport_processes_i = pd.read_csv(Path(path_network_dict['network_path']) /
                                                     pt['airport_processes'])

                if 'iata_icao_static' in pt:
                    df_iata_icao = pd.read_csv(Path(path_network_dict['network_path']) /
                                               pt['iata_icao_static'])

                    # Deal wit the fact that some airports might be in IATA code instead of ICAO
                    df_airport_processes_i['len_station'] = df_airport_processes_i['airport'].apply(lambda x: len(x))
                    if len(df_airport_processes_i[(df_airport_processes_i['len_station'] == 3)]) > 0:
                        df_airport_processes_i = df_airport_processes_i.merge(df_iata_icao[['IATA', 'ICAO']], how='left',
                                                                              left_on='airport', right_on='IATA')
                        df_airport_processes_i['ICAO'] = df_airport_processes_i['ICAO'].fillna(
                            df_airport_processes_i['airport'])
                        df_airport_processes_i['airport'] = df_airport_processes_i['ICAO']

                df_processing_times_airl += [df_airport_processes_i]

            # Read rail process times
            if pt.get('rail_stations_processes') is not None:
                df_rail_processes_i = pd.read_csv(Path(path_network_dict['network_path']) /
                                                  pt['rail_stations_processes'], dtype={"station": str})

                df_processing_times_raill += [df_rail_processes_i]

            # Read default if provided
            if pt.get('default_process_time_k2g') is not None:
                default_k2g += [pt.get('default_process_time_k2g')]
            if pt.get('default_process_time_g2k') is not None:
                default_g2k += [pt.get('default_process_time_g2k')]
            if pt.get('default_process_time_k2p') is not None:
                default_k2p += [pt.get('default_process_time_k2p')]
            if pt.get('default_process_time_p2k') is not None:
                default_p2k += [pt.get('default_process_time_p2k')]

        if len(df_processing_times_airl) > 0:
            df_airport_processes = pd.concat(df_processing_times_airl, ignore_index=True)
        if len(df_processing_times_raill) > 0:
            df_rail_station_processes = pd.concat(df_processing_times_raill, ignore_index=True)

        # Averages as more than one default could be provided in a list of processing times in the toml
        def compute_default_processing(default_l):
            if len(default_l)>0:
                return round(sum(default_l)/len(default_l), 2)
            else:
                return None
        default_k2g = compute_default_processing(default_k2g)
        default_k2p = compute_default_processing(default_k2p)
        default_g2k = compute_default_processing(default_g2k)
        default_p2k = compute_default_processing(default_p2k)

    if 'regions_access' in path_network_dict.keys():
        # Read regions access to infrastructure if provided
        df_regions_accessl = []
        for ra in path_network_dict['regions_access']:

            df_regions_acess_i = pd.read_csv(Path(path_network_dict['network_path']) /
                                        ra['regions_access'], dtype={"station": str})

            if 'total_time' not in df_regions_acess_i.columns:
                # We don't have a regions access with the total time computed
                # We should have had processing times
                if (((df_airport_processes is None) and ((default_k2g is None) or (default_g2k is None)))
                        or ((df_rail_station_processes is None) and ((default_k2p is None) or (default_p2k is None)))):
                    logger.error("Missing processing times (not defined in regions access)")
                    sys.exit(-1)

                # Reshaping the DataFrame
                df_regions_acess_i = pd.melt(
                    df_regions_acess_i,
                    id_vars=["region", "station", "layer", "pax_type"],
                    value_vars=["avg_d2i", "avg_i2d"],
                    var_name="access_type",
                    value_name="avg_time",
                )

                # Mapping 'avg_d2i' to 'access' and 'avg_i2d' to 'egress'
                df_regions_acess_i["access_type"] = df_regions_acess_i["access_type"].map({"avg_d2i": "access", "avg_i2d": "egress"})

                if 'iata_icao_static' in ra:
                    df_iata_icao = pd.read_csv(Path(path_network_dict['network_path']) /
                                            ra['iata_icao_static'])

                    # Deal wit the fact that some airports might be in IATA code instead of ICAO
                    df_regions_acess_i['len_station'] = df_regions_acess_i['station'].apply(lambda x: len(x))

                    if ((len(df_regions_acess_i[df_regions_acess_i['layer'] == 'air']) > 0) and
                            (len(df_regions_acess_i[(df_regions_acess_i['layer'] == 'air') &
                                                    (df_regions_acess_i['len_station'] == 3)]) > 0)):
                        df_regions_acess_i = df_regions_acess_i.merge(df_iata_icao[['IATA','ICAO']], how='left',
                                                                      left_on='station', right_on='IATA')
                        df_regions_acess_i['ICAO'] = df_regions_acess_i['ICAO'].fillna(df_regions_acess_i['station'])
                        df_regions_acess_i['station'] = df_regions_acess_i['ICAO']

                # Merge with airport processes
                if df_airport_processes is not None:
                    df_air_merged = df_regions_acess_i[df_regions_acess_i["layer"] == "air"].merge(
                        df_airport_processes,
                        left_on=["station", "pax_type"],
                        right_on=["airport", "pax_type"],
                        how="left"
                    )

                    # Add the processing time column for air
                    df_air_merged["processing_time"] = np.where(
                        df_air_merged["access_type"] == "access",
                        df_air_merged["k2g"],
                        df_air_merged["g2k"]
                    )

                    # Fill missing values
                    df_air_merged.loc[df_air_merged["access_type"] == "access", "processing_time"] = \
                        df_air_merged.loc[df_air_merged["access_type"] == "access", "processing_time"].fillna(
                            default_k2g)

                    df_air_merged.loc[df_air_merged["access_type"] == "egress", "processing_time"] = \
                        df_air_merged.loc[df_air_merged["access_type"] == "egress", "processing_time"].fillna(
                            default_g2k)

                else:
                    # We should have the defaults
                    df_air_merged = df_regions_acess_i[df_regions_acess_i["layer"] == "air"]
                    df_air_merged['processing_time'] = df_air_merged.apply(
                        lambda row: default_k2g if row["access_type"] == "access" else default_g2k,
                        axis=1
                    )

                if df_rail_station_processes is not None:
                    # Merge with rail processes
                    df_rail_merged = df_regions_acess_i[df_regions_acess_i["layer"] == "rail"].merge(
                        df_rail_station_processes,
                        left_on=["station", "pax_type"],
                        right_on=["station", "pax_type"],
                        how="left"
                    )

                    # Add the processing time column for rail
                    df_rail_merged["processing_time"] = np.where(
                        df_rail_merged["access_type"] == "access",
                        df_rail_merged["k2p"],
                        df_rail_merged["p2k"]
                    )

                    # Fill missing values
                    df_rail_merged.loc[df_rail_merged["access_type"] == "access", "processing_time"] = \
                        df_rail_merged.loc[df_rail_merged["access_type"] == "access", "processing_time"].fillna(
                            default_k2p)

                    df_rail_merged.loc[df_rail_merged["access_type"] == "egress", "processing_time"] = \
                        df_rail_merged.loc[df_rail_merged["access_type"] == "egress", "processing_time"].fillna(
                            default_p2k)


                else:
                    # We should have the defaults
                    df_rail_merged = df_regions_acess_i[df_regions_acess_i["layer"] == "rail"]
                    df_rail_merged['processing_time'] = df_rail_merged.apply(
                        lambda row: default_k2p if row["access_type"] == "access" else default_p2k,
                        axis=1
                    )

                # Combine the two DataFrames
                df_regions_acess_i = pd.concat([df_air_merged, df_rail_merged], ignore_index=True)

                # Drop unnecessary columns
                df_regions_acess_i = df_regions_acess_i[
                    ["region", "station", "layer", "access_type", "pax_type", "avg_time", "processing_time"]
                ]

                df_regions_acess_i['total_time'] = df_regions_acess_i['avg_time'] + df_regions_acess_i['processing_time']

            df_regions_accessl += [df_regions_acess_i]

        df_regions_access = pd.concat(df_regions_accessl, ignore_index=True)

    if 'air_network' in path_network_dict.keys():
        df_fsl = []
        df_asl = []
        df_mctl = []
        df_ra_airl = []
        mct_defaultl = []

        dict_dist_origin_destination = {}
        df_heuristic_air = None
        only_fastest = 0
        if compute_simplified:
            only_fastest = 1
        if compute_simplified and allow_mixed_operators:
            only_fastest = 2

        for an in path_network_dict['air_network']:
            # Read airport data (needed for the heuristic to compute GCD and
            # estimate time needed from current node to destination)
            df_as = pd.read_csv(Path(path_network_dict['network_path']) /
                                an['airports_static'])

            fschedule_filename = 'flight_schedules_proc_'+str(pre_processed_version)+'.csv'
            df_fs = pd.read_csv(Path(path_network_dict['network_path']) / path_network_dict['processed_folder'] /
                                fschedule_filename, keep_default_na=False)

            df_fs.replace('', None, inplace=True)

            df_fs['emissions'] = pd.to_numeric(df_fs['emissions'], errors='coerce')

            if 'lat_orig' not in df_fs.columns:
                df_fs = df_fs.merge(df_as[['icao_id', 'lat', 'lon']], left_on='origin', right_on='icao_id')
                df_fs = df_fs.merge(df_as[['icao_id', 'lat', 'lon']], left_on='destination', right_on='icao_id',
                                    suffixes=('_orig', '_dest'))

            if 'gcdistance' not in df_fs.columns:
                dict_dist_origin_destination = create_dict_distance_origin_destination(df_fs[['origin', 'destination',
                                                                                              'lat_orig', 'lon_orig',
                                                                                              'lat_dest', 'lon_dest']].drop_duplicates())

                df_fs['gcdistance'] = df_fs.apply(lambda x: dict_dist_origin_destination[x['origin'], x['destination']],
                                                  axis=1)

                # Save the df_fs with the GCD computed now
                df_fs.to_csv(Path(path_network_dict['network_path']) / path_network_dict['processed_folder'] /
                                fschedule_filename, index=False)
            else:
                for odr in df_fs[['origin', 'destination', 'gcdistance']].drop_duplicates().iterrows():
                    dict_dist_origin_destination[(odr[1].origin, odr[1].destination)] = odr[1].gcdistance

            df_fs['sobt'] = pd.to_datetime(df_fs['sobt'],  format='%Y-%m-%d %H:%M:%S')
            df_fs['sibt'] = pd.to_datetime(df_fs['sibt'],  format='%Y-%m-%d %H:%M:%S')

            # Give timezones to SOBT and SIBT (by default UTC)
            # TODO: we could change the default in the future based on input data)
            df_fs['sobt'] = df_fs['sobt'].dt.tz_localize('UTC')
            df_fs['sibt'] = df_fs['sibt'].dt.tz_localize('UTC')

            # Read MCTs between air services
            df_mct = pd.read_csv(Path(path_network_dict['network_path']) /
                                 an['mct_air'])

            if an.get('mct_default') is not None:
                mct_defaultl += [an.get('mct_default')]

            # Get regions access for air
            df_ra_air = None
            if df_regions_access is not None:
                df_ra_air = df_regions_access[df_regions_access['layer'] == 'air'].copy().reset_index(drop=True)
                if len(df_ra_air) == 0:
                    df_ra_air = None

            df_fsl += [df_fs]
            df_asl += [df_as]
            df_mctl += [df_mct]
            df_ra_airl += [df_ra_air]

        if heuristics_precomputed is not None:
            p_heuristic_air = Path(heuristics_precomputed['heuristics_precomputed_air'])
            if p_heuristic_air.exists():
                # We have the file for air time heuristics
                df_heuristic_air = pd.read_csv(p_heuristic_air)

        df_fs = pd.concat(df_fsl, ignore_index=True)
        df_as = pd.concat(df_asl, ignore_index=True)
        df_mct = pd.concat(df_mctl, ignore_index=True)
        df_ra_air = pd.concat(df_ra_airl, ignore_index=True)

        mct_default = None
        if len(mct_defaultl)>0:
            # If MCT provided for the different definitions of air create one MCT as
            # average of all provided.
            # TODO: Allow different MCTs per definition of rail layer
            mct_default = sum(mct_defaultl) / len(mct_defaultl)

        # Use first two letters of airport ICAO codes for country origin and destination
        df_fs['country_origin'] = df_fs['origin'].str[:2]
        df_fs['country_destination'] = df_fs['destination'].str[:2]

        air_layer = create_air_layer(df_fs.copy(), df_as, df_mct,
                                     mct_default=mct_default,
                                     df_ra_air=df_ra_air,
                                     keep_only_fastest_service=only_fastest,
                                     dict_dist_origin_destination=dict_dist_origin_destination,
                                     heuristic_precomputed_distance=df_heuristic_air)
        layers += [air_layer]

    if 'rail_network' in path_network_dict.keys():
        df_rail_data_l = []
        df_stopsl = []
        df_mctl = []
        mct_defaultl = []
        mct_default = None
        df_stops = None
        date_rail_str = None
        need_save = False
        for rn in path_network_dict['rail_network']:
            date_rail_str = rn['date_to_set_rail']
            date_rail = pd.to_datetime(date_rail_str, format='%Y%m%d')

            df_stops = None
            if rn.get('create_rail_layer_from') == 'gtfs':
                # Create the services file (regarless if it exists or not) and then process downstream as from services
                fstops_filename = 'rail_timetable_proc_gtfs_' + str(pre_processed_version) + '.csv'
                df_stop_times = pd.read_csv(Path(path_network_dict['network_path']) / path_network_dict['processed_folder'] /
                                           fstops_filename, keep_default_na=False, na_values=[''],
                                              dtype={'stop_id': str})

                df_stops = pd.read_csv(Path(path_network_dict['network_path']) / rn['gtfs'] / 'stops.txt', dtype={'stop_id': str})

                df_rail_data_l += [pre_process_rail_gtfs_to_services(df_stop_times, date_rail, df_stops)]
                need_save = True

            else:
                fstops_filename = 'rail_timetable_proc_' + str(pre_processed_version) + '.csv'

                df_rail_data = pd.read_csv(Path(path_network_dict['network_path']) / path_network_dict['processed_folder'] /
                                           fstops_filename, keep_default_na=False, na_values=[''],
                                              dtype={'origin': str, 'destination': str})

                df_rail_data = df_rail_data.apply(lambda col: col.map(lambda x: None if pd.isna(x) else x))

                df_rail_data['departure_time'] = pd.to_datetime(df_rail_data['departure_time'])
                df_rail_data['arrival_time'] = pd.to_datetime(df_rail_data['arrival_time'])
                # Adjust date to date of rail_str (to match flights datetime)
                most_frequent_arrival_date = df_rail_data['arrival_time'].dt.date.mode()[0]

                df_rail_data['departure_time'] = df_rail_data['departure_time'].apply(lambda dt:
                                                                                      dt.replace(year=date_rail.year,
                                                                                                 month=date_rail.month,
                                                                                                 day=date_rail.day) +
                                                                                      pd.Timedelta(days=(dt.date() -
                                                                                                         most_frequent_arrival_date).days))

                df_rail_data['arrival_time'] = df_rail_data['arrival_time'].apply(lambda dt:
                                                                                  dt.replace(year=date_rail.year,
                                                                                             month=date_rail.month,
                                                                                             day=date_rail.day) +
                                                                                  pd.Timedelta(days=(dt.date() -
                                                                                                     most_frequent_arrival_date).days))

                df_rail_data['origin'] = df_rail_data['origin'].apply(lambda x: str(x))
                df_rail_data['destination'] = df_rail_data['destination'].apply(lambda x: str(x))

                df_rail_data_l += [df_rail_data]

            if df_stops is not None:
                # Need the stops to have its coordinates
                df_stopsl += [df_stops] #pd.read_csv(Path(path_network_dict['network_path']) / rn['gtfs'] / 'stops.txt', dtype={'stop_id': str})]

            # Read MCTs between rail services
            if rn.get('mct_rail') is not None:
                df_mct = pd.read_csv(Path(path_network_dict['network_path']) /
                                      rn['mct_rail'], dtype={'stop_id': str})

                df_mctl += [df_mct]
            if rn.get('mct_default') is not None:
                mct_defaultl += [rn.get('mct_default')]

        df_rail_data = pd.concat(df_rail_data_l, ignore_index=True)
        df_rail_data = compute_cost_emissions_rail(df_rail_data)
        if len(df_stopsl) > 0:
            df_stops = pd.concat(df_stopsl, ignore_index=True)

        df_mct = None
        if len(df_mctl)>0:
            df_mct = pd.concat(df_mctl, ignore_index=True)

        if len(mct_defaultl)>0:
            # If MCT provided for the different definitions of rail create one MCT as
            # average of all provided.
            # TODO: Allow different MCTs per definition of rail layer
            mct_default = round(sum(mct_defaultl) / len(mct_defaultl))

        # Compute departure and arrival times in UTC times if departure and arrival times in UTC are missing
        # Vectorized processing for departure times
        if 'departure_time_utc' not in df_rail_data.columns:
            df_rail_data[['departure_time_utc', 'departure_time_utc_tz',
                          'departure_time_local', 'departure_time_local_tz']] = pd.DataFrame(
                Parallel(n_jobs=-1)(delayed(convert_to_utc_vectorized)(lon, lat, local_time)
                                    for lon, lat, local_time in
                                    zip(df_rail_data['lon_orig'], df_rail_data['lat_orig'], df_rail_data['departure_time']))
            )

            # Vectorized processing for arrival times
            df_rail_data[['arrival_time_utc', 'arrival_time_utc_tz',
                          'arrival_time_local', 'arrival_time_local_tz']] = pd.DataFrame(
                Parallel(n_jobs=-1)(delayed(convert_to_utc_vectorized)(lon, lat, local_time)
                                    for lon, lat, local_time in
                                    zip(df_rail_data['lon_dest'], df_rail_data['lat_dest'], df_rail_data['arrival_time']))
            )
        else:
            # The rail_timetable_proc_#.csv already had the times in UTC and local available
            df_rail_data['departure_time_utc'] = pd.to_datetime(df_rail_data['departure_time_utc'])
            df_rail_data['arrival_time_utc'] = pd.to_datetime(df_rail_data['arrival_time_utc'])
            df_rail_data['departure_time_local'] = pd.to_datetime(df_rail_data['departure_time_local'])
            df_rail_data['arrival_time_local'] = pd.to_datetime(df_rail_data['arrival_time_local'])

        # Move arrival and departure times to UTC
        df_rail_data['arrival_time'] = df_rail_data['arrival_time_utc']
        df_rail_data['departure_time'] = df_rail_data['departure_time_utc']

        if need_save:
            frail = 'rail_timetable_proc_' + str(pre_processed_version) + '.csv'
            df_rail_data.to_csv((Path(path_network_dict['network_path']) / path_network_dict['processed_folder']) / frail,
                         index=False)
        else:
            # We save it still for reference as this is the dataframe used downstream
            frail = 'rail_timetable_proc_' + str(pre_processed_version) + '_used_internally.csv'
            df_rail_data.to_csv(
                (Path(path_network_dict['network_path']) / path_network_dict['processed_folder']) / frail,
                index=False)

        # Get regions access for rail
        df_ra_rail = None
        if df_regions_access is not None:
            df_ra_rail = df_regions_access[df_regions_access['layer'] == 'rail'].copy().reset_index(drop=True)
            if len(df_ra_rail) == 0:
                df_ra_rail = None

        df_heuristic_rail = None
        if heuristics_precomputed is not None:
            p_heuristic_rail = Path(heuristics_precomputed['heuristics_precomputed_rail'])
            if p_heuristic_rail.exists():
                # We have the file for rail time heuristics
                df_heuristic_rail = pd.read_csv(p_heuristic_rail)

        only_fastest = 0
        if compute_simplified:
            only_fastest = 1
        if compute_simplified and allow_mixed_operators:
            only_fastest = 2

        rail_layer = create_rail_layer(df_rail_data, from_gtfs=False, date_considered=date_rail_str,
                                       df_ra_rail=df_ra_rail,
                                       keep_only_fastest_service=only_fastest,
                                       df_stops=df_stops,
                                       df_mct=df_mct,
                                       mct_default=mct_default,
                                       heuristic_precomputed_distance=df_heuristic_rail)

        layers += [rail_layer]

    if 'multimodal' in path_network_dict.keys():
        df_transitionsl = []
        for mm in path_network_dict['multimodal']:
            df_transitionsl += [pd.read_csv(Path(path_network_dict['network_path']) /
                                            mm['air_rail_transitions'], dtype={'origin_station': str,
                                                                               'destination_station': str})]

        df_transitions = pd.concat(df_transitionsl, ignore_index=True)

        df_transitions['extra_avg_travel_a_b'] = 0
        df_transitions['extra_avg_travel_b_a'] = 0

        if policy_package.get('integrated_ticketing') is not None:
            extra_rail_air = policy_package['integrated_ticketing'].get('rail_air_processing_extra', 0)
            extra_air_rail = policy_package['integrated_ticketing'].get('air_rail_processing_extra', 0)
            # If transferring between two airports
            extra_air_air = policy_package['integrated_ticketing'].get('air_air_processing_extra', 0)
            # If transferring between two train stations
            extra_rail_rail = policy_package['integrated_ticketing'].get('air_air_processing_extra', 0)

            # Use np.where for efficient conditional assignment
            df_transitions['extra_avg_travel_a_b'] = np.where(
                (df_transitions['layer_origin'] == 'air') & (df_transitions['layer_destination'] == 'rail'),
                extra_air_rail,
                np.where(
                    (df_transitions['layer_origin'] == 'rail') & (df_transitions['layer_destination'] == 'air'),
                    extra_rail_air,
                    np.where(
                        (df_transitions['layer_origin'] == 'rail') & (df_transitions['layer_destination'] == 'rail'),
                        extra_rail_rail,
                        np.where(
                            (df_transitions['layer_origin'] == 'air') & (df_transitions['layer_destination'] == 'air'),
                            extra_air_air,
                            0
                        )
                    )
                )
            )

            df_transitions['extra_avg_travel_b_a'] = np.where(
                (df_transitions['layer_origin'] == 'rail') & (df_transitions['layer_destination'] == 'air'),
                extra_air_rail,
                np.where(
                    (df_transitions['layer_origin'] == 'air') & (df_transitions['layer_destination'] == 'rail'),
                    extra_rail_air,
                    np.where(
                        (df_transitions['layer_origin'] == 'rail') & (df_transitions['layer_destination'] == 'rail'),
                        extra_rail_rail,
                        np.where(
                            (df_transitions['layer_origin'] == 'air') & (df_transitions['layer_destination'] == 'air'),
                            extra_air_air,
                            0
                        )
                    )
                )
            )

        if 'mct' not in df_transitions.columns:
            # If MCT is in the df_transitions then the format already has the value
            # if not compute it by adding g2k/p2k and k2g/k2p to the travel time between
            # stations

            # We should have had processing times
            if (((df_airport_processes is None) and ((default_k2g is None) or (default_g2k is None)))
                    or ((df_rail_station_processes is None) and ((default_k2p is None) or (default_p2k is None)))):
                logger.error("Missing processing times (not defined in multimodal definition)")
                sys.exit(-1)

            # Create two new DataFrames for each direction
            df_a_b = df_transitions[["origin_station", "destination_station", "layer_origin", "layer_destination",
                         "avg_travel_a_b", "extra_avg_travel_a_b"]].rename(
                columns={"avg_travel_a_b": "avg_travel_time", "extra_avg_travel_a_b": "extra_avg_travel_time"}
            )

            df_b_a = df_transitions[["destination_station", "origin_station", "layer_destination", "layer_origin",
                         "avg_travel_b_a", "extra_avg_travel_b_a"]].rename(
                columns={"destination_station": "origin_station", "origin_station": "destination_station",
                         "layer_destination": "layer_origin", "layer_origin": "layer_destination",
                         "avg_travel_b_a": "avg_travel_time", "extra_avg_travel_b_a": "extra_avg_travel_time"}
            )

            # Combine the two DataFrames
            df_transitions = pd.concat([df_a_b, df_b_a], ignore_index=True)

            # Ensure _multimodal columns exist in airport processes
            if "k2g_multimodal" not in df_airport_processes.columns:
                df_airport_processes["k2g_multimodal"] = df_airport_processes["k2g"]
            if "g2k_multimodal" not in df_airport_processes.columns:
                df_airport_processes["g2k_multimodal"] = df_airport_processes["g2k"]

            # Ensure _multimodal columns exist in rail station processes
            if "k2p_multimodal" not in df_rail_station_processes.columns:
                df_rail_station_processes["k2p_multimodal"] = df_rail_station_processes["k2p"]
            if "p2k_multimodal" not in df_rail_station_processes.columns:
                df_rail_station_processes["p2k_multimodal"] = df_rail_station_processes["p2k"]

            # Merge for origin_station (x2k logic)
            df_transitions = (df_transitions.merge(
                df_airport_processes[["airport", "pax_type", "g2k", "g2k_multimodal"]],
                left_on="origin_station",
                right_on="airport",
                how="left"
            ).merge(
                df_rail_station_processes[["station", "pax_type", "p2k", "p2k_multimodal"]],
                left_on="origin_station",
                right_on="station",
                how="left"
            ).drop(columns=["airport", "station"]))

            df_transitions['pax_type'] = df_transitions.apply(
                lambda row: row['pax_type_x'] if pd.notna(row['pax_type_x']) else row['pax_type_y'],
                axis=1
            )

            df_transitions = df_transitions.drop(columns=['pax_type_x', 'pax_type_y'])

            if default_g2k is not None:
                df_transitions.loc[((df_transitions.layer_origin == "air") &
                                (df_transitions.g2k_multimodal.isna())), 'pax_type'] = 'all'
                df_transitions.loc[((df_transitions.layer_origin == "air") &
                                (df_transitions.g2k_multimodal.isna())), 'g2k_multimodal'] = default_g2k

            if default_p2k is not None:
                df_transitions.loc[((df_transitions.layer_origin == "rail") &
                                    (df_transitions.g2k_multimodal.isna())), 'pax_type'] = 'all'
                df_transitions.loc[((df_transitions.layer_origin == "rail") &
                                    (df_transitions.g2k_multimodal.isna())), 'p2k_multimodal'] = default_p2k


            # Determine x2k based on layer_origin
            df_transitions["x2k"] = df_transitions.apply(
                lambda row: row["g2k_multimodal"] if row["layer_origin"] == "air" else row["p2k_multimodal"],
                axis=1
            )

            # Merge for destination_station (k2x logic)
            df_transitions = df_transitions.merge(
                df_airport_processes[["airport", "pax_type", "k2g", "k2g_multimodal"]],
                left_on=("destination_station", "pax_type"),
                right_on=("airport", "pax_type"),
                how="left"
            ).merge(
                df_rail_station_processes[["station", "pax_type", "k2p", "k2p_multimodal"]],
                left_on=("destination_station", "pax_type"),
                right_on=("station", "pax_type"),
                how="left"
            ).drop(columns=["airport", "station"])

            if default_k2g is not None:
                df_transitions.loc[((df_transitions.layer_destination == "air") &
                                    (df_transitions.k2g_multimodal.isna())), 'k2g_multimodal'] = default_k2g

            if default_k2p is not None:
                df_transitions.loc[((df_transitions.layer_destination == "rail") &
                                    (df_transitions.k2g_multimodal.isna())), 'k2p_multimodal'] = default_k2p


            # Determine k2x based on layer_destination
            df_transitions["k2x"] = df_transitions.apply(
                lambda row: row["k2g_multimodal"] if row["layer_destination"] == "air" else row["k2p_multimodal"],
                axis=1
            )

            # Drop intermediate columns used for merging
            df_transitions = df_transitions.drop(columns=["g2k", "g2k_multimodal", "p2k", "p2k_multimodal",
                                                          "k2g", "k2g_multimodal", "k2p", "k2p_multimodal"])

            # Compute MCT to transition between layers
            df_transitions['mct'] = (df_transitions['x2k'] + df_transitions['avg_travel_time'] +
                                     df_transitions['extra_avg_travel_time'] + df_transitions['k2x'])

            df_transitions.drop(columns=['x2k', 'k2x'], inplace=True)

        df_transitions.rename(columns={'origin_station': 'origin', 'destination_station': 'destination',
                                       'layer_origin': 'layer_id_origin', 'layer_destination': 'layer_id_destination'},
                              inplace=True)

        # Save transitions mct computed
        #(Path(path_network) / processed_folder / fflights

        df_transitions.to_csv((Path(path_network_dict['network_path']) / path_network_dict['processed_folder'] /
                              'transition_layer_connecting_times.csv' ), index=False)

        if 'pax_type' not in df_transitions:
            df_transitions['mct'] = df_transitions['mct'].apply(lambda x: {'all': x})
            df_transitions['avg_travel_time'] = df_transitions['avg_travel_time'].apply(lambda x: {'all': x})
            df_transitions['extra_avg_travel_time'] = df_transitions['extra_avg_travel_time'].apply(lambda x: {'all': x})
        else:
            # Group by origin, destination, layer_id_origin, and layer_id_destination, and aggregate mct into a dictionary by pax_type
            df_transitions1 = df_transitions.groupby(
                ['origin', 'destination', 'layer_id_origin', 'layer_id_destination']
            ).apply(
                lambda group: group.set_index('pax_type')['mct'].to_dict()
            ).reset_index(name='mct')
            df_transitions2 = df_transitions.groupby(
                ['origin', 'destination', 'layer_id_origin', 'layer_id_destination']
            ).apply(
                lambda group: group.set_index('pax_type')['avg_travel_time'].to_dict()
            ).reset_index(name='avg_travel_time')
            df_transitions3 = df_transitions.groupby(
                ['origin', 'destination', 'layer_id_origin', 'layer_id_destination']
            ).apply(
                lambda group: group.set_index('pax_type')['extra_avg_travel_time'].to_dict()
            ).reset_index(name='extra_avg_travel_time')
            df_transitions = pd.concat([df_transitions1,df_transitions2[['avg_travel_time']],df_transitions3[['extra_avg_travel_time']]],axis=1)

    if len(layers) > 0:
        if len(layers) == 1:
            df_transitions_used = None
        else:
            df_transitions_used = df_transitions
        network = Network(layers=layers, transition_btw_layers=df_transitions_used)

    return network


def compute_itineraries(od_itineraries, network, dict_o_d_routes=None, n_itineraries=10,
                        max_connections=2, allow_mixed_operators=False, consider_times_constraints=True):
    dict_itineraries = {}
    start_time = time.time()

    # default_od_routes sos that if dict_o_d_routes is provided then it will return {} instead of None
    if dict_o_d_routes is None:
        dict_o_d_routes = {}
        default_od_routes = None
    else:
        default_od_routes = {}

    n_explored_total = 0
    for i, od in od_itineraries.iterrows():
        start_time_od = time.time()
        same_operators = not allow_mixed_operators
        logger.info("Computing it for: "+od.origin+" "+od.destination)
        itineraries, n_explored = network.find_itineraries(origin=od.origin, destination=od.destination,
                                                           routes=dict_o_d_routes.get((od.origin, od.destination),
                                                                                      default_od_routes),
                                                           nitineraries=n_itineraries,
                                                           max_connections=max_connections,
                                                           consider_operators_connections=same_operators,
                                                           consider_times_constraints=consider_times_constraints)
        dict_itineraries[(od.origin, od.destination)] = itineraries
        n_explored_total += n_explored
        end_time_od = time.time()
        logger.info("Itineraries for "+od.origin+"-"+od.destination+
                    ", computed in, "+str(end_time_od - start_time_od)+" seconds, exploring: "+str(n_explored)+
                    "nodes. Found "+str(len(itineraries))+" itineraries.")

    end_time = time.time()
    elapsed_time = end_time - start_time

    logger.info("Itineraries computed in: "+str(elapsed_time)+" seconds, exploring: "+str(n_explored_total))

    return dict_itineraries


def process_dict_itineraries(dict_itineraries, consider_times_constraints=True, allow_mixed_operators=False,
                             policy_package=None):

    if policy_package is None:
        policy_package = {}

    if 'tax_charges' in policy_package.keys():
        dict_tax_co2 = policy_package['tax_charges']
    else:
        dict_tax_co2 = {}

    options = []
    origins = []
    destinations = []
    total_travel_time = []
    access = []
    egress = []
    d2i = []
    i2d = []
    nservices = []
    n_modes_i = []
    journey_type_i = []
    paths_i = []
    total_waiting_i = []
    total_cost_i = []
    total_emissions_p = []

    n_legs = 0
    for di in dict_itineraries.values():
        for i in di:
            if n_legs < len(i.itinerary):
                n_legs = len(i.itinerary)

    dict_legs_info = {}
    for ln in range(n_legs):
        dict_legs_info['service_id_' + str(ln)] = []
        dict_legs_info['origin_' + str(ln)] = []
        dict_legs_info['destination_' + str(ln)] = []
        dict_legs_info['provider_' + str(ln)] = []
        dict_legs_info['alliance_' + str(ln)] = []
        dict_legs_info['mode_' + str(ln)] = []
        dict_legs_info['departure_time_' + str(ln)] = []
        dict_legs_info['arrival_time_' + str(ln)] = []
        dict_legs_info['travel_time_' + str(ln)] = []
        if ln > 0:
            dict_legs_info['mct_time_' + str(ln - 1) + "_" + str(ln)] = []
            dict_legs_info['ground_mobility_time_' + str(ln - 1) + "_" + str(ln)] = []
            dict_legs_info['connecting_time_' + str(ln - 1) + "_" + str(ln)] = []
            dict_legs_info['waiting_time_' + str(ln - 1) + "_" + str(ln)] = []

        dict_legs_info['cost_' + str(ln)] = []
        dict_legs_info['service_cost_' + str(ln)] = []
        dict_legs_info['emissions_cost_' + str(ln)] = []
        dict_legs_info['emissions_' + str(ln)] = []

        ln += 1

    for od in dict_itineraries.keys():
        origin = od[0]
        destination = od[1]
        option = 0
        for i in dict_itineraries[od]:
            origins.append(origin)
            destinations.append(destination)
            options.append(option)
            total_travel_time.append(i.total_travel_time.total_seconds() / 60)
            access.append(i.access_time.total_seconds() / 60)
            egress.append(i.egress_time.total_seconds() / 60)
            d2i.append(i.d2i_time.total_seconds() / 60)
            i2d.append(i.i2d_time.total_seconds() / 60)
            nservices.append(len(i.itinerary))
            ln = 0
            prev_arrival_time = None
            prev_mode = None
            total_waiting = None
            total_cost = None
            total_emissions = None
            path = None
            n_modes = 0
            journey_type = None
            if len(i.itinerary) == 0:
                # We don't use any mode of transport
                # if len(i.itinerary) == 0:
                #    # We have arrived to the destination but there's no service used, i.e. itinerary = []
                #    i.itinerary = [i.current_node]
                journey_type = 'none'
                dict_legs_info['service_id_' + str(ln)].append(None)
                dict_legs_info['origin_' + str(ln)].append(i.current_node)
                dict_legs_info['destination_' + str(ln)].append(i.current_node)
                dict_legs_info['provider_' + str(ln)].append(None)
                dict_legs_info['alliance_' + str(ln)].append(None)
                dict_legs_info['mode_' + str(ln)].append(i.layers_used[ln])
                dict_legs_info['departure_time_' + str(ln)].append(None)
                dict_legs_info['arrival_time_' + str(ln)].append(None)
                dict_legs_info['travel_time_' + str(ln)].append(None)
                dict_legs_info['cost_' + str(ln)].append(None)
                dict_legs_info['service_cost_' + str(ln)].append(None)
                dict_legs_info['emissions_cost_' + str(ln)].append(None)
                dict_legs_info['emissions_' + str(ln)].append(None)
                path = [i.current_node]
                ln += 1

            for s in i.itinerary:
                if consider_times_constraints:
                    dict_legs_info['service_id_' + str(ln)].append(s.id)
                else:
                    dict_legs_info['service_id_' + str(ln)].append(None)
                dict_legs_info['origin_' + str(ln)].append(s.origin)
                dict_legs_info['destination_' + str(ln)].append(s.destination)
                if consider_times_constraints or not allow_mixed_operators:
                    dict_legs_info['provider_' + str(ln)].append(s.provider)
                    dict_legs_info['alliance_' + str(ln)].append(s.alliance)
                else:
                    dict_legs_info['provider_' + str(ln)].append(None)
                    dict_legs_info['alliance_' + str(ln)].append(None)

                if i.layers_used[ln] != prev_mode:
                    prev_mode = i.layers_used[ln]
                    n_modes += 1
                    if n_modes > 1:
                        journey_type = 'multimodal'
                    else:
                        journey_type = i.layers_used[ln]
                dict_legs_info['mode_' + str(ln)].append(i.layers_used[ln])
                if consider_times_constraints:
                    dict_legs_info['departure_time_' + str(ln)].append(s.departure_time)
                    dict_legs_info['arrival_time_' + str(ln)].append(s.arrival_time)
                else:
                    dict_legs_info['departure_time_' + str(ln)].append(None)
                    dict_legs_info['arrival_time_' + str(ln)].append(None)
                dict_legs_info['travel_time_' + str(ln)].append(s.duration.total_seconds() / 60)
                if path is None:
                    path = [s.origin, s.destination]
                else:
                    path += [s.origin, s.destination]

                if ln > 0:
                    dict_legs_info['mct_time_' + str(ln - 1) + "_" + str(ln)].append(
                        i.mcts[ln - 1].total_seconds() / 60)
                    dict_legs_info['ground_mobility_time_' + str(ln - 1) + "_" + str(ln)].append(
                        i.ground_mobility[ln - 1].total_seconds() / 60)

                    if not consider_times_constraints:
                        dict_legs_info['connecting_time_' + str(ln - 1) + "_" + str(ln)].append(None)
                        dict_legs_info['waiting_time_' + str(ln - 1) + "_" + str(ln)].append(None)
                        total_waiting = None
                    else:
                        dict_legs_info['connecting_time_' + str(ln - 1) + "_" + str(ln)].append(
                            (s.departure_time - prev_arrival_time).total_seconds() / 60)
                        dict_legs_info['waiting_time_' + str(ln - 1) + "_" + str(ln)].append(
                            (s.departure_time - prev_arrival_time).total_seconds() / 60 -
                            i.mcts[ln - 1].total_seconds() / 60)
                        if total_waiting is None:
                            total_waiting = (s.departure_time - prev_arrival_time).total_seconds() / 60 - i.mcts[
                                ln - 1].total_seconds() / 60
                        else:
                            total_waiting += (s.departure_time - prev_arrival_time).total_seconds() / 60 - i.mcts[
                                ln - 1].total_seconds() / 60

                dict_legs_info['service_cost_' + str(ln)].append(s.cost)
                dict_legs_info['emissions_' + str(ln)].append(s.emissions)
                if (s.emissions is None) or (s.mode not in dict_tax_co2.keys()):
                    cost_emissions = 0
                else:
                    cost_emissions = dict_tax_co2[s.mode]['co2_cost'] * s.emissions

                dict_legs_info['emissions_cost_' + str(ln)].append(cost_emissions)

                total_cost_service = s.cost + cost_emissions

                dict_legs_info['cost_' + str(ln)].append(total_cost_service)

                if total_cost is None:
                    total_cost = total_cost_service
                else:
                    total_cost += total_cost_service
                if s.emissions is not None:
                    if total_emissions is None:
                        total_emissions = float(s.emissions)
                    else:
                        total_emissions += float(s.emissions)

                prev_arrival_time = s.arrival_time

                ln += 1

            for lni in range(ln, n_legs):
                dict_legs_info['service_id_' + str(lni)].append(None)
                dict_legs_info['origin_' + str(lni)].append(None)
                dict_legs_info['destination_' + str(lni)].append(None)
                dict_legs_info['provider_' + str(lni)].append(None)
                dict_legs_info['alliance_' + str(lni)].append(None)
                dict_legs_info['mode_' + str(lni)].append(None)
                dict_legs_info['departure_time_' + str(lni)].append(None)
                dict_legs_info['arrival_time_' + str(lni)].append(None)
                dict_legs_info['travel_time_' + str(lni)].append(None)
                if lni > 0:
                    dict_legs_info['mct_time_' + str(lni - 1) + "_" + str(lni)].append(None)
                    dict_legs_info['ground_mobility_time_' + str(lni - 1) + "_" + str(lni)].append(None)
                    dict_legs_info['connecting_time_' + str(lni - 1) + "_" + str(lni)].append(None)
                    dict_legs_info['waiting_time_' + str(lni - 1) + "_" + str(lni)].append(None)
                dict_legs_info['cost_' + str(lni)].append(None)
                dict_legs_info['service_cost_' + str(lni)].append(None)
                dict_legs_info['emissions_cost_' + str(lni)].append(None)
                dict_legs_info['emissions_' + str(lni)].append(None)

            n_modes_i.append(n_modes)
            journey_type_i.append(journey_type)
            total_waiting_i.append(total_waiting)
            if path is not None:
                paths_i.append([v for i, v in enumerate(path) if i == 0 or v != path[i-1]]) # remove consecutive same node
            else:
                paths_i.append(None)
            total_cost_i.append(total_cost)
            total_emissions_p.append(total_emissions)

            option += 1

    dict_it = {'origin': origins,
               'destination': destinations,
               'option': options,
               'nservices': nservices,
               'path': paths_i,
               'total_travel_time': total_travel_time,
               'total_cost': total_cost_i,
               'total_emissions': total_emissions_p,
               'total_waiting_time': total_waiting_i,
               'nmodes': n_modes_i,
               'journey_type': journey_type_i,
               'access_time': access,
               'egress_time': egress,
               'd2i_time': d2i,
               'i2d_time': i2d,
               }

    dict_elements = {**dict_it, **dict_legs_info}

    df = pd.DataFrame(dict_elements)

    columns_to_keep_regardless_none = ['total_cost', 'total_emissions', 'total_waiting_time']
    columns_to_keep_or_not_all_none = [col for col in df.columns if col in columns_to_keep_regardless_none
                                       or df[col].notna().any()]

    return df[columns_to_keep_or_not_all_none]


def compute_possible_itineraries_network(network, o_d, dict_o_d_routes=None, pc=1, n_itineraries=10,
                                         max_connections=2, allow_mixed_operators=False,
                                         consider_times_constraints=True, policy_package=None):

    if policy_package is None:
        policy_package = {}

    start_time_itineraries = time.time()
    if pc == 1:
        dict_itinearies = compute_itineraries(o_d,
                                              network,
                                              dict_o_d_routes,
                                              n_itineraries=n_itineraries,
                                              max_connections=max_connections,
                                              allow_mixed_operators=allow_mixed_operators,
                                              consider_times_constraints=consider_times_constraints)
    else:
        # Parallel computation of itineraries between o-d pairs
        prev_i = 0
        n_od_per_section = max(1, round(len(o_d) / pc))
        if n_od_per_section == 1:
            pc = len(o_d)
        i = n_od_per_section

        itineraries_computation_param = []
        for nr in range(pc):
            if nr == pc - 1:
                i = len(o_d)

            d = o_d.iloc[prev_i:i].copy().reset_index(drop=True)

            if nr == 0:
                itineraries_computation_param = [[d, network, dict_o_d_routes, n_itineraries, max_connections,
                                                  allow_mixed_operators, consider_times_constraints]]
            else:
                if len(d) > 0:
                    itineraries_computation_param.append([d, network, dict_o_d_routes, n_itineraries, max_connections,
                                                          allow_mixed_operators, consider_times_constraints])

            prev_i = i
            i = i + n_od_per_section

        pool = mp.Pool(processes=min(pc, len(itineraries_computation_param)))

        logger.important_info("   Launching parallel o-d itinearies finding")

        res = pool.starmap(compute_itineraries, itineraries_computation_param)

        pool.close()
        pool.join()

        dict_itinearies = {}
        for dictionary in res:
            dict_itinearies.update(dictionary)

    df_itineraries = process_dict_itineraries(dict_itinearies, consider_times_constraints=consider_times_constraints,
                                              allow_mixed_operators=allow_mixed_operators, policy_package=policy_package)
    end_time_itineraries = time.time()
    logger.important_info("In total "+str(len(df_itineraries))+" itineraries computed in, "
                          +str(end_time_itineraries - start_time_itineraries)+" seconds.")

    return df_itineraries


def compute_avg_paths_from_itineraries(df_itineraries):
    df_paths = df_itineraries.copy()

    df_paths['path'] = df_paths['path'].apply(str)

    df_paths['earliest_departure_time'] = df_paths.groupby(['origin', 'destination', 'path'])['departure_time_0'].transform('min')
    arrival_cols = [col for col in df_paths.columns if col.startswith('arrival_time_')]
    df_paths['latest_arrival_time'] = df_paths[arrival_cols].max(axis=1)

    prefixes = ['service_id', 'provider', 'alliance', 'departure_time', 'arrival_time']
    columns_drop = [col for col in df_paths.columns if any(col.startswith(prefix) for prefix in prefixes)]
    df_paths = df_paths.drop(columns=columns_drop)

    exclude_columns = [col for col in df_paths.columns if col.startswith('origin') or col.startswith('destination')
                       or col.startswith('mode') or col.startswith('earliest_departure_time') or
                       col.startswith('latest_arrival_time') or col.startswith('path')]

    for col in df_paths.columns:
        if col not in exclude_columns:
            df_paths[col] = pd.to_numeric(df_paths[col], errors='coerce')

    df_paths['path'] = df_paths['path'].apply(str)
    df_paths_avg = df_paths.groupby(['origin', 'destination', 'path'], as_index=False).agg(lambda x: x.mean() if x.name not in exclude_columns else x.iloc[0])

    df_paths_avg = df_paths_avg.groupby(['origin', 'destination']).apply(lambda x: x.sort_values(by='total_travel_time',
                                                                                                 ascending=True)).reset_index(drop=True)

    df_paths_avg['option'] = df_paths_avg.groupby(['origin', 'destination']).cumcount()

    df_paths_avg.insert(0, 'path_id', range(len(df_paths_avg)))

    grouped_itineraries = df_paths.groupby(['origin', 'destination', 'path']).size().reset_index(name='n_itineraries')

    df_paths_avg = df_paths_avg.merge(grouped_itineraries, on=['origin', 'destination', 'path'])

    # Move n_itineraries to be the second column
    cols = df_paths_avg.columns.tolist()
    cols.insert(1, cols.pop(cols.index('n_itineraries')))
    df_paths_avg = df_paths_avg[cols]

    cols = df_paths_avg.columns.tolist()
    cols.insert(12, cols.pop(cols.index('earliest_departure_time')))
    df_paths_avg = df_paths_avg[cols]

    cols = df_paths_avg.columns.tolist()
    cols.insert(13, cols.pop(cols.index('latest_arrival_time')))
    df_paths_avg = df_paths_avg[cols]

    def update_column_name(column_name):
        if column_name.startswith('total_'):
            return 'total_avg_' + column_name.split('total_')[1]
        elif column_name.startswith('travel_time_'):
            return 'travel_avg_time_' + column_name.split('travel_time_')[1]
        elif column_name.startswith('access_'):
            return 'access_avg_' + column_name.split('access_')[1]
        elif column_name.startswith('egress_'):
            return 'egress_avg_' + column_name.split('egress_')[1]
        elif column_name.startswith('mct_'):
            return 'mct_avg_' + column_name.split('mct_')[1]
        elif column_name.startswith('connecting_'):
            return 'connecting_avg_' + column_name.split('connecting_')[1]
        elif column_name.startswith('waiting_'):
            return 'waiting_avg_' + column_name.split('waiting_')[1]
        elif column_name.startswith('cost_'):
            return 'cost_avg_' + column_name.split('cost_')[1]
        else:
            return column_name

    # Apply the function to update column names
    df_paths_avg = df_paths_avg.rename(columns={col: update_column_name(col) for col in df_paths_avg.columns})

    # Find the fastest and slowest options
    group_cols = ['origin', 'destination', 'path']
    min_cols = ['total_travel_time', 'total_cost', 'total_emissions', 'total_waiting_time'] + [f'travel_time_{i}' for i
                                                                                               in range(10)] + [
                   f'cost_{i}' for i in range(10)] + [f'connecting_time_{i}_{i + 1}' for i in range(10)] + [
                   f'waiting_time_{i}_{i + 1}' for i in range(10)]
    min_cols = [col for col in min_cols if col in df_itineraries.columns]

    df_paths_time = df_itineraries.copy()
    df_paths_time['path'] = df_paths_time['path'].apply(str)

    df_min = df_paths_time.loc[df_paths_time.groupby(group_cols)['total_travel_time'].idxmin(), group_cols + min_cols]
    df_max = df_paths_time.loc[df_paths_time.groupby(group_cols)['total_travel_time'].idxmax(), group_cols + min_cols]

    df_min = df_min.rename(columns={col: col + '_min' for col in min_cols})

    df_max = df_max.rename(columns={col: col + '_max' for col in min_cols})

    df_paths_avg = df_paths_avg.merge(df_min, on=group_cols, how='left').merge(df_max, on=group_cols, how='left')

    cols = df_paths_avg.columns.tolist()
    cols.insert(8, cols.pop(cols.index('total_travel_time_min')))
    df_paths_avg = df_paths_avg[cols]

    cols = df_paths_avg.columns.tolist()
    cols.insert(9, cols.pop(cols.index('total_travel_time_max')))
    df_paths_avg = df_paths_avg[cols]

    # Count number of alternative_id (clusters) that are different per o-d-path triad.
    if 'alternative_id' in df_itineraries.columns:
        df_paths_alternatives = df_itineraries.copy()
        df_paths_alternatives['path'] = df_paths_alternatives['path'].apply(str)
        df_paths_alternatives = df_paths_alternatives.groupby(['origin', 'destination', 'path'])['alternative_id'].\
            nunique().reset_index().rename(columns={'alternative_id': 'n_alternative_id'})
        df_paths_avg = df_paths_avg.merge(df_paths_alternatives[['origin', 'destination', 'path', 'n_alternative_id']],
                                          on=['origin', 'destination', 'path'], how='left')
    else:
        df_paths_avg['n_alternative_id'] = None

    # Reorder n_alternative_id so that it's the third in the dataframe
    col = df_paths_avg.pop('n_alternative_id')
    df_paths_avg.insert(2, 'n_alternative_id', col)

    df_paths_avg['path'] = df_paths_avg['path'].apply(eval)

    return df_paths_avg


def filter_similar_options(group, kpis, thresholds=None):
    filtered_options = []
    clusters = {}
    # Remove the dropna as we want to keep options even if some KPIs are missing. They'll be replaced
    # by 0, which maybe it's not great, but at least not loosing options.
    # group = group.dropna(subset=kpis)
    for category in group['journey_type'].unique():
        category_group = group[group['journey_type'] == category]

        if thresholds is None:
            # Calculate data-driven thresholds as std of group
            # Else provide dictionary with values per KPI
            thresholds_in_func = {kpi: category_group[kpi].std() for kpi in kpis}
        else:
            # process thresholds given to generate thresholds to use
            thresholds_in_func = {}
            # iterate through kpis to use
            for kpi in kpis:
                # Check if kpi defined in thresholds given category
                # if not, check if defined for all
                # if not use std
                thresholds_in_func[kpi] = thresholds.get(category, {}).get(kpi,
                                                                           thresholds.get('all', {}).get(kpi))
                if thresholds_in_func[kpi] is None:
                    thresholds_in_func[kpi] = category_group[kpi].std()

        for _, option in category_group.iterrows():
            similar = False
            for idx, existing_option in enumerate(filtered_options):
                # Loop through all KPIs provided by the user
                is_similar = True  # Flag to check if option is similar based on all KPIs
                for kpi in kpis:
                    value_kpi_option = option[kpi]
                    existing_kpi_option = existing_option[kpi]
                    if pd.isnull(option[kpi]):
                        value_kpi_option = 0
                    if pd.isnull(existing_kpi_option):
                        existing_kpi_option = 0
                    if abs(value_kpi_option - existing_kpi_option) > thresholds_in_func[kpi]:
                        is_similar = False   # If any KPI difference is greater than the threshold, it's not similar
                        break

                # Check if total_waiting_time is also part of the KPIs or not
                if is_similar:
                    clusters[filtered_options[idx]['option']].append(option['option'])
                    similar = True
                    break

            if not similar:
                filtered_options.append(option)
                clusters[option['option']] = [option['option']]

    return [opt['option'] for opt in filtered_options], clusters


def process_group_clustering(name, group, kpis, thresholds):
    try:
        options_cluster = filter_similar_options(group, kpis, thresholds)
        return (name[0], name[1], options_cluster)
    finally:
        del group
        gc.collect()


def cluster_options_itineraries(df_itineraries, kpis=None, thresholds=None, pc=1):
    start_time_clustering = time.time()

    # Default KPIs if none are provided
    if kpis is None:
        kpis = ['total_travel_time', 'total_cost', 'total_emissions', 'total_waiting_time', 'nservices']

    # Group by origin-destination and apply filtering
    df_itineraries.loc[df_itineraries.total_waiting_time.isnull(), 'total_waiting_time'] = 0

    grouped_data = df_itineraries.groupby(['origin', 'destination'])

    # If pc > 1, use parallel computing, otherwise proceed sequentially
    if pc > 1:
        results = Parallel(n_jobs=pc, backend='multiprocessing')(
            delayed(process_group_clustering)(name, group, kpis, thresholds) for name, group in grouped_data)
    else:
        results = [process_group_clustering(name, group, kpis, thresholds) for name, group in grouped_data]

    # Convert results to DataFrame
    result = pd.DataFrame(results, columns=['origin', 'destination', 'options_cluster'])

    #result = df_itineraries.groupby(['origin', 'destination']).apply(lambda group: filter_similar_options(group, kpis,
    #                                                                                                      thresholds)).reset_index()
    #result.columns = ['origin', 'destination', 'options_cluster']

    result['filtered_options'] = result['options_cluster'].apply(lambda x: x[0])
    result['clusters'] = result['options_cluster'].apply(lambda x: x[1])
    result.drop(columns='options_cluster', inplace=True)

    # Prepare the final DataFrame with clusters and averaged KPIs
    '''
        Previous version of the code more readable but much slower --> from 351 sec to 12 sec
        final_clusters = []
        
        for _, row in result.iterrows():
            origin = row['origin']
            destination = row['destination']
            for cluster_id, options in row['clusters'].items():
                cluster_data = df_itineraries[(df_itineraries['option'].isin(options)) &
                                              (df_itineraries['origin'] == origin) &
                                              (df_itineraries['destination'] == destination)]
                journey_type = cluster_data.iloc[0]['journey_type']
                avg_kpis = cluster_data[kpis+['nservices']].mean().to_dict()
                final_clusters.append({
                    'origin': origin,
                    'destination': destination,
                    'journey_type': journey_type,
                    'cluster_id': cluster_id,
                    'options_in_cluster': options,
                    **avg_kpis
                })

        final_df = pd.DataFrame(final_clusters)
    '''

    # Explode clusters dictionary into separate rows
    result['cluster_pairs'] = result['clusters'].apply(lambda x: list(x.items()))
    exploded = result.explode('cluster_pairs')

    # Split the tuple into separate columns
    exploded[['cluster_id', 'options_in_cluster']] = pd.DataFrame(exploded['cluster_pairs'].tolist(), index=exploded.index)

    # Explode 'options_in_cluster' if it contains lists
    exploded = exploded.explode('options_in_cluster')

    # Drop intermediate column
    exploded.drop(columns=['cluster_pairs'], inplace=True)

    # Transform clusters into a DataFrame
    clusters_df = exploded[['origin', 'destination', 'clusters']].copy()

    clusters_df['cluster_id'] = clusters_df['clusters'].apply(lambda x: list(x.keys()))
    clusters_df['options_in_cluster'] = clusters_df['clusters'].apply(lambda x: list(x.values()))

    # Expand cluster_id and options_in_cluster into separate rows
    clusters_df = clusters_df.explode(['cluster_id', 'options_in_cluster'])

    # Flatten options list
    clusters_df = clusters_df.explode('options_in_cluster')

    # Merge with df_itineraries once instead of filtering in a loop
    merged_df = clusters_df.merge(df_itineraries, left_on=['options_in_cluster', 'origin', 'destination'],
                                  right_on=['option', 'origin', 'destination'], how='left')

    # Group by cluster, origin, destination
    grouped = merged_df.groupby(['origin', 'destination', 'cluster_id'])

    # Aggregate KPIs
    final_df = grouped.agg({
        **{kpi: 'mean' for kpi in kpis},  # Compute mean for KPIs
        'nservices': 'mean',  # Compute mean for nservices
        'journey_type': 'first',  # Take the first journey_type
        'options_in_cluster': lambda x: list(set(x))  # Keep unique options in cluster
    }).reset_index()

    # Rename column
    final_df.rename(columns={'options_in_cluster': 'options_in_cluster'}, inplace=True)


    for k in kpis:
        final_df[k] = final_df[k].apply(lambda x: round(x, 2))

    final_df['alternative_id'] = final_df['origin'] + "_" + final_df['destination'] + "_" + final_df['cluster_id'].astype(str)

    columns_to_move = ['alternative_id']
    # Get the current list of columns
    all_columns = list(final_df.columns)

    columns_first = ['origin', 'destination', 'journey_type', 'cluster_id', 'options_in_cluster']

    all_columns = columns_first + [item for item in all_columns if item not in columns_first]

    # Order options in cluster
    final_df['options_in_cluster'] = final_df['options_in_cluster'].apply(lambda x: sorted(x))

    # Find the position after 'cluster_id'
    position_after = all_columns.index('cluster_id') + 1

    # Create a new list of columns with 'a' and 'b' moved
    new_column_order = (
            all_columns[:position_after] +  # Columns up to 'destination'
            columns_to_move +
            [col for col in all_columns if col not in columns_to_move and col not in all_columns[:position_after]]
    # Remaining columns
    )

    # Reorder the DataFrame
    final_df = final_df[new_column_order]

    end_time_clustering = time.time()
    logger.important_info("Filtering and clustering done in, "
                          + str(end_time_clustering - start_time_clustering) + " seconds.")
    return final_df


def keep_pareto_equivalent_solutions(df, thresholds):
    def is_equivalent(val1, val2, threshold):
        return abs(val1 - val2) <= threshold

    def dominates(option1, option2, thresholds):
        better_in_any = False
        for kpi, threshold in thresholds.items():
            if not is_equivalent(option1[kpi], option2[kpi], threshold):
                if option1[kpi] < option2[kpi]:
                    better_in_any = True
                else:
                    return False
        return better_in_any

    def filter_pareto(group, thresholds):
        pareto_options = []
        options = group.to_dict('records')
        for i, option1 in enumerate(options):
            if option1['nservices'] == 1:
                # If only one service, it's direct, so we keep it
                dominated = False
            else:
                dominated = False
                for j, option2 in enumerate(options):
                    if i != j and dominates(option2, option1, thresholds):
                        dominated = True
                        break
            if not dominated:
                pareto_options.append(option1)
        return pd.DataFrame(pareto_options)

    grouped = df.groupby(['origin', 'destination', 'journey_type'])
    pareto_df = grouped.apply(lambda group: filter_pareto(group, thresholds)).reset_index(drop=True)

    return pareto_df


def keep_itineraries_options(df_itineraries, pareto_df):
    # Expand the `options_in_cluster` column
    expanded_rows = []
    for _, row in pareto_df.iterrows():
        for option in row['options_in_cluster']:
            expanded_rows.append({
                'origin': row['origin'],
                'destination': row['destination'],
                'cluster_id': row['cluster_id'],
                'option': option,
                'alternative_id': row['alternative_id']
            })

    expanded_df = pd.DataFrame(expanded_rows)

    # Filter the second dataframe using the expanded dataframe
    filtered_df = df_itineraries.merge(expanded_df, on=['origin', 'destination', 'option'], how='inner')

    # Move cluster_id and alternative_id to beginning of dataframe
    columns_to_move = ['alternative_id', 'cluster_id']

    # Get the current list of columns
    all_columns = list(filtered_df.columns)

    # Find the position after 'origin' and 'destination' (which are the 1st and 2nd columns)
    position_after = all_columns.index('destination') + 1

    # Create a new list of columns with 'a' and 'b' moved
    new_column_order = (
            all_columns[:position_after] +  # Columns up to 'destination'
            columns_to_move +  # Columns to be inserted
            [col for col in all_columns if col not in columns_to_move and col not in all_columns[:position_after]]
    # Remaining columns
    )

    # Reorder the DataFrame
    filtered_df = filtered_df[new_column_order]

    return filtered_df


def obtain_demand_per_cluster_itineraries(df_clusters, df_pax_demand, df_paths):
    # Aggregate Passenger Demand from df_pax_demand
    agg_demand = df_pax_demand.groupby(['origin', 'destination']).sum().reset_index()

    # Extract the demand per alternative column
    alternatives_cols = [col for col in agg_demand.columns if col.startswith('alternative_') and not col.startswith('alternative_prob')]
    agg_demand = agg_demand[['origin', 'destination'] + alternatives_cols]

    # Step 1: Calculate total passengers per alternative for each OD pair
    total_pax_per_alternative = df_pax_demand.groupby(['origin', 'destination'])[alternatives_cols].sum().reset_index()

    # Step 2: Compute percentage of passengers for each archetype per alternative
    def calculate_percentages(row, total_pax_df):
        origin, destination = row['origin'], row['destination']
        total_row = total_pax_df[(total_pax_df['origin'] == origin) & (total_pax_df['destination'] == destination)]
        if not total_row.empty:
            return {col: row[col] / total_row.iloc[0][col] if total_row.iloc[0][col] > 0 else 0.0 for col in alternatives_cols}
        return {}

    df_pax_demand['percentages_per_alternative'] = df_pax_demand.apply(calculate_percentages, total_pax_df=total_pax_per_alternative, axis=1)

    # Create a Mapping for Cluster IDs from df_paths
    simplified_cluster_mapping = {
        row[col]: col
        for _, row in df_paths.iterrows()
        for col in [c for c in df_paths.columns if c.startswith('alternative_id_')]
        if row[col] != 0
    }

    # Make a copy of df_clusters to expand it
    df_clusters_expanded = df_clusters.copy()
    df_clusters_expanded['num_pax'] = 0
    df_clusters_expanded['prob_of_archetype'] = [{} for _ in range(len(df_clusters_expanded))]

    # Iterate through each row in df_clusters to add num_pax and prob_of_archetype
    for index, row in df_clusters_expanded.iterrows():
        cluster_id = row['alternative_id']
        if cluster_id in simplified_cluster_mapping:
            alternative_label = simplified_cluster_mapping[cluster_id]
            alternative_key = alternative_label.replace('alternative_id_', 'alternative_')
            origin, destination = row['origin'], row['destination']
            demand_row = agg_demand[(agg_demand['origin'] == origin) & (agg_demand['destination'] == destination)]

            df_clusters_expanded['num_pax'] = df_clusters_expanded['num_pax'].astype(float)
            if not demand_row.empty and alternative_key in demand_row.columns:
                num_pax = demand_row.iloc[0][alternative_key]
                df_clusters_expanded.at[index, 'num_pax'] = num_pax

                matching_rows = df_pax_demand[(df_pax_demand['origin'] == origin) & (df_pax_demand['destination'] == destination)]
                archetype_percentages = {
                    matching_row['archetype']: matching_row['percentages_per_alternative'].get(alternative_key, 0.0)
                    for _, matching_row in matching_rows.iterrows()
                }
                df_clusters_expanded.at[index, 'prob_of_archetype'] = archetype_percentages

    return df_clusters_expanded


def assing_pax_to_services(df_schedules, df_demand, df_possible_itineraries, paras):

    if paras['train_seats_per_segment'].lower() == 'combined'.lower():
        # If combined need to remove the stops from the train ids so from xxx_stop1_stop2 to xxx on the id
        # of the trains and on the df_possible_itineraries
        # Also for seats keep the mean number of seats and for distance the max
        # The original xxx_stop1_stop2 and distance will be saved on df_possible_itineraries_orig and df_schedules_orig

        df_schedules.loc[df_schedules['mode']=='rail',
                        'nid'] = df_schedules.loc[df_schedules['mode']=='rail']['nid'].apply(lambda x: x.split('_')[0])

        # Keep mean number of seats across different 'stops', if they were to be different
        service_max_seats_group = df_schedules.groupby('nid')['max_seats'].mean()
        service_max_gcdistance = df_schedules.groupby('nid')['gcdistance'].max()

        # Convert the result to a dictionary
        service_max_seats_avg_dict = service_max_seats_group.to_dict()
        service_max_gcdistance_dict = service_max_gcdistance.to_dict()

        #import pickle
        #with open("./pax_assigment_tests/df_schedules_pre_cap.pkl", "wb") as file:
        #    pickle.dump({'df_schedules': df_schedules, 'df_demand': df_demand,
        #                 'df_possible_itineraries': df_possible_itineraries}, file)

        df_schedules['seats'] = df_schedules['nid'].apply(lambda x: service_max_seats_avg_dict[x])
        df_schedules['gcdistance'] = df_schedules['nid'].apply(lambda x: service_max_gcdistance_dict[x])

        df_schedules = df_schedules.drop_duplicates()

        # Remove the stops part of the id of the trains (from xxx_y_z to xxx)
        service_cols = [col for col in df_possible_itineraries.columns if col.startswith('service_id_')]
        mode_cols = [col for col in df_possible_itineraries.columns if col.startswith('mode_')]

        for service_col, mode_col in zip(sorted(service_cols), sorted(mode_cols)):
            # Apply the transformation only to rows where mode_col is 'rail'
            mask = (df_possible_itineraries[mode_col] == 'rail') & df_possible_itineraries[service_col].notna()
            df_possible_itineraries.loc[mask, service_col] = \
                df_possible_itineraries.loc[mask, service_col].astype(str).str.replace(
                    r'_.*', '', regex=True)

    # Initialize unique IDs for df_demand
    #print(df_demand)
    #sys.exit(0)
    #df_demand['id'] = np.arange(1, len(df_demand) + 1)

    # Step 0: If journey_type is none then mode should be none too
    df_possible_itineraries.loc[(df_possible_itineraries['journey_type']=='none'), 'mode_0'] = None

    # Step 1: Merge df_demand and df_poss_it on alternative_id
    merged_df = pd.merge(df_possible_itineraries, df_demand, on='alternative_id', suffixes=('_opt', '_pax'))

    # Step 2: Create a per-passenger DataFrame by repeating each row for each passenger
    # This is not needed as we're working with groups, no need to create a row per individual pax
    #merged_df = merged_df.loc[merged_df.index.repeat(merged_df['num_pax'])].reset_index(drop=True)

    # Step 3: Assign 'nid_f1', 'nid_f2', ..., based on service_id_x from df_poss_it
    # Determine the maximum number of services (to know how many nid_fx columns we need)
    max_services = int(merged_df['nservices_pax'].max())

    # Create columns for each service (e.g., nid_f1, nid_f2, ...)
    #for i in range(max_services):
    #    col_name = f'nid_f{i + 1}'
    #    service_col_name = f'service_id_{i}'  # Service columns in df_poss_it
    #    merged_df[col_name] = merged_df[service_col_name]

    # Generate a mapping of service columns to new column names
    columns_map = {f'service_id_{i}': f'nid_f{i + 1}' for i in range(max_services)}

    # Rename the columns in one operation
    merged_df = merged_df.rename(columns=columns_map)

    # Step 4: Construct 'type' column using vectorized operations for better performance
    modes = [f'mode_{i}' for i in range(max_services)]

    # Replace 'air' with 'flight' in the mode columns
    for mode in modes:
        merged_df[mode] = merged_df[mode].replace('air', 'flight')

    # Add type as concatenation of modes flight, rail, rail_rail, rail_flight, flight_flight, rail_flight_rail, etc.
    merged_df['type'] = merged_df[modes].fillna('').agg('_'.join, axis=1)

    # Remove leading and trailing underscores
    #merged_df['type'] = merged_df['type'].str.replace(r'^_|_+$', '', regex=True)
    merged_df['type'] = merged_df['type'].str.strip('_') # Remove leading and trailing underscores

    merged_df['type'] = merged_df['type'].str.replace(r'_+', '_', regex=True)  # Replace multiple underscores with one
    # if type=='' put None
    merged_df.loc[(merged_df['type']==''), 'type'] = None

    merged_df = merged_df.rename(columns={'cluster_id_opt': 'cluster_id', 'total_cost_opt': 'fare',
                                          'total_waiting_time_opt': 'total_waiting_time',
                                          'total_travel_time_opt': 'total_time', 'num_pax': 'volume',
                                          'origin_opt': 'origin', 'destination_opt': 'destination',
                                          'option_number': 'option_cluster_number'})

    # Step 5: Rename and select the required columns for the final output
    nid_columns = [f'nid_f{i + 1}' for i in range(max_services)]
    # We keep also alternative_id as this is a key to identify demand groups

    final_columns = ['cluster_id', 'option_cluster_number', 'alternative_id', 'option', 'origin', 'destination', 'path'] + nid_columns + [
        'total_waiting_time', 'total_time', 'type', 'volume',
        'fare', 'access_time', 'egress_time', 'd2i_time', 'i2d_time']

    options = merged_df[final_columns].copy()

    options['cluster_id'] = options['alternative_id']

    # Create unique id for option_number
    #    options['option_number'] = range(1, len(options) + 1)

    it_gen, d_seats_max, options = assign_passengers_options_solver(df_schedules, options, paras, verbose=False)

    # it_gen = fill_flights_too_empy(it_gen, df_schedules, d_seats_max, options, paras, verbose=False)

    # For ach option add pax assigned
    df_options_w_pax = options.merge(it_gen[['cluster_id', 'option', 'pax', 'avg_fare', 'generated_info']],
                                     left_on=['cluster_id', 'option'],
                                        right_on=['cluster_id', 'option'], how='left')#.drop(columns=['option'])
    df_options_w_pax['pax'] = df_options_w_pax['pax'].fillna(0)
    df_options_w_pax['pax_group_id'] = np.arange(df_options_w_pax.shape[0])

    return it_gen, d_seats_max, df_options_w_pax


'''
Create the input for the tactical evaluator (Mercury) from the pax assigment
'''
def transform_pax_assigment_to_tactical_input(df_options_w_pax):#df_pax_assigment, df_options):
    df_pax = df_options_w_pax[df_options_w_pax.pax>0].copy() #df_pax_assigment.merge(df_options, right_on=['id', 'option_number'], left_on=['it', 'option'], how='left')

    # Define the regex patterns for dynamic column selection
    leg_pattern = r'^leg\d+$'  # Matches columns like leg1, leg2, ...
    nid_f_pattern = r'^nid_f\d+$'  # Matches columns like nid_f1, nid_f2, ...

    # Specify fixed columns to include
    fixed_columns = ['cluster_id', 'option', 'pax_group_id', 'pax', 'avg_fare', 'generated_info', 'alternative_id', 'type', 'path', 'origin', 'destination', 'access_time', 'egress_time', 'd2i_time', 'i2d_time']

    # Filter dynamically using regex for `leg` and `nid_f` columns
    dynamic_columns = df_pax.filter(regex=f'({leg_pattern}|{nid_f_pattern})').columns.tolist()

    # Combine fixed and dynamic columns
    selected_columns = fixed_columns + dynamic_columns

    # Filter the dataframe to include only the selected columns
    df_pax = df_pax[selected_columns]
    df_pax['ticket_type'] = 'economy'

    # Separate rows where 'type' is None
    df_pax_non_supported_tactically = df_pax[df_pax['type'].isnull()].copy()
    df_supported = df_pax[df_pax['type'].notnull()].copy()

    # Define a function to check validity of the `type` column
    def is_valid_type(type_str):
        modes = type_str.split('_')
        # Only allow 'rail' in the first or last position if it appears
        if 'rail' in modes[1:-1]:  # If 'rail' is in the middle
            return False
        if len(modes) >= 1 and all(mode == 'rail' for mode in modes):  # All rail
            return False
        return True

    # Apply the validity check to the supported rows
    df_supported['valid_type'] = df_supported['type'].apply(is_valid_type)

    # Separate valid and invalid rows
    df_valid_supported = df_supported[df_supported['valid_type']].drop(columns=['valid_type'])
    df_pax_non_supported_tactically = pd.concat(
        [df_pax_non_supported_tactically, df_supported[~df_supported['valid_type']]]
    ).drop(columns=['valid_type'])

    df_pax_non_supported_tactically['type'].drop_duplicates()

    # Helper function to process a row
    def process_row(row):
        modes = row['type'].split('_')
        legs = [row.get(f'nid_f{i + 1}', None) for i in range(len(modes))]

        # Initialize rail_pre and rail_post
        rail_pre = None
        rail_post = None

        # Move rail IDs to rail_pre and rail_post
        if modes[0] == 'rail':
            rail_pre = legs[0]
            legs[0] = None
        if modes[-1] == 'rail':
            rail_post = legs[-1]
            legs[-1] = None

        # Remove rail IDs from legs
        flight_ids = [leg if mode == 'flight' else None for mode, leg in zip(modes, legs)]

        # Compact flight IDs to fill gaps
        flight_ids = [fid for fid in flight_ids if fid is not None]
        flight_ids += [None] * (len(legs) - len(flight_ids))  # Pad with None

        # Update the row
        row['rail_pre'] = rail_pre
        row['rail_post'] = rail_post
        for i, flight_id in enumerate(flight_ids):
            row[f'leg{i + 1}'] = flight_id
        for j in range(len(flight_ids), len(legs)):  # Set remaining leg columns to None
            row[f'leg{j + 1}'] = None

        return row

    # Apply processing
    df_valid_supported = df_valid_supported.copy()
    df_valid_supported['rail_pre'] = None
    df_valid_supported['rail_post'] = None
    df_valid_supported = df_valid_supported.apply(process_row, axis=1)

    # Drop unnecessary leg columns (if needed)
    max_legs = df_valid_supported['type'].str.count('_').max() + 1
    extra_cols = [f'leg{i + 1}' for i in
                  range(max_legs, len([col for col in df_valid_supported.columns if col.startswith('leg')]))]
    df_valid_supported = df_valid_supported.drop(columns=extra_cols, errors='ignore')

    # Add new columns and initialize with NaN
    df_valid_supported['origin1'] = np.nan
    df_valid_supported['destination1'] = np.nan
    df_valid_supported['origin2'] = np.nan
    df_valid_supported['destination2'] = np.nan

    # Update origin1 and destination1 based on rail_pre
    def get_n_from_path(path, n):
        if type(path)==str:
            # The list is in a string form
            path = ast.literal_eval(path)
        return path[n]

    df_valid_supported['origin1'] = df_valid_supported['origin1'].astype('object')
    df_valid_supported.loc[df_valid_supported['rail_pre'].notna(), 'origin1'] = \
        df_valid_supported.loc[df_valid_supported['rail_pre'].notna(), 'path'].apply(lambda x: get_n_from_path(x,0))

    df_valid_supported['destination1'] = df_valid_supported['destination1'].astype('object')
    df_valid_supported.loc[df_valid_supported['rail_pre'].notna(), 'destination1'] = \
        df_valid_supported.loc[df_valid_supported['rail_pre'].notna(), 'path'].apply(lambda x: get_n_from_path(x, 1))

    # Update origin2 and destination2 based on rail_post
    df_valid_supported['origin2'] = df_valid_supported['origin2'].astype('object')
    df_valid_supported.loc[df_valid_supported['rail_post'].notna(), 'origin2'] = \
        df_valid_supported.loc[df_valid_supported['rail_post'].notna(), 'path'].apply(lambda x: get_n_from_path(x, -2))

    df_valid_supported['destination2'] = df_valid_supported['destination2'].astype('object')
    df_valid_supported.loc[df_valid_supported['rail_post'].notna(), 'destination2'] = \
        df_valid_supported.loc[df_valid_supported['rail_post'].notna(), 'path'].apply(lambda x: get_n_from_path(x, -1))

    df_valid_supported['gtfs_pre'] = np.nan
    df_valid_supported['gtfs_post'] = np.nan

    # Filter rows tactical pax assignment
    df_valid_supported = df_valid_supported.rename(
        columns={'pax_group_id': 'nid_x', 'generated_info': 'source'})  # , inplace=True)
    leg_columns = [col for col in df_valid_supported.columns if col.startswith('leg')]
    df_valid_supported = df_valid_supported[
        ['nid_x', 'pax', 'avg_fare', 'ticket_type', 'origin', 'destination', 'access_time', 'egress_time', 'd2i_time', 'i2d_time'] + leg_columns + ['rail_pre', 'rail_post', 'source', 'gtfs_pre',
                                                                     'gtfs_post',
                                                                     'origin1', 'destination1', 'origin2',
                                                                     'destination2',
                                                                     'type']]

    return df_valid_supported, df_pax_non_supported_tactically


def transform_fight_schedules_tactical_input(config, path_folder_network, pre_processed_version):
    path_flight_schedules = (path_folder_network / ('flight_schedules_proc_' + str(pre_processed_version) + '.csv'))

    df_fs = pd.read_csv(path_flight_schedules)

    # Read aircraft related information
    df_ac_icao_iata = pd.read_csv(config['aircraft']['ac_type_icao_iata_conversion'])
    dict_ac_iata_icao = pd.Series(df_ac_icao_iata['icao_ac_code'].values,
                                  index=df_ac_icao_iata['iata_ac_code']).to_dict()

    df_ac_wt = pd.read_csv(config['aircraft']['ac_wtc'])
    dict_ac_wtc = pd.Series(df_ac_wt['wake'].values, index=df_ac_wt['ac_icao']).to_dict()

    df_mtow = pd.read_csv(config['aircraft']['ac_mtow'])
    dict_mtow = pd.Series(df_mtow['mtow'].values, index=df_mtow['aircraft_type']).to_dict()

    # Read airline related information
    df_airline_codes = pd.read_csv(config['airlines']['airline_iata_icao'])
    dict_aln_codes = pd.Series(df_airline_codes['ICAO_code'].values, index=df_airline_codes['IATA_code']).to_dict()

    df_airline_types = pd.read_csv(config['airlines']['airline_ao_type'])
    dict_ao_types = pd.Series(df_airline_types['AO_type'].values, index=df_airline_types['ICAO']).to_dict()

    # Read flight schedules
    df_fs['nid'] = df_fs['service_id']
    df_fs['flight_id'] = df_fs['service_id']
    df_fs['callsign'] = df_fs['service_id'].str.split("_").apply(lambda x: x[0] + x[1])
    df_fs['airline'] = df_fs['provider'].apply(lambda x: dict_aln_codes[x])
    df_fs['airline_type'] = df_fs['airline'].apply(lambda x: dict_ao_types[x])
    df_fs['long_short_dist'] = None
    df_fs['aircraft_type'] = df_fs['act_type'].apply(lambda x: dict_ac_iata_icao[x])
    df_fs['mtow'] = df_fs['aircraft_type'].apply(lambda x: dict_mtow[x])
    df_fs['wk_tbl_cat'] = df_fs['aircraft_type'].apply(lambda x: dict_ac_wtc[x])
    df_fs['registration'] = df_fs['flight_id']
    df_fs['max_seats'] = df_fs['seats']
    df_fs['exclude'] = 0

    df_fs['long_short_dist'] = df_fs['gcdistance'].apply(lambda x: 'I' if x * 0.539957 > 500 else 'D')

    df_fs = df_fs[['nid', 'flight_id', 'callsign', 'airline', 'airline_type', 'origin',
           'destination', 'gcdistance', 'long_short_dist', 'sobt', 'sibt',
           'aircraft_type', 'mtow', 'wk_tbl_cat', 'registration', 'max_seats',
           'exclude']]

    return df_fs
