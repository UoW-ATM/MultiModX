from pathlib import Path
import pandas as pd
from itertools import combinations
import multiprocessing as mp
import time

from strategic_evaluator.mobility_network import Service, NetworkLayer, Network
from strategic_evaluator.mobility_network_particularities import (mct_air_network, fastest_air_time_heuristic,
                                                                  initialise_air_network, mct_rail_network,
                                                                  services_from_after_function_rail,
                                                                  initialise_rail_network,
                                                                  fastest_rail_time_heuristic,
                                                                  fastest_precomputed_distance_time_heuristic)
from libs.gtfs import get_stop_times_on_date


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

        dict_info_access[r.pax_type] = r.avg_time

    return regions_access_air


def create_air_layer(df_fs, df_as, df_mct, df_ra_air=None, keep_only_fastest_service=0,
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

    # Rename sobt and sibt
    df_fs.rename(columns={'sobt': 'departure_time', 'sibt': 'arrival_time'}, inplace=True)

    if keep_only_fastest_service > 0:
        df_fs['duration'] = None
        df_fs['duration'] = df_fs.apply(lambda x: x.arrival_time - x.departure_time, axis=1)
        df_s_min_duration = df_fs.groupby(['origin', 'destination'])['duration'].min().reset_index()
        df_fs = pd.merge(df_s_min_duration, df_fs, on=['origin', 'destination', 'duration'], how='inner')

    if keep_only_fastest_service == 2:
        # Keep only one fastest regardless of airline/alliance
        df_fs.drop_duplicates(subset=['origin', 'destination'], keep='first', inplace=True)

    # Create Services for flights (one per flight in the dataframe)
    df_fs['service'] = df_fs.apply(lambda x: Service(x.service_id, x.origin, x.destination,
                                                     x.departure_time, x.arrival_time, x.cost,
                                                     x.provider, x.alliance, x.emissions,
                                                     gcdistance=x.gcdistance), axis=1)

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


def create_rail_layer(df_stop_times, date_considered='01/01/2024', df_stops_considered=None, df_ra_rail=None,
                      keep_only_fastest_service=False,
                      df_stops=None,
                      heuristic_precomputed_distance=None):

    # Regions access -- Create dictionary
    regions_access_rail = None
    if df_ra_rail is not None:
        regions_access_rail = create_region_access_dict(df_ra_rail)

    if df_stops_considered is not None:
        df_stop_times = df_stop_times[df_stop_times.stop_id.isin(df_stops_considered.stop_id)].copy()

    n_stops_trip = df_stop_times.groupby('trip_id').count().reset_index()

    # Keep trips with at least two stops of interest in the trip
    df_stop_times = df_stop_times[
        df_stop_times.trip_id.isin(n_stops_trip[n_stops_trip.arrival_time > 1].trip_id)].copy()

    date = pd.to_datetime(date_considered, format='%d/%m/%Y')

    # Create rail services as all possible segments in the trips
    rail_services = []

    df_sorted = df_stop_times.sort_values(by=['trip_id', 'stop_sequence'])

    def add_date_and_handle_overflow(time_str):
        hour, minute, second = map(int, time_str.split(':'))
        if hour >= 24:
            # If time exceeds 24 hours, add a day to the date component and adjust the time
            date_adjusted = date + pd.Timedelta(days=1)
            hour %= 24
        else:
            date_adjusted = date
        return date_adjusted + pd.Timedelta(hours=hour, minutes=minute, seconds=second)

    df_sorted['arrival_time'] = df_sorted['arrival_time'].apply(add_date_and_handle_overflow)
    df_sorted['departure_time'] = df_sorted['departure_time'].apply(add_date_and_handle_overflow)

    grouped = df_sorted.groupby('trip_id')

    for trip_id, group_df in grouped:
        stop_ids = group_df['stop_id'].tolist()
        stop_sequences = group_df['stop_sequence'].tolist()
        for i, j in combinations(range(len(stop_ids)), 2):
            service_id = f"{trip_id}_{stop_sequences[i]}_{stop_sequences[j]}"
            origin = str(stop_ids[i])
            destination = str(stop_ids[j])
            departure_time = group_df.iloc[i]['departure_time']
            arrival_time = group_df.iloc[j]['arrival_time']

            # Create Service object and add it to the list of services
            service = Service(service_id, origin, destination, departure_time, arrival_time, cost=0,
                              provider='renfe',
                              alliance='renfe')
            service_df_row = (
                service_id, str(origin), str(destination), departure_time, arrival_time, 0, 'renfe', 'renfe', service)

            rail_services.append(service_df_row)

    rail_services_df = pd.DataFrame(rail_services,
                                    columns=['service_id', 'origin', 'destination', 'departure_time', 'arrival_time',
                                             'cost', 'provider', 'alliance', 'service'])

    # Keep only fastest services if requested to do so
    if keep_only_fastest_service:
        rail_services_df['duration'] = rail_services_df['arrival_time'] - rail_services_df['departure_time']
        # Group by 'origin' and 'destination', then keep only the first row of each group
        rail_services_df = rail_services_df.groupby(['origin', 'destination']).apply(lambda x:
                                                                                     x.sort_values('duration').iloc[0])
        rail_services_df = rail_services_df.reset_index(drop=True)

    if df_stops is not None:
        df_stops = df_stops.copy().rename({'stop_id': 'node', 'stop_lat': 'lat', 'stop_lon': 'lon'},
                                          axis=1)[['node', 'lat', 'lon']]
        df_stops['node'] = df_stops['node'].astype(str)

    # Use precomputed distance vs time function or default air
    if heuristic_precomputed_distance is not None:
        heuristic_func = fastest_precomputed_distance_time_heuristic
    else:
        heuristic_func = fastest_rail_time_heuristic

    nl_rail = NetworkLayer('rail', rail_services_df, regions_access=regions_access_rail,
                           custom_initialisation=initialise_rail_network,
                           custom_mct_func=mct_rail_network,
                           custom_services_from_after_func=services_from_after_function_rail,
                           nodes_coordinates=df_stops,
                           heuristic_precomputed_distance=heuristic_precomputed_distance,
                           custom_heuristic_func=heuristic_func)

    return nl_rail


def preprocess_input(network_definition_config, pre_processed_version=0):
    if 'air_network' in network_definition_config.keys():
        preprocess_air_layer(network_definition_config['network_path'], network_definition_config['air_network'],
                             network_definition_config['processed_folder'],
                             pre_processed_version=pre_processed_version)

    if 'rail_network' in network_definition_config.keys():
        pre_process_rail_layer(network_definition_config['network_path'], network_definition_config['rail_network'],
                               network_definition_config['processed_folder'],
                               pre_processed_version=pre_processed_version)


def preprocess_air_layer(path_network, air_network, processed_folder, pre_processed_version=0):
    df_fs = pd.read_csv(Path(path_network) / air_network['flight_schedules'], keep_default_na=False)
    df_fs.replace('', None, inplace=True)
    df_fs['alliance'].fillna(df_fs['provider'], inplace=True)

    if 'provider' not in df_fs.columns:
        df_fs['provider'] = None
    if 'alliance' not in df_fs.columns:
        df_fs['alliance'] = None
    if 'cost' not in df_fs.columns:
        df_fs['cost'] = 0
    if 'seats' not in df_fs.columns:
        df_fs['seats'] = 140
    if 'emissions' not in df_fs.columns:
        df_fs['emissions'] = 0

    fflights = 'flight_schedules_proc_' + str(pre_processed_version) + '.csv'
    df_fs.to_csv(Path(path_network) / processed_folder / fflights, index=False)


def pre_process_rail_layer(path_network, rail_network, processed_folder, pre_processed_version=0):
    df_stop_times = pd.read_csv(Path(path_network) / rail_network['gtfs'] / 'stop_times.txt')
    df_trips = pd.read_csv(Path(path_network) / rail_network['gtfs'] / 'trips.txt')
    df_calendar = pd.read_csv(Path(path_network) / rail_network['gtfs'] / 'calendar.txt')
    df_calendar_dates = pd.read_csv(Path(path_network) / rail_network['gtfs'] / 'calendar_dates.txt')

    df_calendar['start_date'] = pd.to_datetime(df_calendar['start_date'], format='%Y%m%d')
    df_calendar['end_date'] = pd.to_datetime(df_calendar['end_date'], format='%Y%m%d')

    df_calendar_dates['date'] = pd.to_datetime(df_calendar_dates['date'], format='%Y%m%d')

    # TODO: fix rail dates
    date_rail = '20230503'
    date_rail = pd.to_datetime(date_rail, format='%Y%m%d')
    df_stop_times = get_stop_times_on_date(date_rail, df_calendar, df_calendar_dates, df_trips, df_stop_times)

    if 'rail_stations_considered' in rail_network.keys():
        df_stops_considered = pd.read_csv(Path(path_network) / rail_network['rail_stations_considered'])
        df_stop_times = df_stop_times[df_stop_times.stop_id.isin(df_stops_considered.stop_id)]

    # TODO: add provider, cost, emissions...
    if 'provider' not in df_stop_times.columns:
        df_stop_times['provider'] = None
    if 'alliance' not in df_stop_times.columns:
        df_stop_times['alliance'] = None
    if 'cost' not in df_stop_times.columns:
        df_stop_times['cost'] = 0
    if 'seats' not in df_stop_times.columns:
        df_stop_times['seats'] = 140
    if 'emissions' not in df_stop_times.columns:
        df_stop_times['emissions'] = 0

    frail = 'rail_timetable_proc_'+str(pre_processed_version)+'.csv'
    df_stop_times.to_csv(Path(path_network) / processed_folder / frail, index=False)


def create_network(path_network_dict, compute_simplified=False, use_heuristics_precomputed=False,
                   pre_processed_version=0):
    df_regions_access = None
    df_transitions = None
    layers = []
    network = None

    if 'regions_access' in path_network_dict.keys():
        # Read regions access to infrastructure if provided
        df_regions_access = pd.read_csv(Path(path_network_dict['network_path']) /
                                        path_network_dict['regions_access']['regions_access'])

    if 'air_network' in path_network_dict.keys():
        # Read airport data (needed for the heuristic to compute GCD and
        # estimate time needed from current node to destination)
        df_as = pd.read_csv(Path(path_network_dict['network_path']) /
                            path_network_dict['air_network']['airports_static'])

        fschedule_filename = 'flight_schedules_proc_'+str(pre_processed_version)+'.csv'
        df_fs = pd.read_csv(Path(path_network_dict['network_path']) / path_network_dict['processed_folder'] /
                            fschedule_filename, keep_default_na=False)

        df_fs.replace('', None, inplace=True)
        df_fs = df_fs.merge(df_as[['icao_id', 'lat', 'lon']], left_on='origin', right_on='icao_id')
        df_fs = df_fs.merge(df_as[['icao_id', 'lat', 'lon']], left_on='destination', right_on='icao_id',
                            suffixes=('_orig', '_dest'))

        dict_dist_origin_destination = {}
        if 'gcdistance' not in df_fs.columns:
            from libs.uow_tool_belt.general_tools import haversine
            # Initialise dictionary of o-d distance
            for odr in df_fs[['origin', 'destination']].drop_duplicates().iterrows():
                lat_orig = df_as[df_as.icao_id == odr[1].origin].iloc[0].lat
                lon_orig = df_as[df_as.icao_id == odr[1].origin].iloc[0].lon
                lat_dest = df_as[df_as.icao_id == odr[1].destination].iloc[0].lat
                lon_dest = df_as[df_as.icao_id == odr[1].destination].iloc[0].lon
                dict_dist_origin_destination[(odr[1].origin, odr[1].destination)] = haversine(lon_orig, lat_orig,
                                                                                              lon_dest, lat_dest)
            df_fs['gcdistance'] = df_fs.apply(lambda x: dict_dist_origin_destination[x['origin'], x['destination']],
                                              axis=1)
        else:
            for odr in df_fs[['origin', 'destination', 'gcdistance']].drop_duplicates().iterrows():
                dict_dist_origin_destination[(odr[1].origin, odr[1].destination)] = odr[1].gcdistance

        df_fs['sobt'] = pd.to_datetime(df_fs['sobt'],  format='%Y-%m-%d %H:%M:%S')
        df_fs['sibt'] = pd.to_datetime(df_fs['sibt'],  format='%Y-%m-%d %H:%M:%S')

        # Read MCTs between air services
        df_mct = pd.read_csv(Path(path_network_dict['network_path']) /
                             path_network_dict['air_network']['mct_air'])

        # Get regions access for air
        df_ra_air = None
        if df_regions_access is not None:
            df_ra_air = df_regions_access[df_regions_access['layer'] == 'air'].copy().reset_index(drop=True)
            if len(df_ra_air) == 0:
                df_ra_air = None

        only_fastest = 0
        if compute_simplified:
            only_fastest = 2

        df_heuristic_air = None
        if use_heuristics_precomputed:
            p_heuristic_air = (Path(path_network_dict['network_path']) / path_network_dict['processed_folder'] /
                               'heuristics_computed' / 'air_time_heuristics.csv')

            if p_heuristic_air.exists():
                # We have the file for air time heuristics
                df_heuristic_air = pd.read_csv(p_heuristic_air)

        air_layer = create_air_layer(df_fs.copy(), df_as, df_mct, df_ra_air,
                                     keep_only_fastest_service=only_fastest,
                                     dict_dist_origin_destination=dict_dist_origin_destination,
                                     heuristic_precomputed_distance=df_heuristic_air)
        layers += [air_layer]

    if 'rail_network' in path_network_dict.keys():
        fstops_filename = 'rail_timetable_proc_' + str(pre_processed_version) + '.csv'
        df_stop_times = pd.read_csv(Path(path_network_dict['network_path']) / path_network_dict['processed_folder'] /
                                    fstops_filename, keep_default_na=False)

        # Get regions access for rail
        df_ra_rail = None
        if df_regions_access is not None:
            df_ra_rail = df_regions_access[df_regions_access['layer'] == 'rail'].copy().reset_index(drop=True)
            if len(df_ra_rail) == 0:
                df_ra_rail = None

        df_heuristic_rail = None
        if use_heuristics_precomputed:
            p_heuristic_rail = (Path(path_network_dict['network_path']) / path_network_dict['processed_folder'] /
                                'heuristics_computed' / 'rail_time_heuristics.csv')
            if p_heuristic_rail.exists():
                # We have the file for air time heuristics
                df_heuristic_rail = pd.read_csv(p_heuristic_rail)
            # Need the stops to have its coordinates
            df_stops = pd.read_csv(Path(path_network_dict['network_path']) / path_network_dict['rail_network']['gtfs'] /
                                   'stops.txt')
        else:
            df_stops = None

        rail_layer = create_rail_layer(df_stop_times, date_considered='12/09/2014',
                                       df_ra_rail=df_ra_rail,
                                       keep_only_fastest_service=compute_simplified,
                                       df_stops=df_stops,
                                       heuristic_precomputed_distance=df_heuristic_rail)

        layers += [rail_layer]

    if 'multimodal' in path_network_dict.keys():
        df_transitions = pd.read_csv(Path(path_network_dict['network_path']) /
                                     path_network_dict['multimodal']['air_rail_transitions'])

        df_transitions.rename(columns={'origin_station': 'origin', 'destination_station': 'destination',
                                       'layer_origin': 'layer_id_origin', 'layer_destination': 'layer_id_destination'},
                              inplace=True)
        df_transitions['mct'] = df_transitions['mct'].apply(lambda x: {'all': x})

    if len(layers) > 0:
        if len(layers) == 1:
            df_transitions_used = None
        else:
            df_transitions_used = df_transitions
        network = Network(layers=layers, transition_btw_layers=df_transitions_used)

    return network


def compute_paths(od_paths, network, n_paths=10,
                  max_connections=2, allow_mixed_operators=False, consider_times_constraints=True):
    dict_paths = {}
    start_time = time.time()

    n_explored_total = 0
    for i, od in od_paths.iterrows():
        start_time_od = time.time()
        same_operators = not allow_mixed_operators
        paths, n_explored = network.find_paths(origin=od.origin, destination=od.destination, npaths=n_paths,
                                               max_connections=max_connections,
                                               consider_operators_connections=same_operators,
                                               consider_times_constraints=consider_times_constraints)
        dict_paths[(od.origin, od.destination)] = paths
        n_explored_total += n_explored
        end_time_od = time.time()
        print("Paths for", od.origin, "-", od.destination,
              ", computed in, ", (end_time_od - start_time_od), " seconds, exploring:", n_explored,
              "nodes. Found", len(paths), "paths.\n")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Paths computed in:", elapsed_time, "seconds, exploring:", n_explored_total)

    return dict_paths


def process_dict_paths(dict_paths, consider_times_constraints=True):
    options = []
    origins = []
    destinations = []
    total_travel_time = []
    access = []
    egress = []
    nservices = []
    n_modes_p = []
    total_waiting_p = []
    total_cost_p = []
    total_emissions_p = []

    n_legs = 0
    for dp in dict_paths.values():
        for p in dp:
            if n_legs < len(p.path):
                n_legs = len(p.path)

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
            dict_legs_info['connecting_time_' + str(ln - 1) + "_" + str(ln)] = []
            dict_legs_info['waiting_time_' + str(ln - 1) + "_" + str(ln)] = []

        dict_legs_info['cost_' + str(ln)] = []
        dict_legs_info['emissions_' + str(ln)] = []

        ln += 1

    for od in dict_paths.keys():
        origin = od[0]
        destination = od[1]
        option = 0
        for p in dict_paths[od]:
            origins.append(origin)
            destinations.append(destination)
            options.append(option)
            total_travel_time.append(p.total_travel_time.total_seconds() / 60)
            access.append(p.access_time.total_seconds() / 60)
            egress.append(p.egress_time.total_seconds() / 60)
            nservices.append(len(p.path))
            ln = 0
            prev_arrival_time = None
            prev_mode = None
            total_waiting = None
            n_modes = 0
            for s in p.path:
                dict_legs_info['service_id_' + str(ln)].append(s.id)
                dict_legs_info['origin_' + str(ln)].append(s.origin)
                dict_legs_info['destination_' + str(ln)].append(s.destination)
                dict_legs_info['provider_' + str(ln)].append(s.provider)
                dict_legs_info['alliance_' + str(ln)].append(s.alliance)
                if p.layers_used[ln] != prev_mode:
                    prev_mode = p.layers_used[ln]
                    n_modes += 1
                dict_legs_info['mode_' + str(ln)].append(p.layers_used[ln])
                dict_legs_info['departure_time_' + str(ln)].append(s.departure_time)
                dict_legs_info['arrival_time_' + str(ln)].append(s.arrival_time)
                dict_legs_info['travel_time_' + str(ln)].append(s.duration.total_seconds() / 60)
                if ln > 0:
                    dict_legs_info['mct_time_' + str(ln - 1) + "_" + str(ln)].append(
                        p.mcts[ln - 1].total_seconds() / 60)

                    if not consider_times_constraints:
                        dict_legs_info['connecting_time_' + str(ln - 1) + "_" + str(ln)].append(None)
                        dict_legs_info['waiting_time_' + str(ln - 1) + "_" + str(ln)].append(None)
                        total_waiting = None
                    else:
                        dict_legs_info['connecting_time_' + str(ln - 1) + "_" + str(ln)].append(
                            (s.departure_time - prev_arrival_time).total_seconds() / 60)
                        dict_legs_info['waiting_time_' + str(ln - 1) + "_" + str(ln)].append(
                            (s.departure_time - prev_arrival_time).total_seconds() / 60 -
                            p.mcts[ln - 1].total_seconds() / 60)
                        if total_waiting is None:
                            total_waiting = (s.departure_time - prev_arrival_time).total_seconds() / 60 - p.mcts[
                                ln - 1].total_seconds() / 60
                        else:
                            total_waiting += (s.departure_time - prev_arrival_time).total_seconds() / 60 - p.mcts[
                                ln - 1].total_seconds() / 60

                dict_legs_info['cost_' + str(ln)].append(s.cost)
                dict_legs_info['emissions_' + str(ln)].append(None)

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
                    dict_legs_info['connecting_time_' + str(lni - 1) + "_" + str(lni)].append(None)
                    dict_legs_info['waiting_time_' + str(lni - 1) + "_" + str(lni)].append(None)
                dict_legs_info['cost_' + str(lni)].append(None)
                dict_legs_info['emissions_' + str(lni)].append(None)

            n_modes_p.append(n_modes)
            total_waiting_p.append(total_waiting)
            total_cost_p.append(None)
            total_emissions_p.append(None)

            option += 1

    dict_it = {'origin': origins,
               'destination': destinations,
               'option': options,
               'nservices': nservices,
               'total_travel_time': total_travel_time,
               'total_cost': total_cost_p,
               'total_emissions': total_emissions_p,
               'total_waiting_time': total_waiting_p,
               'nmodes': n_modes_p,
               'access_time': access,
               'egress_time': egress}

    dict_elements = {**dict_it, **dict_legs_info}

    df = pd.DataFrame(dict_elements)

    return df


def compute_possible_paths_network(network, o_d, pc=1, n_paths=10, max_connections=2,
                                   allow_mixed_operators=False, consider_times_constraints=True):
    if pc == 1:
        dict_paths = compute_paths(o_d,
                                   network,
                                   n_paths=n_paths,
                                   max_connections=max_connections,
                                   allow_mixed_operators=allow_mixed_operators,
                                   consider_times_constraints=consider_times_constraints)
    else:
        # Parallel computation of paths between o-d pairs
        prev_i = 0
        n_od_per_section = max(1, round(len(o_d) / pc))
        if n_od_per_section == 1:
            pc = len(o_d)
        i = n_od_per_section

        path_computation_param = []
        for nr in range(pc):
            if nr == pc - 1:
                i = len(o_d)

            d = o_d.iloc[prev_i:i].copy().reset_index(drop=True)

            if nr == 0:
                path_computation_param = [[d, network, n_paths, max_connections, allow_mixed_operators,
                                           consider_times_constraints]]
            else:
                if len(d) > 0:
                    path_computation_param.append([d, network, n_paths, max_connections, allow_mixed_operators,
                                                   consider_times_constraints])

            prev_i = i
            i = i + n_od_per_section

        pool = mp.Pool(processes=min(pc, len(path_computation_param)))

        print("   Launching parallel o-d path finding")

        res = pool.starmap(compute_paths, path_computation_param)

        pool.close()
        pool.join()

        dict_paths = {}
        for dictionary in res:
            dict_paths.update(dictionary)

    df_paths = process_dict_paths(dict_paths, consider_times_constraints=consider_times_constraints)
    print("In total", len(df_paths), " paths computed")
    return df_paths
