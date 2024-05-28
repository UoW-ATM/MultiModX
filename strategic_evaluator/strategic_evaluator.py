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

from libs.gtfs import get_stop_times_on_date, add_date_and_handle_overflow


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


def create_rail_layer(df_rail, from_gtfs=True, date_considered='20240101', df_stops_considered=None, df_ra_rail=None,
                      keep_only_fastest_service=False,
                      df_stops=None,
                      heuristic_precomputed_distance=None):

    if from_gtfs:
        # Process GTFS to create services
        if type(date_considered) is str:
            date_considered = pd.to_datetime(date_considered, format='%Y%m%d')
        rail_services_df = pre_process_rail_gtfs_to_services(df_rail, date_considered)
    else:
        rail_services_df = df_rail

    # Regions access -- Create dictionary
    regions_access_rail = None
    if df_ra_rail is not None:
        regions_access_rail = create_region_access_dict(df_ra_rail)

    if df_stops_considered is not None:
        rail_services_df = rail_services_df[(rail_services_df.origin.isin(df_stops_considered.stop_id)) &
                                            (rail_services_df.destination.isin(df_stops_considered.stop_id))].copy()

    # Create Service objects
    rail_services_df['service'] = rail_services_df.apply(lambda x:
                                                         Service(x.service_id, x.origin, x.destination,
                                                                 x.departure_time, x.arrival_time,
                                                                 x.cost, x.provider, x.alliance, x.emissions, x.seats), axis=1)


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

    df_fs['alliance'].fillna(df_fs['provider'], inplace=True)

    if 'alliances' in air_network.keys():
        # Replace alliance for the alliances defined in the alliances.csv file
        df_alliances = pd.read_csv(Path(path_network) / air_network['alliances'])
        df_fs = df_fs.merge(df_alliances, how='left', on='provider', suffixes=("","_alliance"))
        df_fs.loc[~df_fs.alliance_alliance.isna(), 'alliance'] = df_fs.loc[~df_fs.alliance_alliance.isna()]['alliance_alliance']
        df_fs = df_fs.drop(['alliance_alliance'], axis=1)

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
    # TODO: filter by parent stations
    date_rail = '20230503'
    date_rail = pd.to_datetime(date_rail, format='%Y%m%d')
    df_stop_times = get_stop_times_on_date(date_rail, df_calendar, df_calendar_dates, df_trips, df_stop_times)

    if 'rail_stations_considered' in rail_network.keys():
        df_stops_considered = pd.read_csv(Path(path_network) / rail_network['rail_stations_considered'])
        df_stop_times = df_stop_times[df_stop_times.stop_id.isin(df_stops_considered.stop_id)]

    # TODO: add provider, cost, emissions...
    rail_provider = 'renfe'
    rail_alliance = 'renfe'
    if 'provider' not in df_stop_times.columns:
        df_stop_times['provider'] = rail_provider
    if 'alliance' not in df_stop_times.columns:
        df_stop_times['alliance'] = rail_alliance

    frail = 'rail_timetable_proc_gtfs_'+str(pre_processed_version)+'.csv'
    # Save information in GTFS form
    df_stop_times.to_csv(Path(path_network) / processed_folder / frail, index=False)

    # Keep processing to translate GTFS form to 'Services' form
    pre_process_rail_gtfs_to_services(df_stop_times, date_rail, (Path(path_network) / processed_folder),
                                      pre_processed_version, save_services=True)


def pre_process_rail_gtfs_to_services(df_stop_times, date_rail, path_folder=None, pre_processed_version=0, save_services=False):

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
    rail_seats = None
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

            service_df_row = (
                service_id, str(origin), str(destination), departure_time, arrival_time, rail_provider, rail_alliance,
                rail_cost, rail_seats, rail_emissions)

            rail_services.append(service_df_row)

    rail_services_df = pd.DataFrame(rail_services,
                                    columns=['service_id', 'origin', 'destination', 'departure_time', 'arrival_time',
                                             'provider', 'alliance', 'cost', 'seats', 'emissions'])

    if save_services:
        frail = ('rail_timetable_proc_') + str(pre_processed_version) + '.csv'
        rail_services_df.to_csv(path_folder / frail, index=False)

    return rail_services_df


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
        # TODO: deal with date_considered
        date_rail_str = '20140912'
        date_rail = pd.to_datetime(date_rail_str, format='%Y%m%d')

        if path_network_dict['rail_network'].get('create_rail_layer_from') == 'gtfs':
            # Create the services file (regarless if it exists or not) and then process downstream as from services
            fstops_filename = 'rail_timetable_proc_gtfs_' + str(pre_processed_version) + '.csv'
            df_stop_times = pd.read_csv(Path(path_network_dict['network_path']) / path_network_dict['processed_folder'] /
                                       fstops_filename, keep_default_na=False, na_values=[''])

            df_rail_data = pre_process_rail_gtfs_to_services(df_stop_times, date_rail,
                                              (Path(path_network_dict['network_path']) / path_network_dict['processed_folder']),
                                              pre_processed_version=pre_processed_version, save_services=True)

        else:
            fstops_filename = 'rail_timetable_proc_' + str(pre_processed_version) + '.csv'

            df_rail_data = pd.read_csv(Path(path_network_dict['network_path']) / path_network_dict['processed_folder'] /
                                       fstops_filename, keep_default_na=False, na_values=[''])

            df_rail_data = df_rail_data.applymap(lambda x: None if pd.isna(x) else x)

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

        rail_layer = create_rail_layer(df_rail_data, from_gtfs=False, date_considered=date_rail_str,
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
        print("Computing it for:", od.origin, od.destination)
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
        print("Itineraries for", od.origin, "-", od.destination,
              ", computed in, ", (end_time_od - start_time_od), " seconds, exploring:", n_explored,
              "nodes. Found", len(itineraries), "itineraries.\n")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Itineraries computed in:", elapsed_time, "seconds, exploring:", n_explored_total)

    return dict_itineraries


def process_dict_itineraries(dict_itineraries, consider_times_constraints=True):
    options = []
    origins = []
    destinations = []
    total_travel_time = []
    access = []
    egress = []
    nservices = []
    n_modes_i = []
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
            dict_legs_info['connecting_time_' + str(ln - 1) + "_" + str(ln)] = []
            dict_legs_info['waiting_time_' + str(ln - 1) + "_" + str(ln)] = []

        dict_legs_info['cost_' + str(ln)] = []
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
            nservices.append(len(i.itinerary))
            ln = 0
            prev_arrival_time = None
            prev_mode = None
            total_waiting = None
            path = None
            n_modes = 0
            for s in i.itinerary:
                if consider_times_constraints:
                    dict_legs_info['service_id_' + str(ln)].append(s.id)
                else:
                    dict_legs_info['service_id_' + str(ln)].append(None)
                dict_legs_info['origin_' + str(ln)].append(s.origin)
                dict_legs_info['destination_' + str(ln)].append(s.destination)
                # TODO consider_time_constarint but yes provider...
                if consider_times_constraints:
                    dict_legs_info['provider_' + str(ln)].append(s.provider)
                    dict_legs_info['alliance_' + str(ln)].append(s.alliance)
                else:
                    dict_legs_info['provider_' + str(ln)].append(None)
                    dict_legs_info['alliance_' + str(ln)].append(None)

                if i.layers_used[ln] != prev_mode:
                    prev_mode = i.layers_used[ln]
                    n_modes += 1
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

            n_modes_i.append(n_modes)
            total_waiting_i.append(total_waiting)
            if path is not None:
                paths_i.append([v for i, v in enumerate(path) if i == 0 or v != path[i-1]]) # remove consecutive same node
            else:
                paths_i.append(None)
            total_cost_i.append(None)
            total_emissions_p.append(None)


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
               'access_time': access,
               'egress_time': egress}

    dict_elements = {**dict_it, **dict_legs_info}

    df = pd.DataFrame(dict_elements)

    columns_to_keep_regardless_none = ['total_cost', 'total_emissions', 'total_waiting_time']
    columns_to_keep_or_not_all_none = [col for col in df.columns if col in columns_to_keep_regardless_none
                                       or df[col].notna().any()]

    return df[columns_to_keep_or_not_all_none]


def compute_possible_itineraries_network(network, o_d, dict_o_d_routes=None, pc=1, n_itineraries=10,
                                         max_connections=2, allow_mixed_operators=False,
                                         consider_times_constraints=True):

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

        print("   Launching parallel o-d itinearies finding")

        res = pool.starmap(compute_itineraries, itineraries_computation_param)

        pool.close()
        pool.join()

        dict_itinearies = {}
        for dictionary in res:
            dict_itinearies.update(dictionary)

    df_itineraries = process_dict_itineraries(dict_itinearies, consider_times_constraints=consider_times_constraints)
    print("In total", len(df_itineraries), " itineraries computed")
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
                       col.startswith('latest_arrival_time')]

    df_paths['path'] = df_paths['path'].apply(str)
    df_paths_avg = df_paths.groupby(['origin', 'destination', 'path'], as_index=False).agg(lambda x: x.mean() if x.name not in exclude_columns else x.iloc[0])

    df_paths_avg = df_paths_avg.groupby(['origin', 'destination']).apply(lambda x: x.sort_values(by='total_travel_time', ascending=True)).reset_index(drop=True)

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

    df_paths_avg['path'] = df_paths_avg['path'].apply(eval)

    return df_paths_avg

