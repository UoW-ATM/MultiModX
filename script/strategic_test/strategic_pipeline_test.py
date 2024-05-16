import time
from pathlib import Path
import argparse
import tomli
import pandas as pd
from itertools import combinations
import multiprocessing as mp

import sys
sys.path.insert(1, '../..')

from strategic_evaluator.mobility_network import Service, NetworkLayer, Network
from strategic_evaluator.mobility_network_particularities import (mct_air_network, fastest_air_time_heuristic,
                                                                  initialise_air_network, mct_rail_network)
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


def create_air_layer(df_fs, df_as, df_mct, df_ra_air=None, keep_only_fastest_service=0):

    # Regions access -- Create dictionary
    regions_access_air = None
    if df_ra_air is not None:
        regions_access_air = create_region_access_dict(df_ra_air)

    #  MCTs between air services
    dict_mct_std = dict(zip(df_mct['icao_id'], df_mct['standard']))
    dict_mct_international = dict(zip(df_mct['icao_id'], df_mct['international']))
    dict_mct_domestic = dict(zip(df_mct['icao_id'], df_mct['domestic']))
    dict_mct = {'std': dict_mct_std, 'int': dict_mct_international, 'dom': dict_mct_domestic}

    # Create Services for flights (one per flight in the dataframe)
    df_fs['cost'] = 0
    df_fs['service'] = df_fs.apply(lambda x: Service(x.service_id, x.origin, x.destination, x.sobt, x.sibt,  x.cost,
                                                     x.provider, x.alliance, gcdistance=x.gcdistance), axis=1)
    df_fs.rename(columns={'sobt': 'departure_time', 'sibt': 'arrival_time'}, inplace=True)

    if keep_only_fastest_service > 0:
        df_fs['duration'] = None
        df_fs['duration'] = df_fs.apply(lambda x: x.arrival_time - x.departure_time, axis=1)
        df_s_min_duration = df_fs.groupby(['origin', 'destination'])['duration'].min().reset_index()
        df_fs = pd.merge(df_s_min_duration, df_fs, on=['origin', 'destination', 'duration'], how='inner')

    if keep_only_fastest_service == 2:
        # Keep only one fastest regardless of airline/alliance
        df_fs.drop_duplicates(subset=['origin', 'destination'], keep='first', inplace=True)

    # Create network
    anl = NetworkLayer('air',
                       df_fs[['service_id', 'origin', 'destination', 'departure_time', 'arrival_time', 'cost',
                              'provider', 'alliance', 'service']].copy(),
                       dict_mct, regions_access=regions_access_air,
                       custom_mct_func=mct_air_network,
                       custom_heuristic_func=fastest_air_time_heuristic,
                       custom_initialisation=initialise_air_network,
                       airport_coordinates=df_as.copy(),
                       keep_only_fastest_service=keep_only_fastest_service)

    return anl


def create_rail_layer(df_stop_times, date_considered='01/01/2024', df_stops_considered=None, df_ra_rail=None):

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

    nl_rail = NetworkLayer('rail', rail_services_df, regions_access=regions_access_rail,
                           custom_mct_func=mct_rail_network)
    return nl_rail


def create_networks(path_network_dict, compute_simplified=False):
    df_ra = None
    df_transitions = None
    layers = []
    network = None

    if 'regions_access' in path_network_dict.keys():
        df_ra = pd.read_csv(Path(path_network_dict['regions_access']['regions_access']))

    if 'air_network' in path_network_dict.keys():
        df_fs = pd.read_csv(Path(path_network_dict['air_network']['flight_schedules']), keep_default_na=False)
        df_fs.replace('', None, inplace=True)
        df_fs['alliance'].fillna(df_fs['provider'], inplace=True)
        if 'gcdistance' not in df_fs.columns:
            df_fs['gcdistance'] = None
        if 'provider' not in df_fs.columns:
            df_fs['provider'] = None
        if 'alliance' not in df_fs.columns:
            df_fs['alliance'] = None

        df_fs['sobt'] = pd.to_datetime(df_fs['sobt'],  format='%Y-%m-%d %H:%M:%S')
        df_fs['sibt'] = pd.to_datetime(df_fs['sibt'],  format='%Y-%m-%d %H:%M:%S')

        # Read airport data (needed for the heuristic to compute GCD and
        # estimate time needed from current node to destination)
        df_as = pd.read_csv(Path(path_network_dict['air_network']['airports_static']))

        # Read MCTs between air services
        df_mct = pd.read_csv(Path(path_network_dict['air_network']['mct_air']))

        df_ra_air = None
        if df_ra is not None:
            df_ra_air = df_ra[df_ra['layer'] == 'air'].copy().reset_index(drop=True)
            if len(df_ra_air) == 0:
                df_ra_air = None

        only_fastest = 0
        if compute_simplified:
            only_fastest = 2

        air_layer = create_air_layer(df_fs.copy(), df_as, df_mct, df_ra_air, keep_only_fastest_service=only_fastest)
        layers += [air_layer]

    if 'rail_network' in path_network_dict.keys():
        # TODO: fix rail dates
        date_rail = '20230503'
        date_rail = pd.to_datetime(date_rail, format='%Y%m%d')
        df_stop_times = pd.read_csv(Path(path_network_dict['rail_network']['gtfs']) / 'stop_times.txt')
        df_trips = pd.read_csv(Path(path_network_dict['rail_network']['gtfs']) / 'trips.txt')
        df_calendar = pd.read_csv(Path(path_network_dict['rail_network']['gtfs']) / 'calendar.txt')
        df_calendar['start_date'] = pd.to_datetime(df_calendar['start_date'], format='%Y%m%d')
        df_calendar['end_date'] = pd.to_datetime(df_calendar['end_date'], format='%Y%m%d')
        df_calendar_dates = pd.read_csv(Path(path_network_dict['rail_network']['gtfs']) / 'calendar_dates.txt')
        df_calendar_dates['date'] = pd.to_datetime(df_calendar_dates['date'], format='%Y%m%d')
        df_stop_times = get_stop_times_on_date(date_rail, df_calendar, df_calendar_dates, df_trips, df_stop_times)

        df_stops_considered = None
        if 'rail_stations_considered' in path_network_dict['rail_network'].keys():
            df_stops_considered = pd.read_csv(Path(path_network_dict['rail_network']['rail_stations_considered']))
            df_stop_times = df_stop_times[df_stop_times.stop_id.isin(df_stops_considered.stop_id)]

        if 'regions_access' in path_network_dict.keys():
            df_ra = pd.read_csv(Path(path_network_dict['regions_access']['regions_access']))

        df_ra_rail = None
        if df_ra is not None:
            df_ra_rail = df_ra[df_ra['layer'] == 'rail'].copy().reset_index(drop=True)
            if len(df_ra_rail) == 0:
                df_ra_rail = None

        if compute_simplified:
            # TODO: rail_layer_simp
            pass

        rail_layer = create_rail_layer(df_stop_times, date_considered='12/09/2014',
                                       df_stops_considered=df_stops_considered,
                                       df_ra_rail=df_ra_rail)

        layers += [rail_layer]

    if 'multimodal' in path_network_dict.keys():
        df_transitions = pd.read_csv(Path(path_network_dict['multimodal']['air_rail_transitions']))

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


def read_origin_demand_matrix(path_demand):
    df_demand = pd.read_csv(Path(path_demand), keep_default_na=False)
    df_demand.replace('', None, inplace=True)
    return df_demand


def process_outcome(dict_paths):
    options = []
    origins = []
    destinations = []
    total_travel_time = []
    access = []
    egress = []
    nservices = []
    n_modes_p = []
    total_waiting_p = []

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
                if p.layers_used[ln - 1] != prev_mode:
                    prev_mode = p.layers_used[ln - 1]
                    n_modes += 1
                dict_legs_info['mode_' + str(ln)].append(p.layers_used[ln - 1])
                dict_legs_info['departure_time_' + str(ln)].append(s.departure_time)
                dict_legs_info['arrival_time_' + str(ln)].append(s.arrival_time)
                dict_legs_info['travel_time_' + str(ln)].append(s.duration.total_seconds() / 60)
                if ln > 0:
                    dict_legs_info['mct_time_' + str(ln - 1) + "_" + str(ln)].append(
                        p.mcts[ln - 1].total_seconds() / 60)
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

            option += 1

    dict_it = {'origin': origins,
               'destination': destinations,
               'option': options,
               'total_travel_time': total_travel_time,
               'nservices': nservices,
               'total_waiting_time': total_waiting_p,
               'nmodes': n_modes_p,
               'access_time': access,
               'egress_time': egress}

    dict_elements = {**dict_it, **dict_legs_info}

    df = pd.DataFrame(dict_elements)

    return df


def compute_paths(od_paths, network, n_path=10, max_connections=2, allow_mixed_operators=False):
    dict_paths = {}
    start_time = time.time()

    n_explored_total = 0
    for i, od in od_paths.iterrows():
        start_time_od = time.time()
        same_operators = not allow_mixed_operators
        paths, n_explored = network.find_paths(origin=od.origin, destination=od.destination, npaths=n_path,
                                               max_connections=max_connections,
                                               consider_operators_connections=same_operators)
        dict_paths[(od.origin, od.destination)] = paths
        n_explored_total += n_explored
        end_time_od = time.time()
        print("Paths for", od.origin, "-", od.destination,
              ", computed in, ", (end_time_od - start_time_od), " seconds, exploring:", n_explored,
              "found", len(paths), "paths.\n")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Paths computed in:", elapsed_time, "seconds, exploring:", n_explored_total)

    return dict_paths


def run(path_network_dict, path_demand, pc=1, n_path=50, max_connections=2, allow_mixed_operators=False,
        compute_simplified=False):

    network = create_networks(path_network_dict, compute_simplified=compute_simplified)

    demand_matrix = read_origin_demand_matrix(path_demand)

    od_paths = demand_matrix[['origin', 'destination']].drop_duplicates()

    if pc == 1:
        dict_paths = compute_paths(od_paths,
                                   network,
                                   n_path=n_path,
                                   max_connections=max_connections,
                                   allow_mixed_operators=allow_mixed_operators)
    else:
        # Parallel computation of paths between o-d pairs
        prev_i = 0
        n_od_per_section = max(1, round(len(od_paths) / pc))
        if n_od_per_section == 1:
            pc = len(od_paths)
        i = n_od_per_section

        path_computation_param = []
        for nr in range(pc):
            if nr == pc - 1:
                i = len(od_paths)

            d = od_paths.iloc[prev_i:i].copy().reset_index(drop=True)

            if nr == 0:
                path_computation_param = [[d, network, n_path, max_connections, allow_mixed_operators]]
            else:
                if len(d) > 0:
                    path_computation_param.append([d, network, n_path, max_connections, allow_mixed_operators])

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

    df_paths = process_outcome(dict_paths)
    print("In total", len(df_paths), " paths computed")
    return df_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Strategic pipeline test', add_help=True)

    parser.add_argument('-tf', '--toml_file', help='TOML defining the network', required=True)
    parser.add_argument('-pc', '--n_proc', help='Number of processors', required=False)
    parser.add_argument('-df', '--demand_file', help='Pax demand file', required=False)
    parser.add_argument('-mo', '--allow_mixed_operators', help='Allow mix operators',
                        required=False, action='store_true')
    parser.add_argument('-np', '--num_paths', help='Number of paths to find', required=False, default=50)
    parser.add_argument('-mc', '--max_connections', help='Number of connections allowed', required=False,
                        default=1)
    parser.add_argument('-cs', '--compute_simplified', help='Compute simplified network', required=False,
                        action='store_true')

    # Parse parameters
    args = parser.parse_args()

    with open(Path(args.toml_file), mode="rb") as fp:
        network_paths_config = tomli.load(fp)

    if args.demand_file is not None:
        network_paths_config['demand']['demand'] = Path(args.demand_file)

    pc = 1
    if args.n_proc is not None:
        pc = int(args.n_proc)

    df_paths = run(network_paths_config['network_definition'],
                   network_paths_config['demand']['demand'], pc=pc, allow_mixed_operators=args.allow_mixed_operators,
                   n_path=int(args.num_paths), max_connections=int(args.max_connections),
                   compute_simplified=args.compute_simplified)

    df_paths.to_csv(Path(network_paths_config['output']['output_folder']) /
                    network_paths_config['output']['output_df_file'])

    # Improvements
    # TODO: day in rail
    # TODO: all direct services to be provided (note that now it might find all direct air before rail,
    #       even if rail shorter, to check why)
    # TODO: don't change trains that go to the same destination on the same route
    # TODO: heuristic on rail to speed up search
    # TODO: factor of worsening w.r.t. fastest
    # TODO: simplify rail layer to find graph of alternatives regardless of services times
