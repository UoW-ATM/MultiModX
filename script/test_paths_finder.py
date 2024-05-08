import time
import pandas as pd
from itertools import combinations
import sys
sys.path.insert(1, '..')
from strategic_evaluator.mobility_network import Service, NetworkLayer, Network
from strategic_evaluator.mobility_network_particularities import (mct_air_network,
                                                                  fastest_air_time_heuristic, initialise_air_network)


def create_example_air_layer(path_fs, path_airports):
    # Read air layer
    # Read schedules
    fs = pd.read_parquet(path_fs)

    # Define regions_access for air layer
    regions_access_air = {
        'Barcelona': [{'station': 'LEBL', 'access': {'all': 80, 'business': 60}, 'egress': {'all': 70, 'business': 40}},
                      {'station': 'LEGE', 'access': {'all': 120, 'business': 100},
                       'egress': {'all': 100, 'business': 90}},
                      {'station': 'LERS', 'access': {'all': 110, 'business': 90},
                       'egress': {'all': 90, 'business': 80}}],
        'Girona': [{'station': 'LEGE', 'access': {'all': 60}}],
        'Madrid': [{'station': 'LEMD', 'access': {'all': 80, 'business': 60}, 'egress': {'all': 70, 'business': 40}}]
    }

    # Read airport data (needed for the heuristic to compute GCD and
    # estimate time needed from current node to destination)
    airport_data = pd.read_parquet(path_airports)

    # Read MCTs between air services
    dict_mct_std = dict(zip(airport_data['icao_id'], airport_data['MCT_standard']))
    dict_mct_international = dict(zip(airport_data['icao_id'], airport_data['MCT_international']))
    dict_mct_domestic = dict(zip(airport_data['icao_id'], airport_data['MCT_domestic']))

    dict_mct = {'std': dict_mct_std, 'int': dict_mct_international, 'dom': dict_mct_domestic}

    # Create Services for flights (one per flight in the dataframe)
    fs['service_id'] = fs['nid']
    fs['departure_time'] = fs['sobt']
    fs['arrival_time'] = fs['sibt']
    fs['cost'] = 0
    fs['provider'] = fs['airline']
    fs['alliance'] = fs['airline']

    fs['service'] = fs.apply(lambda x: Service(x.nid, x.origin, x.destination, x.sobt, x.sibt, 0,
                                               x.airline, x.airline, gcdistance=x.gcdistance), axis=1)

    # Create network
    anl = NetworkLayer('air',
                       fs[['service_id', 'origin', 'destination', 'departure_time', 'arrival_time', 'cost', 'provider',
                           'alliance', 'service']],
                       dict_mct, regions_access=regions_access_air,
                       custom_mct_func=mct_air_network,
                       custom_heuristic_func=fastest_air_time_heuristic,
                       custom_initialisation=initialise_air_network,
                       airport_coordinates=airport_data[['icao_id', 'lat', 'lon']].copy())

    return anl

def example_only_air_layer(path_fs, path_airports):
    anl = create_example_air_layer(path_fs, path_airports)

    network = Network(layers=[anl])

    # Example of finding 100 paths between Madrid and Barcelona region
    start_time = time.time()
    paths, n_explored = network.find_paths(origin='Madrid', destination='Barcelona', npaths=100, max_connections=2)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Paths computed in:", elapsed_time, "seconds, exploring:", n_explored, "nodes and finding",
          len(paths), "paths")
    print(paths, "\n")


def create_example_rail_layer(stops_times_path, date_considered, stops_considered):
    # Read Rail data --> could be improved, e.g. filtering by day of operation, now just all possible trains in GFTS
    # as an example
    df_stops_times = pd.read_csv(stops_times_path)

    df_stops_times = df_stops_times[df_stops_times.stop_id.isin(stops_considered)].copy()
    n_stops_trip = df_stops_times.groupby('trip_id').count().reset_index()
    # Keep trips with at least two stops of interest in the trip
    df_stops_times = df_stops_times[
        df_stops_times.trip_id.isin(n_stops_trip[n_stops_trip.arrival_time > 1].trip_id)].copy()

    date = pd.to_datetime(date_considered, format='%d/%m/%Y')

    # Create rail services as all possible segments in the trips
    rail_services = []

    df_sorted = df_stops_times.sort_values(by=['trip_id', 'stop_sequence'])

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

    # Regions access rail
    regions_access_rail = {'Barcelona': [{'station': '71801', 'access': {'all': 40, 'business': 30},
                                          'egress': {'all': 30, 'business': 25}}],
                           'Madrid': [{'station': '60000', 'access': {'all': 40, 'business': 30},
                                       'egress': {'all': 30, 'business': 25}},
                                      {'station': '17000', 'access': {'all': 40, 'business': 30},
                                       'egress': {'all': 30, 'business': 25}},
                                      {'station': '10000', 'access': {'all': 40, 'business': 30},
                                       'egress': {'all': 30, 'business': 25}}
                                      ]
                           }
    nl_rail = NetworkLayer('rail', rail_services_df, regions_access=regions_access_rail)
    return nl_rail

def example_only_rail_layer(stops_times_path, date_considered = '09/11/2014'):
    # Manually defining rail stations of interest
    stops_considered = [71801, 61307, 61200, 62103, 60911, 65300, 71500, 17000, 37606, 37500, 35400, 18000, 13200,
                        11511, 11208, 4104, 78400, 4040, 81202, 81108, 81100, 80100, 20309, 22100, 60000, 4007, 70600,
                        22308, 11014, 14100, 15100, 20100, 20200, 31400, 30100, 10600, 10500, 51405, 60600, 50500,
                        51003, 51300, 3216, 79300, 54413, 37200, 5000, 43019, 74200, 8004, 8240, 15410, 15211, 14223,
                        11600, 8223, 23004, 51205, 3100, 65003, 62102, 79400, 71400, 78805, 70200, 10000, 67200]
    print("Considering", len(stops_considered), "rail stations")
    rail_layer = create_example_rail_layer(stops_times_path, date_considered, stops_considered)

    network = Network(layers=[rail_layer])

    # Example 100 paths between Barcelona 71801 and Madrid 60000
    start_time = time.time()
    paths, n_explored = network.find_paths(origin='71801', destination='60000', npaths=100, max_connections=2)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Paths computed in:", elapsed_time, "seconds, exploring:", n_explored, "nodes and finding",
          len(paths), "paths")
    print(paths, "\n")

    # Example 100 paths between Barcelona and Madrid region
    start_time = time.time()
    paths, n_explored = n.find_paths(origin='Barcelona', destination='Madrid', npaths=100, max_connections=2)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Paths computed in:", elapsed_time, "seconds, exploring:", n_explored, "nodes and finding",
          len(paths), "paths")
    print(paths, "\n")


def example_multilayer(path_fs, path_airports, stops_times_path, date_considered, ground_mobility_path):
    # Read air layer
    air_layer = create_example_air_layer(path_fs, path_airports)

    # Read rail layer
    # Manually defining rail stations of interest
    stops_considered = [71801, 61307, 61200, 62103, 60911, 65300, 71500, 17000, 37606, 37500, 35400, 18000, 13200,
                        11511, 11208, 4104, 78400, 4040, 81202, 81108, 81100, 80100, 20309, 22100, 60000, 4007, 70600,
                        22308, 11014, 14100, 15100, 20100, 20200, 31400, 30100, 10600, 10500, 51405, 60600, 50500,
                        51003, 51300, 3216, 79300, 54413, 37200, 5000, 43019, 74200, 8004, 8240, 15410, 15211, 14223,
                        11600, 8223, 23004, 51205, 3100, 65003, 62102, 79400, 71400, 78805, 70200, 10000, 67200]
    print("Considering", len(stops_considered), "rail stations")
    rail_layer = create_example_rail_layer(stops_times_path, date_considered, stops_considered)

    # Read air layer

    # Read ground mobility between layers
    gm = pd.read_parquet(
        ground_mobility_path)
    gm['origin_rail'] = pd.to_numeric(gm['origin'], errors='coerce')
    gm['destination_rail'] = pd.to_numeric(gm['destination'], errors='coerce')
    gm['layer_id_origin'] = None
    gm['layer_id_destination'] = None

    gm.loc[gm.origin_rail.notnull(), 'layer_id_origin'] = 'rail'
    gm.loc[gm.origin_rail.notnull(), 'layer_id_destination'] = 'air'
    gm.loc[gm.origin_rail.isnull(), 'layer_id_origin'] = 'air'
    gm.loc[gm.origin_rail.isnull(), 'layer_id_destination'] = 'rail'
    gm['mct'] = gm['mean'].apply(lambda x: {'all': x})

    transition_btw_layers = gm[['origin', 'destination', 'layer_id_origin', 'layer_id_destination', 'mct']]

    # Example intra layer transition
    # transition_btw_layers = pd.concat(
    #    [transition_btw_layers, pd.DataFrame({'origin': 'LEMD', 'destination': 'LEIB', 'layer_id_origin': 'air',
    #                                          'layer_id_destination': 'air',
    #                                          'mct': [{'all': 40}]})], ignore_index=True)


    print("Transitions considered")
    print(transition_btw_layers)

    network = Network(layers=[air_layer, rail_layer], transition_btw_layers=transition_btw_layers)

    # Example 50 paths between Barcelona and Madrid region
    start_time = time.time()
    paths, n_explored = network.find_paths(origin='Barcelona', destination='Madrid', npaths=50, max_connections=2)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Paths computed in:", elapsed_time, "seconds, exploring:", n_explored, "nodes and finding",
          len(paths), "paths")
    print(paths, "\n")

    # Example 50 paths between LEST and Barcelona region
    start_time = time.time()
    paths, n_explored = network.find_paths(origin='LEST', destination='Barcelona', npaths=50, max_connections=2)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Paths computed in:", elapsed_time, "seconds, exploring:", n_explored, "nodes and finding",
          len(paths), "paths")
    print(paths, "\n")


if __name__ == '__main__':
    # TODO consider pax types for access/mct/gm
    # TODO consider heuristic for rail
    # TODO trim options rail, e.g. avoid using same route in connection
    # TODO consider dates differences in rail/air services

    path_fs = '../tests/ATRS_conference/data/scenario=9/data/schedules/flight_schedule_old1409.parquet'
    path_airports = '../tests/ATRS_conference/data/scenario=9/data/airports/airport_info_static.parquet'
    example_only_air_layer(path_fs, path_airports)

    stops_times_path = '../tests/ATRS_conference/data/scenario=9/case_studies/case_study=-1/data/gtfs/gtfs/gtfs_es/stop_times.txt'
    date_considered = '09/11/2014'
    example_only_rail_layer(stops_times_path=stops_times_path, date_considered=date_considered)

    ground_mobility_path = '../tests/ATRS_conference/data/scenario=9/case_studies/case_study=-1/data/ground_mobility/connecting_times.parquet'
    example_multilayer(path_fs, path_airports, stops_times_path, date_considered, ground_mobility_path)


