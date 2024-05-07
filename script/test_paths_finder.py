import time
import pandas as pd
import sys
sys.path.insert(1, '..')
from strategic_evaluator.mobility_network import Service, NetworkLayer, Network
from strategic_evaluator.mobility_network_particularities import (mct_air_network,
                                                                  fastest_air_time_heuristic, initialise_air_network)




if __name__ == '__main__':

    # Read air layer

    fs = pd.read_parquet('../tests/ATRS_conference/data/scenario=9/data/schedules/flight_schedule_old1409.parquet')

    regions_access_air = {'Barcelona': [{'station':'LEBL', 'access':{'all':80, 'business':60}, 'egress':{'all':70, 'business': 40}},
                                        {'station':'LEGE', 'access':{'all':120, 'business':100}, 'egress':{'all':100, 'business': 90}},
                                        {'station':'LERS', 'access':{'all':110, 'business':90}, 'egress':{'all':90, 'business': 80}}],
                          'Girona': [{'station': 'LEGE', 'access':{'all':60}}],
                          'Madrid': [{'station':'LEMD', 'access':{'all':80, 'business':60}, 'egress':{'all':70, 'business': 40}}]
                          }

    airport_data = pd.read_parquet('../tests/ATRS_conference/data/scenario=9/data/airports/airport_info_static.parquet')
    # MCTs
    # air
    dict_mct_std = dict(zip(airport_data['icao_id'], airport_data['MCT_standard']))
    dict_mct_international = dict(zip(airport_data['icao_id'], airport_data['MCT_international']))
    dict_mct_domestic = dict(zip(airport_data['icao_id'], airport_data['MCT_domestic']))

    dict_mct = {'std': dict_mct_std, 'int': dict_mct_international, 'dom':dict_mct_domestic}

    fs['service_id'] = fs['nid']
    fs['departure_time'] = fs['sobt']
    fs['arrival_time'] = fs['sibt']
    fs['cost'] = 0
    fs['provider'] = fs['airline']
    fs['alliance'] = fs['airline']

    fs['service'] = fs.apply(lambda x: Service(x.nid, x.origin, x.destination, x.sobt, x.sibt, 0,
                                               x.airline, x.airline, gcdistance=x.gcdistance), axis=1)

    #fs['service'] = fs.apply(lambda x: Service(x.nid, x.origin, x.destination, x.sobt, x.sibt, 0,
    #                                           "A", "A", gcdistance=x.gcdistance), axis=1)

    # Read rail layer
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

    # Read ground mobility between layers
    gm = pd.read_parquet(
        '../tests/ATRS_conference/data/scenario=9/case_studies/case_study=-1/data/ground_mobility/connecting_times.parquet')
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

    #transition_btw_layers = pd.concat(
    #    [transition_btw_layers, pd.DataFrame({'origin': 'LEMD', 'destination': 'LEIB', 'layer_id_origin': 'air',
    #                                          'layer_id_destination': 'rail',
    #                                          'mct': [{'all': 40}]})], ignore_index=True)


    # Create network

    # TODO pax types for access/mct/gm

    anl = NetworkLayer('air',
                       fs[['service_id', 'origin', 'destination', 'departure_time', 'arrival_time', 'cost', 'provider',
                           'alliance', 'service']],
                       dict_mct, regions_access=regions_access_air,
                       custom_mct_func=mct_air_network,
                       #custom_heuristic_func=fastest_air_time_heuristic,
                       custom_initialisation=initialise_air_network,
                       airport_coordinates=airport_data[['icao_id', 'lat', 'lon']].copy())

    anl2 = NetworkLayer('rail',
                        fs[['service_id', 'origin', 'destination', 'departure_time', 'arrival_time', 'cost', 'provider',
                            'alliance', 'service']], dict_mct, regions_access=regions_access_air,
                        custom_mct_func=mct_air_network,
                        custom_heuristic_func=fastest_air_time_heuristic,
                        custom_initialisation=initialise_air_network,
                        airport_coordinates=airport_data[['icao_id', 'lat', 'lon']].copy())

    n = Network(layers=[anl], transition_btw_layers=transition_btw_layers)



    # Record start time
    start_time = time.time()

    paths, n_epxlored = n.find_paths(origin='Madrid', destination='Barcelona', npaths=100, max_connections=2)
    print("Explored:", n_epxlored, "nodes")
    print("Found", len(paths), "paths")
    print(paths)

    # Record end time
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time

    print("Elapsed time:", elapsed_time, "seconds")


