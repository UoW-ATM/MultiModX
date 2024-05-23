from pathlib import Path
import argparse
import tomli
import pandas as pd

import sys
sys.path.insert(1, '../..')

from strategic_evaluator.strategic_evaluator import (create_network, preprocess_input,
                                                     compute_possible_itineraries_network,
                                                     compute_avg_paths_from_itineraries)


def read_origin_demand_matrix(path_demand):
    df_demand = pd.read_csv(Path(path_demand), keep_default_na=False)
    df_demand.replace('', None, inplace=True)
    return df_demand


def run(network_paths_config, pc=1, n_itineraries=10, max_connections=1, pre_processed_version=0,
        allow_mixed_operators=False,
        consider_time_constraints=True, use_heuristics_precomputed=False):

    # Preprocess input
    preprocess_input(network_paths_config['network_definition'])

    # Create network
    network = create_network(network_paths_config['network_definition'],
                             compute_simplified=not consider_time_constraints,
                             use_heuristics_precomputed=use_heuristics_precomputed,
                             pre_processed_version=pre_processed_version)

    # Read demand
    demand_matrix = read_origin_demand_matrix(network_paths_config['demand']['demand'])

    # Compute possible itineraries based on demand
    o_d = demand_matrix[['origin', 'destination']].drop_duplicates()

    df_itineraries = compute_possible_itineraries_network(network, o_d, pc, n_itineraries=n_itineraries,
                                                          max_connections=max_connections,
                                                          allow_mixed_operators=allow_mixed_operators,
                                                          consider_times_constraints=consider_time_constraints)

    if consider_time_constraints:
        ofp = 'possible_itineraries_' + str(pre_processed_version) + '.csv'
    else:
        ofp = 'potential_paths_' + str(pre_processed_version) + '.csv'

    df_itineraries.to_csv(Path(network_paths_config['output']['output_folder']) / ofp, index=False)

    if consider_time_constraints:
        # Compute average paths from possible itineraries
        df_avg_paths = compute_avg_paths_from_itineraries(df_itineraries)
        ofp = 'possible_paths_avg_' + str(pre_processed_version) + '.csv'
        df_avg_paths.to_csv(Path(network_paths_config['output']['output_folder']) / ofp, index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Strategic pipeline test', add_help=True)

    parser.add_argument('-tf', '--toml_file', help='TOML defining the network', required=True)
    parser.add_argument('-pc', '--n_proc', help='Number of processors', required=False)
    parser.add_argument('-df', '--demand_file', help='Pax demand file', required=False)
    parser.add_argument('-mo', '--allow_mixed_operators', help='Allow mix operators',
                        required=False, action='store_true')
    parser.add_argument('-ni', '--num_itinearies', help='Number of itineraries to find', required=False, default=50)
    parser.add_argument('-mc', '--max_connections', help='Number of connections allowed', required=False,
                        default=1)
    parser.add_argument('-cs', '--compute_simplified', help='Compute simplified network', required=False,
                        action='store_true')

    parser.add_argument('-hpc', '--use_heuristics_precomputed', help='Use heuristics precomputed based'
                                                                     ' on distance',
                        action='store_true', required=False)

    parser.add_argument('-ppv', '--preprocessed_version', help='Preprocessed version of schedules to use',
                        required=False, default=0)

    # Parse parameters
    args = parser.parse_args()

    with open(Path(args.toml_file), mode="rb") as fp:
        network_paths_config = tomli.load(fp)

    if args.demand_file is not None:
        network_paths_config['demand']['demand'] = Path(args.demand_file)

    pc = 1
    if args.n_proc is not None:
        pc = int(args.n_proc)

    if args.compute_simplified:
        args.allow_mixed_operators = True

    run(network_paths_config, pc=pc, n_itineraries=int(args.num_itinearies), max_connections=int(args.max_connections),
        pre_processed_version=int(args.preprocessed_version),
        allow_mixed_operators=args.allow_mixed_operators, consider_time_constraints=not args.compute_simplified,
        use_heuristics_precomputed=args.use_heuristics_precomputed)

    # Improvements
    # TODO: day in rail
    # TODO: heuristic in toml and enable different selection for different layers
    # TODO: precompute distance between rail stations
