from pathlib import Path
import argparse
import tomli
import pandas as pd

import sys
sys.path.insert(1, '../..')

from strategic_evaluator.strategic_evaluator import create_network, preprocess_input, compute_possible_paths_network


def read_origin_demand_matrix(path_demand):
    df_demand = pd.read_csv(Path(path_demand), keep_default_na=False)
    df_demand.replace('', None, inplace=True)
    return df_demand


def run(network_paths_config, pc=1, n_paths=10, max_connections=1, pre_processed_version=0,
        allow_mixed_operators=False,
        consider_time_constraints=True):

    # Preprocess input
    preprocess_input(network_paths_config['network_definition'])

    # Create network
    network = create_network(network_paths_config['network_definition'],
                             compute_simplified=args.compute_simplified,
                             use_heuristics_precomputed=args.use_heuristics_precomputed,
                             pre_processed_version=pre_processed_version)

    # Read demand
    demand_matrix = read_origin_demand_matrix(network_paths_config['demand']['demand'])

    # Compute possible paths based on demand
    o_d = demand_matrix[['origin', 'destination']].drop_duplicates()

    df_paths = compute_possible_paths_network(network, o_d, pc, n_paths=n_paths, max_connections=max_connections,
                                              allow_mixed_operators=allow_mixed_operators,
                                              consider_times_constraints=consider_time_constraints)

    ofp = 'possible_paths_'+str(pre_processed_version)+'.csv'
    df_paths.to_csv(Path(network_paths_config['output']['output_folder']) / ofp, index=False)


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

    run(network_paths_config, pc=pc, n_paths=int(args.num_paths), max_connections=int(args.max_connections),
        pre_processed_version=int(args.preprocessed_version),
        allow_mixed_operators=args.allow_mixed_operators, consider_time_constraints=not args.compute_simplified)

    # Improvements
    # TODO: day in rail
    # TODO: heuristic in toml and enable different selection for different layers
    # TODO: precompute distance between rail stations
