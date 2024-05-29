from pathlib import Path
import argparse
import tomli
import pandas as pd
from collections import defaultdict
import ast
import logging
import sys
sys.path.insert(1, '../..')


# Define custom logging levels
IMPORTANT_INFO = 25  # Between INFO (20) and WARNING (30)


def important_info(self, message, *args, **kwargs):
    if self.isEnabledFor(IMPORTANT_INFO):
        self._log(IMPORTANT_INFO, message, args, **kwargs)


def setup_logging(verbosity):
    levels = [logging.WARNING, IMPORTANT_INFO, logging.INFO, logging.DEBUG]
    level = levels[min(len(levels) - 1, verbosity)]  # Ensure the level does not exceed DEBUG
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    logging.debug(f"Logging level set to {logging.getLevelName(level)}")


def read_origin_demand_matrix(path_demand):
    df_demand = pd.read_csv(Path(path_demand), keep_default_na=False)
    df_demand.replace('', None, inplace=True)
    return df_demand


def run(network_paths_config, pc=1, n_itineraries=10, max_connections=1, pre_processed_version=0,
        allow_mixed_operators=False,
        consider_time_constraints=True, use_heuristics_precomputed=False, use_potential_paths=False):

    # Preprocess input
    preprocess_input(network_paths_config['network_definition'], pre_processed_version=pre_processed_version)

    # Create network
    network = create_network(network_paths_config['network_definition'],
                             compute_simplified=not consider_time_constraints,
                             use_heuristics_precomputed=use_heuristics_precomputed,
                             pre_processed_version=pre_processed_version)

    # Read demand
    demand_matrix = read_origin_demand_matrix(network_paths_config['demand']['demand'])

    # Compute possible itineraries based on demand
    o_d = demand_matrix[['origin', 'destination']].drop_duplicates()

    # If use_potential_paths then read potential paths and create dictionary
    dict_o_d_routes = None
    if use_potential_paths:
        # Read path of potential paths:
        df_pp = pd.read_csv((Path(network_paths_config['network_definition']['network_path'])/
               network_paths_config['network_definition']['potential_paths']))

        df_pp['path'] = df_pp['path'].apply(ast.literal_eval)
        dict_o_d_routes = defaultdict(list)

        for _, row in df_pp.iterrows():
            key = (row['origin'], row['destination'])
            dict_o_d_routes[key].append(row['path'])

        dict_o_d_routes = dict(dict_o_d_routes)

    df_itineraries = compute_possible_itineraries_network(network, o_d, dict_o_d_routes=dict_o_d_routes,
                                                          pc=pc, n_itineraries=n_itineraries,
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


def run_two_step(network_paths_config, pc=1, n_paths=15, n_itineraries=50,
                     max_connections=1, pre_processed_version=0,
                     allow_mixed_operators_itineraries=False,
                     use_heuristics_precomputed=False):

    # Preprocess input
    logger.info("Pre-processing input")
    preprocess_input(network_paths_config['network_definition'], pre_processed_version=pre_processed_version)

    # Read demand
    logger.info("Reading demand")
    demand_matrix = read_origin_demand_matrix(network_paths_config['demand']['demand'])

    # Compute possible itineraries based on demand
    o_d = demand_matrix[['origin', 'destination']].drop_duplicates()


    # First compute potential paths
    # Create network
    logger.info("Create network simplified to compute paths")
    network = create_network(network_paths_config['network_definition'],
                             compute_simplified=True,
                             use_heuristics_precomputed=use_heuristics_precomputed,
                             pre_processed_version=pre_processed_version)

    # Compute potential paths
    logger.info("Computing potential paths")
    df_potential_paths = compute_possible_itineraries_network(network, o_d, pc=pc, n_itineraries=n_paths,
                                                          max_connections=max_connections,
                                                          allow_mixed_operators=True,
                                                          consider_times_constraints=False)

    ofp = 'potential_paths_' + str(pre_processed_version) + '.csv'
    df_potential_paths.to_csv(Path(network_paths_config['output']['output_folder']) / ofp, index=False)

    # Then compute itineraries based on potential paths
    # Create potential routes dictionary
    logger.info("Create dictionary of potential routes to use by itineraries")
    dict_o_d_routes = defaultdict(list)

    for _, row in df_potential_paths.iterrows():
        key = (row['origin'], row['destination'])
        dict_o_d_routes[key].append(row['path'])

    dict_o_d_routes = dict(dict_o_d_routes)

    # Create network
    logger.info("Create network to compute itineraries")
    network = create_network(network_paths_config['network_definition'],
                             compute_simplified=False,
                             use_heuristics_precomputed=use_heuristics_precomputed,
                             pre_processed_version=pre_processed_version)

    # Compute itineraries
    logger.info("Computing potential paths")
    df_itineraries = compute_possible_itineraries_network(network, o_d, dict_o_d_routes=dict_o_d_routes,
                                                          pc=pc, n_itineraries=n_itineraries,
                                                          max_connections=max_connections,
                                                          allow_mixed_operators=allow_mixed_operators_itineraries,
                                                          consider_times_constraints=True)

    ofp = 'possible_itineraries_' + str(pre_processed_version) + '.csv'
    df_itineraries.to_csv(Path(network_paths_config['output']['output_folder']) / ofp, index=False)

    # Compute average paths from possible itineraries
    logger.info("Compute average path for possible itineraries")
    df_avg_paths = compute_avg_paths_from_itineraries(df_itineraries)
    ofp = 'possible_paths_avg_' + str(pre_processed_version) + '.csv'
    df_avg_paths.to_csv(Path(network_paths_config['output']['output_folder']) / ofp, index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Strategic pipeline test', add_help=True)

    parser.add_argument('-tf', '--toml_file', help='TOML defining the network', required=True)
    parser.add_argument('-ni', '--num_itineraries', help='Number of itineraries to find', required=False,
                        default=50)
    parser.add_argument('-mc', '--max_connections', help='Number of connections allowed', required=False,
                        default=1)
    parser.add_argument('-hpc', '--use_heuristics_precomputed', help='Use heuristics precomputed based on'
                                                                     ' distance', action='store_true', required=False)
    parser.add_argument('-ppv', '--preprocessed_version', help='Preprocessed version of schedules to use',
                        required=False, default=0)
    parser.add_argument('-cpci', '--compute_potential_and_itineraries', help='First compute potential paths'
                                                                             'and then itineraries on those paths',
                        action='store_true', required=False)
    parser.add_argument('-np', '--num_paths', help='Number of paths to compute if computing potential paths',
                        required=False, default=30)
    parser.add_argument('-v', '--verbose', action='count', default=0, help="increase output verbosity")

    parser.add_argument('-pc', '--n_proc', help='Number of processors', required=False)
    parser.add_argument('-df', '--demand_file', help='Pax demand file instead of the one in the toml_file',
                        required=False)

    parser.add_argument('-mo', '--allow_mixed_operators', help='Allow mix operators',
                        required=False, action='store_true')

    parser.add_argument('-cpp', '--compute_potential_paths', help='Compute only potential paths',
                        required=False, action='store_true')
    parser.add_argument('-upp', '--use_potential_paths', help='Compute itineraries from list of potential '
                                                              'paths only',
                        required=False, action='store_true')

    # Examples of usage
    # python ./strategic_pipeline.py -tf ../../data/es_full_AW/es_full_AW.toml -ni 50 -mc 3 -hpc -pc 20 -v
    # Compute the files from es_full_AW.toml, compute 50 itineraries with up to 3 connections, using heuristics

    # python ./strategic_pipeline.py -tf ../../data/es_full_AW/es_full_AW.toml -np 20 -mc 3 -hpc -cpp -pc 20 -v
    # Compute the files from es_full_AW.toml, compute 20 potential paths with up to 3 connections, using heuristics

    # python ./strategic_pipeline.py -tf ../../data/es_full_AW/es_full_AW.toml -ni 50 -mc 3 -hpc -upp -pc 20 -v
    # Compute the files from es_full_AW.toml, compute 50 itineraries with up to 3 connections, using heuristics using
    # potential paths defined in toml to guide search

    # python ./strategic_pipeline.py -tf ../../data/es_full_AW/es_full_AW.toml -ni 50 -np 20 -mc 3 -cpci -hpc -pc 20 -v
    # Compute the files from es_full_AW.toml, first compute 20 potential routes and then use that to compute 50 itineras
    # with up to 3 connections, using heuristics

    # Parse parameters
    args = parser.parse_args()

    # Setting up logging
    logging.addLevelName(IMPORTANT_INFO, "IMPORTANT_INFO")
    logging.Logger.important_info = important_info

    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    # Loading functions here so that logging setting is inherited
    from strategic_evaluator.strategic_evaluator import (create_network, preprocess_input,
                                                         compute_possible_itineraries_network,
                                                         compute_avg_paths_from_itineraries)

    with open(Path(args.toml_file), mode="rb") as fp:
        network_paths_config = tomli.load(fp)

    if args.demand_file is not None:
        network_paths_config['demand']['demand'] = Path(args.demand_file)

    pc = 1
    if args.n_proc is not None:
        pc = int(args.n_proc)


    if not args.compute_potential_and_itineraries:
        if args.compute_potential_paths:
            args.allow_mixed_operators = True

        if args.compute_potential_paths:
            logger.important_info("Running only computation of potential paths")
            logger.important_info("Note potential paths doesn't look into operators compatibility!")
        else:
            logger.important_info("Running only computation of itineraries")

        if args.compute_potential_paths:
            n_to_compute = int(args.num_paths)
        else:
            n_to_compute = int(args.num_itineraries)

        run(network_paths_config, pc=pc, n_itineraries=n_to_compute, max_connections=int(args.max_connections),
            pre_processed_version=int(args.preprocessed_version),
            allow_mixed_operators=args.allow_mixed_operators, consider_time_constraints=not args.compute_potential_paths,
            use_heuristics_precomputed=args.use_heuristics_precomputed,
            use_potential_paths=args.use_potential_paths)
    else:
        logger.important_info("Running first potential paths and then itineraries")
        logger.important_info("Note potential paths doesn't look into operators compatibility!")
        run_two_step(network_paths_config, pc=pc, n_paths=int(args.num_paths), n_itineraries=int(args.num_itineraries),
                     max_connections=int(args.max_connections), pre_processed_version=int(args.preprocessed_version),
                     allow_mixed_operators_itineraries=args.allow_mixed_operators,
                     use_heuristics_precomputed=args.use_heuristics_precomputed)

    # Improvements
    # TODO: day in rail
    # TODO: potential paths considering operators compatiblitily...
