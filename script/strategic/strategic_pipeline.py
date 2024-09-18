from pathlib import Path
import argparse
import tomli
import pandas as pd
from collections import defaultdict
import logging
import sys
sys.path.insert(1, '../..')


# Define custom logging levels
IMPORTANT_INFO = 35  # Between WARNING (30) and ERROR (40)

def important_info(self, message, *args, **kwargs):
    if self.isEnabledFor(IMPORTANT_INFO):
        self._log(IMPORTANT_INFO, message, args, **kwargs)


def setup_logging(verbosity, log_to_console=True, log_to_file=None, file_reset=False, file_level=None):
    # Define the log levels in order of increasing verbosity
    levels = [logging.ERROR, IMPORTANT_INFO, logging.WARNING, logging.INFO, logging.DEBUG]

    # ERROR and WARNING are always considered, if verbosity=1 then IMPORTANT_INFO too
    level = levels[min(len(levels) - 1, verbosity)]  # Ensure the level does not exceed DEBUG

    # Create the main logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Remove default handlers if they exist
    logger.handlers = []

    # Format for log messages
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

    # Console handler (if enabled)
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)  # Set console log level
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

        logger.important_info(f"Logging to console enabled set to {logging.getLevelName(level)}")

    # File handler (if enabled)
    if log_to_file:
        # Set mode: 'w' to overwrite (reset) the file or 'a' to append
        file_mode = 'w' if file_reset else 'a'
        file_handler = logging.FileHandler(log_to_file, mode=file_mode)

        # Ensure file_level defaults to WARNING if not passed
        file_logging_level = file_level if file_level is not None else logging.WARNING
        file_handler.setLevel(file_logging_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

        logger.important_info(f"Logging to file: {log_to_file} set to {logging.getLevelName(file_handler.level)}")



def read_origin_demand_matrix(path_demand):
    df_demand = pd.read_csv(Path(path_demand), keep_default_na=False)
    df_demand.replace('', None, inplace=True)
    return df_demand


def run_full_strategic_pipeline(network_paths_config, pc=1, n_paths=15, n_itineraries=50,
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

    network_definition = network_paths_config['network_definition']

    # First compute potential paths
    # Create network
    logger.info("Create network simplified to compute paths")
    network = create_network(network_definition,
                             compute_simplified=True,
                             allow_mixed_operators=allow_mixed_operators_itineraries,
                             use_heuristics_precomputed=use_heuristics_precomputed,
                             pre_processed_version=pre_processed_version)

    # Compute potential paths
    # Instead of computing these they could be read from a readily available csv file.
    logger.info("Computing potential paths")
    df_potential_paths = compute_possible_itineraries_network(network, o_d, pc=pc, n_itineraries=n_paths,
                                                          max_connections=max_connections,
                                                          allow_mixed_operators=allow_mixed_operators_itineraries,
                                                          consider_times_constraints=False)

    ofp = 'potential_paths_' + str(pre_processed_version) + '.csv'
    df_potential_paths.to_csv(Path(network_paths_config['output']['output_folder']) / ofp, index=False)

    # Then compute itineraries based on potential paths
    # Create potential routes dictionary
    logger.info("Create dictionary of potential routes to use by itineraries")
    dict_o_d_routes = defaultdict(list)

    df_potential_paths['path'] = df_potential_paths['path'].apply(tuple)
    df_potential_paths_unique = df_potential_paths[['origin', 'destination', 'path']].drop_duplicates()
    df_potential_paths_unique['path'] = df_potential_paths_unique['path'].apply(list)

    for _, row in df_potential_paths_unique.iterrows():
        key = (row['origin'], row['destination'])
        dict_o_d_routes[key].append(row['path'])

    dict_o_d_routes = dict(dict_o_d_routes)

    # Create network
    logger.info("Create network to compute itineraries")
    network = create_network(network_definition,
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

    # Filter options that are 'similar' from the df_itineraries
    logger.important_info("Filtering/Clustering itineraries options")
    df_cluster_options = cluster_options_itineraries(df_itineraries, kpis=['total_travel_time', 'total_cost',
                                                                           'total_emissions', 'total_waiting_time'],
                                                     pc=pc)


    ofp = 'possible_itineraries_clustered_' + str(pre_processed_version) + '.csv'
    df_cluster_options.to_csv(Path(network_paths_config['output']['output_folder']) / ofp, index=False)

    # Pareto options from similar options
    logger.important_info("Computing Pareto itineraries options")

    thresholds = {
        'total_travel_time': 15,
        'total_cost': 10,
        'total_emissions': 5,
        'total_waiting_time': 30
    }

    # Apply the Pareto filtering
    pareto_df = keep_pareto_equivalent_solutions(df_cluster_options, thresholds)

    ofp = 'possible_itineraries_clustered_pareto_' + str(pre_processed_version) + '.csv'
    pareto_df.to_csv(Path(network_paths_config['output']['output_folder']) / ofp, index=False)

    df_itineraries_filtered = keep_itineraries_options(df_itineraries, pareto_df)

    ofp = 'possible_itineraries_clustered_pareto_filtered_' + str(pre_processed_version) + '.csv'
    df_itineraries_filtered.to_csv(Path(network_paths_config['output']['output_folder']) / ofp, index=False)


# Setting up logging
logging.addLevelName(IMPORTANT_INFO, "IMPORTANT_INFO")
logging.Logger.important_info = important_info


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
    parser.add_argument('-np', '--num_paths', help='Number of paths to compute if computing potential paths',
                        required=False, default=30)
    parser.add_argument('-v', '--verbose', action='count', default=0, help="increase output verbosity")

    parser.add_argument('-lf', '--log_file', help='Path to log file', required=False)

    parser.add_argument('-pc', '--n_proc', help='Number of processors', required=False)

    parser.add_argument('-df', '--demand_file', help='Pax demand file instead of the one in the toml_file',
                        required=False)

    parser.add_argument('-amo', '--allow_mixed_operators', help='Allow mix operators',
                        required=False, action='store_true')

    parser.add_argument('-cpp', '--compute_potential_paths', help='Compute only potential paths',
                        required=False, action='store_true')
    parser.add_argument('-upp', '--use_potential_paths', help='Compute itineraries from list of potential '
                                                              'paths only',
                        required=False, action='store_true')

    # Examples of usage
    # python ./strategic_pipeline.py -tf ../../data/es_full_AW/es_full_AW.toml -ni 50 -np 20 -mc 3 -hpc -pc 20 -v
    # Compute the files from es_full_AW.toml, first compute 20 potential routes and then use that to compute 50
    # itineraries with up to 3 connections, using heuristics

    # Parse parameters
    args = parser.parse_args()

    setup_logging(args.verbose, log_to_console=(args.verbose > 0), log_to_file=args.log_file, file_reset=True)

    logger = logging.getLogger(__name__)

    # Loading functions here so that logging setting is inherited
    from strategic_evaluator.strategic_evaluator import (create_network, preprocess_input,
                                                         compute_possible_itineraries_network,
                                                         compute_avg_paths_from_itineraries,
                                                         cluster_options_itineraries,
                                                         keep_pareto_equivalent_solutions,
                                                         keep_itineraries_options)

    with open(Path(args.toml_file), mode="rb") as fp:
        network_paths_config = tomli.load(fp)

    if args.demand_file is not None:
        network_paths_config['demand']['demand'] = Path(args.demand_file)

    pc = 1
    if args.n_proc is not None:
        pc = int(args.n_proc)

    logger.important_info("Running first potential paths and then itineraries")
    run_full_strategic_pipeline(network_paths_config,
                                pc=pc,
                                n_paths=int(args.num_paths),
                                n_itineraries=int(args.num_itineraries),
                                max_connections=int(args.max_connections),
                                pre_processed_version=int(args.preprocessed_version),
                                allow_mixed_operators_itineraries=args.allow_mixed_operators,
                                use_heuristics_precomputed=args.use_heuristics_precomputed)

    # Improvements
    # TODO: day in rail
