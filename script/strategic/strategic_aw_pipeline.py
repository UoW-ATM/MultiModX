import os
from pathlib import Path
import argparse
import pandas as pd
from collections import defaultdict
import logging
import time
import sys
sys.path.insert(1, '../..')

from libs.uow_tool_belt.general_tools import recreate_output_folder
from libs.general_tools_logging_config import (save_information_config_used, important_info, setup_logging, IMPORTANT_INFO,
                                               process_strategic_config_file)


def read_origin_demand_matrix(path_demand):
    df_demand = pd.read_csv(Path(path_demand), keep_default_na=False)
    df_demand.replace('', None, inplace=True)
    return df_demand


def run_full_strategic_pipeline(toml_config, pc=1, n_paths=15, n_itineraries=50,
                                max_connections=1, pre_processed_version=0,
                                allow_mixed_operators_itineraries=False,
                                use_heuristics_precomputed=False,
                                recreate_output_fld=True):

    start_pipeline_time = time.time()

    if recreate_output_fld:
        # Check if output folder exists, if not create it
        recreate_output_folder(Path(toml_config['network_definition']['network_path']) /
                               toml_config['network_definition']['processed_folder'],
                               delete_previous=True,
                               logger=logger)
        recreate_output_folder(Path(toml_config['output']['output_folder']),
                               delete_previous=True,
                               logger=logger)


    # Preprocess input
    logger.info("Pre-processing input")
    preprocess_input(toml_config['network_definition'],
                     pre_processed_version=pre_processed_version,
                     policy_package=toml_config.get('policy_package'))

    # Read demand
    logger.info("Reading demand")
    demand_matrix = read_origin_demand_matrix(toml_config['demand']['demand'])

    # Compute possible itineraries based on demand
    o_d = demand_matrix[['origin', 'destination']].drop_duplicates()

    network_definition = toml_config['network_definition']

    # First compute potential paths
    # Create network
    logger.info("Create network simplified to compute paths")
    heuristics_precomputed = None
    if use_heuristics_precomputed:
        heuristics_precomputed = toml_config['other_param']['heuristics_precomputed']

    network = create_network(network_definition,
                             compute_simplified=True,
                             allow_mixed_operators=allow_mixed_operators_itineraries,
                             heuristics_precomputed=heuristics_precomputed,
                             pre_processed_version=pre_processed_version,
                             policy_package=toml_config.get('policy_package'))

    # Compute potential paths
    # Instead of computing these they could be read from a readily available csv file.
    logger.info("Computing potential paths")
    # df_potential_paths = compute_possible_itineraries_network(network, o_d, pc=pc, n_itineraries=n_paths,
    #                                                       max_connections=max_connections,
    #                                                       allow_mixed_operators=allow_mixed_operators_itineraries,
    #                                                       consider_times_constraints=False,
    #                                                           policy_package=toml_config.get('policy_package'))
    #
    # # Remove paths without mode of transport
    # df_potential_paths = df_potential_paths[df_potential_paths.nmodes>=1].copy().reset_index(drop=True)
    #
    # # Remove paths not in external itineraries except domestic
    # if 'external_itineraries' in toml_config['network_definition']:
    #     external_itineraries_path = (Path(toml_config['network_definition']['network_path']) /
    #            toml_config['network_definition']['external_itineraries'])
    #     domestic_nuts_path = (Path(toml_config['network_definition']['network_path']) /
    #            toml_config['network_definition']['domestic_nuts'])
    #     external_itineraries = pd.read_csv(external_itineraries_path)
    #     # print(external_itineraries)
    #
    #     domestic_nuts = pd.read_csv(domestic_nuts_path)
    #     df_potential_paths['pathx'] = df_potential_paths['path'].apply(str)
    #     # print(df_potential_paths[['path','pathx']])
    #     df_potential_paths = df_potential_paths[(df_potential_paths['pathx'].isin(external_itineraries['path'].values)) | ((df_potential_paths['origin'].isin(domestic_nuts['origin'].values)) & (df_potential_paths['destination'].isin(domestic_nuts['origin'].values)))]
    #     # df_potential_paths['indicator'] = df_potential_paths['path'].apply(lambda x: True if x in external_itineraries['pathx'].values else False)
    #     #
    #
    # ofp = 'potential_paths_' + str(pre_processed_version) + '.csv'
    # df_potential_paths.to_csv(Path(toml_config['output']['output_folder']) / ofp, index=False)
    if 'external_itineraries' in toml_config['network_definition']:
        external_itineraries_path = (Path(toml_config['network_definition']['network_path']) /
               toml_config['network_definition']['external_itineraries'])
        external_itineraries = pd.read_csv(external_itineraries_path)
        # print(toml_cdonfig)
        df_regions_acess = pd.read_csv(Path(toml_config['network_definition']['network_path']) / toml_config['network_definition']['regions_access'][0]['regions_access'], dtype={"station": str})

        station_region_dict = dict(zip(df_regions_acess['station'], df_regions_acess['region']))


        external_itineraries['origin_nuts'] = external_itineraries['origin'].apply(lambda x: station_region_dict.get(x,x))
        external_itineraries['destination_nuts'] = external_itineraries['destination'].apply(lambda x: station_region_dict.get(x,x))
        external_itineraries.drop(columns=['origin','destination']).rename(columns={'origin_nuts':'origin','destination_nuts':'destination'})
        domestic_nuts_path = (Path(toml_config['network_definition']['network_path']) /
               toml_config['network_definition']['domestic_nuts'])
        domestic_nuts = pd.read_csv(domestic_nuts_path)

    # Then compute itineraries based on potential paths
    df_potential_paths = external_itineraries.copy()
    # Create potential routes dictionary
    logger.info("Create dictionary of potential routes to use by itineraries")
    dict_o_d_routes = defaultdict(list)

    # df_potential_paths['path'] = df_potential_paths['path'].apply(lambda x: tuple(x) if x else None)
    df_potential_paths_unique = df_potential_paths[['origin', 'destination', 'path','pax']].drop_duplicates(subset=['origin', 'destination', 'path'])
    # df_potential_paths_unique['path'] = df_potential_paths_unique['path'].apply(lambda x: list(x) if x else None)
    ofp = 'df_potential_paths_unique' + str(pre_processed_version) + '.csv'
    df_potential_paths_unique.to_csv(Path(toml_config['output']['output_folder']) / ofp, index=False)

    for _, row in df_potential_paths_unique.iterrows():
        key = (row['origin'], row['destination'])
        dict_o_d_routes[key].append(row['path'])

    dict_o_d_routes = dict(dict_o_d_routes)

    # Create network
    logger.info("Create network to compute itineraries")
    network = create_network(
        network_definition,
        compute_simplified=False,
        heuristics_precomputed=heuristics_precomputed,
        pre_processed_version=pre_processed_version,
        policy_package=toml_config.get('policy_package')
    )

    # Compute itineraries
    logger.info("Computing potential paths")

    df_itineraries = compute_possible_itineraries_network(network, o_d, dict_o_d_routes=dict_o_d_routes,
                                                          pc=pc, n_itineraries=n_itineraries,
                                                          max_connections=max_connections,
                                                          allow_mixed_operators=allow_mixed_operators_itineraries,
                                                          consider_times_constraints=True,
                                                          policy_package=toml_config.get('policy_package'))

    # Remove paths without mode of transport
    df_itineraries = df_itineraries[df_itineraries.nservices >= 1].copy().reset_index(drop=True)


    ofp = 'possible_itineraries_' + str(pre_processed_version) + '.csv'
    df_itineraries.to_csv(Path(toml_config['output']['output_folder']) / ofp, index=False)

    # Compute average paths from possible itineraries
    logger.info("Compute average path for possible itineraries")
    df_avg_paths = compute_avg_paths_from_itineraries(df_itineraries)
    ofp = 'possible_paths_avg_' + str(pre_processed_version) + '.csv'
    df_avg_paths.to_csv(Path(toml_config['output']['output_folder']) / ofp, index=False)

    # Filter options that are 'similar' from the df_itineraries
    logger.important_info("Filtering/Clustering itineraries options")

    # kpis_and_thresholds_to_cluster = toml_config['other_param'].get('kpi_cluster_itineraries', {})
    # kpis_to_cluster = kpis_and_thresholds_to_cluster.get('kpis_to_use')
    #
    # kpis_thresholds = kpis_and_thresholds_to_cluster.get('thresholds')
    #
    # df_cluster_options = cluster_options_itineraries(df_itineraries, kpis=kpis_to_cluster, thresholds=kpis_thresholds,
    #                                                  pc=pc)
    #
    # ofp = 'possible_itineraries_clustered_' + str(pre_processed_version) + '.csv'
    # df_cluster_options.to_csv(Path(toml_config['output']['output_folder']) / ofp, index=False)
    #
    # # Pareto options from similar options
    # logger.important_info("Computing Pareto itineraries options")
    #
    # thresholds_pareto_dominance = toml_config['other_param']['thresholds_pareto_dominance']
    #
    # # Apply the Pareto filtering
    # pareto_df = keep_pareto_equivalent_solutions(df_cluster_options, thresholds_pareto_dominance)
    #
    # ofp = 'possible_itineraries_clustered_pareto_' + str(pre_processed_version) + '.csv'
    # pareto_df.to_csv(Path(toml_config['output']['output_folder']) / ofp, index=False)
    #
    # df_itineraries_filtered = keep_itineraries_options(df_itineraries, pareto_df)
    #
    # ofp = 'possible_itineraries_clustered_pareto_filtered_' + str(pre_processed_version) + '.csv'
    # df_itineraries_filtered.to_csv(Path(toml_config['output']['output_folder']) / ofp, index=False)
    #
    # # Compute average paths from itineraries filtered
    # logger.info("Compute average path for filtered itineraries")
    # df_avg_paths = compute_avg_paths_from_itineraries(df_itineraries_filtered)
    # ofp = 'possible_paths_avg_from_filtered_it_' + str(pre_processed_version) + '.csv'
    # df_avg_paths.to_csv(Path(toml_config['output']['output_folder']) / ofp, index=False)
    #
    #
    # # Assign passengers to paths clusters
    # logger.important_info("Assigning passengers to Path Clustered")
    #
    # n_alternatives = pareto_df.groupby(["origin", "destination"])["cluster_id"].nunique().max()
    # logger.important_info(f"Assigning demand to paths with {n_alternatives} alternatives.")
    # df_pax_demand_paths, df_paths_final = assign_demand_to_paths(pareto_df, n_alternatives, max_connections, toml_config)
    # df_pax_demand_paths.to_csv(Path(toml_config['output']['output_folder']) / "pax_demand_paths.csv", index=False)
    #
    # # Add demand per cluster
    # df_cluster_pax = obtain_demand_per_cluster_itineraries(pareto_df, df_pax_demand_paths, df_paths_final)
    #
    df_avg_paths['path'] = df_avg_paths['path'].apply(lambda x: str(x) if x else None)
    ofp = 'df_avg_paths' + str(pre_processed_version) + '.csv'
    df_avg_paths.to_csv(Path(toml_config['output']['output_folder']) / ofp, index=False)
    df_avg_paths = df_avg_paths.merge(df_potential_paths_unique[['path','pax']],on='path',how='left')
    grouped = df_avg_paths.groupby(['origin', 'destination', 'path'])
    kpis = ['total_avg_travel_time', 'total_avg_cost', 'total_avg_emissions', 'total_avg_waiting_time', 'nservices']
    df_cluster_pax = grouped.agg({
        **{kpi: 'mean' for kpi in kpis},  # Compute mean for KPIs
        'nservices': 'mean',  # Compute mean for nservices
        'journey_type': 'first',  # Take the first journey_type
        'pax':'sum',
        #'options_in_cluster': lambda x: list(set(x))  # Keep unique options in cluster
    }).reset_index().rename(columns={'total_avg_travel_time':'total_travel_time', 'total_avg_cost':'total_cost', 'total_avg_emissions':'total_emissions', 'total_avg_waiting_time':'total_waiting_time'})
    df_cluster_pax['num_pax'] = df_cluster_pax['pax']
    df_cluster_pax['label'] = df_cluster_pax['origin'] + "_" + df_cluster_pax['destination']
    df_cluster_pax['cluster_id'] = df_cluster_pax.groupby('label').cumcount()
    df_cluster_pax['option_number'] = df_cluster_pax['cluster_id']
    df_cluster_pax['alternative_id'] = df_cluster_pax['origin'] + "_" + df_cluster_pax['destination'] + "_" + df_cluster_pax['cluster_id'].astype(str)

    #add alternative_id back to itineraries
    # df_itineraries['path'] = df_itineraries['path']
    df_itineraries['path_str'] = df_itineraries['path'].apply(lambda x: str(x) if x else None)
    df_itineraries = df_itineraries.merge(df_cluster_pax[['origin', 'destination', 'path', 'alternative_id']],
                                          left_on=['origin', 'destination', 'path_str'], right_on=['origin', 'destination', 'path'], how='left')

    ofp = 'possible_itineraries_clustered_pareto_w_demand_' + str(pre_processed_version) + '.csv'
    df_cluster_pax.to_csv(Path(toml_config['output']['output_folder']) / ofp, index=False)

    #concat with intra-spain files
    cs10_its_path = (Path(toml_config['intra_spain']['possible_itineraries']))
    cs10_its = pd.read_csv(cs10_its_path)
    cs10_clusters_path = Path(toml_config['intra_spain']['clusters'])
    cs10_clusters = pd.read_csv(cs10_clusters_path)
    df_clusters = pd.concat([cs10_clusters,df_cluster_pax]).drop(columns=['path'])
    df_itineraries = pd.concat([cs10_its,df_itineraries])


    # Assign passengers to services
    logger.important_info("Assigning passengers to services")
    # Read flight and train schedules (if exist)
    fs_path = (Path(toml_config['network_definition']['network_path']) /
               toml_config['network_definition']['processed_folder'] /
               ('flight_schedules_proc_' + str(pre_processed_version) + '.csv'))
    ts_path = (Path(toml_config['network_definition']['network_path']) /
               toml_config['network_definition']['processed_folder'] /
               ('rail_timetable_proc_' + str(pre_processed_version) + '.csv'))

    # Initialize an empty list to hold dataframes of the schedules
    dataframes = []

    # Check if each file exists and read it if it does
    if os.path.exists(fs_path):
        fs = pd.read_csv(fs_path)
        fs['mode'] = 'flight'
        dataframes.append(fs[['service_id', 'seats', 'gcdistance', 'mode']])

    if os.path.exists(ts_path):
        ts = pd.read_csv(ts_path)
        ts['mode'] = 'rail'
        dataframes.append(ts[['service_id', 'seats', 'gcdistance', 'mode']])

    # Concatenate dataframes if any exist, otherwise set ds to None
    if dataframes:
        ds = pd.concat(dataframes).rename(columns={'service_id': 'nid', 'seats': 'max_seats'})
    else:
        ds = None

    df_pax_assigment, d_seats_max, df_options_w_pax = assing_pax_to_services(ds, df_clusters.copy(),
                                                                    df_itineraries.copy(),
                                                                    paras=toml_config['other_param']['pax_assigner'])

    # ofp = 'pax_assigned_from_flow_' + str(pre_processed_version) + '.csv'
    # df_pax_assigment.to_csv(Path(toml_config['output']['output_folder']) / ofp, index=False)
    ofp = 'pax_assigned_seats_max_target_' + str(pre_processed_version) + '.csv'
    d_seats_max.to_csv(Path(toml_config['output']['output_folder']) / ofp, index=False)
    ofp = 'pax_assigned_to_itineraries_options_' + str(pre_processed_version) + '.csv'
    df_options_w_pax_save = df_options_w_pax.copy().drop(columns=['generated_info','avg_fare'])
    df_options_w_pax_save.rename(columns={'volume': 'total_volume_pax_cluster',
                                          'volume_ceil': 'total_volume_pax_cluster_ceil'})
    df_options_w_pax_save.to_csv(Path(toml_config['output']['output_folder']) / ofp, index=False)


    logger.important_info("Transforming format to tactical input")
    # Transform passenger assigned into tactical input
    df_pax_tactical, df_pax_tactical_not_supported = transform_pax_assigment_to_tactical_input(df_options_w_pax)

    ofp = 'pax_assigned_tactical_' + str(pre_processed_version) + '.csv'
    df_pax_tactical.to_csv(Path(toml_config['output']['output_folder']) / ofp, index=False)

    ofp = 'pax_assigned_tactical_not_supported_' + str(pre_processed_version) + '.csv'
    df_pax_tactical_not_supported.to_csv(Path(toml_config['output']['output_folder']) / ofp, index=False)


    # Transform flight schedules into tactical input
    # TODO: flight schedule might not exist if only rail layer used
    df_flights_tactical = transform_fight_schedules_tactical_input(toml_config['other_param']['tactical_input'],
                                             (Path(toml_config['network_definition']['network_path']) /
                                              toml_config['network_definition']['processed_folder']),
                                             pre_processed_version
                                             )

    ofp = 'flight_schedules_tactical_' + str(pre_processed_version) + '.csv'
    df_flights_tactical.to_csv(Path(toml_config['output']['output_folder']) / ofp, index=False)

    end_pipeline_time = time.time()
    elapsed_time = end_pipeline_time - start_pipeline_time
    logger.important_info("Whole Strategic Pipeline computed in: " + str(elapsed_time) + " seconds.")



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


    parser.add_argument('-eo', '--end_output_folder', help='Ending to be added to output folder',
                        required=False)

    # Examples of usage
    # python ./strategic_pipeline.py -tf ../../data/es_full_AW/es_full_AW.toml -ni 50 -np 20 -mc 3 -hpc -pc 20 -v
    # Compute the files from es_full_AW.toml, first compute 20 potential routes and then use that to compute 50
    # itineraries with up to 3 connections, using heuristics

    # Parse parameters
    args = parser.parse_args()

    setup_logging(args.verbose, log_to_console=(args.verbose > 0), log_to_file=args.log_file, file_reset=True)

    logger = logging.getLogger(__name__)

    # Loading functions here so that logging setting is inherited
    from strategic_evaluator.strategic_evaluator import (
        create_network, preprocess_input, compute_possible_itineraries_network,
        compute_avg_paths_from_itineraries, cluster_options_itineraries,
        keep_pareto_equivalent_solutions, keep_itineraries_options,
        obtain_demand_per_cluster_itineraries, assing_pax_to_services,
        transform_pax_assigment_to_tactical_input,
        transform_fight_schedules_tactical_input
    )
    from strategic_evaluator.logit_model import (
        assign_demand_to_paths
    )

    toml_config = process_strategic_config_file(args.toml_file, args.end_output_folder)

    if args.demand_file is not None:
        toml_config['demand']['demand'] = Path(args.demand_file)

    pc = 1
    if args.n_proc is not None:
        pc = int(args.n_proc)

    # Check if output folder exists, if not create it
    recreate_output_folder(Path(toml_config['network_definition']['network_path']) /
                           toml_config['network_definition']['processed_folder'],
                           delete_previous=False,
                           logger=logger)
    recreate_output_folder(Path(toml_config['output']['output_folder']),
                           delete_previous=False,
                           logger=logger)

    save_information_config_used(toml_config, args)

    logger.important_info("Running first potential paths and then itineraries")

    run_full_strategic_pipeline(toml_config,
                                pc=pc,
                                n_paths=int(args.num_paths),
                                n_itineraries=int(args.num_itineraries),
                                max_connections=int(args.max_connections),
                                pre_processed_version=int(args.preprocessed_version),
                                allow_mixed_operators_itineraries=args.allow_mixed_operators,
                                use_heuristics_precomputed=args.use_heuristics_precomputed,
                                recreate_output_fld=False)

    # Improvements
    # TODO: day in rail
