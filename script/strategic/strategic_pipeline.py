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

from strategic_evaluator.logit_model import format_path_clusters


def read_origin_demand_matrix(path_demand):
    df_demand = pd.read_csv(Path(path_demand), keep_default_na=False)
    df_demand.replace('', None, inplace=True)
    return df_demand


def run_full_strategic_pipeline(toml_config, pc=1, n_paths=15, n_itineraries=50,
                                max_connections=1, pre_processed_version=0,
                                allow_mixed_operators_itineraries=False,
                                use_heuristics_precomputed=False,
                                save_all_output=True):

    #### START PIPELINE ####
    start_pipeline_time = time.time()

    # PREPROCESS INPUT
    logger.info("Pre-processing input")
    preprocess_input(toml_config['network_definition'],
                     pre_processed_version=pre_processed_version,
                     policy_package=toml_config.get('policy_package'))

    # READ DEMAND AND LOGIT MODELS
    logger.info("Reading demand and logit models")
    demand_matrices = {}
    logit_models = {}
    for demand in toml_config["demand"]:
        logit_id=demand["sensitivities_logit"]["logit_id"]
        df_demand = read_origin_demand_matrix(demand["demand"])
        # If compress_archetypes is set in the TOML config file then
        # group all the demand by date, origin, destination and
        # rename archetype to archetype_0
        if demand["sensitivities_logit"].get('compress_archetypes',False):
            df_demand = df_demand.groupby(['date', 'origin', 'destination'])['trips'].sum().reset_index()
            df_demand['archetype'] = 'archetype_0'
        demand_matrices[logit_id] = df_demand
        logit_models[logit_id] = {'sensitivities': demand["sensitivities_logit"]["sensitivities"],
                                  'n_archetypes': demand["sensitivities_logit"].get("n_archetypes")}

    ##### COMPUTE POSSIBLE ITINERARIES BASED ON DEMAND #####
    # Obtain origin destination
    o_d=[]
    o_d_per_logit_dict={}
    for logit_id in demand_matrices.keys():
        o_d_per_logit_dict[logit_id]=demand_matrices[logit_id][["origin","destination"]].drop_duplicates() #dictionary with o_d_pairs per logit
        o_d += [demand_matrices[logit_id][["origin","destination"]]]

    o_d = pd.concat(o_d).drop_duplicates()

    network_definition = toml_config['network_definition']

    # FIRST COMPUTE POTENTIAL PATHS
    # Create network
    logger.info("Create network simplified to compute paths")
    heuristics_precomputed = None
    if use_heuristics_precomputed:
        # Heuristics are to guide the A* algorithm
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
    df_potential_paths = compute_possible_itineraries_network(network,
                                                              o_d,
                                                              pc=pc,
                                                              n_itineraries=n_paths,
                                                              max_connections=max_connections,
                                                              allow_mixed_operators=allow_mixed_operators_itineraries,
                                                              consider_times_constraints=False,
                                                              policy_package=toml_config.get('policy_package'))

    # Remove paths without mode of transport --> Access and egress only
    df_potential_paths = df_potential_paths[df_potential_paths.nmodes>=1].copy().reset_index(drop=True)

    if save_all_output:
        ofp = 'potential_paths_' + str(pre_processed_version) + '.csv'
        df_potential_paths.to_csv(Path(toml_config['output']['output_folder']) / ofp, index=False)


    # THEN COMPUTE ITINERARIES BASED ON POTENTIAL PATHS
    # Create potential routes dictionary
    logger.info("Create dictionary of potential routes to use by itineraries")
    dict_o_d_routes = defaultdict(list)

    df_potential_paths['path'] = df_potential_paths['path'].apply(lambda x: tuple(x) if x else None)
    df_potential_paths_unique = df_potential_paths[['origin', 'destination', 'path']].drop_duplicates()
    df_potential_paths_unique['path'] = df_potential_paths_unique['path'].apply(lambda x: list(x) if x else None)

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
    logger.info("Computing possible itineraries")
    df_itineraries = compute_possible_itineraries_network(network,
                                                          o_d,
                                                          dict_o_d_routes=dict_o_d_routes,
                                                          pc=pc,
                                                          n_itineraries=n_itineraries,
                                                          max_connections=max_connections,
                                                          allow_mixed_operators=allow_mixed_operators_itineraries,
                                                          consider_times_constraints=True,
                                                          policy_package=toml_config.get('policy_package'))

    # Remove paths without mode of transport --> Only access and egress
    df_itineraries = df_itineraries[df_itineraries.nservices >= 1].copy().reset_index(drop=True)

    # Saved always, as part of minimum output
    ofp = 'possible_itineraries_' + str(pre_processed_version) + '.csv'
    df_itineraries.to_csv(Path(toml_config['output']['output_folder']) / ofp, index=False)

    # COMPUTE AVERAGE PATHS FROM POSSIBLE ITINERARIES
    logger.info("Compute average path for possible itineraries")
    df_avg_paths = compute_avg_paths_from_itineraries(df_itineraries)
    if save_all_output:
        ofp = 'possible_paths_avg_' + str(pre_processed_version) + '.csv'
        df_avg_paths.to_csv(Path(toml_config['output']['output_folder']) / ofp, index=False)

    # FILTER/CLUSTER OPTIONS THAT ARE 'SIMILAR' FROM THE DF_ITINERARIES
    logger.important_info("Filtering/Clustering itineraries options")

    kpis_and_thresholds_to_cluster = toml_config['other_param'].get('kpi_cluster_itineraries', {})
    kpis_to_cluster = kpis_and_thresholds_to_cluster.get('kpis_to_use')

    kpis_thresholds = kpis_and_thresholds_to_cluster.get('thresholds')

    df_cluster_options = cluster_options_itineraries(df_itineraries,
                                                     kpis=kpis_to_cluster,
                                                     thresholds=kpis_thresholds,
                                                     pc=pc)

    if save_all_output:
        ofp = 'possible_itineraries_clustered_' + str(pre_processed_version) + '.csv'
        df_cluster_options.to_csv(Path(toml_config['output']['output_folder']) / ofp, index=False)

    # PARETO FOR SIMILAR OPTIONS
    logger.important_info("Computing Pareto itineraries options")

    thresholds_pareto_dominance = toml_config['other_param']['thresholds_pareto_dominance']

    # Apply the Pareto filtering
    pareto_df = keep_pareto_equivalent_solutions(df_cluster_options, thresholds_pareto_dominance)

    if save_all_output:
        ofp = 'possible_itineraries_clustered_pareto_' + str(pre_processed_version) + '.csv'
        pareto_df.to_csv(Path(toml_config['output']['output_folder']) / ofp, index=False)

    df_itineraries_filtered = keep_itineraries_options(df_itineraries, pareto_df)

    # Saved always as part of minimum output
    ofp = 'possible_itineraries_clustered_pareto_filtered_' + str(pre_processed_version) + '.csv'
    df_itineraries_filtered.to_csv(Path(toml_config['output']['output_folder']) / ofp, index=False)


    # COMPUTE AVERAGE PATHS FROM ITINERARIES FILTERED
    logger.info("Compute average path for filtered itineraries")
    df_avg_paths = compute_avg_paths_from_itineraries(df_itineraries_filtered)

    if save_all_output:
        ofp = 'possible_paths_avg_from_filtered_it_' + str(pre_processed_version) + '.csv'
        df_avg_paths.to_csv(Path(toml_config['output']['output_folder']) / ofp, index=False)


    ##### ASSIGN PASSENGERS TO PATHS CLUSTERED ####
    # Based on logit model preferences
    logger.important_info("Assigning passengers to Path Clustered")

    # Define the parameters
    # Separate pareto_df for each logit model
    pareto_df_per_logit={logit_id: pareto_df.merge(
        o_d_per_logit_dict[logit_id],
        on=["origin","destination"],how="inner") 
        for logit_id in o_d_per_logit_dict}


    # Variables to save demand as logit models are applied
    df_pax_demand_paths_list=[]
    df_paths_final_list=[]

    # For each logit applied to subset demand (demand for that logit) assign demand to paths
    for logit_id, subset_pareto_df in pareto_df_per_logit.items():
        n_alternatives = subset_pareto_df.groupby(["origin", "destination"])["cluster_id"].nunique().max()
        logger.important_info(f"Assigning demand to paths with {n_alternatives} alternatives.")
        # TODO: subset_pareto now have train, plan, multimodal and option_number columns. We could remerge these into pareto_df
        df_pax_demand_paths, df_paths_final = assign_demand_to_paths(subset_pareto_df, 
                                                                    max_connections= max_connections,
                                                                    logit_model=logit_models[logit_id],
                                                                    df_demand=demand_matrices[logit_id].copy(),
                                                                    n_alternatives=n_alternatives,
                                                                    network_paths_config=toml_config) #this is where I have to look at
        df_pax_demand_paths["logit_id"]=logit_id #tag logit model
        df_paths_final["logit_id"]=logit_id

        df_pax_demand_paths_list.append(df_pax_demand_paths) #append new entry to the list
        df_paths_final_list.append(df_paths_final)

    df_pax_demand_paths=pd.concat(df_pax_demand_paths_list,ignore_index=True) #concat the list
    df_paths_final=pd.concat(df_paths_final_list, ignore_index=True)

    if 'option_number' not in pareto_df.columns:
        # The assign_demand_to_paths modify the pareto_df
        # adding train, plane, multimodal and option_number
        # in the sub_df version this is done in the sub_dfs
        # so we need to read that now here.
        pareto_df = format_path_clusters(pareto_df)

    # Saved always as minimum output
    df_pax_demand_paths.to_csv(Path(toml_config['output']['output_folder']) / "pax_demand_paths.csv", index=False)


    # ADD DEMAND TO CLUSTER OF ITINERARIES
    df_cluster_pax = obtain_demand_per_cluster_itineraries(pareto_df, df_pax_demand_paths, df_paths_final)

    # Saved always as minimum output
    ofp = 'possible_itineraries_clustered_pareto_w_demand_' + str(pre_processed_version) + '.csv'
    df_cluster_pax.to_csv(Path(toml_config['output']['output_folder']) / ofp, index=False)



    ##### ASSIGN PASSENGERS TO SERVICES (PAX DISAGGREGATION)  #####

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


    df_pax_assigment, d_seats_max, df_options_w_pax = assing_pax_to_services(ds,
                                                                             df_cluster_pax.copy(),
                                                                             df_itineraries_filtered.copy(),
                                                                             paras=toml_config['other_param']['pax_assigner'])

    # ofp = 'pax_assigned_from_flow_' + str(pre_processed_version) + '.csv'
    # df_pax_assigment.to_csv(Path(toml_config['output']['output_folder']) / ofp, index=False)
    if save_all_output:
        ofp = 'pax_assigned_seats_max_target_' + str(pre_processed_version) + '.csv'
        d_seats_max.to_csv(Path(toml_config['output']['output_folder']) / ofp, index=False)

    # Saved always as minimum output
    ofp = 'pax_assigned_to_itineraries_options_' + str(pre_processed_version) + '.csv'
    df_options_w_pax_save = df_options_w_pax.copy().drop(columns=['generated_info','avg_fare'])
    df_options_w_pax_save.rename(columns={'volume': 'total_volume_pax_cluster',
                                          'volume_ceil': 'total_volume_pax_cluster_ceil'})
    df_options_w_pax_save.to_csv(Path(toml_config['output']['output_folder']) / ofp, index=False)



    ##### TRANSFORM FORMAT TO TACTICAL (MERCURY) INPUT FORMAT #####

    logger.important_info("Transforming format to tactical input")
    # Transform passenger assigned into tactical input
    df_pax_tactical, df_pax_tactical_not_supported = transform_pax_assigment_to_tactical_input(df_options_w_pax)

    # Saved always as minimum output
    ofp = 'pax_assigned_tactical_' + str(pre_processed_version) + '.csv'
    df_pax_tactical.to_csv(Path(toml_config['output']['tactical_output_folder']) / ofp, index=False)

    # Saved always as minimum output
    ofp = 'pax_assigned_tactical_not_supported_' + str(pre_processed_version) + '.csv'
    df_pax_tactical_not_supported.to_csv(Path(toml_config['output']['tactical_output_folder']) / ofp, index=False)


    # Transform flight schedules into tactical input
    # TODO: flight schedule might not exist if only rail layer used
    df_flights_tactical = transform_fight_schedules_tactical_input(toml_config['other_param']['tactical_input'],
                                             (Path(toml_config['network_definition']['network_path']) /
                                              toml_config['network_definition']['processed_folder']),
                                             pre_processed_version
                                             )

    # Saved always as minimum output
    ofp = 'flight_schedules_tactical_' + str(pre_processed_version) + '.csv'
    df_flights_tactical.to_csv(Path(toml_config['output']['tactical_output_folder']) / ofp, index=False)


    #### END PIPELINE ####
    end_pipeline_time = time.time()
    elapsed_time = end_pipeline_time - start_pipeline_time
    logger.important_info("Whole Strategic Pipeline computed in: " + str(elapsed_time) + " seconds.")



# Setting up logging
logging.addLevelName(IMPORTANT_INFO, "IMPORTANT_INFO")
logging.Logger.important_info = important_info


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Strategic pipeline', add_help=True)

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
    parser.add_argument('-pc', '--n_proc', help='Number of processors', required=False)
    parser.add_argument('-df', '--demand_file', help='Pax demand file instead of the one in the toml_file',
                        required=False)
    parser.add_argument('-amo', '--allow_mixed_operators', help='Allow mix operators',
                        required=False, action='store_true')
    parser.add_argument('-v', '--verbose', action='count', default=0, help="increase output verbosity")
    parser.add_argument('-lf', '--log_file', help='Path to log file', required=False)
    parser.add_argument('-mo', '--minimum_output', help='Save only minimum files instead of all of them',
                        action='store_true', required=False)
    parser.add_argument('-rof', '--recreate_output_folder', help='Save only minimum files instead of all of them',
                        action='store_true', required=False)
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
                           delete_previous=args.recreate_output_folder,
                           logger=logger)
    recreate_output_folder(Path(toml_config['output']['output_folder']),
                           delete_previous=args.recreate_output_folder,
                           logger=logger)
    recreate_output_folder(Path(toml_config['output']['tactical_output_folder']),
                           delete_previous=args.recreate_output_folder,
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
                                save_all_output=not args.minimum_output)
