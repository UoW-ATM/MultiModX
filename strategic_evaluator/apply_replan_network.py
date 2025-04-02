import pandas as pd
import logging
from collections import defaultdict

import sys
sys.path.insert(1, '../..')

logger = logging.getLogger(__name__)

# Loading functions here so that logging setting is inherited
from strategic_evaluator.strategic_evaluator import (
    create_network, preprocess_input, compute_possible_itineraries_network,
    cluster_options_itineraries, keep_pareto_equivalent_solutions,
    keep_itineraries_options
)



def replan_rail_timetable(rs_planned, rail_replanned=None, rail_cancelled=None, rail_added=None):
    ## Remove duplicates rows in rail planned timetable
    rs_planned = rs_planned.drop_duplicates()
    rs_planned['status'] = 'planned'

    # Remove cancelled services
    if rail_cancelled is not None:
        # Remove trains cancelled
        rs_planned = rs_planned.merge(rail_cancelled, left_on='trip_id', right_on='service_id', how='left')

        # keep rs that are not cancelled
        rs_planned = rs_planned[~(((~rs_planned['from'].isna()) & (rs_planned['to'].isna()) & (
                rs_planned['stop_sequence'] >= rs_planned['from'])) |
                                  ((~rs_planned['from'].isna()) & (~rs_planned['to'].isna()) & (
                                          rs_planned['stop_sequence'] >= rs_planned['from']) & (
                                           rs_planned['stop_sequence'] <= rs_planned['to'])) |
                                  ((rs_planned['from'].isna()) & (~rs_planned['to'].isna()) & (
                                          rs_planned['stop_sequence'] <= rs_planned['to']))
                                  )]
        rs_planned = rs_planned.drop(['service_id', 'from', 'to'], axis=1)

    # Remove trains replanned (as they'll be added as replanned)
    if rail_replanned is not None:
        rs_planned = rs_planned[~rs_planned.trip_id.isin(rail_replanned.trip_id)]

        # Add new trains replanned
        rail_replanned['status'] = 'replanned'
        rs_planned = pd.concat([rs_planned, rail_replanned])

    if rail_added is not None:
        # Add additional new rail services
        rail_added['status'] = 'added'
        rs_planned = pd.concat([rs_planned, rail_added])

    # Remove duplicates (in case)
    rs_planned = rs_planned.drop_duplicates()

    return rs_planned


def replan_flight_schedules(fs_planned, fs_replanned=None, fs_cancelled=None, fs_added=None):
    fs_planned['status'] = 'planned'

    if fs_cancelled is not None:
        fs_planned = fs_planned[~fs_planned.service_id.isin(fs_cancelled.service_id)]

    if fs_replanned is not None:
        fs_planned = fs_planned[~fs_planned.service_id.isin(fs_replanned.service_id)]
        fs_replanned['status'] = 'replanned'
        fs_planned = pd.concat([fs_planned, fs_replanned])

    if fs_added is not None:
        fs_added['status'] = 'added'
        fs_planned = pd.concat([fs_planned, fs_added])

    return fs_planned


def compute_itineraries_in_replanned_network(toml_config, pc=1, n_paths=15, n_itineraries = 50,  max_connections = 1,
                                             allow_mixed_operators_itineraries = False,
                                             use_heuristics_precomputed = False,
                                             pre_processed_version = 0,
                                             capacity_available = None):
    # Preprocess input
    logger.info("Pre-processing input")
    preprocess_input(toml_config['network_definition'],
                     pre_processed_version=pre_processed_version,
                     policy_package=toml_config.get('policy_package'))

    # Read demand
    logger.info("Reading demand")
    demand_matrix = pd.read_csv(toml_config['demand']['demand'], keep_default_na=False)
    demand_matrix.replace('', None, inplace=True)

    # Compute possible itineraries based on demand
    o_d = demand_matrix[['origin', 'destination']].drop_duplicates()

    #First compute potential paths
    network_definition = toml_config['network_definition']

    # Get if there are services that are full and shouldn't be considered
    # at all in the network
    services_wo_capacity = None
    if capacity_available is not None:
        services_wo_capacity = capacity_available[capacity_available.capacity == 0]

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
                             policy_package=toml_config.get('policy_package'),
                             remove_services=services_wo_capacity)

    # Compute potential paths
    # Instead of computing these they could be read from a readily available csv file.
    logger.info("Computing potential paths")
    df_potential_paths = compute_possible_itineraries_network(network, o_d, pc=pc, n_itineraries=n_paths,
                                                          max_connections=max_connections,
                                                          allow_mixed_operators=allow_mixed_operators_itineraries,
                                                          consider_times_constraints=False,
                                                              policy_package=toml_config.get('policy_package'))

    # Remove paths without mode of transport
    df_potential_paths = df_potential_paths[df_potential_paths.nmodes >= 1].copy().reset_index(drop=True)

    ofp = 'potential_paths_' + str(pre_processed_version) + '.csv'
    df_potential_paths.to_csv(toml_config['output']['output_folder'] / ofp, index=False)

    # Then compute itineraries based on potential paths
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
        policy_package=toml_config.get('policy_package'),
        remove_services=services_wo_capacity
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
    df_itineraries.to_csv(toml_config['output']['output_folder'] / ofp, index=False)

    # Filter options that are 'similar' from the df_itineraries
    logger.important_info("Filtering/Clustering itineraries options")

    kpis_and_thresholds_to_cluster = toml_config['other_param'].get('kpi_cluster_itineraries', {})
    kpis_to_cluster = kpis_and_thresholds_to_cluster.get('kpis_to_use')

    kpis_thresholds = kpis_and_thresholds_to_cluster.get('thresholds')

    df_cluster_options = cluster_options_itineraries(df_itineraries, kpis=kpis_to_cluster, thresholds=kpis_thresholds,
                                                     pc=pc)

    ofp = 'possible_itineraries_clustered_' + str(pre_processed_version) + '.csv'
    df_cluster_options.to_csv(toml_config['output']['output_folder'] / ofp, index=False)

    # Pareto options from similar options
    logger.important_info("Computing Pareto itineraries options")

    thresholds_pareto_dominance = toml_config['other_param']['thresholds_pareto_dominance']

    # Apply the Pareto filtering
    pareto_df = keep_pareto_equivalent_solutions(df_cluster_options, thresholds_pareto_dominance)

    ofp = 'possible_itineraries_clustered_pareto_' + str(pre_processed_version) + '.csv'
    pareto_df.to_csv(toml_config['output']['output_folder'] / ofp, index=False)

    df_itineraries_filtered = keep_itineraries_options(df_itineraries, pareto_df)

    ofp = 'possible_itineraries_clustered_pareto_filtered_' + str(pre_processed_version) + '.csv'
    df_itineraries_filtered.to_csv(toml_config['output']['output_folder'] / ofp, index=False)

    return df_itineraries, df_itineraries_filtered



