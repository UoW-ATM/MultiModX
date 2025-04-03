import pandas as pd
import logging
from collections import defaultdict
from datetime import timedelta
import ast

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

    '''
    We won't compute Pareto and filter when reassinging, keep all
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
    '''

    return df_itineraries


def compute_alternatives_possibles_pax_itineraries(pax_need_replanning, df_itineraries, fs_planned, rs_planned):
    ###############################
    # Process pax need replanning #
    ###############################

    # Drop columns we're not going to use pax assigned
    pax_need_replanning['type_pax'] = pax_need_replanning['type']
    pax_need_replanning['path'] = pax_need_replanning['path'].apply(ast.literal_eval)
    pax_need_replanning['total_travel_time_pax'] = pax_need_replanning['total_time']
    pax_need_replanning['access_time_pax'] = pax_need_replanning['access_time'].apply(
        lambda x: timedelta(minutes=x))
    pax_need_replanning['egress_time_pax'] = pax_need_replanning['egress_time'].apply(
        lambda x: timedelta(minutes=x))
    pax_need_replanning['nservices_pax'] = pax_need_replanning['type_pax'].apply(lambda x: len(x.split("_")))
    pax_need_replanning['modes_pax'] = pax_need_replanning['type_pax'].apply(lambda x: list(set(x.split("_"))))

    pax_need_replanning = pax_need_replanning.drop(columns=['cluster_id', 'option_cluster_number',
                                                            'alternative_id', 'option',
                                                            'fare', 'volume', 'd2i_time', 'i2d_time', 'volume_ceil',
                                                            'total_waiting_time', 'type',
                                                            'total_time', 'access_time', 'egress_time'])

    def nodes_types_service_initial_end(x):
        type_split = x.type_pax.split("_")
        first_service_type = type_split[0]
        last_service_type = type_split[-1]
        if first_service_type == 'rail':
            split_f1 = x.nid_f1.split("_")
            service_id_o = split_f1[0]
        else:
            service_id_o = x.nid_f1
        if last_service_type == 'rail':
            split_fn = x['nid_f' + str(len(type_split))].split("_")
            service_id_n = split_fn[0]
        else:
            service_id_n = x['nid_f' + str(len(type_split))]

        return x.path[0], x.path[-1], first_service_type, last_service_type, service_id_o, service_id_n

    # Add infrastructure nodes (initial and final), type (initial and final), and service_id (initial and final)
    pax_need_replanning[['node_pax_0', 'node_pax_n', 'type_pax_0', 'type_pax_n', 'service_id_pax_0',
                         'service_id_pax_n']] = pd.DataFrame(
        pax_need_replanning.apply(nodes_types_service_initial_end, axis=1).tolist(),
        index=pax_need_replanning.index
    )


    ###########################
    # Process rail schedules #
    ##########################

    # Filter from rs_planned only the ones we need (based on pax_need_replannig)
    rs_planned = rs_planned[rs_planned.stop_id.isin(
        set(pax_need_replanning[pax_need_replanning.type_pax_0 == 'rail']['node_pax_0']).union(
            set(pax_need_replanning[pax_need_replanning.type_pax_n == 'rail']['node_pax_n'])))].copy()

    # Service_stop id
    rs_planned['service_stop'] = rs_planned['trip_id'].astype(str) + '_' + rs_planned['stop_sequence'].astype(str)

    # Keep only columns we might need
    rs_planned = rs_planned[['trip_id', 'stop_id', 'stop_sequence', 'service_stop', 'provider', 'alliance',
                             'departure_time_utc', 'arrival_time_utc',
                             'departure_time_local', 'arrival_time_local']]

    # Create dictionary of departure and arrival times at UTC
    dict_rs_planned_departure_utc = rs_planned.set_index(['service_stop'])['departure_time_utc'].to_dict()
    dict_rs_planned_arrival_utc = rs_planned.set_index(['service_stop'])['arrival_time_utc'].to_dict()
    # Dictionary of alliances for rs
    dict_alliance_rs = rs_planned[['trip_id', 'alliance']].set_index('trip_id')['alliance'].to_dict()


    #############################
    # Process flight schedules #
    ###################'########

    # Process flight schedules
    fs_planned['sobt_utc'] = fs_planned['sobt']
    fs_planned['sibt_utc'] = fs_planned['sibt']

    # Create dictionary of departure and arrival times at UTC
    dict_fs_planned_departure_utc = fs_planned.set_index(['service_id'])['sobt_utc'].to_dict()
    dict_fs_planned_arrival_utc = fs_planned.set_index(['service_id'])['sibt_utc'].to_dict()

    # Dictionary of alliances for fs
    dict_alliance_fs = fs_planned[['service_id', 'alliance']].set_index('service_id')['alliance'].to_dict()

    ##################################################################
    # Add in pax_need_replanning schedules of first and last service #
    #         Based on dict_fs_planned and dict_rs_planned           #
    #        And other info relating to flights and/or trains        #
    ##################################################################

    def sobt_sibt_infrastructure_total_itinerary_utc(x):
        type_split = x.type_pax.split("_")
        first_service_type = type_split[0]
        last_service_type = type_split[-1]
        if first_service_type == 'flight':
            sobt = dict_fs_planned_departure_utc[x.nid_f1]
        else:
            split_f1 = x.nid_f1.split("_")
            nid_f1 = split_f1[0] + '_' + split_f1[1]
            sobt = dict_rs_planned_departure_utc[nid_f1]

        if last_service_type == 'flight':
            sibt = dict_fs_planned_arrival_utc[x['nid_f' + str(len(type_split))]]
        else:
            split_fn = x['nid_f' + str(len(type_split))].split("_")
            nid_fn = split_fn[0] + '_' + split_fn[2]
            sibt = dict_rs_planned_arrival_utc[nid_fn]

        return sobt, sibt

    # Get SOBT and SIBT of first and last stop on pax itineraries based on dict_fs_planned and dict_rs_planned
    pax_need_replanning[['sobt_utc_pax_0', 'sibt_utc_pax_n']] = pd.DataFrame(
        pax_need_replanning.apply(sobt_sibt_infrastructure_total_itinerary_utc, axis=1).tolist(),
        index=pax_need_replanning.index
    )

    # Final elements of pax itineraries originally planned performance
    pax_need_replanning['dept_home_utc_pax'] = pax_need_replanning['sobt_utc_pax_0'] - pax_need_replanning[
        'access_time_pax']
    pax_need_replanning['arr_home_utc_pax'] = pax_need_replanning['sibt_utc_pax_n'] + pax_need_replanning[
        'egress_time_pax']


    # Add which alliance the segments of pax_needed_replanning are using
    for col in pax_need_replanning.columns[pax_need_replanning.columns.str.startswith("nid_f")]:
        pax_need_replanning['alliance_' + col] = pax_need_replanning[col].apply(lambda x: dict_alliance_fs.get(x,
                                                                                                               dict_alliance_rs.get(
                                                                                                                   x.split(
                                                                                                                       '_')[
                                                                                                                       0])) if
        not pd.isna(x) else x)

    pax_need_replanning["alliances_used"] = pax_need_replanning.filter(like="alliance_nid_f").apply(
        lambda row: list(set(row.dropna())), axis=1)

    df_itineraries["alliances_used"] = df_itineraries.filter(like="alliance_").apply(
        lambda row: list(set(row.dropna())), axis=1)

    pax_need_replanning = pax_need_replanning.drop(columns=(list(pax_need_replanning.filter(like="alliance_nid_f").columns)))


    ##############################
    #   PROCESS df_itineraries   #
    ##############################

    df_itineraries['sobt_utc_it_0'] = pd.to_datetime(df_itineraries['departure_time_0'])
    df_itineraries['sibt_utc_it_n'] = df_itineraries.apply(
        lambda x: pd.to_datetime(x['arrival_time_' + str(x.nservices - 1)]), axis=1)
    df_itineraries['access_time_it'] = df_itineraries['access_time'].apply(lambda x: timedelta(minutes=x))
    df_itineraries['egress_time_it'] = df_itineraries['egress_time'].apply(lambda x: timedelta(minutes=x))
    df_itineraries['dept_home_utc_it'] = df_itineraries['sobt_utc_it_0'] - df_itineraries['access_time_it']
    df_itineraries['arr_home_utc_it'] = df_itineraries['sibt_utc_it_n'] + df_itineraries['egress_time_it']
    df_itineraries['total_travel_time_it'] = df_itineraries['total_travel_time']
    df_itineraries['nservices_it'] = df_itineraries['nservices']
    df_itineraries['node_it_0'] = df_itineraries.path.apply(lambda x: x[0])
    df_itineraries['node_it_n'] = df_itineraries.path.apply(lambda x: x[-1])
    df_itineraries['modes_it'] = df_itineraries.filter(like="mode_").apply(lambda row: list(set(row.dropna())), axis=1)

    # Drop columns we won't need
    df_itineraries = df_itineraries.drop(
        columns=(['total_travel_time', 'total_cost', 'total_emissions', 'total_waiting_time',
                  'd2i_time', 'i2d_time'] +
                 list(df_itineraries.filter(like="emissions_").columns) +
                 list(df_itineraries.filter(like="service_cost_").columns) +
                 list(df_itineraries.filter(like="cost_").columns) +
                 list(df_itineraries.filter(like="waiting_time_").columns) +
                 list(df_itineraries.filter(like="ground_mobility_time").columns) +
                 list(df_itineraries.filter(like="mct_time_").columns) +
                 list(df_itineraries.filter(like="arrival_time_").columns) +
                 list(df_itineraries.filter(like="departure_time_").columns) +
                 [col for col in df_itineraries.columns if col.startswith("travel_time_")] +
                 list(df_itineraries.filter(like="connecting_time_").columns)))

    #################################################
    #   MERGE Pax itinearies with replanning needed #
    #################################################

    # Merge now pax planned itineraries that need replanning
    # with the new possible itineraries
    pax_need_replanning['od'] = pax_need_replanning['origin'] + '_' + pax_need_replanning['destination']
    df_itineraries['od'] = df_itineraries['origin'] + '_' + df_itineraries['destination']
    pax_need_replanning_w_it_options = pax_need_replanning.drop(columns={'origin', 'destination'}).merge(
        df_itineraries, how='left', on='od', suffixes=('_pax', '_it'))

    # Compute some further indicators on compatibility between itinerary pax and options computed

    # Check if alliances between itineraries and pax match (i.e. all alliances used the itineraries are in pax)
    # Convert lists to sets for easy subset checking
    mask = pax_need_replanning_w_it_options.apply(
        lambda row: set(row["alliances_used_it"]).issubset(set(row["alliances_used_pax"])), axis=1
    )

    # DataFrame where all alliances in 'alliances_used_it' are within 'alliances_used_pax'
    pax_need_replanning_w_it_options.loc[mask, 'alliances_match'] = True
    # DataFrame where at least one alliance in 'alliances_used_it' is NOT in 'alliances_used_pax'
    pax_need_replanning_w_it_options.loc[~mask, 'alliances_match'] = False

    pax_need_replanning_w_it_options['same_path'] = pax_need_replanning_w_it_options['path_pax'] == \
                                                    pax_need_replanning_w_it_options['path_it']

    pax_need_replanning_w_it_options['delay_departure_home'] = (pax_need_replanning_w_it_options['dept_home_utc_it'] -
                                                                pax_need_replanning_w_it_options[
                                                                    'dept_home_utc_pax']).dt.total_seconds() / 60
    pax_need_replanning_w_it_options['delay_arrival_home'] = (pax_need_replanning_w_it_options['arr_home_utc_it'] -
                                                              pax_need_replanning_w_it_options[
                                                                  'arr_home_utc_pax']).dt.total_seconds() / 60
    pax_need_replanning_w_it_options['delay_total_travel_time'] = pax_need_replanning_w_it_options[
                                                                      'total_travel_time_it'] - \
                                                                  pax_need_replanning_w_it_options[
                                                                      'total_travel_time_pax']
    pax_need_replanning_w_it_options['extra_services'] = pax_need_replanning_w_it_options['nservices_it'] - \
                                                         pax_need_replanning_w_it_options['nservices_pax']
    pax_need_replanning_w_it_options['same_initial_node'] = pax_need_replanning_w_it_options['node_pax_0'] == \
                                                            pax_need_replanning_w_it_options['node_it_0']
    pax_need_replanning_w_it_options['same_final_node'] = pax_need_replanning_w_it_options['node_pax_n'] == \
                                                          pax_need_replanning_w_it_options['node_it_0']
    pax_need_replanning_w_it_options['same_modes'] = pax_need_replanning_w_it_options.apply(
        lambda row: set(row['modes_pax']) == set(row['modes_it']), axis=1)

    return pax_need_replanning_w_it_options


def filter_options_pax_it_w_constraints(pax_need_replanning_w_it_options, dict_constraints):
    # Filter rows that don't comply with restrictions

    # respect (or not) new itineraries same alliances as planned pax it
    respect_alliances_pax_it_new_it = dict_constraints.get('respect_alliances_pax_it_new_it', False)

    # keep only the same paths from planned to actual
    respect_path = dict_constraints.get('respect_path', False)

    # Respect modes between pax planned and itinerary
    respect_modes = dict_constraints.get('respect_modes', False)

    # keep initial node the same
    initial_node_same = dict_constraints.get('initial_node_same', False)

    # keep the final node the same
    final_node_same = dict_constraints.get('final_node_same', False)

    # allow departure (from home) before initial pax planned
    departure_before_pax_it = dict_constraints.get('departure_before_pax_it', True)


    pax_need_replanning_w_it_options_kept = pax_need_replanning_w_it_options.copy()

    if respect_alliances_pax_it_new_it:
        pax_need_replanning_w_it_options_kept = pax_need_replanning_w_it_options_kept[
            pax_need_replanning_w_it_options_kept.alliances_match]
    if respect_path:
        pax_need_replanning_w_it_options_kept = pax_need_replanning_w_it_options_kept[
            pax_need_replanning_w_it_options_kept.same_path]
    if respect_modes:
        pax_need_replanning_w_it_options_kept = pax_need_replanning_w_it_options_kept[
            pax_need_replanning_w_it_options_kept.same_modes]
    if initial_node_same:
        pax_need_replanning_w_it_options_kept = pax_need_replanning_w_it_options_kept[
            pax_need_replanning_w_it_options_kept.same_initial_node]
    if final_node_same:
        pax_need_replanning_w_it_options_kept = pax_need_replanning_w_it_options_kept[
            pax_need_replanning_w_it_options_kept.same_final_node]
    if not departure_before_pax_it:
        pax_need_replanning_w_it_options_kept = pax_need_replanning_w_it_options_kept[
            (pax_need_replanning_w_it_options_kept.dept_home_utc_it >=
             pax_need_replanning_w_it_options_kept.dept_home_utc_pax)]

    return pax_need_replanning_w_it_options_kept




