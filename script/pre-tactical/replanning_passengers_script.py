from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
import sys
sys.path.insert(1, '../..')

from strategic_evaluator.pax_reassigning_replanned_network import (compute_pax_status_in_replanned_network,
                                                                   compute_capacities_available_services)

from libs.uow_tool_belt.general_tools import recreate_output_folder
from libs.general_tools_logging_config import (save_information_config_used, important_info, setup_logging, IMPORTANT_INFO,
                                               process_strategic_config_file)


def parse_time_with_date(time_str, base_date):
    """ Convert HH:MM:SS string to datetime, allowing 24+ hour format """
    hours, minutes, seconds = map(int, time_str.split(":"))
    full_datetime = datetime.combine(base_date, datetime.min.time()) + timedelta(hours=hours, minutes=minutes,
                                                                                 seconds=seconds)
    return full_datetime


def pre_process_rail_input(rail_time_table, base_date):
    # Remove rows without times
    rail_time_table = rail_time_table[(~rail_time_table['arrival_time'].isna()) & (~rail_time_table['departure_time'].isna())].copy()

    # Apply function to both time columns
    rail_time_table['arrival_datetime'] = rail_time_table['arrival_time'].apply(
        lambda t: parse_time_with_date(t, base_date))
    rail_time_table['departure_datetime'] = rail_time_table['departure_time'].apply(
        lambda t: parse_time_with_date(t, base_date))

    rail_time_table_ids = rail_time_table[['trip_id']].drop_duplicates().rename({'trip_id': 'service_id'}, axis=1)
    rail_time_table_ids['service_id'] = rail_time_table_ids['service_id'].astype(str)
    rail_time_table_ids['from'] = np.nan
    rail_time_table_ids['to'] = np.nan

    return rail_time_table, rail_time_table_ids


def run_reassigning_pax_replanning_pipeline(toml_config, pc=1, n_paths=15, n_itineraries=50,
                                max_connections=1, pre_processed_version=0,
                                allow_mixed_operators_itineraries=False,
                                use_heuristics_precomputed=False,
                                recreate_output_fld=True):

    start_pipeline_time = time.time()

    ###########################
    # Recreate output folders #
    ##########################
    output_folder_path = (Path(toml_config['general']['experiment_path']) /
                               toml_config['general']['output_folder'])

    if recreate_output_fld:
        # Check if output folder exists, if not create it
        recreate_output_folder(output_folder_path,
                               delete_previous=True,
                               logger=logger)


    #######################################
    # Read all inputs that will be needed #
    ######################################
    ### Definition all paths ###

    # Planned operations
    path_planned_pax_assigned = (Path(toml_config['general']['experiment_path']) /
                                 toml_config['planned_network_info']['planned_network'] /
                                 toml_config['planned_network_info']['path_results'] /
                                 ('pax_assigned_to_itineraries_options_' +
                                  str(toml_config['planned_network_info']['precomputed']) + '.csv'))

    path_planned_flights = (Path(toml_config['general']['experiment_path']) /
                               toml_config['planned_network_info']['planned_network'] /
                               ('flight_schedules_proc_' +
                                str(toml_config['planned_network_info']['precomputed']) +
                                '.csv'))

    path_planned_trains = (Path(toml_config['general']['experiment_path']) /
                               toml_config['planned_network_info']['planned_network'] /
                               ('rail_timetable_all_gtfs_' +
                                str(toml_config['planned_network_info']['precomputed']) +
                                '.csv'))

    # Replanned operations
    path_cancelled_flights = (Path(toml_config['general']['experiment_path']) /
                             toml_config['general']['replanned_input_folder'] /
                             ('flights_cancelled_' + str(pre_processed_version) + '.csv'))
    path_cancelled_rail = (Path(toml_config['general']['experiment_path']) /
                           toml_config['general']['replanned_input_folder'] /
                           ('rail_cancelled_' + str(pre_processed_version) + '.csv'))
    path_flights_replanned = (Path(toml_config['general']['experiment_path']) /
                              toml_config['general']['replanned_input_folder'] /
                              ('flight_replanned_proc_' + str(pre_processed_version) + '.csv'))
    path_trains_replanned = (Path(toml_config['general']['experiment_path']) /
                             toml_config['general']['replanned_input_folder'] /
                             ('rail_timetable_replanned_all_gtfs_' + str(pre_processed_version) + '.csv'))
    path_flights_additional = (Path(toml_config['general']['experiment_path']) /
                             toml_config['general']['replanned_input_folder'] /
                             ('flight_added_schedules_proc_' + str(pre_processed_version) + '.csv'))
    path_trains_additional = (Path(toml_config['general']['experiment_path']) /
                             toml_config['general']['replanned_input_folder'] /
                             ('rail_timetable_added_all_gtfs_' + str(pre_processed_version) + '.csv'))

    # Infrastructure paths (MCTs)
    path_mct_rail = (Path(toml_config['general']['experiment_path']) /
                     toml_config['network_definition']['rail_network'][0]['mct_rail'])
    mct_default_rail = int(toml_config['network_definition']['rail_network'][0]['mct_default'])
    path_mct_air = (Path(toml_config['general']['experiment_path']) /
                     toml_config['network_definition']['air_network'][0]['mct_air'])
    mct_default_air = int(toml_config['network_definition']['air_network'][0]['mct_default'])

    path_mct_layers = (Path(toml_config['general']['experiment_path']) /
                       toml_config['planned_network_info']['planned_network'] /
                       'transition_layer_connecting_times.csv')

    # Others
    path_seats_service = (Path(toml_config['general']['experiment_path']) /
                                 toml_config['planned_network_info']['planned_network'] /
                                 toml_config['planned_network_info']['path_results'] /
                                 ('pax_assigned_seats_max_target_' +
                                  str(toml_config['planned_network_info']['precomputed']) + '.csv'))

    ### Read planned operations ###
    # Passenger assigned from planned network
    pax_assigned_planned = pd.read_csv(path_planned_pax_assigned)
    pax_assigned_planned = pax_assigned_planned[pax_assigned_planned.pax > 0].copy()
    pax_assigned_planned['pax_group_id'] = pax_assigned_planned.index

    # Read planned flights
    fs_planned = pd.read_csv(path_planned_flights)
    fs_planned['sobt'] = pd.to_datetime(fs_planned['sobt'])
    fs_planned['sibt'] = pd.to_datetime(fs_planned['sibt'])
    fs_planned['sobt_local'] = pd.to_datetime(fs_planned['sobt_local'])
    fs_planned['sibt_local'] = pd.to_datetime(fs_planned['sibt_local'])

    # Read planned trains
    rs_planned = pd.read_csv(path_planned_trains,
                             dtype={'trip_id': 'string', 'stop_id': 'string'})

    # Baseline for rail
    date_to_set_rail = toml_config['network_definition']['rail_network'][0]['date_to_set_rail']  # "20190906"
    base_date = datetime.strptime(date_to_set_rail, "%Y%m%d").date()
    rs_planned, rs_planned_ids = pre_process_rail_input(rs_planned, base_date)


    ### Read replanned operations ###
    # Read cancelled services
    flights_cancelled = None
    if path_cancelled_flights.exists():
        flights_cancelled = pd.read_csv(path_cancelled_flights)

    trains_cancelled = None
    if path_cancelled_rail.exists():
        trains_cancelled = pd.read_csv(path_cancelled_rail, dtype={'service_id': 'string'})

    # Read replanned services
    flights_replanned = None
    if path_flights_replanned.exists():
        flights_replanned = pd.read_csv(path_flights_replanned)
        flights_replanned['sobt'] = pd.to_datetime(flights_replanned['sobt'])
        flights_replanned['sibt'] = pd.to_datetime(flights_replanned['sibt'])
        flights_replanned['sobt_local'] = pd.to_datetime(flights_replanned['sobt_local'])
        flights_replanned['sibt_local'] = pd.to_datetime(flights_replanned['sibt_local'])

    trains_replanned = None
    trains_replanned_ids = None
    if path_trains_replanned.exists():
        trains_replanned = pd.read_csv(path_trains_replanned,
                                     dtype={'trip_id': 'string', 'stop_id': 'string'})
        trains_replanned, trains_replanned_ids = pre_process_rail_input(trains_replanned, base_date)

    # Read additional services
    flights_added = None
    if path_flights_additional.exists():
        flights_added = pd.read_csv(path_flights_additional)

    trains_added = None
    if path_trains_additional.exists():
        trains_added = pd.read_csv(path_trains_additional, dtype={'trip_id': 'string', 'stop_id': 'string'})
        trains_added, trains_added_ids = pre_process_rail_input(trains_added, base_date)

    # Read MCTs
    mct_default_rail = mct_default_rail
    dict_mct_rail = pd.read_csv(path_mct_rail, dtype={'stop_id': 'string'}).set_index('stop_id')['default transfer time'].to_dict()

    mct_default_air = mct_default_air
    dict_mct_air = pd.read_csv(path_mct_air).set_index('icao_id').to_dict(orient='index')


    df_mct_layers = pd.read_csv(path_mct_layers)
    df_mct_layers['od'] = df_mct_layers['origin'] + '_' + df_mct_layers['destination']
    dict_mct_rail_rail = \
        df_mct_layers[
            ((df_mct_layers['layer_id_origin'] == 'rail') & (df_mct_layers['layer_id_destination'] == 'rail'))][
            ['od', 'mct']].set_index('od')['mct'].to_dict()
    dict_mct_air_rail = \
        df_mct_layers[
            ((df_mct_layers['layer_id_origin'] == 'air') & (df_mct_layers['layer_id_destination'] == 'rail'))][
            ['od', 'mct']].set_index('od')['mct'].to_dict()
    dict_mct_rail_air = \
        df_mct_layers[
            ((df_mct_layers['layer_id_origin'] == 'rail') & (df_mct_layers['layer_id_destination'] == 'air'))][
            ['od', 'mct']].set_index('od')['mct'].to_dict()
    dict_mct_air_air = \
        df_mct_layers[
            ((df_mct_layers['layer_id_origin'] == 'air') & (df_mct_layers['layer_id_destination'] == 'air'))][
            ['od', 'mct']].set_index('od')['mct'].to_dict()

    dict_mcts = {'air': dict_mct_air,
                 'rail': dict_mct_rail,
                 'default_air': mct_default_air,
                 'default_rail': mct_default_rail,
                 'rail_rail': dict_mct_rail_rail,
                 'air_rail': dict_mct_air_rail,
                 'rail_air': dict_mct_rail_air,
                 'air_air': dict_mct_air_air}

    # Read Seats in Services
    seats_in_service = pd.read_csv(path_seats_service)
    dict_seats_service = seats_in_service[['nid','max_seats']].set_index(['nid'])['max_seats'].to_dict()


    ########################################
    # First identify pax need reassigning #
    #######################################
    logger.important_info("Identifying status of passengers planned in replanned network")

    pax_assigned_planned, pax_kept, pax_need_replannning = compute_pax_status_in_replanned_network(pax_assigned_planned,
                                                                                                   fs_planned,
                                                                                                   rs_planned,
                                                                                                   flights_cancelled, trains_cancelled,
                                                                                                   flights_replanned, trains_replanned, trains_replanned_ids,
                                                                                                   dict_mcts)

    # Save pax_assigned_planned with their status due to replanning
    pax_assigned_planned.to_csv((output_folder_path /
                                 ('pax_assigned_to_itineraries_options_status_replanned_'+ str(pre_processed_version) +'.csv')),
                                 index=False)


    # Compute capacities available in services
    services_w_capacity, services_wo_capacity = compute_capacities_available_services(pax_kept, dict_seats_service)


    # Get o-d with total demand that needs reacommodating
    od_demand_need_reaccomodating = pax_need_replannning.groupby(['origin', 'destination'])['pax'].sum().reset_index()


    end_pipeline_time = time.time()
    elapsed_time = end_pipeline_time - start_pipeline_time
    logger.important_info("Whole Replanning Pax Reassigment Pipeline computed in: " + str(elapsed_time) + " seconds.")


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
                        required=False, default=1)
    parser.add_argument('-np', '--num_paths', help='Number of paths to compute if computing potential paths',
                        required=False, default=30)
    parser.add_argument('-v', '--verbose', action='count', default=0, help="increase output verbosity")

    parser.add_argument('-lf', '--log_file', help='Path to log file', required=False)

    parser.add_argument('-pc', '--n_proc', help='Number of processors', required=False)

    parser.add_argument('-amo', '--allow_mixed_operators', help='Allow mix operators',
                        required=False, action='store_true')

    parser.add_argument('-eo', '--end_output_folder', help='Ending to be added to output folder',
                        required=False)


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

    logger.important_info("Running reassing passengers in replanned network")

    run_reassigning_pax_replanning_pipeline(toml_config,
                                            pc=pc,
                                            n_paths=int(args.num_paths),
                                            n_itineraries=int(args.num_itineraries),
                                            max_connections=int(args.max_connections),
                                            pre_processed_version=int(args.preprocessed_version),
                                            allow_mixed_operators_itineraries=args.allow_mixed_operators,
                                            use_heuristics_precomputed=args.use_heuristics_precomputed)

