from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
import pytz
from pyomo.environ import minimize, Var, Constraint, maximize

import sys
sys.path.insert(1, '../..')

from strategic_evaluator.pax_reassigning_replanned_network import (compute_pax_status_in_replanned_network,
                                                                   compute_capacities_available_services)
from strategic_evaluator.apply_replan_network import  (replan_rail_timetable, replan_flight_schedules,
                                                       compute_itineraries_in_replanned_network,
                                                       compute_alternatives_possibles_pax_itineraries,
                                                       filter_options_pax_it_w_constraints)

from libs.uow_tool_belt.general_tools import recreate_output_folder
from libs.general_tools_logging_config import (save_information_config_used, important_info, setup_logging, IMPORTANT_INFO,
                                               process_strategic_config_file)
from libs.time_converstions import  convert_to_utc_vectorized
from libs.passenger_assigner.passenger_assigner import create_model_passenger_reassigner_pyomo
from libs.passenger_assigner.lexicographic_lib import lexicographic_optimization


def parse_time_with_date(time_str, base_date):
    """ Convert HH:MM:SS string to datetime, allowing 24+ hour format """
    hours, minutes, seconds = map(int, time_str.split(":"))
    full_datetime = datetime.combine(base_date, datetime.min.time()) + timedelta(hours=hours, minutes=minutes,
                                                                                 seconds=seconds)
    return full_datetime


def pre_process_rail_input(rail_time_table, base_date, df_stops=None):
    # Remove rows without times
    rail_time_table = rail_time_table[(~rail_time_table['arrival_time'].isna()) & (~rail_time_table['departure_time'].isna())].copy()

    # Apply function to both time columns
    rail_time_table['arrival_datetime'] = rail_time_table['arrival_time'].apply(
        lambda t: parse_time_with_date(t, base_date))
    rail_time_table['departure_datetime'] = rail_time_table['departure_time'].apply(
        lambda t: parse_time_with_date(t, base_date))

    if df_stops is not None:
        rail_time_table = rail_time_table.merge(df_stops[['stop_id', 'stop_lat', 'stop_lon']], on='stop_id')
        # Arrival time in UTC and Local
        rail_time_table[['departure_time_utc', 'departure_time_utc_tz',
                    'departure_time_local', 'departure_time_local_tz']] = pd.DataFrame(
            rail_time_table.apply(lambda x: convert_to_utc_vectorized(x.stop_lon, x.stop_lat, x.departure_datetime),
                             axis=1).tolist(),
            index=rail_time_table.index)
        rail_time_table[['arrival_time_utc', 'arrival_time_utc_tz',
                    'arrival_time_local', 'arrival_time_local_tz']] = pd.DataFrame(
            rail_time_table.apply(lambda x: convert_to_utc_vectorized(x.stop_lon, x.stop_lat, x.arrival_datetime),
                             axis=1).tolist(),
            index=rail_time_table.index)

    rail_time_table_ids = rail_time_table[['trip_id']].drop_duplicates().rename({'trip_id': 'service_id'}, axis=1)
    rail_time_table_ids['service_id'] = rail_time_table_ids['service_id'].astype(str)
    rail_time_table_ids['from'] = np.nan
    rail_time_table_ids['to'] = np.nan

    return rail_time_table, rail_time_table_ids


def run_reassigning_pax_replanning_pipeline(toml_config, pc=1, n_paths=15, n_itineraries=50,
                                max_connections=1, pre_processed_version=0,
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
        recreate_output_folder(toml_config['output']['output_folder'],
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

    path_stops_trains = (Path(toml_config['general']['experiment_path']) /
                     toml_config['network_definition']['rail_network'][0]['gtfs'] / 'stops.txt')

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

    # Read planned flights
    fs_planned = pd.read_csv(path_planned_flights)

    # Convert to datetime (without timezone first)
    def parse_offset_to_tzinfo(tz_str):
        """Convert '+HH:MM' or '-HH:MM' to a fixed timezone offset."""
        sign = 1 if tz_str[0] == '+' else -1
        hours, minutes = map(int, tz_str[1:].split(':'))
        total_minutes = sign * (hours * 60 + minutes)
        return pytz.FixedOffset(total_minutes)

    fs_planned['sibt'] = pd.to_datetime(fs_planned['sibt'], errors='coerce')
    fs_planned['sobt'] = pd.to_datetime(fs_planned['sobt'], errors='coerce')
    fs_planned['sibt_local'] = pd.to_datetime(fs_planned['sibt_local'], errors='coerce')
    fs_planned['sobt_local'] = pd.to_datetime(fs_planned['sobt_local'], errors='coerce')

    # Apply timezone using offsets
    fs_planned['sibt'] = fs_planned.apply(
        lambda row: row['sibt'].tz_localize(parse_offset_to_tzinfo(row["sibt_tz"]))
        if pd.notna(row['sibt']) else pd.NaT,
        axis=1
    )

    fs_planned['sobt'] = fs_planned.apply(
        lambda row: row['sobt'].tz_localize(parse_offset_to_tzinfo(row["sobt_tz"]))
        if pd.notna(row['sobt']) else pd.NaT,
        axis=1
    )

    fs_planned['sibt_local'] = fs_planned.apply(
        lambda row: row['sibt_local'].tz_localize(parse_offset_to_tzinfo(row['sibt_local_tz']))
        if pd.notna(row['sibt_local']) else pd.NaT,
        axis=1
    )

    fs_planned['sobt_local'] = fs_planned.apply(
        lambda row: row['sobt_local'].tz_localize(parse_offset_to_tzinfo(row['sobt_local_tz']))
        if pd.notna(row['sobt_local']) else pd.NaT,
        axis=1
    )

    # Read planned trains
    rs_planned = pd.read_csv(path_planned_trains,
                             dtype={'trip_id': 'string', 'stop_id': 'string'})

    rs_planned = rs_planned[(~rs_planned['arrival_time'].isna()) & (~rs_planned['departure_time'].isna())].copy()

    # Baseline for rail and local/UTC times
    # Read stops info in GTFS rail
    # Needed for coordinates to compute time in UTC from GTFS rail data
    df_stops = pd.read_csv(path_stops_trains, dtype={'stop_id': 'string'})

    date_to_set_rail = toml_config['network_definition']['rail_network'][0]['date_to_set_rail']  # "20190906"
    base_date = datetime.strptime(date_to_set_rail, "%Y%m%d").date()
    rs_planned, rs_planned_ids = pre_process_rail_input(rs_planned, base_date, df_stops)


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
        flights_replanned['sobt'] = pd.to_datetime(flights_replanned['sobt'] + flights_replanned['sobt_tz'])
        flights_replanned['sibt'] = pd.to_datetime(flights_replanned['sibt'] + flights_replanned['sibt_tz'])
        flights_replanned['sobt_local'] = pd.to_datetime(flights_replanned['sobt_local'] + flights_replanned['sobt_local_tz'])
        flights_replanned['sibt_local'] = pd.to_datetime(flights_replanned['sibt_local'] + flights_replanned['sibt_local_tz'])

    trains_replanned = None
    trains_replanned_ids = None
    if path_trains_replanned.exists():
        trains_replanned = pd.read_csv(path_trains_replanned,
                                     dtype={'trip_id': 'string', 'stop_id': 'string'})
        trains_replanned, trains_replanned_ids = pre_process_rail_input(trains_replanned, base_date, df_stops)

    # Read additional services
    flights_added = None
    additonal_seats = None
    if path_flights_additional.exists():
        flights_added = pd.read_csv(path_flights_additional)
        flights_added['sobt'] = pd.to_datetime(flights_added['sobt'] + flights_added['sobt_tz'])
        flights_added['sibt'] = pd.to_datetime(flights_added['sibt'] + flights_added['sibt_tz'])
        flights_added['sobt_local'] = pd.to_datetime(flights_added['sobt_local'] + flights_added['sobt_local_tz'])
        flights_added['sibt_local'] = pd.to_datetime(flights_added['sibt_local'] + flights_added['sibt_local_tz'])
        additonal_seats = flights_added[['service_id', 'seats']].copy().rename(columns={'seats': 'capacity'})
        additonal_seats['type'] = 'flight'

    trains_added = None
    if path_trains_additional.exists():
        trains_added = pd.read_csv(path_trains_additional, dtype={'trip_id': 'string', 'stop_id': 'string'})
        trains_added, trains_added_ids = pre_process_rail_input(trains_added, base_date, df_stops)

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
    if additonal_seats is not None:
        # If we have additional seats add them so they are considered
        # in the dictionary of seats per sevice
        seats_in_service = pd.concat([seats_in_service, (additonal_seats[['service_id', 'capacity', 'type']].
                                                         rename(columns={'service_id': 'nid',
                                                                         'capacity': 'max_seats',
                                                                         'type': 'mode_transport'}))])

    dict_seats_service = (seats_in_service.groupby('mode_transport')[['nid', 'max_seats']]
                          .apply(lambda g: dict(zip(g['nid'], g['max_seats'])))
                          .to_dict())


    #####################################################################
    # First adjust rail and flight network with modifications replanned #
    ####################################################################

    rs_replanned, dict_seats_service = replan_rail_timetable(rs_planned,
                                                             rail_replanned=trains_replanned,
                                                             rail_cancelled=trains_cancelled,
                                                             rail_added=trains_added,
                                                             dict_seats_service=dict_seats_service)

    fs_replanned, dict_seats_service = replan_flight_schedules(fs_planned,
                                                               fs_replanned=flights_replanned,
                                                               fs_cancelled=flights_cancelled,
                                                               fs_added=flights_added,
                                                               dict_seats_service=dict_seats_service)


    # Save updated replanned network
    # In this case remove the TimeZone from the SOBT/SIBT
    fs_replanned_to_save = fs_replanned.copy()
    fs_replanned_to_save['sobt'] = fs_replanned_to_save['sobt'].apply(
        lambda x: str(x).split("+")[0] if '+' in str(x)
        else "-".join(str(x).split("-")[0:-1]) if '-' in str(x)
        else str(x))
    fs_replanned_to_save['sibt'] = fs_replanned_to_save['sibt'].apply(
        lambda x: str(x).split("+")[0] if '+' in str(x)
        else "-".join(str(x).split("-")[0:-1]) if '-' in str(x)
        else str(x))

    fs_replanned_to_save['sobt_local'] = fs_replanned_to_save['sobt_local'].apply(
        lambda x: str(x).split("+")[0] if '+' in str(x)
        else "-".join(str(x).split("-")[0:-1]) if '-' in str(x)
        else str(x))
    fs_replanned_to_save['sibt_local'] = fs_replanned_to_save['sibt_local'].apply(
        lambda x: str(x).split("+")[0] if '+' in str(x)
        else "-".join(str(x).split("-")[0:-1]) if '-' in str(x)
        else str(x))

    fs_replanned_to_save.to_csv((output_folder_path /
                         ('flight_schedules_proc_' + str(
                                     pre_processed_version) + '.csv')),
                                index=False)

    rs_replanned.to_csv((output_folder_path /
                         ('rail_timetable_all_gtfs_' + str(
                             pre_processed_version) + '.csv')),
                        index=False)

    ########################################
    #  Then identify pax need reassigning  #
    #######################################
    logger.important_info("Identifying status of passengers planned in replanned network")

    pax_assigned_planned, pax_kept, pax_need_replannning = compute_pax_status_in_replanned_network(pax_assigned_planned,
                                                                                                   fs_replanned,
                                                                                                   rs_replanned,
                                                                                                   flights_cancelled, trains_cancelled,
                                                                                                   flights_replanned, trains_replanned_ids,
                                                                                                   dict_mcts)

    # Save pax_assigned_planned with their status due to replanning
    pax_assigned_planned.to_csv((toml_config['output']['output_folder'] /
                                 ('pax_assigned_to_itineraries_options_status_replanned_'+ str(pre_processed_version) +'.csv')),
                                 index=False)


    ########################################
    #  Then identify pax need reassigning  #
    #######################################

    # Pax assigned final
    pax_assigned_final = pax_assigned_planned.copy()
    # Empty dataframe of pax_stranded
    pax_stranded = pax_assigned_final.iloc[0:0].copy()

    if len(pax_need_replannning) > 0:
        # Have some pax that need replanning
        # Else we're done

        # Compute capacities available in services
        #df_stops_considered = pd.read_csv(path_stops_trains_used, dtype=str)
        #rs_replanned_filtered = rs_replanned[rs_replanned.stop_id.isin(df_stops_considered.stop_id)].copy()

        services_w_capacity, services_wo_capacity = compute_capacities_available_services(demand=pax_kept,
                                                                                          dict_services_w_capacity=dict_seats_service)

        services_w_capacity.to_csv((toml_config['output']['output_folder'] /
                                    ('services_w_capacity_'+ str(pre_processed_version)+'.csv')), index=False)
        services_wo_capacity.to_csv((toml_config['output']['output_folder'] /
                                     ('services_wo_capacity_'+ str(pre_processed_version)+'.csv')), index=False)

        # Compute capacity available overall per service
        capacity_available = pd.concat([services_w_capacity, services_wo_capacity])
        capacity_available['capacity'] = capacity_available['max_seats_service'] - capacity_available['max_pax_in_service']
        capacity_available = capacity_available[['service_id', 'type', 'capacity']]

        capacity_available.to_csv((toml_config['output']['output_folder'] /
                                   ('services_capacities_'+ str(pre_processed_version)+'.csv')), index=False)

        capacity_available.to_csv((toml_config['output']['output_folder'] / 'services_capacity_available.csv'), index=False)

        # Compute o-d demand that needs to be reassigned so that suitable itineraries can be computed
        # Get o-d with total demand that needs reacommodating
        od_demand_need_reaccomodating = pax_need_replannning.groupby(['origin', 'destination'])['pax'].sum().reset_index()
        # Put only one archetype, we only need one example per o-d pair
        od_demand_need_reaccomodating['archetype'] = 'archetype_0'
        # Set date as date_to_set_rail, if rail exist, as that's the date of the flight (not that it matters, I think)
        od_demand_need_reaccomodating['date'] = toml_config.get('network_definition').get('rail_network',[{}])[0].get('date_to_set_rail', '20190906')
        od_demand_need_reaccomodating = od_demand_need_reaccomodating.rename({'pax': 'trips'}, axis=1)
        od_demand_need_reaccomodating.to_csv((toml_config['output']['output_folder'] /
                                              ('demand_missing_reaccomodate_'+ str(pre_processed_version) +'.csv')),
                                             index=False)

        # Modify the toml config so that demand is the newly created file
        toml_config['demand']['demand'] = Path((toml_config['output']['output_folder'] /
                                           ('demand_missing_reaccomodate_' + str(pre_processed_version) + '.csv')))

        # Add path to  pre-processed input (flight and rail timetable processed)
        toml_config['general']['pre_processed_input_folder'] = toml_config['general']['output_folder']
        toml_config['network_definition']['pre_processed_input_folder'] = toml_config['general']['pre_processed_input_folder']
        toml_config['network_definition']['processed_folder'] = (toml_config['general']['output_folder'] + '/processed')

        recreate_output_folder((Path(toml_config['network_definition']['network_path']) /
                                toml_config['network_definition']['processed_folder']))

        allow_mixed_operators_itineraries = not toml_config['replanning_considerations']['constraints']['new_itineraries_respect_alliances']

        df_itineraries = compute_itineraries_in_replanned_network(toml_config,
                                                                  pc = pc,
                                                                  n_paths = n_paths,
                                                                  n_itineraries = n_itineraries,
                                                                  max_connections = max_connections,
                                                                  allow_mixed_operators_itineraries = allow_mixed_operators_itineraries,
                                                                  use_heuristics_precomputed = use_heuristics_precomputed,
                                                                  pre_processed_version = pre_processed_version,
                                                                  capacity_available = capacity_available)


        pax_need_replanning_w_it_options = compute_alternatives_possibles_pax_itineraries(pax_need_replannning,
                                                                                          df_itineraries,
                                                                                          fs_planned,
                                                                                          rs_planned)

        pax_need_replanning_w_it_options.to_csv((toml_config['output']['output_folder'] /
                                                 ('pax_need_replanning_w_it_options' +
                                                      str(pre_processed_version) + '.csv')), index=False)

        pax_need_replanning_w_it_options_kept = pax_need_replanning_w_it_options
        if toml_config['replanning_considerations'].get('constraints') is not None:
            pax_need_replanning_w_it_options_kept = filter_options_pax_it_w_constraints(pax_need_replanning_w_it_options,
                                                                                            toml_config['replanning_considerations']['constraints'])

        pax_need_replanning_w_it_options_kept.to_csv((toml_config['output']['output_folder'] /
                                                      ('pax_need_replanning_w_it_options_filtered_w_constraints_' +
                                                      str(pre_processed_version) + '.csv')), index=False)

        # id_pax_groups_stranded --> there are no option for them at all
        pax_groups_stranded = pax_need_replannning[~pax_need_replannning.pax_group_id.isin(pax_need_replanning_w_it_options_kept.pax_group_id)]

        pax_groups_stranded.to_csv((toml_config['output']['output_folder'] /
                                    ('pax_stranded_' + str(pre_processed_version) + '.csv')), index=False)

        if len(pax_need_replanning_w_it_options_kept) > 0:
            logger.important_info("Assigning passengers to services")

            # We have pax with options, need to optimise the assigment
            capacity_services_w_capacity = capacity_available[capacity_available.capacity>0].copy()

            pax_reassigning = pax_need_replanning_w_it_options_kept[(['pax','pax_group_id', 'option'] +
            [col for col in pax_need_replanning_w_it_options_kept.columns if col.startswith("service_id_")] +
            [col for col in pax_need_replanning_w_it_options_kept.columns if col.startswith("mode_")] +
            ['alliances_match', 'same_path', 'delay_departure_home', 'delay_arrival_home', 'delay_total_travel_time',
             'extra_services', 'same_initial_node', 'same_final_node', 'same_modes'])].copy()

            # Filter only columns that start with 'service_id_' but NOT 'service_id_pax_'
            service_cols = [col for col in pax_reassigning.columns if
                            col.startswith("service_id_") and not col.startswith("service_id_pax_")]

            # Stack those columns and drop NaNs
            all_service_ids = pax_reassigning[service_cols].stack().dropna()

            # Get unique values as a list
            unique_service_ids = all_service_ids.unique().tolist()

            # Get services that are available (used) and with their capacities
            services_available_w_capacity = capacity_available[((capacity_available.capacity >= 1) &
                                                               (capacity_available.service_id.isin(unique_service_ids)))].copy()

            if len(unique_service_ids) != len(services_available_w_capacity):
                logger.important_info("WARNING/ERROR: The number of unique services used in new itineraries ("+
                                      str(len(unique_service_ids))+") is different to the number of services with capacity "
                                                                   "which overlap with the services used in the itineraries ("+
                                      str(len(services_available_w_capacity))+").")


            #dict_mode_transport = services_available_w_capacity.set_index('service_id')['type'].to_dict()
            #dict_service_capacity = services_available_w_capacity.set_index('service_id')['capacity'].to_dict()
            # Use capacity_available instead of service_available_w_capacity as the later is filtered by used and maybe
            # some stops combinations are not used but passed through... to need to keep their capacity checked.
            dict_mode_transport = capacity_available.set_index('service_id')['type'].to_dict()
            dict_service_capacity = capacity_available.set_index('service_id')['capacity'].to_dict()

            dict_volume = pax_reassigning[['pax_group_id', 'pax']].drop_duplicates().set_index('pax_group_id')['pax'].to_dict()

            objectives_names = [o[0] for o in toml_config['replanning_considerations']['optimisation']['objectives']]

            model = create_model_passenger_reassigner_pyomo(it_data = pax_reassigning.copy(),
                                                            dict_volume = dict_volume,
                                                            dict_sc = dict_service_capacity,
                                                            dict_mode_transport = dict_mode_transport,
                                                            objectives = objectives_names,
                                                            pc = pc)

            # Count variables and constraints
            num_vars = len([v for v in model.component_data_objects(Var)])
            num_constraints = len([c for c in model.component_data_objects(Constraint)])
            print(f"Number of variables: {num_vars}")
            print(f"Number of constraints: {num_constraints}")

            objectives = [(o[0], maximize) if o[1] == 'maximize' else (o[0], minimize) for o in
                          toml_config['replanning_considerations']['optimisation']['objectives']]

            thresholds = toml_config['replanning_considerations']['optimisation']['thresholds']
            solver = toml_config['replanning_considerations']['optimisation']['solver']
            nprocs = min(pc, toml_config['replanning_considerations']['optimisation']['nprocs'])

            results = lexicographic_optimization(model, objectives, solver=solver, thresholds=thresholds,
                                                 num_threads=nprocs)


            # Process results

            # Create a dictionary of (it, opt) to pax values from the model
            assigned_pax_dict = {(id, opt): model.x[id, opt].value for id, opt in model.x}

            # Convert the dictionary to a DataFrame for efficient merging
            assigned_pax_df = pd.DataFrame(
                list(assigned_pax_dict.items()), columns=["it_opt", "pax"]
            )

            # Split the tuple column into separate 'it' and 'opt' columns
            assigned_pax_df[["id", "option"]] = pd.DataFrame(assigned_pax_df["it_opt"].tolist(),
                                                             index=assigned_pax_df.index)

            # Drop the tuple column as it's no longer needed
            assigned_pax_df = assigned_pax_df.drop(columns=["it_opt"])

            assigned_pax_df.rename(columns={'id': 'pax_group_id', 'pax':'pax_assigned'}, inplace=True)


            # Merge with the original it_data
            pax_reassigning = pax_reassigning.merge(assigned_pax_df, on=["pax_group_id", "option"], how="left")
            pax_reassigning["pax_assigned"] = pax_reassigning["pax_assigned"].apply(lambda x: 0 if x == -0.0 else x)

            # Fill missing pax values with 0
            pax_reassigning["pax_assigned"] = pax_reassigning["pax_assigned"].fillna(0)

            print("Optimization Results: " + str(results))
            print("Total num pax assigned: " + str(pax_reassigning.pax_assigned.sum()))

            logging.info("Optimization Results: " + str(results))
            logging.info("Total num pax assigned: " + str(pax_reassigning.pax.sum()))

            pax_reassigning.to_csv((toml_config['output']['output_folder'] /
                                                 ('pax_reassigned_results_solver_' +
                                                      str(pre_processed_version) + '.csv')), index=False)


    pax_assigned_final.to_csv((output_folder_path /
                               ('pax_assigned_to_itineraries_options_replanned_' + str(
                                     pre_processed_version) + '.csv')),
                                index=False)
    pax_stranded.to_csv((output_folder_path /
                               ('pax_assigned_to_itineraries_options_replanned_stranded_' + str(
                                     pre_processed_version) + '.csv')),
                                index=False)

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

    parser.add_argument('-eo', '--end_output_folder', help='Ending to be added to output folder',
                        required=False)


    # Parse parameters
    args = parser.parse_args()

    setup_logging(args.verbose, log_to_console=(args.verbose > 0), log_to_file=args.log_file, file_reset=True)

    logger = logging.getLogger(__name__)

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
                                            use_heuristics_precomputed=args.use_heuristics_precomputed)

