from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
import pytz
import re

import sys
sys.path.insert(1, '../..')

from strategic_evaluator.pax_reassigning_replanned_network import (compute_pax_status_in_replanned_network,
                                                                   compute_capacities_available_services,
                                                                   reassign_passengers_services)
from strategic_evaluator.apply_replan_network import (replan_rail_timetable, replan_flight_schedules,
                                                      compute_itineraries_in_replanned_network,
                                                      compute_alternatives_possibles_pax_itineraries,
                                                      filter_options_pax_it_w_constraints,
                                                      preprocess_network_replanned,
                                                      create_df_delays_flights_rail_replanned_schedules,
                                                      process_pax_kept)

from libs.uow_tool_belt.general_tools import recreate_output_folder
from libs.general_tools_logging_config import (save_information_config_used, important_info, setup_logging, IMPORTANT_INFO,
                                               process_strategic_config_file)
from libs.time_converstions import  convert_to_utc_vectorized


def review_fiels_flight_schedules_SOL3(fs_sol3, fs_orig, replanning=False):
    """Function to fix the fields from flight schedules provided by SOL3 as some columns are missing, etc"""
    if replanning:
        # Only all flights from SOL3 should be in the original (or are not replanning...)
        fs_sol3 = fs_sol3[fs_sol3.service_id.isin(fs_orig.service_id)].copy()
        if len(fs_sol3) == 0:
            return None

    fs_sol3 = fs_sol3.merge(fs_orig[['service_id',
                                     'dep_terminal',
                                     'arr_terminal',
                                     'sobt_tz', 'sibt_tz',
                                     'sobt_local_tz', 'sibt_local_tz',
                                     'provider',
                                     'act_type',
                                     'gcdistance',
                                     'cost', 'emissions',
                                     'alliance']], on='service_id')

    # Keep the alliance from the original schedules
    fs_sol3.drop(['alliance_x'], axis=1, inplace=True)
    fs_sol3.rename({'alliance_y': 'alliance'}, axis='columns', inplace=True)

    # Create sobt_local and sibt_local
    def fix_24_hour_format(time_str):
        if re.search(r' 24:\d{2}:\d{2}', time_str):
            date_part = time_str[:10]
            time_part = time_str[11:]
            hours, minutes, seconds = map(int, time_part.split(':'))
            if hours == 24:
                new_date = (datetime.strptime(date_part, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
                return f"{new_date} 00:{minutes:02d}:{seconds:02d}"
        return time_str

    # Apply the fix to sobt and sibt before converting
    fs_sol3['sobt'] = fs_sol3['sobt'].apply(fix_24_hour_format)
    fs_sol3['sibt'] = fs_sol3['sibt'].apply(fix_24_hour_format)

    # Convert to datetime (assuming UTC)
    fs_sol3['sobt'] = pd.to_datetime(fs_sol3['sobt'], utc=True)
    fs_sol3['sibt'] = pd.to_datetime(fs_sol3['sibt'], utc=True)

    # Step 2: Define a function to convert timezone offset string to timedelta
    def parse_tz_offset(offset_str):
        sign = 1 if offset_str.startswith('+') else -1
        hours = int(offset_str[1:3])
        return timedelta(hours=sign * hours)

    # Step 3: Apply the timezone offset
    fs_sol3['sobt_local'] = fs_sol3.apply(lambda row: row['sobt'] + parse_tz_offset(row['sobt_local_tz']), axis=1)
    fs_sol3['sibt_local'] = fs_sol3.apply(lambda row: row['sibt'] + parse_tz_offset(row['sibt_local_tz']), axis=1)

    # Step 4: Convert back to string without timezone info
    fs_sol3['sobt_local'] = fs_sol3['sobt_local'].dt.strftime('%Y-%m-%d %H:%M:%S')
    fs_sol3['sibt_local'] = fs_sol3['sibt_local'].dt.strftime('%Y-%m-%d %H:%M:%S')

    fs_sol3['sobt'] = fs_sol3['sobt'].dt.strftime('%Y-%m-%d %H:%M:%S')
    fs_sol3['sibt'] = fs_sol3['sibt'].dt.strftime('%Y-%m-%d %H:%M:%S')

    return fs_sol3



def parse_time_with_date(time_str, base_date):
    """ Convert HH:MM:SS string to datetime, allowing 24+ hour format """
    hours, minutes, seconds = map(int, time_str.split(":"))
    full_datetime = datetime.combine(base_date, datetime.min.time()) + timedelta(hours=hours, minutes=minutes,
                                                                                 seconds=seconds)
    return full_datetime


def pre_process_rail_input(rail_time_table, base_date, df_stops=None, df_stops_considered=None):
    # Remove rows without times
    rail_time_table = rail_time_table[(~rail_time_table['arrival_time'].isna()) & (~rail_time_table['departure_time'].isna())].copy()

    if df_stops_considered is not None:
        # Filter only stops considered
        rail_time_table = rail_time_table[rail_time_table.stop_id.isin(df_stops_considered.stop_id)].copy()

    # Apply function to both time columns
    rail_time_table['arrival_datetime'] = rail_time_table['arrival_time'].apply(
        lambda t: parse_time_with_date(t, base_date))
    rail_time_table['departure_datetime'] = rail_time_table['departure_time'].apply(
        lambda t: parse_time_with_date(t, base_date))

    if df_stops is not None:
        rail_time_table = rail_time_table.merge(df_stops[['stop_id', 'stop_lat', 'stop_lon']], on='stop_id')
        rail_time_table = rail_time_table[~rail_time_table.stop_lat.isna()].copy()

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

    return rail_time_table, rail_time_table_ids['service_id'].tolist()


def run_reassigning_pax_replanning_pipeline(toml_config, pc=1, n_paths=15, n_itineraries=50,
                                max_connections=1, pre_processed_version=0,
                                use_heuristics_precomputed=False,
                                recreate_output_fld=True):

    start_pipeline_time = time.time()

    ###########################
    # Recreate output folders #
    ##########################
    output_processed_folder_path = toml_config['network_definition']['processed_folder']
    output_pax_folder_path = toml_config['output']['output_folder_pax']
    output_paths_folder_path = toml_config['output']['output_folder']

    if recreate_output_fld:
        # Check if output folder exists, if not create it
        recreate_output_folder(output_processed_folder_path,
                               delete_previous=True,
                               logger=logger)
        recreate_output_folder(output_paths_folder_path,
                               delete_previous=True,
                               logger=logger)
        recreate_output_folder(output_pax_folder_path,
                                delete_previous=True,
                               logger=logger)



    #######################################
    # Read all inputs that will be needed #
    #######################################
    logger.important_info("READING THE INPUTS")

    ### Definition all paths ###

    # Planned operations
    path_planned_pax_assigned = (Path(toml_config['general']['experiment_path']) /
                                 toml_config['planned_network_info']['planned_network'] /
                                 toml_config['planned_network_info']['path_results'] /
                                 ('pax_assigned_to_itineraries_options_' +
                                  str(toml_config['planned_network_info']['precomputed']) + '.csv'))

    path_planned_flights = (Path(toml_config['general']['experiment_path']) /
                               toml_config['planned_network_info']['planned_network'] /
                               toml_config['planned_network_info']['path_processed'] /
                               ('flight_schedules_proc_' +
                                str(toml_config['planned_network_info']['precomputed']) +
                                '.csv'))

    path_planned_trains = (Path(toml_config['general']['experiment_path']) /
                               toml_config['planned_network_info']['planned_network'] /
                               toml_config['planned_network_info']['path_processed'] /
                               ('rail_timetable_all_gtfs_' +
                                str(toml_config['planned_network_info']['precomputed']) +
                                '.csv'))

    path_stops_trains = (Path(toml_config['general']['experiment_path']) /
                     toml_config['network_definition']['rail_network'][0]['gtfs'] / 'stops.txt')

    path_stops_train_considered = (Path(toml_config['general']['experiment_path']) /
                     toml_config['network_definition']['rail_network'][0]['rail_stations_considered'])


    # Replanned operations
    replanned_actions_folder_path = (Path(toml_config['general']['experiment_path']) /
                                     toml_config['general']['replanned_input_folder'] /
                                     toml_config['general']['replanned_actions_folder'])

    path_cancelled_flights = (replanned_actions_folder_path /
                             ('flight_cancelled_' + str(pre_processed_version) + '.csv'))
    path_cancelled_rail = (replanned_actions_folder_path /
                           ('rail_cancelled_' + str(pre_processed_version) + '.csv'))
    path_flights_replanned = (replanned_actions_folder_path /
                              ('flight_replanned_proc_' + str(pre_processed_version) + '.csv'))
    path_trains_replanned = (replanned_actions_folder_path /
                             ('rail_timetable_replanned_all_gtfs_' + str(pre_processed_version) + '.csv'))
    path_flights_additional = (replanned_actions_folder_path /
                             ('flight_added_schedules_proc_' + str(pre_processed_version) + '.csv'))
    path_trains_additional = (replanned_actions_folder_path /
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
                       toml_config['planned_network_info']['path_processed'] /
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

    # Read stops considered
    df_stops_considered = pd.read_csv(path_stops_train_considered, dtype={'stop_id': 'string'})

    date_to_set_rail = toml_config['network_definition']['rail_network'][0]['date_to_set_rail']  # "20190906"
    base_date = datetime.strptime(date_to_set_rail, "%Y%m%d").date()

    # Do not pass df_stops_considered so that all stops are kept, if not they are removed and rail is not gtfs_all anymore
    rs_planned, rs_planned_ids = pre_process_rail_input(rs_planned, base_date, df_stops, df_stops_considered=None)


    ### Read replanned operations ###
    # Read cancelled services
    flights_cancelled = None
    if path_cancelled_flights.exists():
        flights_cancelled = pd.read_csv(path_cancelled_flights)
        if len(flights_cancelled) == 0:
            flights_cancelled = None

    trains_cancelled = None
    if path_cancelled_rail.exists():
        trains_cancelled = pd.read_csv(path_cancelled_rail, dtype={'service_id': 'string'})
        # Trains cancelled have the form service_id, from, to indicating which service from which stop to which stop
        # is cancelled. Note that from and to could be None, meaning from first or until last stop
        # Once could provide a cancellation file which only has service_id that would mean that to and from should
        # be added as None. Do that.
        # Ensure 'from' and 'to' columns exist, and fill missing or None values with np.nan
        for col in ['from', 'to']:
            if col not in trains_cancelled.columns:
                trains_cancelled[col] = np.nan
            else:
                trains_cancelled[col] = trains_cancelled[col].replace({None: np.nan})

        if len(trains_cancelled)==0:
            trains_cancelled = None

    # Read replanned services
    flights_replanned = None
    if path_flights_replanned.exists():
        flights_replanned = pd.read_csv(path_flights_replanned)
        if toml_config['general'].get('review_fields_from_SOL3', False):
            # Need to clean the replanned information provided as SOL3 is not
            # considering things like TZ, seats, ac_type
            flights_replanned = review_fiels_flight_schedules_SOL3(flights_replanned, fs_planned, replanning=True)

        if len(flights_replanned) > 0:
            flights_replanned['sobt'] = pd.to_datetime(flights_replanned['sobt'] + flights_replanned['sobt_tz'])
            flights_replanned['sibt'] = pd.to_datetime(flights_replanned['sibt'] + flights_replanned['sibt_tz'])
            flights_replanned['sobt_local'] = pd.to_datetime(flights_replanned['sobt_local'] + flights_replanned['sobt_local_tz'])
            flights_replanned['sibt_local'] = pd.to_datetime(flights_replanned['sibt_local'] + flights_replanned['sibt_local_tz'])
        else:
            flights_replanned = None

    trains_replanned = None
    trains_replanned_ids = None
    if path_trains_replanned.exists():
        trains_replanned = pd.read_csv(path_trains_replanned,
                                     dtype={'trip_id': 'string', 'stop_id': 'string'})
        if len(trains_replanned)==0:
            trains_replanned = None
        else:
            trains_replanned, trains_replanned_ids = pre_process_rail_input(trains_replanned, base_date, df_stops)

    # Read additional services
    flights_added = None
    additonal_seats = None
    if path_flights_additional.exists():
        flights_added = pd.read_csv(path_flights_additional)
        if len(flights_added) == 0:
            flights_added = None
        else:
            flights_added['sobt'] = pd.to_datetime(flights_added['sobt'] + flights_added['sobt_tz'])
            flights_added['sibt'] = pd.to_datetime(flights_added['sibt'] + flights_added['sibt_tz'])
            flights_added['sobt_local'] = pd.to_datetime(flights_added['sobt_local'] + flights_added['sobt_local_tz'])
            flights_added['sibt_local'] = pd.to_datetime(flights_added['sibt_local'] + flights_added['sibt_local_tz'])
            additonal_seats = flights_added[['service_id', 'seats']].copy().rename(columns={'seats': 'capacity'})
            additonal_seats['type'] = 'flight'

    trains_added = None
    if path_trains_additional.exists():
        trains_added = pd.read_csv(path_trains_additional, dtype={'trip_id': 'string', 'stop_id': 'string'})
        if len(trains_added) > 0:
            trains_added, trains_added_ids = pre_process_rail_input(trains_added, base_date, df_stops)
        else:
            trains_added = None

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
        # in the dictionary of seats per service
        seats_in_service = pd.concat([seats_in_service, (additonal_seats[['service_id', 'capacity', 'type']].
                                                         rename(columns={'service_id': 'nid',
                                                                         'capacity': 'max_seats',
                                                                         'type': 'mode_transport'}))])

    dict_seats_service = (seats_in_service.groupby('mode_transport')[['nid', 'max_seats']]
                          .apply(lambda g: dict(zip(g['nid'], g['max_seats'])))
                          .to_dict())

    # Expand dictionary for rail to include stops not only services
    if 'rail' in dict_seats_service.keys():
        # We have rail as mode
        first_element_dict_rail = list(dict_seats_service['rail'].keys())[0]
        # If it's not str or not have _ twice then it's only a rail id without stops
        if (type(first_element_dict_rail) != str) or (not '_' in first_element_dict_rail):
            # We don't have a str or _ in the name so it is missing the stops
            # Final output dictionary
            expanded_rail_capacity_trip_dict = {}

            # Group by trip_id and iterate
            for trip_id, group in rs_planned.groupby('trip_id'):
                stops = sorted(group['stop_sequence'].tolist())
                for i in range(len(stops) - 1):
                    for j in range(i + 1, len(stops)):
                        key = f"{trip_id}_{stops[i]}_{stops[j]}"
                        expanded_rail_capacity_trip_dict[key] = dict_seats_service['rail'][trip_id]

            dict_seats_service['rail'] = expanded_rail_capacity_trip_dict



    #####################################################################
    # First adjust rail and flight network with modifications replanned #
    ####################################################################

    if trains_replanned is not None:
        logger.important_info("Replanning rail")
    if trains_cancelled is not None:
        logger.important_info("Cancelling rail")
    if trains_added is not None:
        logger.important_info("Adding rail")

    rs_replanned, dict_seats_service, services_removed = replan_rail_timetable(rs_planned,
                                                             rail_replanned=trains_replanned,
                                                             rail_cancelled=trains_cancelled,
                                                             rail_added=trains_added,
                                                             dict_seats_service=dict_seats_service)
    trains_cancelled = services_removed

    if flights_replanned is not None:
        logger.important_info("Replanning flights")
    if flights_cancelled is not None:
        logger.important_info("Cancelling flights")
    if flights_added is not None:
        logger.important_info("Adding flights")

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

    fs_replanned_to_save.to_csv((output_processed_folder_path /
                         ('flight_schedules_proc_' + str(
                                     pre_processed_version) + '.csv')),
                                index=False)

    rs_replanned.to_csv((output_processed_folder_path /
                         ('rail_timetable_all_gtfs_w_additional_utc_local_' + str(
                             pre_processed_version) + '.csv')),
                        index=False)

    rs_replanned[['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence', 'pickup_type',
                  'drop_off_type', 'border_point', 'load_unload', 'check_in', 'check_out', 'provider',
                  'alliance', 'country', 'status']].to_csv((output_processed_folder_path /
                         ('rail_timetable_all_gtfs_' + str(
                             pre_processed_version) + '.csv')),
                        index=False)



    #####################################
    # Preprocess the replanned network  #
    #####################################

    # Regardless of having to replan_pax preprocess replanned network
    # Add path to  pre-processed input (flight and rail timetable processed)
    toml_config['general']['pre_processed_input_folder'] = output_processed_folder_path
    toml_config['network_definition']['pre_processed_input_folder'] = toml_config['general']['replanned_input_folder'] + "/processed/"
    toml_config['network_definition']['processed_folder'] = toml_config['general']['replanned_input_folder'] + "/processed/"

    preprocess_network_replanned(toml_config, pre_processed_version = pre_processed_version)

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



    # Pax itineraries initial with impact of replanning
    # Save pax_assigned_planned with their status due to replanning
    pax_assigned_planned.to_csv((output_pax_folder_path /
                                 ('0.pax_assigned_to_itineraries_options_status_replanned_'+ str(pre_processed_version) +'.csv')),
                                 index=False)

    # Pax kept their initial itinerary after replanning
    # Could be unnafected, delayed or replanned doable
    pax_kept.to_csv((output_pax_folder_path /
                     ('1.pax_assigned_to_itineraries_options_kept_' + str(
                         pre_processed_version) + '.csv')),
                    index=False)

    # These are pax itineraries that needed replanning
    pax_need_replannning.to_csv((output_pax_folder_path /
                               ('2.pax_assigned_need_replanning_' + str(
                                     pre_processed_version) + '.csv')),
                                index=False)



    ########################################
    #  Then identify pax need reassigning  #
    #######################################

    # Pax assigned final
    pax_assigned_final = pax_assigned_planned.copy()
    # Empty dataframe of pax_stranded
    pax_stranded = pax_assigned_final.iloc[0:0].copy()
    # Empty dataframe of pax_reassigned (in case there're no to reassign)
    pax_reassigned = pax_stranded.copy()
    pax_reassigned['pax_assigned'] = None

    df_pax_need_replanning_demand = pax_need_replannning.groupby(['pax_group_id', 'origin', 'destination'])['pax'].sum().reset_index()
    df_pax_need_replanning_demand.rename(columns={'pax': 'demand_to_assign'}, inplace=True)
    df_pax_need_replanning_demand['unfulfilled'] = df_pax_need_replanning_demand['demand_to_assign']

    if len(pax_need_replannning) > 0:
        # Have some pax that need replanning
        # Else we're done

        # Compute capacities available in services
        #df_stops_considered = pd.read_csv(path_stops_trains_used, dtype=str)
        #rs_replanned_filtered = rs_replanned[rs_replanned.stop_id.isin(df_stops_considered.stop_id)].copy()

        services_w_capacity, services_wo_capacity = compute_capacities_available_services(demand=pax_kept,
                                                                                          dict_services_w_capacity=dict_seats_service)

        services_w_capacity.to_csv((output_paths_folder_path /
                                    ('services_w_capacity_'+ str(pre_processed_version)+'.csv')), index=False)
        services_wo_capacity.to_csv((output_paths_folder_path /
                                     ('services_wo_capacity_'+ str(pre_processed_version)+'.csv')), index=False)

        # Compute capacity available overall per service
        capacity_available = pd.concat([services_w_capacity, services_wo_capacity])
        capacity_available['capacity'] = capacity_available['max_seats_service'] - capacity_available['max_pax_in_service']
        capacity_available = capacity_available[['service_id', 'type', 'capacity']]

        capacity_available.to_csv((output_paths_folder_path /
                                   ('services_capacities_'+ str(pre_processed_version)+'.csv')), index=False)

        capacity_available.to_csv((output_paths_folder_path / 'services_capacity_available.csv'), index=False)

        # Compute o-d demand that needs to be reassigned so that suitable itineraries can be computed
        # Get o-d with total demand that needs reacommodating
        od_demand_need_reaccomodating = pax_need_replannning.groupby(['origin', 'destination'])['pax'].sum().reset_index()
        # Put only one archetype, we only need one example per o-d pair
        od_demand_need_reaccomodating['archetype'] = 'archetype_0'
        # Set date as date_to_set_rail, if rail exist, as that's the date of the flight (not that it matters, I think)
        od_demand_need_reaccomodating['date'] = toml_config.get('network_definition').get('rail_network',[{}])[0].get('date_to_set_rail', '20190906')
        od_demand_need_reaccomodating = od_demand_need_reaccomodating.rename({'pax': 'trips'}, axis=1)
        od_demand_need_reaccomodating.to_csv((output_paths_folder_path /
                                              ('demand_missing_reaccomodate_'+ str(pre_processed_version) +'.csv')),
                                             index=False)

        # Modify the toml config so that demand is the newly created file
        toml_config['demand']['demand'] = Path((toml_config['output']['output_folder'] /
                                           ('demand_missing_reaccomodate_' + str(pre_processed_version) + '.csv')))


        allow_mixed_operators_itineraries = not toml_config['replanning_considerations']['constraints']['new_itineraries_respect_alliances']

        df_itineraries = compute_itineraries_in_replanned_network(toml_config,
                                                                  pc = pc,
                                                                  n_paths = n_paths,
                                                                  n_itineraries = n_itineraries,
                                                                  max_connections = max_connections,
                                                                  allow_mixed_operators_itineraries = allow_mixed_operators_itineraries,
                                                                  use_heuristics_precomputed = use_heuristics_precomputed,
                                                                  pre_processed_version = pre_processed_version,
                                                                  capacity_available = capacity_available,
                                                                  pre_process_input = False)


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
            logger.important_info("Computing stranded pax without option")
            # Check if there're pax without options, those would be stranded
            pax_wo_options = pax_need_replannning.merge(pax_need_replanning_w_it_options_kept[['pax_group_id',
                                                                                               'same_path']],
                                                        how='left', on='pax_group_id')
            pax_wo_options = pax_wo_options[pax_wo_options.same_path.isna()]
            pax_wo_options.drop(columns=['same_path'], inplace=True)
            pax_wo_options['pax_stranded'] = pax_wo_options['pax']
            pax_stranded = pax_wo_options[['cluster_id', 'option_cluster_number', 'alternative_id', 'option',
                                           'origin', 'destination', 'path'] +
                                          [col for col in pax_wo_options.columns if col.startswith("nid_f")] +
                                          ['total_waiting_time', 'total_time', 'type', 'volume', 'fare',
                                           'access_time', 'egress_time', 'd2i_time', 'i2d_time', 'volume_ceil',
                                           'pax', 'pax_group_id', 'pax_status_replanned', 'pax_stranded']].copy()

            pax_stranded['stranded_type'] = 'no_option'

            logger.important_info("Assigning passengers to services")

            nprocs = min(pc, toml_config['replanning_considerations']['optimisation']['nprocs'])

            pax_reassigned, pax_demand_assigned = reassign_passengers_services(
                pax_need_replanning_w_it_options_kept,
                capacity_available,
                objectives=toml_config['replanning_considerations']['optimisation']['objectives'],
                thresholds=toml_config['replanning_considerations']['optimisation']['thresholds'],
                pc=nprocs,
                solver=toml_config['replanning_considerations']['optimisation']['solver'])

            pax_reassigned['pax_group_id_new'] = (pax_reassigned['pax_group_id'].astype(str) +
                                                  '_' +
                                                  pax_reassigned['option'].astype(str))

            pax_reassigned.to_csv((toml_config['output']['output_folder'] /
                                                 ('pax_reassigned_results_solver_' +
                                                      str(pre_processed_version) + '.csv')), index=False)
            pax_demand_assigned.to_csv((toml_config['output']['output_folder'] /
                                                 ('pax_demand_assigned_' +
                                                      str(pre_processed_version) + '.csv')), index=False)

            # Generate new itineraries and stranded
            demand_not_fulfilled = pax_demand_assigned[pax_demand_assigned.unfulfilled>0]
            pax_stranded_unfulfilled = pax_assigned_planned.merge(demand_not_fulfilled[['pax_group_id', 'unfulfilled']], on=['pax_group_id'])
            pax_stranded_unfulfilled.rename(columns={'unfulfilled': 'pax_stranded'}, inplace=True)
            pax_stranded_unfulfilled['stranded_type'] = 'no_capacity'

            pax_stranded = pd.concat([pax_stranded, pax_stranded_unfulfilled])

            demand_reassigned = pax_reassigned[pax_reassigned.pax_assigned>0].copy()
            demand_reassigned.rename(columns={'pax':'demand_reassigning',
                                              'option': 'option_reassigning'}, inplace=True)

            #demand_reassigned['pax_group_new_id'] = demand_reassigned['pax_group_id']+
            pax_reassigned = pax_assigned_planned.merge(demand_reassigned, on=['pax_group_id'])

        else:
            # All pax are stranded
            logger.important_info("Computing stranded pax without option -- None have option")
            # Check if there're pax without options, those would be stranded
            pax_wo_options = pax_need_replannning
            pax_wo_options['pax_stranded'] = pax_wo_options['pax']
            pax_stranded = pax_wo_options[['cluster_id', 'option_cluster_number', 'alternative_id', 'option',
                                           'origin', 'destination', 'path'] +
                                          [col for col in pax_wo_options.columns if col.startswith("nid_f")] +
                                          ['total_waiting_time', 'total_time', 'type', 'volume', 'fare',
                                           'access_time', 'egress_time', 'd2i_time', 'i2d_time', 'volume_ceil',
                                           'pax', 'pax_group_id', 'pax_status_replanned', 'pax_stranded']].copy()

            pax_stranded['stranded_type'] = 'no_option'

            # Empty pax_reassigned dataframe
            pax_reassigned = pax_stranded.iloc[0:0].copy()
            pax_reassigned['pax_assigned'] = 0
            pax_reassigned['pax_group_id_new'] = None


    # Compute summary of pax needed reassigning and actually assigned
    pa = pax_reassigned.groupby(['pax_group_id'])['pax_assigned'].sum().reset_index()
    df_pax_need_replanning_demand = df_pax_need_replanning_demand.merge(pa, how='left', on='pax_group_id')
    df_pax_need_replanning_demand['pax_assigned'] = df_pax_need_replanning_demand['pax_assigned'].fillna(0)
    df_pax_need_replanning_demand['unfulfilled'] = (df_pax_need_replanning_demand['demand_to_assign'] -
                                                    df_pax_need_replanning_demand['pax_assigned'])
    df_pax_need_replanning_demand[['pax_group_id', 'origin', 'destination',
                                   'demand_to_assign', 'pax_assigned', 'unfulfilled']].to_csv((output_pax_folder_path /
                                          ('5.pax_demand_assigned_summary_' + str(pre_processed_version) + '.csv'
                                          )), index=False)





    # These are the pax itineraries which have been reassigned to other options
    pax_reassigned.to_csv((output_pax_folder_path /
                               ('3.pax_reassigned_to_itineraries_' + str(
                                     pre_processed_version) + '.csv')),
                                index=False)

    # These are pax itineraries which are stranded
    pax_stranded.to_csv((output_pax_folder_path /
                               ('4.pax_assigned_to_itineraries_replanned_stranded_' + str(
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

    output_folder_path = (Path(toml_config['general']['experiment_path']) /
                          toml_config['general']['replanned_input_folder'] /
                          'processed')
    toml_config['network_definition']['processed_folder'] = output_folder_path
    toml_config['general']['output_folder'] = toml_config['general']['replanned_input_folder']

    toml_config['output']['output_folder'] = (Path(toml_config['general']['experiment_path']) /
                                              toml_config['general']['replanned_input_folder'] /
                                              'paths_itineraries')
    toml_config['output']['output_folder_pax'] = (Path(toml_config['general']['experiment_path']) /
                                              toml_config['general']['replanned_input_folder'] /
                                              'pax_replanned')


    # Recreate output folders
    recreate_output_folder(output_folder_path,
                           delete_previous=True,
                           logger=logger)
    recreate_output_folder(toml_config['output']['output_folder'],
                           delete_previous=True,
                           logger=logger)
    recreate_output_folder(toml_config['output']['output_folder_pax'],
                           delete_previous=True,
                           logger=logger)

    save_information_config_used(toml_config, args)

    logger.important_info("Running reassing passengers in replanned network")

    run_reassigning_pax_replanning_pipeline(toml_config,
                                            pc=pc,
                                            n_paths=int(args.num_paths),
                                            n_itineraries=int(args.num_itineraries),
                                            max_connections=int(args.max_connections),
                                            pre_processed_version=int(args.preprocessed_version),
                                            use_heuristics_precomputed=args.use_heuristics_precomputed,
                                            recreate_output_fld=False)



