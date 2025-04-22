import pandas as pd
import numpy as np
import logging
from datetime import timedelta

import sys
sys.path.insert(1, '../..')

logger = logging.getLogger(__name__)

from libs.passenger_assigner.passenger_assigner import reassign_pax_option_solver


def get_affected_pax_due_flight(pax_assigned_planned, flights_impacted):
    # Convert flights_cancelled to a set for fast lookup
    flights_impacted = set(flights_impacted["service_id"])

    # Check if any nid_fX column contains a cancelled flight
    mask = pax_assigned_planned.filter(like="nid_f").apply(lambda col: col.isin(flights_impacted), axis=0).any(axis=1)

    # Split DataFrames
    affected_pax_flight = pax_assigned_planned[mask]  # Rows with cancelled flights
    unaffected_pax_flight = pax_assigned_planned[~mask]  # Rows without cancelled flights
    if len(affected_pax_flight) == 0:
        afected_pax_flight = None
    if len(unaffected_pax_flight) == 0:
        unaffected_pax_flight = None
    return affected_pax_flight, unaffected_pax_flight


def get_affected_pax_due_train(pax_assigned_planned, trains_cancelled):
    # trains_cancelled is the list of id (service_stop_stop) of the trains cancelled

    # Function to determine which nid_fX columns contain trains
    def get_train_columns(row):
        train_columns = []
        transport_types = row["type"].split("_")  # Split multi-mode types (e.g., flight_rail)

        for i, transport in enumerate(transport_types):
            if "rail" in transport:  # Adjust condition if needed for specific types
                col_name = f"nid_f{i + 1}"  # Column names start from nid_f1
                if col_name in nid_columns:  # Ensure column exists
                    train_columns.append(col_name)

        return train_columns

    # Apply cancellation check only to relevant nid_fX columns
    def is_row_impacted(row, trains_cancelled):
        if pd.isna(row["type"]):
            # We don't have a type so nothing to cancel
            return False

        train_columns = get_train_columns(row)  # Find relevant columns

        for col in train_columns:
            # Either the whole service (service_stop_stop) or the service (split("_")[0]) can be cancelled
            if (row[col] in trains_cancelled) or (row[col].split("_")[0] in trains_cancelled):
                return True  # Mark row as cancelled if any train is affected
        return False


    # Identify all nid_fX columns
    nid_columns = [col for col in pax_assigned_planned.columns if col.startswith("nid_f")]

    # Apply cancellation check with additional argument
    mask = pax_assigned_planned.apply(lambda row: is_row_impacted(row, trains_cancelled), axis=1)

    # Split the DataFrame
    affected_pax_train = pax_assigned_planned[mask]
    unaffected_pax_train = pax_assigned_planned[~mask]
    if len(affected_pax_train) == 0:
        affected_pax_train = None
    if len(unaffected_pax_train) == 0:
        unaffected_pax_train = None

    return affected_pax_train, unaffected_pax_train


def get_affected_pax_due_to_cancellations(pax_assigned_planned, flight_cancelled=None, trains_cancelled=None):
    if all(v is None for v in [flight_cancelled, trains_cancelled]):
        # No flights or trains cancelled
        return None, pax_assigned_planned

    unaffected_pax = pax_assigned_planned

    affected_pax_flight = None
    if flight_cancelled is not None:
        affected_pax_flight, unaffected_pax = get_affected_pax_due_flight(unaffected_pax, flight_cancelled)

    affected_pax_train = None
    if trains_cancelled is not None:
        affected_pax_train, unaffected_pax = get_affected_pax_due_train(unaffected_pax, trains_cancelled)

    # Total affected pax:
    affected_pax = pd.concat([affected_pax_flight, affected_pax_train])

    return affected_pax, unaffected_pax


def get_affected_pax_due_to_replaning(pax_assigned_planned, flight_replanned=None, trains_replanned=None):
    if all(v is None for v in [flight_replanned, trains_replanned]):
        # No flights or trains replanned
        return None, pax_assigned_planned

    unaffected_pax = pax_assigned_planned

    affected_pax_flight = None
    if flight_replanned is not None:
        affected_pax_flight, unaffected_pax = get_affected_pax_due_flight(unaffected_pax, flight_replanned)

    affected_pax_train = None
    if trains_replanned is not None:
        affected_pax_train, unaffected_pax = get_affected_pax_due_train(unaffected_pax, trains_replanned)

    # Total affected pax:
    if (affected_pax_flight is not None) or (unaffected_pax is not None):
        affected_pax = pd.concat([affected_pax_flight, affected_pax_train])
    else:
        affected_pax = None


    return affected_pax, unaffected_pax


def extract_rail_info(nid):
    parts = nid.split("_")
    if len(parts) == 3:
        return parts[0], int(parts[1]), int(parts[2])  # trip_id, from_stop_seq, to_stop_seq
    return None, None, None


def get_mct_hub(dict_mct, coming_from, hub, going_to):
    dict_times = dict_mct.get(hub)
    if dict_times is None:
        return None
    else:
        if (coming_from[0:2] == going_to[0:2]) and (hub[0:2] == coming_from[0:2]):
            # All domestic (looking at country by first two letters of origin, destination, connecting code (hub)
            # TODO could be improved, e.g. Canary islands to Peninsular Spain, different origin code same country.
            return timedelta(minutes=dict_times['domestic'])
        else:
            return timedelta(minutes=dict_times['international'])


def get_mct_between_layers(dict_mcts, coming_from, going_to, from_layer, to_layer):
    dict_use = {}
    if (from_layer == 'flight') and (to_layer == 'flight'):
        dict_use = dict_mcts['air_air']
    elif (from_layer == 'flight') and (to_layer == 'rail'):
        dict_use = dict_mcts['air_rail']
    elif (from_layer == 'rail') and (to_layer == 'flight'):
        dict_use = dict_mcts['rail_air']
    elif (from_layer == 'rail') and (to_layer == 'rail'):
        dict_use = dict_mcts['rail_rail']

    return timedelta(minutes=dict_use[coming_from + "_" + going_to])


def check_potentially_affected_pax_w_connections_replanned(pot_affected_pax_replanned_w_conn, rs, fs, dict_mcts):

    if 'sobt' not in pot_affected_pax_replanned_w_conn.columns:
        pot_affected_pax_replanned_w_conn['sobt'] = None
    if 'sibt' not in pot_affected_pax_replanned_w_conn.columns:
        pot_affected_pax_replanned_w_conn['sibt'] = None
    if 'service_id' not in pot_affected_pax_replanned_w_conn.columns:
        pot_affected_pax_replanned_w_conn['service_id'] = None

    if 'departure_time_utc' not in pot_affected_pax_replanned_w_conn.columns:
        pot_affected_pax_replanned_w_conn['departure_time_utc'] = None
    if 'arrival_time_utc' not in pot_affected_pax_replanned_w_conn.columns:
        pot_affected_pax_replanned_w_conn['arrival_time_utc'] = None
    if 'stop_id' not in pot_affected_pax_replanned_w_conn.columns:
        pot_affected_pax_replanned_w_conn['stop_id'] = None
    if 'stop_sequence' not in pot_affected_pax_replanned_w_conn.columns:
        pot_affected_pax_replanned_w_conn['stop_sequence'] = None

    # Iterate through nid_fX columns to build sequence of modes and times in local time of sobt and sibt, and o-d info
    for col in [c for c in pot_affected_pax_replanned_w_conn.columns if c.startswith("nid_f")]:
        # Merge flights
        pot_affected_pax_replanned_w_conn = pot_affected_pax_replanned_w_conn.merge(
            fs[['service_id', 'sobt', 'sibt', 'origin', 'destination']], left_on=col, right_on="service_id",
            how="left", suffixes=("", f"_{col}")
        )

        # Extract rail info
        trip_info = pot_affected_pax_replanned_w_conn[col].apply(
            lambda x: extract_rail_info(x) if "_" in str(x) else (None, None, None))
        pot_affected_pax_replanned_w_conn["trip_id"] = trip_info.apply(lambda x: x[0])
        #pot_affected_pax_replanned_w_conn["from_stop_seq"] = trip_info.apply(lambda x: int(x[1]) if x[1] is not None else None)
        pot_affected_pax_replanned_w_conn["from_stop_seq"] = trip_info.apply(lambda x: int(x[1]) if x[1] is not None else np.nan).astype('Int64')
        #pot_affected_pax_replanned_w_conn["to_stop_seq"] = trip_info.apply(lambda x: int(x[2]) if x[2] is not None else None)
        pot_affected_pax_replanned_w_conn["to_stop_seq"] = trip_info.apply(lambda x: int(x[2]) if x[2] is not None else np.nan).astype('Int64')



        # Merge rail start stop
        pot_affected_pax_replanned_w_conn = pot_affected_pax_replanned_w_conn.merge(
            rs[['trip_id', 'departure_time_utc', 'stop_id', 'stop_sequence']], left_on=["trip_id", "from_stop_seq"],
            right_on=["trip_id", "stop_sequence"], how="left", suffixes=("", f"_{col}_start")
        )

        # Merge rail end stop
        pot_affected_pax_replanned_w_conn = pot_affected_pax_replanned_w_conn.merge(
            rs[['trip_id', 'arrival_time_utc', 'stop_id', 'stop_sequence']], left_on=["trip_id", "to_stop_seq"],
            right_on=["trip_id", "stop_sequence"], how="left", suffixes=("", f"_{col}_end")
        )

    # Consolidate data for each _f column
    for col in [c for c in pot_affected_pax_replanned_w_conn.columns if c.startswith("nid_f")]:
        # suffix = col.split("_")[1]  # Extract the suffix, e.g., "f1"

        # Replace empty flight values with rail values
        pot_affected_pax_replanned_w_conn[f"service_id_{col}"] = pot_affected_pax_replanned_w_conn[
            f"service_id_{col}"].fillna(pot_affected_pax_replanned_w_conn[col])
        pot_affected_pax_replanned_w_conn[f"sobt_{col}"] = pot_affected_pax_replanned_w_conn[
            f"sobt_{col}"].fillna(pot_affected_pax_replanned_w_conn[f"departure_time_utc_{col}_start"])
        pot_affected_pax_replanned_w_conn[f"sibt_{col}"] = pot_affected_pax_replanned_w_conn[
            f"sibt_{col}"].fillna(pot_affected_pax_replanned_w_conn[f"arrival_time_utc_{col}_end"])
        pot_affected_pax_replanned_w_conn[f"origin_{col}"] = pot_affected_pax_replanned_w_conn[f"origin_{col}"].fillna(
            pot_affected_pax_replanned_w_conn[f"stop_id_{col}_start"])
        pot_affected_pax_replanned_w_conn[f"destination_{col}"] = pot_affected_pax_replanned_w_conn[
            f"destination_{col}"].fillna(pot_affected_pax_replanned_w_conn[f"stop_id_{col}_end"])

    # Drop the extra columns
    columns_to_drop = [col for col in pot_affected_pax_replanned_w_conn.columns if "_start" in col or "_end" in col]
    pot_affected_pax_replanned_w_conn.drop(columns=columns_to_drop, inplace=True)

    # Iterate over nid_fX columns
    # Iterate over nid_fX columns
    for col in [c for c in pot_affected_pax_replanned_w_conn.columns if c.startswith("nid_f")]:
        # Extract the index number from "nid_fX" (e.g., "nid_f1" -> 1)
        idx = int(col.split("_f")[1]) - 1  # Convert to 0-based index

        # Extract mode from the type column
        pot_affected_pax_replanned_w_conn[f"mode_{col}"] = pot_affected_pax_replanned_w_conn["type"].apply(
            lambda x: str(x).split("_")[idx] if idx < len(str(x).split("_")) else None
        )

    # Clean up unnecessary columns
    pot_affected_pax_replanned_w_conn.drop(
        columns=["sobt", "sibt", "service_id", "trip_id", "from_stop_seq", "to_stop_seq",
                 "departure_time_utc", "arrival_time_utc", "stop_id", "stop_sequence"], inplace=True)

    pot_affected_pax_replanned_w_conn = pot_affected_pax_replanned_w_conn.astype(object).where(
        pot_affected_pax_replanned_w_conn.notna(), None)



    # Now here we have for each itinerary the succession of nodes (o-d), modes and sobt and sibt
    # need to add the MCTs

    # Check all the connections
    def compute_mct_in_row_for_leg_i(x, i):
        if x['origin_nid_f' + str(i+1)] is None:
            # We are not connecting anymore (e.g. leg1 to leg2 connection but leg2 to leg3 no exists)
            return None
        else:
            if x['destination_nid_f' + str(i)] == x['origin_nid_f' + str(i + 1)]:
                # We are connecting in the same node
                if x['mode_nid_f' + str(i)] == 'rail':
                    # We are connecting at a rail station
                    return timedelta(minutes=dict_mcts['rail'].get(x['origin_nid_f' + str(i + 1)], dict_mcts['default_rail']))
                elif x['mode_nid_f' + str(i)] == 'flight':
                    mct = get_mct_hub(dict_mcts['air'], x['origin_nid_f' + str(i)], x['origin_nid_f' + str(i + 1)],
                                       x['destination_nid_f' + str(i + 1)])
                    if mct is None:
                        mct = dict_mcts['default_air']
                    return mct
                else:
                    return None
            else:
                # We are connecting between layers
                return get_mct_between_layers(dict_mcts,
                                              x['destination_nid_f' + str(i)], x['origin_nid_f' + str(i + 1)],
                                              x['mode_nid_f' + str(i)], x['mode_nid_f' + str(i + 1)])

    max_number_connections = max(pot_affected_pax_replanned_w_conn['type'].apply(lambda x: len(x.split('_'))))

    # Convert all sobt/sibt columns
    for i in range(1, max_number_connections + 1):
        sobt_col = f'sobt_nid_f{i}'
        if sobt_col in pot_affected_pax_replanned_w_conn.columns:
            pot_affected_pax_replanned_w_conn[sobt_col] = pd.to_datetime(pot_affected_pax_replanned_w_conn[sobt_col],
                                                                         errors='coerce')

    for i in range(1, max_number_connections):
        sibt_col = f'sibt_nid_f{i}'
        if sibt_col in pot_affected_pax_replanned_w_conn.columns:
            pot_affected_pax_replanned_w_conn[sibt_col] = pd.to_datetime(pot_affected_pax_replanned_w_conn[sibt_col],
                                                                         errors='coerce')

    for i in range(1, max_number_connections):
        mct_col = f'mct_{i}_{i + 1}'
        sobt_col = f'sobt_nid_f{i + 1}'
        sibt_col = f'sibt_nid_f{i}'
        buffer_col = f'buffer_{i}_{i + 1}'

        pot_affected_pax_replanned_w_conn[mct_col] = pot_affected_pax_replanned_w_conn.apply(
            lambda x: compute_mct_in_row_for_leg_i(x, i), axis=1)

        # Convert new mct col to timedelta for entire column
        pot_affected_pax_replanned_w_conn[mct_col] = pd.to_timedelta(pot_affected_pax_replanned_w_conn[mct_col],
                                                                     errors='coerce')

        i_not_null = ~pot_affected_pax_replanned_w_conn[mct_col].isnull()

        pot_affected_pax_replanned_w_conn.loc[i_not_null, buffer_col] = (
                pot_affected_pax_replanned_w_conn.loc[i_not_null, sobt_col] -
                pot_affected_pax_replanned_w_conn.loc[i_not_null, sibt_col] -
                pot_affected_pax_replanned_w_conn.loc[i_not_null, mct_col]
        )

        # Create a new column for minutes
        buffer_min_col = buffer_col + "_min"

        pot_affected_pax_replanned_w_conn.loc[i_not_null, buffer_min_col] = (
                pot_affected_pax_replanned_w_conn.loc[i_not_null, buffer_col]
                .dt.total_seconds() / 60
        )


    # Identify all buffer_x_y columns dynamically
    buffer_min_cols = [col for col in pot_affected_pax_replanned_w_conn.columns if col.startswith("buffer_") and
                   col.endswith("_min")]


    # Create masks
    all_positive_mask = (
        pot_affected_pax_replanned_w_conn[buffer_min_cols]
        .fillna(0)
        .ge(0)
        .all(axis=1)
    )

    contains_negative_mask = ~all_positive_mask  # Opposite of the above

    # Split the DataFrame
    pot_affected_pax_replanned_w_conn_doable = pot_affected_pax_replanned_w_conn[all_positive_mask].copy()  # Rows where all buffer_x_y > 0
    pot_affected_pax_replanned_w_conn_no_doable = pot_affected_pax_replanned_w_conn[contains_negative_mask].copy()  # Rows where at least one buffer_x_y ≤ 0

    return pot_affected_pax_replanned_w_conn_doable, pot_affected_pax_replanned_w_conn_no_doable


def compute_pax_status_in_replanned_network(pax_demand_planned,
                                            fs_replanned,
                                            rs_replanned,
                                            flights_cancelled, trains_cancelled,
                                            flights_replanned, trains_replanned_ids,
                                            dict_mcts):

    # Originally all pax itineraries are unnafected
    # As we go along we'll identify cancelled, with connections, delayed, etc.
    pax_demand_planned['pax_status_replanned'] = 'unnafected'


    ##############################################################################
    # First check which passengers are affected due to cancellation of services  #
    #############################################################################
    affected_pax_cancellations, not_cancelled_pax = get_affected_pax_due_to_cancellations(pax_demand_planned,
                                                                                          flight_cancelled=flights_cancelled,
                                                                                          trains_cancelled=trains_cancelled)
    if affected_pax_cancellations is not None:
        pax_demand_planned.loc[
            pax_demand_planned.pax_group_id.isin(affected_pax_cancellations.pax_group_id), 'pax_status_replanned'] = 'cancelled'


    ###########################################################################
    # Then check which passengers are affected due to services being modified #
    ##########################################################################
    pot_affctd_pax_replanned, unnafected_pax = get_affected_pax_due_to_replaning(not_cancelled_pax,
                                                                                 flights_replanned,
                                                                                 trains_replanned_ids)

    # One legs are not affected so only delayed
    pot_affected_pax_replanned_w_conn = None

    if pot_affctd_pax_replanned is not None:
        # Some passengers might be affected
        # Pax that are delayed are ones affected as being replanned
        if 'nid_f2' in pot_affctd_pax_replanned.columns:
            pax_affected_delayed = pot_affctd_pax_replanned[pot_affctd_pax_replanned.nid_f2.isna()].copy()
            pot_affected_pax_replanned_w_conn = pot_affctd_pax_replanned[~pot_affctd_pax_replanned.nid_f2.isna()].copy()
        else:
            pax_affected_delayed = pot_affctd_pax_replanned

        pax_demand_planned.loc[
            pax_demand_planned.pax_group_id.isin(pax_affected_delayed.pax_group_id), 'pax_status_replanned'] = 'delayed'

    # Now check if there are replanned with connections which
    # as for those some might make the connection while others might
    # not be possible anymore
    if pot_affected_pax_replanned_w_conn is not None:
        # We have connections
        # All itineraries with connections are potentially not doable now
        # The ones without connections would only be delayed
        # Assuming new schedules are always >= original
        # If not those pax might also not be able to do their trip

        # For the pax_replanned_w_conn we need to check if their itineraries are still possible
        pot_affected_pax_replanned_w_conn = pot_affctd_pax_replanned[~pot_affctd_pax_replanned.nid_f2.isna()].copy()

        # Get from the potentially affected which ones are still doable and for which ones their
        # connections are not feasible anymore
        pot_affected_pax_replanned_w_conn_doable, pot_affected_pax_replanned_w_conn_no_doable = check_potentially_affected_pax_w_connections_replanned(pot_affected_pax_replanned_w_conn,
                                                                                                                                                       rs_replanned,
                                                                                                                                                       fs_replanned,
                                                                                                                                                       dict_mcts)

       # Assign status to pax itineraries
        pax_demand_planned.loc[pax_demand_planned.pax_group_id.isin(
            pot_affected_pax_replanned_w_conn_doable.pax_group_id), 'pax_status_replanned'] = 'replanned_doable'
        pax_demand_planned.loc[pax_demand_planned.pax_group_id.isin(
            pot_affected_pax_replanned_w_conn_no_doable.pax_group_id), 'pax_status_replanned'] = 'replanned_no_doable'


    # Divide pax between pax that would need replanning and pax who are kept in the planned network
    # Split the pax between those who can and those who cannot do their trips anymore
    pax_need_replannning = pax_demand_planned[
        pax_demand_planned.pax_status_replanned.isin(['cancelled', 'replanned_no_doable'])]
    pax_kept = pax_demand_planned[
        ~pax_demand_planned.pax_status_replanned.isin(['cancelled', 'replanned_no_doable'])]

    return pax_demand_planned.copy(), pax_kept.copy(), pax_need_replannning.copy()


def compute_load_factor(df_pax_per_service, dict_seats_service):
    # Compute load factor of all services in dict_seats_service
    # Divide dataframe between flights and rail services
    df_pax_per_service_flight = df_pax_per_service[df_pax_per_service.type=='flight'].copy()
    df_pax_per_service_rail = df_pax_per_service[df_pax_per_service.type=='rail'].copy()
    df_final = []

    if 'flight' in dict_seats_service.keys():
        # We have flights for which their capacity should be computed
        df_flights = pd.DataFrame(list(dict_seats_service['flight'].items()), columns=['service_id', 'max_seats_service'])
        df_flights['type'] = 'flight'
        df_flights = df_flights.merge(df_pax_per_service_flight[['service_id', 'pax']], on='service_id', how='left')
        df_flights.rename(columns={'pax': 'max_pax_in_service'}, inplace=True)
        df_flights['max_pax_in_service'] = df_flights['max_pax_in_service'].fillna(0)

        index_flights_no_service = ((df_flights.max_seats_service == 0) | (df_flights.max_pax_in_service == 0))
        df_flights.loc[index_flights_no_service, 'load_factor'] = 1.0  # There's no room if the're are no seats

        df_flights.loc[~index_flights_no_service, 'load_factor'] = df_flights['max_pax_in_service'] / \
                                                                   df_flights['max_seats_service']

        df_final = [df_flights]

    if 'rail' in dict_seats_service.keys():
        # We have rail for which their capacity should be computed
        # Create a dataframe with the dictionary seats services
        # as that one has all the services regardless of being used or
        # not
        df_rail = pd.DataFrame(list(dict_seats_service['rail'].items()), columns=['service_id', 'max_seats_service'])
        df_rail['type'] = 'rail'

        # Merge with df_pax_per_service_rail so we have the pax per service_stop_stop
        df_rail = df_rail.merge(df_pax_per_service_rail[['service_id', 'pax']], on='service_id', how='left')
        df_rail['pax'] = df_rail['pax'].fillna(0)

        # First get capacity of rail services without the stops (only service_id)
        df_rail['rail_service_id'] = df_rail['service_id'].apply(lambda x: x.split('_')[0])
        df_rail['stop_orig'] = df_rail['service_id'].apply(lambda x: int(x.split('_')[1]))
        df_rail['stop_dest'] = df_rail['service_id'].apply(lambda x: int(x.split('_')[2]))


        # For each service compute the number of pax per stop
        def compute_pax_per_consecutive_segment(stops_pax):
            # compute_all_stops if true do all stops not only the ones used
            stops = set(stops_pax['stop_orig'])
            stops.update(stops_pax['stop_dest'])
            stops = sorted(list(stops))
            rail_service_id = stops_pax['rail_service_id'].iloc[0]

            pax_between_segments = []
            for i in range(len(stops) - 1):
                pax_in_between_stops = int(
                    stops_pax[~((stops_pax['stop_dest'] <= stops[i]) | (stops_pax['stop_orig'] >= stops[i + 1]))][
                        'pax'].sum())
                pax_between_segments += [{'rail_service': rail_service_id,
                                          'total_pax_in_service': pax_in_between_stops,
                                          'stop_orig': stops[i],
                                          'stop_destination': stops[i + 1]}]

            return pax_between_segments

        # Compute how many pax are per consecutive segment for each rail service
        capacity_services_consecutive_segments = df_rail.groupby('rail_service_id')[['rail_service_id',
                                                                                     'stop_orig',
                                                                                     'stop_dest',
                                                                                     'pax']].apply(compute_pax_per_consecutive_segment)

        capacity_services_consecutive_segments = capacity_services_consecutive_segments.reset_index()

        capacity_services_consecutive_segments = capacity_services_consecutive_segments.explode(0,
                                                                                                ignore_index=True)
        capacity_services_consecutive_segments = pd.json_normalize(capacity_services_consecutive_segments[0])

        # Now for each rail service (railservice_stoporig_stopdest) compute load factor and seats used/available
        def get_max_pax_in_segment(service_id, cap_serv_cnsive_segments):
            rail_service = service_id.split('_')[0]
            stop_orig = int(service_id.split('_')[1])
            stop_dest = int(service_id.split('_')[2])
            x = cap_serv_cnsive_segments[((cap_serv_cnsive_segments['rail_service'] == rail_service) &
                                          ~((cap_serv_cnsive_segments['stop_destination'] <= stop_orig) |
                                            (cap_serv_cnsive_segments['stop_orig'] >= stop_dest)))]
            return max(x['total_pax_in_service'])

        df_rail['max_pax_in_service'] = df_rail['service_id'].apply(lambda x: get_max_pax_in_segment(x,
                                                                                                     capacity_services_consecutive_segments))

        index_trains_no_service = ((df_rail.max_seats_service==0) | (df_rail.max_pax_in_service==0))
        df_rail.loc[index_trains_no_service, 'load_factor'] = 1.0 # There's no room if there are no seats
        df_rail.loc[index_trains_no_service, 'max_pax_in_service'] = 0 # There are no pax if there are no seats avaliable
                                                                        # Even if overlaps with other sergments, e.g.
                                                                        # if for service 1307 stops 1 to 2 is cancelled
                                                                        # 1307_1_3 needs to be 0, even if 1307_2_3 is valid
                                                                        # and has 124 pax. 1307_1_3 is not full 124
                                                                        # it is just 0 as it's not possible

        df_rail.loc[~index_trains_no_service, 'load_factor'] = df_rail['max_pax_in_service'] / \
                                                               df_rail['max_seats_service']

        df_rail.drop(columns={'rail_service_id', 'stop_orig', 'stop_dest'}, inplace=True)

        df_rail.rename(columns={'pax': 'pax_between_stops'}, inplace=True)

        df_final += [df_rail]

    df_final = pd.concat(df_final).reset_index()

    return df_final


def compute_capacities_available_services(demand, dict_services_w_capacity):
    # If rail_service_min_max_stops is passed then capacity computed for all those service_stops instead of
    # the ones in the df_pax.

    # Get list of pax per service regardless if used in nid_f1, nid_f2...
    # Identify nid_f columns dynamically (as there could be from nid_f1 to nid_fn
    nid_cols = [col for col in demand.columns if col.startswith('nid_f')]

    # Split the 'type' column into multiple type_n columns
    df_type_split = demand['type'].str.split('_', expand=True)
    df_type_split.columns = [f'type_{i + 1}' for i in range(df_type_split.shape[1])]
    pax_kept = demand.join(df_type_split)

    # Concatenate nid_f columns with their respective type_n
    df_melted = pax_kept.melt(id_vars=['pax', 'type'], value_vars=nid_cols,
                              var_name='nid_col', value_name='service_id')

    # Keep only rows where service_id is not NaN
    df_melted = df_melted.dropna(subset=['service_id']).reset_index(drop=True)

    # Extract the index of the nid_f column (e.g., nid_f1 -> 1)
    df_melted['nid_index'] = df_melted['nid_col'].str.extract(r'(\d+)').astype(int)

    # Assign the corresponding type_n based on nid_index
    df_melted['type'] = df_melted.apply(lambda x: x['type'].split('_')[x['nid_index'] - 1], axis=1)

    # Select final columns
    df_pax_per_service = df_melted[['service_id', 'type', 'pax']]

    # Add all pax in each service
    df_pax_per_service = df_pax_per_service.groupby(['service_id', 'type'])['pax'].sum().reset_index()

    # Compute load factor per service
    df_services_capacity = compute_load_factor(df_pax_per_service, dict_services_w_capacity)

    services_w_capacity = df_services_capacity[df_services_capacity.load_factor < 1].copy()
    services_wo_capacity = df_services_capacity[df_services_capacity.load_factor >= 1].copy()

    return services_w_capacity, services_wo_capacity



def reassign_passengers_services(pax_need_replanning_w_options, capacity_available, objectives, thresholds,
                                 pc=1,
                                 solver='gurobi', ):
    # We have pax with options, need to optimise the assigment
    pax_reassigning = pax_need_replanning_w_options[(['pax', 'pax_group_id', 'option'] +
                                                     [col for col in pax_need_replanning_w_options.columns if
                                                      col.startswith("service_id_")] +
                                                     [col for col in pax_need_replanning_w_options.columns if
                                                      col.startswith("mode_")] +
                                                     ['alliances_match', 'same_path', 'delay_departure_home',
                                                      'delay_arrival_home', 'delay_total_travel_time',
                                                      'extra_services', 'same_initial_node', 'same_final_node',
                                                      'same_modes'])].copy()

    # Filter only columns that start with 'service_id_' but NOT 'service_id_pax_'
    service_cols = [col for col in pax_reassigning.columns if
                    col.startswith("service_id_") and not col.startswith("service_id_pax_")]

    # Stack those columns and drop NaNs
    all_service_ids = pax_reassigning[service_cols].stack().dropna()

    # Get unique values as a list
    unique_service_ids = all_service_ids.unique().tolist()

    # Get services that are available (used) and with their capacities
    services_available_w_capacity = capacity_available[((capacity_available.capacity >= 1) &
                                                        (capacity_available.service_id.isin(
                                                            unique_service_ids)))].copy()

    if len(unique_service_ids) != len(services_available_w_capacity):
        logger.important_info("WARNING/ERROR: The number of unique services used in new itineraries (" +
                              str(len(unique_service_ids)) + ") is different to the number of services with capacity "
                                                             "which overlap with the services used in the itineraries (" +
                              str(len(services_available_w_capacity)) + ").")

    # dict_mode_transport = services_available_w_capacity.set_index('service_id')['type'].to_dict()
    # dict_service_capacity = services_available_w_capacity.set_index('service_id')['capacity'].to_dict()
    # Use capacity_available instead of service_available_w_capacity as the later is filtered by used and maybe
    # some stops combinations are not used but passed through... to need to keep their capacity checked.
    dict_mode_transport = capacity_available.set_index('service_id')['type'].to_dict()
    dict_service_capacity = capacity_available.set_index('service_id')['capacity'].to_dict()

    dict_volume = pax_reassigning[['pax_group_id', 'pax']].drop_duplicates().set_index('pax_group_id')['pax'].to_dict()





    pax_reassigned, pax_demand_assigned = reassign_pax_option_solver(pax_reassigning,
                                                                    dict_volume=dict_volume,
                                                                    dict_sc=dict_service_capacity,
                                                                    dict_mode_transport=dict_mode_transport,
                                                                    objectives=objectives,
                                                                    thresholds=thresholds,
                                                                    pc=pc,
                                                                    solver=solver)

    return pax_reassigned, pax_demand_assigned

