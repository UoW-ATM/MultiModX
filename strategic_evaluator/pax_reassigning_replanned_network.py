import pandas as pd
from datetime import timedelta


def get_affected_pax_due_flight(pax_assigned_planned, flights_impacted):
    # Convert flights_cancelled to a set for fast lookup
    flights_impacted = set(flights_impacted["service_id"])

    # Check if any nid_fX column contains a cancelled flight
    mask = pax_assigned_planned.filter(like="nid_f").apply(lambda col: col.isin(flights_impacted), axis=0).any(axis=1)

    # Split DataFrames
    affected_pax_flight = pax_assigned_planned[mask]  # Rows with cancelled flights
    unaffected_pax_flight = pax_assigned_planned[~mask]  # Rows without cancelled flights
    return affected_pax_flight, unaffected_pax_flight


def get_affected_pax_due_train(pax_assigned_planned, trains_impacted):
    # Function to check if a train is cancelled
    def is_train_impacted(nid, impacted_trains_set, trains_impacted):
        parts = nid.split("_")
        train_id = parts[0]

        # Extract stops (if available)
        stops = list(map(int, parts[1:])) if len(parts) > 1 else []

        # Check if train is cancelled based on service_id alone
        if train_id in impacted_trains_set:
            cancellation = trains_impacted[trains_impacted["service_id"] == train_id].iloc[0]

            from_stop = cancellation["from"]
            to_stop = cancellation["to"]

            # If no stops are given, cancel all instances of the train
            if pd.isna(from_stop) and pd.isna(to_stop):
                return True

            # Convert from/to to int if they're not NaN
            from_stop = int(from_stop) if not pd.isna(from_stop) else None
            to_stop = int(to_stop) if not pd.isna(to_stop) else None

            # If from and to exist, check for overlap
            if from_stop is not None and to_stop is not None:
                return any(from_stop <= stop <= to_stop for stop in stops)

            # If only from exists, cancel services that start from that stop onward
            if from_stop is not None:
                return any(stop >= from_stop for stop in stops)

            # If only to exists, cancel services before that stop
            if to_stop is not None:
                return any(stop <= to_stop for stop in stops)

        return False

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
    def is_row_impacted(row, impacted_trains_set, trains_impacted):
        if pd.isna(row["type"]):
            # We don't have a type so nothing to cancel
            return False

        train_columns = get_train_columns(row)  # Find relevant columns
        for col in train_columns:
            if pd.notna(row[col]) and is_train_impacted(row[col], impacted_trains_set, trains_impacted):
                return True  # Mark row as cancelled if any train is affected
        return False

    impacted_trains_set = set(trains_impacted["service_id"])

    # Identify all nid_fX columns
    nid_columns = [col for col in pax_assigned_planned.columns if col.startswith("nid_f")]

    # Apply cancellation check with additional argument
    mask = pax_assigned_planned.apply(lambda row: is_row_impacted(row, impacted_trains_set, trains_impacted), axis=1)

    # Split the DataFrame
    affected_pax_train = pax_assigned_planned[mask]
    unaffected_pax_train = pax_assigned_planned[~mask]

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
    affected_pax = pd.concat([affected_pax_flight, affected_pax_train])

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

    if 'sobt_local' not in pot_affected_pax_replanned_w_conn.columns:
        pot_affected_pax_replanned_w_conn['sobt_local'] = None
    if 'sibt_local' not in pot_affected_pax_replanned_w_conn.columns:
        pot_affected_pax_replanned_w_conn['sibt_local'] = None
    if 'service_id' not in pot_affected_pax_replanned_w_conn.columns:
        pot_affected_pax_replanned_w_conn['service_id'] = None

    if 'departure_datetime' not in pot_affected_pax_replanned_w_conn.columns:
        pot_affected_pax_replanned_w_conn['departure_datetime'] = None
    if 'arrival_datetime' not in pot_affected_pax_replanned_w_conn.columns:
        pot_affected_pax_replanned_w_conn['arrival_datetime'] = None
    if 'stop_id' not in pot_affected_pax_replanned_w_conn.columns:
        pot_affected_pax_replanned_w_conn['stop_id'] = None
    if 'stop_sequence' not in pot_affected_pax_replanned_w_conn.columns:
        pot_affected_pax_replanned_w_conn['stop_sequence'] = None

    # Iterate through nid_fX columns to build sequence of modes and times in local time of sobt and sibt, and o-d info
    for col in [c for c in pot_affected_pax_replanned_w_conn.columns if c.startswith("nid_f")]:
        # Merge flights
        pot_affected_pax_replanned_w_conn = pot_affected_pax_replanned_w_conn.merge(
            fs[['service_id', 'sobt_local', 'sibt_local', 'origin', 'destination']], left_on=col, right_on="service_id",
            how="left", suffixes=("", f"_{col}")
        )

        # Extract rail info
        trip_info = pot_affected_pax_replanned_w_conn[col].apply(
            lambda x: extract_rail_info(x) if "_" in str(x) else (None, None, None))
        pot_affected_pax_replanned_w_conn["trip_id"] = trip_info.apply(lambda x: x[0])
        pot_affected_pax_replanned_w_conn["from_stop_seq"] = trip_info.apply(lambda x: x[1])
        pot_affected_pax_replanned_w_conn["to_stop_seq"] = trip_info.apply(lambda x: x[2])

        # Merge rail start stop
        pot_affected_pax_replanned_w_conn = pot_affected_pax_replanned_w_conn.merge(
            rs[['trip_id', 'departure_datetime', 'stop_id', 'stop_sequence']], left_on=["trip_id", "from_stop_seq"],
            right_on=["trip_id", "stop_sequence"], how="left", suffixes=("", f"_{col}_start")
        )

        # Merge rail end stop
        pot_affected_pax_replanned_w_conn = pot_affected_pax_replanned_w_conn.merge(
            rs[['trip_id', 'arrival_datetime', 'stop_id', 'stop_sequence']], left_on=["trip_id", "to_stop_seq"],
            right_on=["trip_id", "stop_sequence"], how="left", suffixes=("", f"_{col}_end")
        )

    # Consolidate data for each _f column
    for col in [c for c in pot_affected_pax_replanned_w_conn.columns if c.startswith("nid_f")]:
        # suffix = col.split("_")[1]  # Extract the suffix, e.g., "f1"

        # Replace empty flight values with rail values
        pot_affected_pax_replanned_w_conn[f"service_id_{col}"] = pot_affected_pax_replanned_w_conn[
            f"service_id_{col}"].fillna(pot_affected_pax_replanned_w_conn[col])
        pot_affected_pax_replanned_w_conn[f"sobt_local_{col}"] = pot_affected_pax_replanned_w_conn[
            f"sobt_local_{col}"].fillna(pot_affected_pax_replanned_w_conn[f"departure_datetime_{col}_start"])
        pot_affected_pax_replanned_w_conn[f"sibt_local_{col}"] = pot_affected_pax_replanned_w_conn[
            f"sibt_local_{col}"].fillna(pot_affected_pax_replanned_w_conn[f"arrival_datetime_{col}_end"])
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
        columns=["sobt_local", "sibt_local", "service_id", "trip_id", "from_stop_seq", "to_stop_seq",
                 "departure_datetime", "arrival_datetime", "stop_id", "stop_sequence"], inplace=True)


    # Now here we have for each itinerary the succession of nodes (o-d), modes and sobt and sibt
    # need to add the MCTs

    # Check all the connections
    def compute_mct_in_row_for_leg_i(x, i):
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

    for i in range(1, max_number_connections):
        pot_affected_pax_replanned_w_conn['mct_' + str(i) + '_' + str(i + 1)] = pot_affected_pax_replanned_w_conn.apply(
            lambda x: compute_mct_in_row_for_leg_i(x, i), axis=1)
        pot_affected_pax_replanned_w_conn['buffer_' + str(i) + '_' + str(i + 1)] = (
                    pot_affected_pax_replanned_w_conn['sobt_local_nid_f' + str(i + 1)] -
                    pot_affected_pax_replanned_w_conn['sibt_local_nid_f' + str(i)] -
                    pot_affected_pax_replanned_w_conn['mct_' + str(i) + '_' + str(i + 1)])
        pot_affected_pax_replanned_w_conn['buffer_' + str(i) + '_' + str(i + 1)] = pot_affected_pax_replanned_w_conn[
            'buffer_' + str(i) + '_' + str(i + 1)].apply(lambda x: x.total_seconds() / 60)

    # Identify all buffer_x_y columns dynamically
    buffer_cols = [col for col in pot_affected_pax_replanned_w_conn.columns if col.startswith("buffer_")]

    # Create masks
    all_positive_mask = (pot_affected_pax_replanned_w_conn[buffer_cols] > 0).all(axis=1)  # True if all buffer_x_y > 0
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

    return pax_demand_planned, pax_kept, pax_need_replannning


def compute_load_factor(df_pax_per_service, dict_seats_service):
    def add_total_pax_in_service(df_pax_per_service):
        df_pax_per_service['total_pax_in_service'] = df_pax_per_service['pax']
        df_pax_per_service.loc[(df_pax_per_service['type'] == 'rail'), 'rail_service_id'] = df_pax_per_service.loc[
            (df_pax_per_service['type'] == 'rail'), 'service_id'].apply(lambda x: x.split('_')[0])
        df_pax_per_service.loc[(df_pax_per_service['type'] == 'rail'), 'stop_orig'] = df_pax_per_service.loc[
            (df_pax_per_service['type'] == 'rail'), 'service_id'].apply(lambda x: int(x.split('_')[1]))
        df_pax_per_service.loc[(df_pax_per_service['type'] == 'rail'), 'stop_destination'] = df_pax_per_service.loc[
            (df_pax_per_service['type'] == 'rail'), 'service_id'].apply(lambda x: int(x.split('_')[2]))

        df_pax_per_service.loc[(df_pax_per_service['type'] == 'rail'), 'total_pax_in_service'] = \
        df_pax_per_service.loc[(df_pax_per_service['type'] == 'rail')].apply(
            lambda x: df_pax_per_service[((df_pax_per_service.rail_service_id == x.rail_service_id) &
                                          ~((df_pax_per_service.stop_orig >= x.stop_destination) &
                                            (df_pax_per_service.stop_destination >= x.stop_orig)
                                            ))].pax.sum(), axis=1)
        return df_pax_per_service

    df_pax_per_service = add_total_pax_in_service(df_pax_per_service)
    df_pax_per_service['max_seats_service'] = df_pax_per_service['service_id'].apply(
        lambda x: dict_seats_service[x])

    return df_pax_per_service['total_pax_in_service'] / df_pax_per_service['max_seats_service']


def compute_capacities_available_services(df_pax, dict_seats_service):
    # Get list of pax per service regardless if used in nid_f1, nid_f2...
    # Identify nid_f columns dynamically (as there could be from nid_f1 to nid_fn
    nid_cols = [col for col in df_pax.columns if col.startswith('nid_f')]

    # Split the 'type' column into multiple type_n columns
    df_type_split = df_pax['type'].str.split('_', expand=True)
    df_type_split.columns = [f'type_{i + 1}' for i in range(df_type_split.shape[1])]
    pax_kept = df_pax.join(df_type_split)

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
    df_pax_per_service['load_factor'] = compute_load_factor(df_pax_per_service, dict_seats_service)

    services_w_capacity = df_pax_per_service[df_pax_per_service.load_factor < 1].copy()
    services_wo_capacity = df_pax_per_service[df_pax_per_service.load_factor >= 1].copy()

    return services_w_capacity, services_wo_capacity


