from datetime import timedelta


def mct_rail_network(*args, **kwargs):
    return timedelta(minutes=10)

def mct_air_network(obj, service_from, service_to):
    if service_from.destination != service_to.origin:
        # Something wrong, destination previous flight should be same as origin next one
        return None
    coming_from = service_from.origin
    connecting_at = service_to.origin
    going_to = service_to.destination

    if (coming_from[0:2] == going_to[0:2]) and (connecting_at[0:2] == coming_from[0:2]):
        # All domestic (looking at country by first two letters of origin, destination, connecting code,
        # TODO could be improved, e.g. Canary islands to Peninsular Spain, different origin code same country.
        dict_type_connection = obj.dict_mct['dom']
    else:
        dict_type_connection = obj.dict_mct['int']
    return timedelta(minutes=dict_type_connection.get(connecting_at, obj.dict_mct['std'].get(connecting_at, 30)))


def initialise_air_network(obj):
    obj.distance_origin_destination = {}
    for s in obj.df_services.service:
        if (s.origin, s.destination) not in obj.distance_origin_destination.keys():
            if s.gcdistance is not None:
                obj.distance_origin_destination[(s.origin, s.destination)] = s.gcdistance

    # Calculate duration (arrival_time - departure_time)
    obj.df_services['duration'] = obj.df_services['arrival_time'] - obj.df_services['departure_time']

    # Group by ('origin', 'destination') and find the minimum duration for each group
    obj.dict_min_duration_o_d = obj.df_services.groupby(['origin', 'destination'])['duration'].min().to_dict()

    obj.dict_heuristics_explored = {}

    # Create dictionary of coordinates
    obj.dict_coordinates = {row['node']: {'lat': row['lat'], 'lon': row['lon']} for _, row in obj.nodes_coordinates.iterrows()}


def initialise_rail_network(obj):
    obj.df_services['service_id_generic'] = obj.df_services['service_id'].apply(lambda x: x.split("_")[0])

    # Create a list to store all stops
    obj.all_stops = set(obj.df_services['origin']).union(obj.df_services['destination'])

    # Initialize an empty dictionary to store the mapping
    obj.stop_service_mapping = {stop: set() for stop in obj.all_stops}

    # Iterate over the DataFrame and populate the dictionary
    for index, row in obj.df_services.iterrows():
        obj.stop_service_mapping[row['origin']].add(row['service_id_generic'])
        obj.stop_service_mapping[row['destination']].add(row['service_id_generic'])

    # Calculate duration (arrival_time - departure_time)
    obj.df_services['duration'] = obj.df_services['arrival_time'] - obj.df_services['departure_time']

    # Group by ('origin', 'destination') and find the minimum duration for each group
    obj.dict_min_duration_o_d = obj.df_services.groupby(['origin', 'destination'])['duration'].min().to_dict()

    obj.dict_heuristics_explored = {}

    # Create dictionary of coordinates
    obj.dict_coordinates = {row['node']: {'lat': row['lat'], 'lon': row['lon']} for _, row in obj.nodes_coordinates.iterrows()}


def services_from_after_function_rail(obj, node, time, service):
    if service is None:
        return obj._default_services_from_after_function(node, time)
    else:
        # Avoid same service (getting out and back up in the same service) and avoid getting a service that comes/goes
        # to the origin stop from where we are coming from. This could be refined, i.e., maybe I change to a faster
        # train coming from the same place, but I could have gotten it directly in the origin...
        services = obj.df_services[(obj.df_services.origin == node) & (obj.df_services.departure_time >= time) &
                                   (obj.df_services.service_id_generic != service.id.split("_")[0]) &
                                   (~obj.df_services.service_id_generic.isin(obj.stop_service_mapping[service.origin]))]

        if len(services) > 0:
            return set(services.service)
        else:
            return set()


def fastest_air_time_heuristic(obj, node, destination_nodes):
    # print("FH")
    from libs.uow_tool_belt.general_tools import haversine
    min_times = []

    for d in destination_nodes:

        min_time = obj.dict_min_duration_o_d.get((node, d))

        if min_time is None:
            # We don't have time for this origin-destination
            # use distance heuristic

            dist = obj.distance_origin_destination.get((node, d))

            if dist is not None:
                min_time = dist * timedelta(seconds=8)
                min_times += [min_time]
            elif d in obj.nodes_coordinates.node:
                cn = obj.nodes_coordinates[obj.nodes_coordinates.node == node].iloc[0]
                cd = obj.nodes_coordinates[obj.nodes_coordinates.node == d].iloc[0]
                dist = haversine(cn.lon, cn.lat, cd.lon, cd.lat)
                obj.distance_origin_destination[(node, d)] = dist
                min_time = dist * timedelta(seconds=8)
                min_times += [min_time]
        else:
            min_times += [min_time]

    if len(min_times) == 0:
        return 0 * timedelta(seconds=8)

    return min(min_times)

def fastest_rail_time_heuristic(obj, node, destination_nodes):
    # print("RH")
    # TODO if node-d does not exist use a heuristic based on distance. Note
    # that this means there's no direct train so in reality it's even worse as a connection would be
    # needed
    min_times = []
    for d in destination_nodes:
        if d in obj.all_stops:
            min_time = obj.dict_min_duration_o_d.get((node, d), timedelta(minutes=0))
            min_times += [min_time]

    if len(min_times) == 0:
        min_time = timedelta(minutes=0)
    else:
        min_time = min(min_times)
    return min_time


def fastest_precomputed_distance_time_heuristic(obj, node, destination_nodes):
    # print("HERE")
    from libs.uow_tool_belt.general_tools import haversine

    # Function to retrieve time given a distance based on the heuristic
    def _get_time_heuristic(obj, distance):
        # Iterate through each row of the DataFrame
        for index, row in obj.heuristic_precomputed_distance.iterrows():
            # Check if the distance falls within the current interval
            if row['min_dist'] < distance <= row['max_dist']:
                return row['time']
        # If the distance is greater than the maximum distance in the DataFrame, return None
        return None

    if node in destination_nodes:
        # We have already arrived
        return timedelta(minutes=0)

    distance_to_dest_gcd = []
    min_time_heuristic = obj.dict_heuristics_explored.get((node, str(destination_nodes)))
    if min_time_heuristic is not None:
        return min_time_heuristic

    for d in destination_nodes:
        # Check which destinations nodes in same layer as us
        if d in obj.list_destinations:
            dist_gcd = obj.dict_dist_origin_destination.get((node, d))
            if dist_gcd is None:
                cn = obj.nodes_coordinates[obj.nodes_coordinates.node == node].iloc[0]
                cd = obj.nodes_coordinates[obj.nodes_coordinates.node == d].iloc[0]
                dist_gcd = haversine(cn.lon, cn.lat, cd.lon, cd.lat)
                obj.dict_dist_origin_destination[(node, d)] = dist_gcd
            distance_to_dest_gcd.append(dist_gcd)

    if len(distance_to_dest_gcd) > 0:
        min_dist = min(distance_to_dest_gcd)
        min_time_heuristic = _get_time_heuristic(obj, min_dist)
        if min_time_heuristic is not None:
            min_time_heuristic = timedelta(minutes=min_time_heuristic)
        else:
            min_time_heuristic = timedelta(minutes=0)
        obj.dict_heuristics_explored[(node, str(destination_nodes))] = min_time_heuristic
        return min_time_heuristic
    else:
        # all the destination nodes are in a different layer
        # so we don't know how long it will take... return 0 for now.
        obj.dict_heuristics_explored[(node, str(destination_nodes))] = timedelta(minutes=0)
        return timedelta(minutes=0)

