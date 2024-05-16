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


def fastest_air_time_heuristic(obj, node, destination_nodes):
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
            elif d in obj.airport_coordinates.icao_id:
                cn = obj.airport_coordinates[obj.airport_coordinates.icao_id == node].iloc[0]
                cd = obj.airport_coordinates[obj.airport_coordinates.icao_id == d].iloc[0]
                dist = haversine(cn.lon, cn.lat, cd.lon, cd.lat)
                obj.distance_origin_destination[(node, d)] = dist
                min_time = dist * timedelta(seconds=8)
                min_times += [min_time]
        else:
            min_times += [min_time]

    if len(min_times) == 0:
        return 0 * timedelta(seconds=8)

    return min(min_times)


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


def fastest_rail_time_heuristic(obj, node, destination_nodes):
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

