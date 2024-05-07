from datetime import timedelta


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
    for d in destination_nodes:
        if (node, d) not in obj.distance_origin_destination:
            cn = obj.airport_coordinates[obj.airport_coordinates.icao_id == node].iloc[0]
            cd = obj.airport_coordinates[obj.airport_coordinates.icao_id == d].iloc[0]
            dist = haversine(cn.lon, cn.lat, cd.lon, cd.lat)
            obj.distance_origin_destination[(node, d)] = dist

    min_dist = min(obj.distance_origin_destination.get((node, destination)) for destination in destination_nodes)
    min_time = min_dist * timedelta(seconds=8)  # This is actually super fast
    return min_time


def initialise_air_network(obj):
    obj.distance_origin_destination = {}
    for s in obj.df_services.service:
        if (s.origin, s.destination) not in obj.distance_origin_destination.keys():
            obj.distance_origin_destination[(s.origin, s.destination)] = s.gcdistance