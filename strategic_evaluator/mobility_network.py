import heapq
import copy
from datetime import timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class Service:
    def __init__(self, service_id, origin, destination, departure_time, arrival_time, cost, provider, alliance,
                 emissions=None, seats=None, **kwargs):
        self.id = service_id
        self.origin = origin
        self.destination = destination
        self.departure_time = departure_time
        self.arrival_time = arrival_time
        self.duration = arrival_time - departure_time
        self.cost = cost
        self.provider = provider
        self.alliance = alliance
        self.emissions = emissions
        self.seats = seats

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return f"Service {self.id}: {self.origin} -> {self.destination}"
        # return f"Service {self.id}: {self.origin} -> {self.destination}
        # {self.departure_time} -> {self.arrival_time} {self.duration}"


class NetworkLayer:
    def __init__(self, network_id, df_services, dict_mct=None, regions_access=None,
                 dict_dist_origin_destination=None,
                 custom_mct_func=None,
                 custom_heuristic_func=None,
                 custom_initialisation=None,
                 custom_services_from_after_func=None,
                 **kwargs):

        self.id = network_id
        self.df_services = df_services

        # List of destinations in layer:
        self.list_destinations = list(self.df_services['destination'].drop_duplicates())

        self.dict_s_departing = {}
        for s in self.df_services['service']:
            self.dict_s_departing[s.origin] = self.dict_s_departing.get(s.origin, set())
            self.dict_s_departing[s.origin].add(s)

        if dict_mct is None:
            dict_mct = {}
        self.dict_mct = dict_mct

        self._custom_mct_function = custom_mct_func
        self._custom_heuristic_function = custom_heuristic_func

        self._custom_services_from_after_function = custom_services_from_after_func

        self.dict_mode_access = {}
        self.dict_mode_egress = {}

        self.dict_from_station_to_region = {}
        self.dict_from_region_to_station = {}

        if regions_access is not None:
            for r, accesses in regions_access.items():
                for a in accesses:
                    self.dict_from_station_to_region.setdefault(a['station'], []).append(r)
                    self.dict_from_region_to_station.setdefault(r, []).append(a['station'])
                    if 'access' in a:
                        self.dict_mode_access[(a['station'], r)] = a['access']
                    if 'egress' in a:
                        self.dict_mode_egress[(a['station'], r)] = a['egress']

        if dict_dist_origin_destination is None:
            self.dict_dist_origin_destination = {}
        else:
            self.dict_dist_origin_destination = dict_dist_origin_destination

        for key, value in kwargs.items():
            setattr(self, key, value)

        if custom_initialisation is not None:
            custom_initialisation(self)

    def is_node_in_layer(self, node):
        return len(self.df_services[(self.df_services.origin == node) |
                                    (self.df_services.destination == node)]) > 0

    def get_services_between(self, origins, destinations):
        df = self.df_services[(self.df_services.origin.isin(origins) & self.df_services.destination.isin(destinations))]
        if len(df) > 0:
            return df['service'].tolist()
        else:
            return None

    def get_initial_nodes(self, origin):
        initial_nodes = []
        # Check the origins (as nodes) that satisfy the origin required in the call
        if origin in self.dict_from_region_to_station.keys():
            # Origin is a region in which the network is connected to
            initial_nodes = self.dict_from_region_to_station[origin]
        elif origin in self.dict_s_departing.keys():
            # Origin is a node
            initial_nodes = [origin]
        return initial_nodes

    def get_destination_nodes(self, destination):
        # Check the destinations (as nodes) that satisfy the destination required in the call
        destination_nodes = []
        for node, regions in self.dict_from_station_to_region.items():
            if destination in regions:
                destination_nodes.append(node)
        if destination in self.dict_s_departing.keys():
            destination_nodes = [destination]
        return destination_nodes

    def get_access_time(self, node, origin):
        # Dictionary form region to station to add in the travel time the access to the station (if origin is a region)
        dict_access_time = self.dict_mode_access.get((node, origin), {})
        return timedelta(minutes=dict_access_time.get('all', 0))

    def get_egress_time(self, node, destination):
        dict_egress_time = self.dict_mode_egress.get((node, destination), {})
        return timedelta(minutes=dict_egress_time.get('all', 0))

    def get_services_from(self, node):
        return self.dict_s_departing.get(node, set())

    def _get_services_from_after_function(self):
        # Check if custom function is provided, otherwise use default function
        if self._custom_services_from_after_function:
            return self._custom_services_from_after_function
        else:
            def wrapper(obj, node, time, *args, **kwargs):
                return obj._default_services_from_after_function(node, time)

            return wrapper

    def _default_services_from_after_function(self, node, time):
        services = self.df_services[(self.df_services.origin == node) & (self.df_services.departure_time >= time)]
        if len(services) > 0:
            return set(services.service)
        else:
            return set()


    def get_services_from_after(self, node, time, from_service=None):
        return self._get_services_from_after_function()(self, node, time, from_service)


    def _default_mct_function(self, service_from, service_to):
        # Return value that depends only on airport of connection
        if service_from.destination != service_to.origin:
            # Something wrong, destination previous flight should be same as origin next one
            return None
        return self.dict_mct.get(service_from.destination, timedelta(minutes=0))

    def _get_mct_function(self):
        # Check if custom function is provided, otherwise use default function
        if self._custom_mct_function:
            return self._custom_mct_function
        else:
            def wrapper(obj, *args, **kwargs):
                return obj._default_mct_function(*args, **kwargs)

            return wrapper

    def _default_heuristic_function(self):
        return timedelta(minutes=0)

    def _get_heuristic_function(self):
        if self._custom_heuristic_function:
            return self._custom_heuristic_function
        else:
            def wrapper(obj, *args, **kwargs):
                return obj._default_heuristic_function()

            return wrapper

    def get_heuristic(self, *args, **kwargs):
        return self._get_heuristic_function()(self, *args, **kwargs)

        # return self._custom_mct_function if self._custom_mct_function else self._default_mct_function

    def get_mct(self, *args, **kwargs):
        return self._get_mct_function()(self, *args, **kwargs)


class Network:
    def __init__(self, layers, transition_btw_layers=None):
        self.dict_layers = {}
        for layer in layers:
            self.dict_layers[layer.id] = layer

        self.dict_transitions = {}
        if transition_btw_layers is not None:
            for _, row in transition_btw_layers.iterrows():
                key = (row['layer_id_origin'], row['origin'])
                value = {
                    'layer_id': row['layer_id_destination'],
                    'destination': row['destination'],
                    'mct': row['mct']
                }
                if key in self.dict_transitions:
                    self.dict_transitions[key].append(value)
                else:
                    self.dict_transitions[key] = [value]

    def get_connecting_time_btw_layers(self, mct):
        return timedelta(minutes=mct.get('all', 0))

    def find_layer_node(self, node):
        found_layer = False
        j = -1
        layers = list(self.dict_layers.values())
        while not found_layer and j < len(layers):
            j += 1
            found_layer = layers[j].is_node_in_layer(node)

        if found_layer:
            return layers[j]
        else:
            return None

    def find_itineraries(self, origin, destination, routes=None, nitineraries=1, max_connections=1, layers_ids=None,
                         allow_transitions_layers=True, consider_operators_connections=True,
                         consider_times_constraints=True):

        def remove_consecutive_duplicates(elements):
            return [elements[i] for i in range(len(elements)) if i == 0 or elements[i] != elements[i - 1]]

        # Store solutions
        itineraries = []

        dict_next_valid_nodes = None
        if routes is not None:
            # Dictionary to store the possible next stops
            dict_next_valid_nodes = defaultdict(list)

            # Iterate through each path
            for path in routes:
                # Iterate through each segment of the path
                for i in range(len(path) - 1):
                    # Create the key as a tuple of the sub-path up to the current segment
                    key = tuple(path[:i + 1])
                    # The next stop is the element right after the current segment
                    next_stop = path[i + 1]
                    # Append the next stop to the list of possible next stops for this key
                    if next_stop not in dict_next_valid_nodes[key]:
                        dict_next_valid_nodes[key].append(next_stop)
                # Also add the full path as a key
                full_path_key = tuple(path)
                if full_path_key not in dict_next_valid_nodes:
                    dict_next_valid_nodes[full_path_key] = []

            # Algorithm to find all posssible itineraries between origin-destination using routes provided
            #return self.find_itineraries_in_route(origin, destination, routes, consider_operators_connections,
            #                                      consider_times_constraints)

        if layers_ids is None:
            layers_considered = self.dict_layers.keys()
        else:
            layers_considered = layers_ids

        dict_initial_nodes_layers = {}
        dict_destination_nodes_layers = {}
        initial_nodes = []
        destination_nodes = []
        for layer in layers_considered:
            i_n = self.dict_layers[layer].get_initial_nodes(origin)
            if len(i_n) > 0:
                initial_nodes += i_n
                dict_initial_nodes_layers[layer] = i_n
            d_n = self.dict_layers[layer].get_destination_nodes(destination)
            if len(d_n) > 0:
                destination_nodes += d_n
                dict_destination_nodes_layers[layer] = d_n

        if len(initial_nodes) == 0:
            # Origin not in the layers considered of the network
            logger.warning(f"Origin {origin} is not in layers of network --> Path not possible")
            #print(f"Origin {origin} is not in layers of network --> Path not possible")
            return itineraries, 0
        if len(destination_nodes) == 0:
            logger.warning(f"Destination {destination} is not in layers of network  --> Path not possible")
            #print(f"Destination {destination} is not in layers of network  --> Path not possible")
            return itineraries, 0

        # Try to find first direct services in single layer
        dict_direct_services = {}
        for layer in layers_considered:
            if (layer in dict_destination_nodes_layers.keys()) and (layer in dict_initial_nodes_layers.keys()):
                direct_services = self.dict_layers[layer].get_services_between(dict_initial_nodes_layers[layer],
                                                                                           dict_destination_nodes_layers[layer])
                if direct_services is not None:
                    dict_direct_services[layer] = direct_services

        max_time_direct = None
        if len(dict_direct_services) > 0:
            # We have direct services. Let's process them
            for layer, services in dict_direct_services.items():
                for service in services:
                    access_time = self.dict_layers[layer].get_access_time(service.origin, origin)
                    egress_time = self.dict_layers[layer].get_egress_time(service.destination, destination)
                    travel_time = access_time + service.duration + egress_time
                    i = Itinerary(itinerary=[service], current_node=service.destination, total_travel_time=travel_time,
                                  layer_id=layer, access_time=access_time, egress_time=egress_time)
                    i.arrived = True
                    itineraries += [i]
                    if (max_time_direct is None) or (max_time_direct < travel_time):
                        max_time_direct = travel_time

        if len(itineraries) > 0:
            # Order the itineraries based on travel time
            def __get_total_travel_time(obj):
                return obj.total_travel_time

            # Sort the list of objects based on the 'travel_time' attribute
            itineraries = sorted(itineraries, key=__get_total_travel_time)
            if len(itineraries) > 1:
                # We can get the max_time of one before the last to avoid outliers on rail particularly
                max_time_direct = itineraries[-2].total_travel_time

        if max_connections == 0:
            return itineraries, 0

        # Priority queue to store flights to be explored, ordered by total travel time
        # Add first nodes to start exploring the graph
        pq = []

        for layer, origins in dict_initial_nodes_layers.items():
            for o in origins:
                # Either we don't have a dictionary of valid nodes or the origin is in the keys of the dictionary
                if (dict_next_valid_nodes is None) or (tuple([o]) in dict_next_valid_nodes.keys()) > 0:
                    access_time = self.dict_layers[layer].get_access_time(o, origin)
                    i = Itinerary(itinerary=[], current_node=o, total_travel_time=access_time,
                                  layer_id=layer, access_time=access_time)
                    pq += [i]

        # Heapify the priority queue using the wrapper function
        heapq.heapify(pq)

        # Number of departures from origin
        n_nodes_explored = 0

        while pq:
            # total_time, path, current_airport
            i = heapq.heappop(pq)
            n_nodes_explored += 1

            # Check if it's already arrived
            if i.arrived:
                if (((max_time_direct is not None) and (len(itineraries) >= nitineraries) and
                      (i.total_travel_time <= max_time_direct)) or (max_time_direct is None) or (len(itineraries) < nitineraries)):
                    itineraries += [i]

                if (((len(itineraries) >= nitineraries) and max_time_direct is None) or
                        ((max_time_direct is not None) and (i.total_travel_time > max_time_direct) and
                         (len(itineraries) >= nitineraries))):
                    return itineraries, n_nodes_explored

            # Check if target node is reached --> first if current node layer is in destination layers
            elif ((i.current_layer_id in dict_destination_nodes_layers.keys()) and
                    (i.current_node in dict_destination_nodes_layers[i.current_layer_id])):
                egress_time = self.dict_layers[i.current_layer_id].get_egress_time(i.current_node, destination)
                i.egress_time = egress_time
                i.total_travel_time += egress_time
                i.expected_minimum_travel_time = i.total_travel_time
                i.arrived = True
                #if len(i.itinerary) == 0:
                #    # We have arrived to the destination but there's no service used, i.e. itinerary = []
                #    i.itinerary = [i.current_node]

                heapq.heappush(pq, i)

            # Explore all services from current node
            elif len(i.itinerary) <= max_connections:

                # Check first on same layer
                if (not i.itinerary) or (not consider_times_constraints):
                    # First time we are in this path. Check all services available from the current node.
                    possible_following_services_same_layer_all = self.dict_layers[i.current_layer_id].get_services_from(
                        i.current_node)

                    # As it is the firs time, we want to make sure we don't use a service which takes us from
                    # a station in the origin to another station in the origin.
                    # Get all stations in the layer that are reachable directly from the origin.
                    # If the destination is not in that set, then keep it, otherwise remove it.

                    nodes_reachable_from_origin_in_layer = self.dict_layers[i.current_layer_id].get_initial_nodes(
                        origin)

                    possible_following_services_same_layer = set()
                    for service in possible_following_services_same_layer_all:
                        if i.current_layer_id == 'air' or service.destination not in nodes_reachable_from_origin_in_layer:
                            # If air we keep it as if we get a flight then it's fine.
                            # avoid airports that origin-destination are 'reachable' but outside region
                            # TODO: Consider access times when deciding if it makes sense or not to keep service instead of only if reachable from origin
                            possible_following_services_same_layer = possible_following_services_same_layer.union({service})
                else:
                    # We have already some elements in the path, check which services (edges) are available
                    # on the same layer after this one.
                    possible_following_services_same_layer = self.dict_layers[i.current_layer_id].get_services_from_after(
                        i.current_node, i.itinerary[-1].arrival_time, i.itinerary[-1])

                for service in possible_following_services_same_layer:
                    # First edge in this path, so nothing to check, we just take it and add it to the list in the path.
                    if not i.itinerary:
                        # Check that we're not reaching the final destination already as this would be a direct
                        # service and we have already considered these outside the loop. If not then save in heap.
                        if ((i.current_layer_id not in dict_destination_nodes_layers.keys()) or
                                (service.destination not in dict_destination_nodes_layers[i.current_layer_id])):
                            # Either not have list of routes to follow or path in possible paths
                            if ((dict_next_valid_nodes is None) or
                                    (tuple([service.origin, service.destination]) in dict_next_valid_nodes.keys())):
                                new_total_time = i.access_time + service.arrival_time - service.departure_time
                                new_itinerary = [service]
                                it = Itinerary(itinerary=new_itinerary,
                                                             current_node=service.destination,
                                                             total_travel_time=new_total_time,
                                                             layer_id=i.current_layer_id,
                                                             access_time=i.access_time)
                                heapq.heappush(pq, it)

                    else:
                        # Avoid going back to an airport already visited
                        if service.destination not in i.nodes_visited:

                            if (len(i.itinerary) + 1 <= max_connections) or (service.destination in destination_nodes):
                                # If we'd reach the destination using that service or at least we have an
                                # extra connection possible. This is to avoid opening an edge which will require
                                # another connection when no more connections are possible.

                                # Either not have list of routes to follow or path in possible paths
                                if ((dict_next_valid_nodes is None) or
                                        (tuple(remove_consecutive_duplicates(i.nodes_visited + [service.destination]))
                                         in dict_next_valid_nodes.keys())):

                                    if ((service.provider == i.itinerary[-1].provider) or
                                            (service.alliance == i.itinerary[-1].alliance) or
                                            (not consider_operators_connections)):
                                        # Checking that the providers are compatible.
                                        # If consider_operators_connections=False, this check is not taken into account

                                        # Get the MCT between the two services: previous service (i.itinerary[-1]), new service
                                        mct = self.dict_layers[i.current_layer_id].get_mct(i.itinerary[-1], service)
                                        if (mct is not None) or (not consider_times_constraints):
                                            # If there is no MCT we cannot connect between them. Probably shouldn't
                                            # already have been part of possible_following_services_same_layer

                                            if ((service.departure_time >= i.itinerary[-1].arrival_time + mct) or
                                                    (not consider_times_constraints)):
                                                # Checked that the departure and arrival times of the services are
                                                # compatible
                                                # or we don't consider mct times constraints. Allow connecting even
                                                # if in reality not possible due to times.

                                                new_path = i.shallow_copy()# copy.deepcopy(i)
                                                ht = self.dict_layers[i.current_layer_id].get_heuristic(service.destination,
                                                                                                        destination_nodes)

                                                new_path.add_service_itinerary(service, heuristic_time=ht,
                                                                               time_from_path=consider_times_constraints,
                                                                               mct=mct)
                                                heapq.heappush(pq, new_path)

                # Check now if we can do a transition to another layer
                if (len(i.itinerary) > 0 and
                        allow_transitions_layers and
                        self.dict_transitions.get((i.current_layer_id, i.current_node)) is not None):
                    # Check if we can change layer from i.current_node in layer i.current_layer_id
                    # We already have some elements in the path (len(i.itinerary)>0), otherwise we would be
                    # transitioning accross layers without even moving (e.g. doing door-to-LEBL and then LEBL-Sants)
                    # Transitions are allowed accross layers and the current node has possible transitions
                    # defined in ground mobility accross layers.

                    for pt in self.dict_transitions.get((i.current_layer_id, i.current_node)):
                        # For each possible transition from this node check if we can use it.
                        if pt['layer_id'] in self.dict_layers.keys():
                            # The transition is to a layer that is part of the Network

                            # Get the connecting time to transition the layers
                            connecting_time_btw_layers = self.get_connecting_time_btw_layers(pt.get('mct', {}))

                            if consider_times_constraints:
                                # Possible services that can be used from the other layer considering
                                # arrival time of current service and connecting time btw layers.
                                posbl_follow_srvcs_othr_layr = self.dict_layers[pt['layer_id']].get_services_from_after(pt['destination'],
                                                                                                                        (i.itinerary[-1].arrival_time + connecting_time_btw_layers))
                            else:
                                # We don't care about the time of the next services so we don't care about
                                # the connecting time between layers
                                posbl_follow_srvcs_othr_layr = self.dict_layers[pt['layer_id']].get_services_from(pt['destination'])

                            for service in posbl_follow_srvcs_othr_layr:
                                if (len(i.itinerary) + 1 <= max_connections) or (service.destination in destination_nodes):
                                    # If we'd reach the destination using that service or at least we have an
                                    # extra connection possible. This is to avoid opening an edge which will require
                                    # another connection when no more connections are possible.

                                    # Either not have list of routes to follow or path in possible paths
                                    if ((dict_next_valid_nodes is None) or
                                            (tuple(remove_consecutive_duplicates(i.nodes_visited+[service.origin, service.destination]))
                                             in dict_next_valid_nodes.keys())):
                                        new_path = i.shallow_copy() #copy.deepcopy(i)
                                        ht = self.dict_layers[pt['layer_id']].get_heuristic(service.destination,
                                                                                            destination_nodes)
                                        new_path.add_service_itinerary(service, heuristic_time=ht, layer_id=pt['layer_id'],
                                                                       time_from_path=consider_times_constraints,
                                                                       mct=connecting_time_btw_layers)
                                        heapq.heappush(pq, new_path)

        # If target airport is not reachable or not all itineraries requested found
        return itineraries, n_nodes_explored


class Itinerary:
    def __init__(self, itinerary, current_node, total_travel_time, layer_id, access_time=None, egress_time=None, **kwargs):
        self.itinerary = itinerary
        self.current_layer_id = layer_id
        self.current_node = current_node
        self.total_travel_time = total_travel_time
        self.layers_used = [self.current_layer_id]
        self.nodes_visited = []
        for s in itinerary:
            self.nodes_visited += [s.origin, s.destination]

        self.mcts = []

        self.access_time = access_time
        if access_time is None:
            self.access_time = timedelta(minutes=0)
        self.egress_time = egress_time
        if egress_time is None:
            self.egress_time = timedelta(minutes=0)

        self.expected_minimum_travel_time = total_travel_time

        self.arrived = False

        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_service_itinerary(self, s, heuristic_time=None, layer_id=None, time_from_path=True, mct=None):
        self.itinerary += [s]
        self.mcts += [mct]
        self.nodes_visited += [s.origin, s.destination]
        self.current_node = s.destination
        if time_from_path:
            self.total_travel_time = self.access_time + s.arrival_time - self.itinerary[0].departure_time  # Total travel time
        else:
            self.total_travel_time += s.duration
            if mct is not None:
                self.total_travel_time += mct

        if heuristic_time is None:
            heuristic_time = timedelta(minutes=0)
        self.expected_minimum_travel_time = self.total_travel_time + heuristic_time
        if layer_id is None:
            # No change layer
            self.layers_used += [self.layers_used[-1]]
        else:
            self.layers_used += [layer_id]
            self.current_layer_id = layer_id

    def __lt__(self, i):
        # return self.total_travel_time < i.total_travel_time # len(self.path)<len(i.itinerary)
        if self.expected_minimum_travel_time != i.expected_minimum_travel_time:
            # First check minimum expected travel time (which is travel so far + heuristic)
            return self.expected_minimum_travel_time < i.expected_minimum_travel_time
        elif len(self.itinerary) != len(i.itinerary):
            # If travel times are the same then check if one has a shorter itinerary than the other
            return len(self.itinerary) < len(i.itinerary)
        elif len(self.nodes_visited) > 0:
            # If both itineraries are same length and we have visited some nodes
            if str(self.nodes_visited) != str(i.nodes_visited):
                # If the list of nodes visited are different compare these
                return str(self.nodes_visited) < str(i.nodes_visited)
            else:
                # If the list of nodes visited are the same keep the one departing first
                return self.itinerary[0].departure_time < i.itinerary[0].departure_time
        else:
            # We don't have any node visited so return True
            return True

    def shallow_copy(self):
        # Copy the basic attributes
        new_itinerary = Itinerary(
            itinerary=self.itinerary[:],  # Shallow copy of the itinerary list
            current_node=self.current_node,
            total_travel_time=self.total_travel_time,
            layer_id=self.current_layer_id,
            access_time=self.access_time,
            egress_time=self.egress_time
        )

        # Shallow copy lists
        new_itinerary.layers_used = self.layers_used[:]
        new_itinerary.nodes_visited = self.nodes_visited[:]
        new_itinerary.mcts = self.mcts[:]

        # Copy remaining attributes
        new_itinerary.expected_minimum_travel_time = self.expected_minimum_travel_time
        new_itinerary.arrived = self.arrived

        # Copy any additional attributes set by kwargs
        for key in self.__dict__.keys():
            if key not in ['itinerary', 'layers_used', 'nodes_visited', 'mcts', 'current_layer_id', 'current_node',
                           'total_travel_time', 'access_time', 'egress_time', 'expected_minimum_travel_time',
                           'arrived']:
                setattr(new_itinerary, key, copy.copy(getattr(self, key)))

        return new_itinerary


    def __repr__(self):
        return f"Path {self.itinerary} --> Travel time: {self.total_travel_time}"
