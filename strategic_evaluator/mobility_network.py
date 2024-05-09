import heapq
import copy
from datetime import timedelta


class Service:
    def __init__(self, service_id, origin, destination, departure_time, arrival_time, cost, provider, alliance,
                 **kwargs):
        self.id = service_id
        self.origin = origin
        self.destination = destination
        self.departure_time = departure_time
        self.arrival_time = arrival_time
        self.duration = arrival_time - departure_time
        self.cost = cost
        self.provider = provider
        self.alliance = alliance

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return f"Service {self.id}: {self.origin}"
        # return f"Service {self.id}: {self.origin} -> {self.destination} {self.departure_time} -> {self.arrival_time} {self.duration}"


class NetworkLayer:
    def __init__(self, network_id, df_services, dict_mct=None, regions_access=None,
                 custom_mct_func=None,
                 custom_heuristic_func=None,
                 custom_initialisation=None,
                 **kwargs):

        self.id = network_id
        self.df_services = df_services

        self.dict_s_departing = {}
        for s in self.df_services['service']:
            self.dict_s_departing[s.origin] = self.dict_s_departing.get(s.origin, set())
            self.dict_s_departing[s.origin].add(s)

        if dict_mct is None:
            dict_mct = {}
        self.dict_mct = dict_mct

        self._custom_mct_function = custom_mct_func
        self._custom_heuristic_function = custom_heuristic_func

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

        for key, value in kwargs.items():
            setattr(self, key, value)

        if custom_initialisation is not None:
            custom_initialisation(self)

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

    def get_services_from_after(self, node, time):
        services = self.df_services[(self.df_services.origin == node) & (self.df_services.departure_time >= time)]
        if len(services) > 0:
            return set(services.service)
        else:
            return set()

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

    def find_paths(self, origin, destination, npaths=1, max_connections=1, layers_ids=None,
                   allow_transitions_layers=True, consider_operators_connections=True,
                   consider_times_constraints=True):
        # Store solutions
        paths = []

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
            print(f"Origin {origin} is not in layers of network --> Path not possible")
            return paths, 0
        if len(destination_nodes) == 0:
            print(f"Destination {destination} is not in layers of network  --> Path not possible")
            return paths, 0

        # Priority queue to store flights to be explored, ordered by total travel time
        # Add first nodes to start exploring the graph
        pq = []

        for layer, origins in dict_initial_nodes_layers.items():
            for o in origins:
                access_time = self.dict_layers[layer].get_access_time(o, origin)
                p = Path(path=[], current_node=o, total_travel_time=access_time,
                         layer_id=layer, access_time=access_time)
                pq += [p]

        # Heapify the priority queue using the wrapper function
        heapq.heapify(pq)

        n_nodes_explored = 0

        while pq:
            # total_time, path, current_airport
            p = heapq.heappop(pq)
            n_nodes_explored += 1

            # Check if target node is reached
            if p.current_node in dict_destination_nodes_layers[p.current_layer_id]:
                egress_time = self.dict_layers[p.current_layer_id].get_egress_time(p.current_node, destination)
                p.egress_time = egress_time
                p.total_travel_time += egress_time

                paths += [p]

                # Check if we have found all the paths we wanted
                if len(paths) == npaths:
                    return paths, n_nodes_explored  # dict_best_to_reach

            # Explore all flights from current airport
            elif len(p.path) <= max_connections:

                # Check first on same layer
                if (not p.path) or (not consider_times_constraints):
                    # First time we are in this path. Check all services available from the current node.
                    possible_following_services_same_layer = self.dict_layers[p.current_layer_id].get_services_from(
                        p.current_node)
                else:
                    # We have already some elements in the path, check which services (edges) are available
                    # on the same layer after this one.
                    possible_following_services_same_layer = self.dict_layers[p.current_layer_id].get_services_from_after(
                        p.current_node, p.path[-1].arrival_time)

                for service in possible_following_services_same_layer:
                    # First edge in this path, so nothing to check, we just take it and add it to the list in the path.
                    if not p.path:
                        new_total_time = p.access_time + service.arrival_time - service.departure_time
                        new_path = [service]
                        heapq.heappush(pq, Path(path=new_path,
                                                current_node=service.destination,
                                                total_travel_time=new_total_time,
                                                layer_id=p.current_layer_id,
                                                access_time=p.access_time))

                    else:
                        # Avoid going back to an airport already visited
                        if service.destination not in p.nodes_visited:

                            if (len(p.path) + 1 <= max_connections) or (service.destination in destination_nodes):
                                # If we'd reach the destination using that service or at least we have an
                                # extra connection possible. This is to avoid opening an edge which will require
                                # another connection when no more connections are possible.

                                if ((service.provider == p.path[-1].provider) or
                                        (service.alliance == p.path[-1].alliance) or
                                        (not consider_operators_connections)):
                                    # Checking that the providers are compatible.
                                    # If consider_operators_connections=False, this check is not taken into account

                                    # Get the MCT between the two services: previous service (p.path[-1]), new service
                                    mct = self.dict_layers[p.current_layer_id].get_mct(p.path[-1], service)
                                    if (mct is not None) or (not consider_times_constraints):
                                        # If there is no MCT we cannot connect between them. Probably shouldn't
                                        # already have been part of possible_following_services_same_layer

                                        if ((service.departure_time >= p.path[-1].arrival_time + mct) or
                                                (not consider_times_constraints)):
                                            # Checked that the departure and arrival times of the services are
                                            # compatible
                                            # or we don't consider mct times constraints. Allow connecting even
                                            # if in reality not possible due to times.

                                            new_path = copy.deepcopy(p)
                                            ht = self.dict_layers[p.current_layer_id].get_heuristic(service.destination,
                                                                                                    destination_nodes)
                                            new_path.add_service_path(service, heuristic_time=ht,
                                                                      time_from_path=consider_times_constraints,
                                                                      mct=mct)
                                            heapq.heappush(pq, new_path)

                # Check now if we can do a transition to another layer
                if (len(p.path) > 0 and
                        allow_transitions_layers and
                        self.dict_transitions.get((p.current_layer_id, p.current_node)) is not None):
                    # Check if we can change layer from p.current_node in layer p.current_layer_id
                    # We already have some elements in the path (len(p.path)>0), otherwise we would be
                    # transitioning accross layers without even moving (e.g. doing door-to-LEBL and then LEBL-Sants
                    # Transitions are allowed accross layers and the current node has possible transitions
                    # defined in ground mobility accross layers.

                    for pt in self.dict_transitions.get((p.current_layer_id, p.current_node)):
                        # For each possible transition from this node check if we can use it.
                        if pt['layer_id'] in self.dict_layers.keys():
                            # The transition is to a layer that is part of the Network

                            # Get the connecting time to transition the layers
                            connecting_time_btw_layers = self.get_connecting_time_btw_layers(pt.get('mct', {}))

                            if consider_times_constraints:
                                # Possible services that can be used from the other layer considering
                                # arrival time of current service and connecting time btw layers.
                                posbl_follow_srvcs_othr_layr = self.dict_layers[pt['layer_id']].get_services_from_after(pt['destination'],
                                                                                                                        (p.path[-1].arrival_time + connecting_time_btw_layers))
                            else:
                                # We don't care about the time of the next services so we don't care about
                                # the connecting time between layers
                                posbl_follow_srvcs_othr_layr = self.dict_layers[pt['layer_id']].get_services_from(pt['destination'])

                            for service in posbl_follow_srvcs_othr_layr:
                                if (len(p.path) + 1 <= max_connections) or (service.destination in destination_nodes):
                                    # If we'd reach the destination using that service or at least we have an
                                    # extra connection possible. This is to avoid opening an edge which will require
                                    # another connection when no more connections are possible.

                                    new_path = copy.deepcopy(p)
                                    ht = self.dict_layers[pt['layer_id']].get_heuristic(service.destination,
                                                                                        destination_nodes)
                                    new_path.add_service_path(service, heuristic_time=ht, layer_id=pt['layer_id'],
                                                              time_from_path=consider_times_constraints,
                                                              mct=connecting_time_btw_layers)
                                    heapq.heappush(pq, new_path)

        # If target airport is not reachable or not all paths requested found
        return paths, n_nodes_explored


class Path:
    def __init__(self, path, current_node, total_travel_time, layer_id, access_time=None, egress_time=None, **kwargs):
        self.path = path
        self.current_layer_id = layer_id
        self.current_node = current_node
        self.total_travel_time = total_travel_time
        self.nodes_visited = []
        self.layers_used = [self.current_layer_id]
        for s in path:
            self.nodes_visited += [s.origin, s.destination]

        self.access_time = access_time
        if access_time is None:
            self.access_time = timedelta(minutes=0)
        self.egress_time = egress_time
        if egress_time is None:
            self.egress_time = timedelta(minutes=0)

        self.expected_minimum_travel_time = total_travel_time

        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_service_path(self, s, heuristic_time=None, layer_id=None, time_from_path=True, mct=None):
        self.path += [s]
        self.nodes_visited += [s.origin, s.destination]
        self.current_node = s.destination
        if time_from_path:
            self.total_travel_time = self.access_time + s.arrival_time - self.path[0].departure_time  # Total travel time
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

    def __lt__(self, p):
        # return self.total_travel_time < p.total_travel_time # len(self.path)<len(p.path) #
        return self.expected_minimum_travel_time < p.expected_minimum_travel_time  # len(self.path)<len(p.path) #

    def __repr__(self):
        return f"Path {self.path} --> Travel time: {self.total_travel_time}"
