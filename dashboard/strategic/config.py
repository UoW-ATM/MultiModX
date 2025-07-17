import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "../../data/CS10/v=0.16/output/")
MAPS_FOLDER = os.path.join(BASE_DIR, "../../data/EUROSTAT/NUTS_RG_01M_2021_4326.shp/NUTS_RG_01M_2021_4326.shp")
RAIL_FOLDER = os.path.join(BASE_DIR, "../../data/CS10/v=0.16/gtfs_es_UIC_v2.3")
INFRASTRUCTURE_FOLDER = os.path.join(BASE_DIR, "../../data/CS10/v=0.16/infrastructure")

VARIABLES = {
    "Demand original": {"type": "demand_origin", "file": "demand__alldemand_od.csv"},
    "Demand served": {"type": "demand_served", "file": "demand_served__by_nuts_n3.csv"},
    "Average Total Travel Time": {"type": "travel_time", "file": "strategic_total_journey_time__avg_by_nuts.csv"},
    "Origin-Destination demand potential paths": {"type": "od_paths", "files": {"pax_assigned_to_paths": "possible_itineraries_clustered_pareto_w_demand_0.csv",
                                                                      "possible_it_clustered": 'possible_itineraries_clustered_pareto_filtered_0.csv'}},
    "Origin-Destination itineraries": {"type": "od_trips", "files": {"pax_assigned_to_itineraries": "pax_assigned_to_itineraries_options_0.csv"}},
    "Catchment areas": {"type": "catchment_areas", "file": "catchment_area__access_egress_rail_stop_access_egress_w_rail.csv"},
}