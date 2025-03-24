from pathlib import Path
import os
import pandas as pd
from kpi_lib_strategic import strategic_total_journey_time, diversity_of_destinations, modal_share, pax_time_efficiency, demand_served, load_factor, resilience_alternatives, catchment_area, cost_per_user,co2_emissions
from kpi_lib_tactical import flight_arrival_delay, kerb2gate_time, total_journey_time, variability, pax_not_supported

def read_strategic_output(path_to_strategic_output):
	pax_assigned_to_itineraries_options = pd.read_csv(Path(path_to_strategic_output) / 'pax_assigned_to_itineraries_options_0.csv')
	possible_itineraries_clustered_pareto_filtered = pd.read_csv(Path(path_to_strategic_output) / 'possible_itineraries_clustered_pareto_filtered_0.csv')
	demand = pd.read_csv(Path(path_to_strategic_output) / '..' / '..' / 'demand'/'demand_ES_MD_intra_v0.4.csv')
	pax_assigned_seats_max_target = pd.read_csv(Path(path_to_strategic_output) / 'pax_assigned_seats_max_target_0.csv')
	pax_assigned_tactical = pd.read_csv(Path(path_to_strategic_output) / 'pax_assigned_tactical_0.csv')
	pax_assigned_tactical_not_supported = pd.read_csv(Path(path_to_strategic_output) / 'pax_assigned_tactical_not_supported_0.csv')

	data = {'pax_assigned_to_itineraries_options':pax_assigned_to_itineraries_options, 'possible_itineraries_clustered_pareto_filtered':possible_itineraries_clustered_pareto_filtered, 'demand':demand, 'pax_assigned_seats_max_target':pax_assigned_seats_max_target,'pax_assigned_tactical':pax_assigned_tactical,'pax_assigned_tactical_not_supported':pax_assigned_tactical_not_supported}
	return data

def read_tactical_data(path_to_tactical_output,path_to_tactical_input):

	df_pax = pd.read_csv(Path(path_to_tactical_output) / 'output_pax.csv.gz',index_col=0,low_memory=False)
	df_flights = pd.read_csv(Path(path_to_tactical_output) / 'output_flights.csv.gz',index_col=0,low_memory=False)
	airport_processes = pd.read_parquet(Path(path_to_tactical_input) / 'data' / 'airports' / 'airport_processes.parquet')
	#input_pax = pd.read_parquet(Path(path_to_tactical_input) / 'case_studies' / 'case_study=0' / 'data' / 'pax' / 'pax_assigned_tactical_0.parquet')
	rail_stations_processes = pd.read_parquet(Path(path_to_tactical_input) / 'case_studies' / 'case_study=0' / 'data' / 'ground_mobility' / 'rail_stations_processes_v0.1.parquet')

	data = {'pax':df_pax,'flights':df_flights,'airport_processes':airport_processes,'rail_stations_processes':rail_stations_processes}
	return data

if __name__ == '__main__':
	print ('KPI calculation... ')
	path_to_strategic_output = '../../data/CS-ES-MD/v=0.8/processed_baseline/paths_itineraries'
	data = read_strategic_output(path_to_strategic_output)
	#strategic_total_journey_time(data['pax_assigned_to_itineraries_options'],data['possible_itineraries_clustered_pareto_filtered'])
	#diversity_of_destinations(data['possible_itineraries_clustered_pareto_filtered'])
	#modal_share(data['pax_assigned_to_itineraries_options'],data['possible_itineraries_clustered_pareto_filtered'])
	#pax_time_efficiency(data['pax_assigned_to_itineraries_options'])
	#demand_served(data['pax_assigned_to_itineraries_options'],data['demand'])
	#load_factor(data['pax_assigned_to_itineraries_options'],data['possible_itineraries_clustered_pareto_filtered'],data['pax_assigned_seats_max_target'])
	#resilience_alternatives(data['possible_itineraries_clustered_pareto_filtered'])
	#catchment_area(data['pax_assigned_to_itineraries_options'],data['possible_itineraries_clustered_pareto_filtered'])
	#cost_per_user(data['pax_assigned_to_itineraries_options'])
	#co2_emissions(data['pax_assigned_to_itineraries_options'],data['possible_itineraries_clustered_pareto_filtered'])

	path_to_tactical_output = '/home/michal/Documents/westminster/multimodx/results/3.1_9_1_0_ground_mobility__delay_mean_0/'
	path_to_tactical_input = '/home/michal/Documents/westminster/multimodx/input/scenario=1/'
	data2 = read_tactical_data(path_to_tactical_output,path_to_tactical_input)
	flight_arrival_delay(data2['flights'])
	#kerb2gate_time(data['pax'],data['airport_processes'])
	#total_journey_time(data2['pax'],data2['airport_processes'],data['pax_assigned_tactical'])
	#variability(data2['pax'])


