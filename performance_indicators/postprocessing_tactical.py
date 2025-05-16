import pandas as pd
from pathlib import Path
import os
from scipy.stats import norm, lognorm, expon
import ast
import random
import tomli

def read_strategic_output(path_to_strategic_output,ppv):
	pax_assigned_to_itineraries_options = pd.read_csv(Path(path_to_strategic_output) / ('pax_assigned_to_itineraries_options_'+ppv+'.csv'))
	possible_itineraries_clustered_pareto_filtered = pd.read_csv(Path(path_to_strategic_output) / ('possible_itineraries_clustered_pareto_filtered_'+ppv+'.csv'))
	demand = pd.read_csv(Path(path_to_strategic_output) / '..' / '..' / '..' / 'demand'/'demand_ES_MD_intra_v0.4.csv')
	pax_assigned_seats_max_target = pd.read_csv(Path(path_to_strategic_output) / ('pax_assigned_seats_max_target_'+ppv+'.csv'))
	pax_assigned_tactical = pd.read_csv(Path(path_to_strategic_output) / ('pax_assigned_tactical_'+ppv+'.csv'))
	pax_assigned_tactical_not_supported = pd.read_csv(Path(path_to_strategic_output) / ('pax_assigned_tactical_not_supported_'+ppv+'.csv'))

	data = {'pax_assigned_to_itineraries_options':pax_assigned_to_itineraries_options, 'possible_itineraries_clustered_pareto_filtered':possible_itineraries_clustered_pareto_filtered, 'demand':demand, 'pax_assigned_seats_max_target':pax_assigned_seats_max_target,'pax_assigned_tactical':pax_assigned_tactical,'pax_assigned_tactical_not_supported':pax_assigned_tactical_not_supported}
	return data

def read_tactical_data(path_to_tactical_output,tactical_output_name,iteration,parameter_name,path_to_tactical_input):

	df_pax = pd.read_csv(Path(path_to_tactical_output) / (tactical_output_name+'_'+str(iteration)+parameter_name) / 'output_pax.csv.gz',index_col=0,low_memory=False)
	df_flights = pd.read_csv(Path(path_to_tactical_output) / (tactical_output_name+'_'+str(iteration)+parameter_name) / 'output_flights.csv.gz',index_col=0,low_memory=False)
	airport_processes = pd.read_parquet(Path(path_to_tactical_input) / 'data' / 'airports' / 'airport_processes.parquet')
	#input_pax = pd.read_parquet(Path(path_to_tactical_input) / 'case_studies' / 'case_study=0' / 'data' / 'pax' / 'pax_assigned_tactical_0.parquet')
	rail_stations_processes = pd.read_parquet(Path(path_to_tactical_input) / 'case_studies' / 'case_study=0' / 'data' / 'ground_mobility' / 'rail_stations_processes_v0.1.parquet')

	data = {'pax':df_pax,'flights':df_flights,'airport_processes':airport_processes,'rail_stations_processes':rail_stations_processes}
	return data

def add_airport_processes(df_pax,airport_processes,pax_assigned_tactical,iteration,parameter_name,config):

	#adding airport processes to the output from mercury for air/multimodal pax

	# generate kerb2gate_time for air pax
	air_pax = df_pax[(df_pax['kerb2gate_time']==0) & (~pd.isnull(df_pax['airport1']))].copy()
	air_pax = airport_process_time_generator(air_pax,config,airport_processes,process='k2g')
	#only options that have time defined
	df = df_pax[df_pax['kerb2gate_time']>0].copy()
	#merge
	df_pax = pd.concat([df,air_pax])

	# generate gate2kerb_time for air pax
	air_pax = df_pax[df_pax['gate2kerb_time']==0].copy()
	air_pax = airport_process_time_generator(air_pax,config,airport_processes,process='g2k')
	#only options that have time defined
	df = df_pax[df_pax['gate2kerb_time']>0].copy()
	#merge
	df_pax = pd.concat([df,air_pax])

	#add access/egress time from input
	pax_assigned_tactical['nid_x'] = pax_assigned_tactical['nid_x'].astype(str)
	df_pax['original_id'] = df_pax['original_id'].astype(str)
	df_pax = df_pax.merge(pax_assigned_tactical[['nid_x','origin','destination','d2i_time','i2d_time']], left_on=['original_id'], right_on=['nid_x'], how='left', indicator=True)

	df_pax['total_time'] = df_pax['kerb2gate_time']+df_pax['tot_journey_time']+df_pax['gate2kerb_time']+df_pax['d2i_time']+df_pax['i2d_time']
	df_pax['source'] = 'mercury'
	print(df_pax)
	df_pax.to_csv(Path(config['input']['path_to_tactical_output']) / (config['input']['tactical_output_name']+'_'+str(iteration)+parameter_name) / 'postprocessing_pax.csv')

def airport_process_time_generator(pax,config,airport_processes,process='k2g'):

	airports__taxi_estimation_scale = config['paras']['airports']['taxi_estimation_scale']
	if process == 'k2g':
		print('generating kerb2gate_time')

		pax['kerb2gate_time'] = pax.apply(lambda row: norm(loc=airport_processes.loc[airport_processes['icao_id']==row['airport1'],'k2g'], scale=airport_processes.loc[airport_processes['icao_id']==row['airport1'],'k2g_std']).rvs()+norm(loc=0., scale=airports__taxi_estimation_scale).rvs(), axis=1)
	elif process == 'g2k':
		print('generating gate2kerb_time')
		if 'airport4' not in pax.columns:
			pax['airport4'] = None

		pax['airport_last'] = pax.apply(lambda row: row['airport4'] if not pd.isnull(row['airport4']) else row['airport3'] if not pd.isnull(row['airport3']) else row['airport2'], axis=1)
		pax['gate2kerb_time'] = pax.apply(lambda row: norm(loc=airport_processes.loc[airport_processes['icao_id']==row['airport_last'],'g2k'], scale=airport_processes.loc[airport_processes['icao_id']==row['airport_last'],'g2k_std']).rvs()+norm(loc=0., scale=airports__taxi_estimation_scale).rvs(), axis=1)
	print(pax[['kerb2gate_time','gate2kerb_time']])
	print(pax.head())
	return pax

def rail_station_process_time_generator(pax,rail_stations_processes):

	def get_n_from_path(path, n):
		if type(path)==str:
			# The list is in a string form
			path = ast.literal_eval(path)
		return path[n]

	pax['origin_station'] = pax.apply(lambda row: get_n_from_path(row['path'],0), axis=1)
	pax['destination_station'] = pax.apply(lambda row: get_n_from_path(row['path'],-1), axis=1)

	print(get_n_from_path(['007131412', '007131400'],0), int(get_n_from_path(['007131412', '007131400'],0)))
	pax['platform2kerb_time'] = pax.apply(lambda row: rail_stations_processes.loc[rail_stations_processes['station']==int(row['destination_station']),'p2k'].iloc[0] if len(rail_stations_processes.loc[rail_stations_processes['station']==int(row['destination_station']),'p2k'])>0 else 0, axis=1)
	pax['kerb2platform_time'] = pax.apply(lambda row: rail_stations_processes.loc[rail_stations_processes['station']==int(row['origin_station']),'k2p'].iloc[0] if len(rail_stations_processes.loc[rail_stations_processes['station']==int(row['origin_station']),'k2p'])>0 else 0, axis=1)

	print(pax[['kerb2platform_time','platform2kerb_time']])
	print(pax.head())
	return pax

def pax_not_supported(df_pax,rail_stations_processes,pax_assigned_to_itineraries_options,iteration,parameter_name,config):

	pax_rail = df_pax[df_pax['type'].isin(['rail','rail_rail','rail_rail_rail','rail_flight_rail'])].copy()
	pax = rail_station_process_time_generator(pax_rail,rail_stations_processes)
	pax = pax.merge(pax_assigned_to_itineraries_options[['pax_group_id','total_time']], how='left', left_on='pax_group_id', right_on='pax_group_id')
	pax = rail_delay(pax,config)
	pax['total_time'] = pax['total_time'] + pax['train_delay']
	pax['source'] = 'postprocessing'
	pax.rename(columns={'pax': 'n_pax',}, inplace=True)
	pax['tot_arrival_delay'] = pax['train_delay']
	pax['modified_itinerary'] = False
	pax['final_destination_reached'] = True

	#other pax not supported
	other_pax = df_pax[df_pax['type'].isin(['rail_rail_flight','flight_rail_flight', 'flight_rail_rail'])].copy()
	other_pax = other_pax.merge(pax_assigned_to_itineraries_options[['pax_group_id','total_time']], how='left', left_on='pax_group_id', right_on='pax_group_id')
	other_pax.rename(columns={'pax': 'n_pax',}, inplace=True)
	pax = pd.concat([pax,other_pax])
	pax.to_csv(Path(config['input']['path_to_tactical_output']) / (config['input']['tactical_output_name']+'_'+str(iteration)+parameter_name) / 'pax_not_supported.csv')

def rail_delay(df_pax,config):

	delay_prob = config['paras']['trains']['delay_prob'] #probability of train being delayed
	delay_mean = config['paras']['trains']['delay_mean'] #mean delay value
	delay_std = config['paras']['trains']['delay_std']
	df_pax['nid_last'] = df_pax.apply(lambda row: row['nid_f1'] if pd.isnull(row['nid_f2']) else row['nid_f2'] if pd.isnull(row['nid_f3']) else row['nid_f3'], axis=1)
	trains = set(df_pax['nid_last'])
	delay={train:norm(loc=delay_mean,scale=delay_std).rvs() for train in trains if random.random()<=delay_prob}
	df_pax['train_delay'] = df_pax.apply(lambda row: delay[row['nid_last']] if row['nid_last'] in delay else 0, axis=1)

	return df_pax

def read_config(toml_path):

	with open(Path(toml_path), mode="rb") as fp:
		toml_config = tomli.load(fp)

	with open(Path(toml_config['input']['path_to_scenario']), mode="rb") as fp:
		scenario_config = tomli.load(fp)

	toml_config.update(scenario_config)
	return toml_config

if __name__ == '__main__':
	print ('postprocessing tactical output... ')

	config = read_config('postprocessing.toml')
	iterations = config['input']['iterations']
	ppv = str(config['input']['ppv'])
	if len(config['input']['parameter_name']) > 0:
		parameter_name = "_"+config['input']['parameter_name']
	else:
		parameter_name = ""
	#print(config)
	data_strategic = read_strategic_output(config['input']['path_to_strategic_output'],ppv)

	for iteration in range(iterations):
		data_tactical = read_tactical_data(config['input']['path_to_tactical_output'],config['input']['tactical_output_name'],iteration,parameter_name,config['input']['path_to_tactical_input'])

		add_airport_processes(data_tactical['pax'],data_tactical['airport_processes'],data_strategic['pax_assigned_tactical'],iteration,parameter_name,config)
		pax_not_supported(data_strategic['pax_assigned_tactical_not_supported'],data_tactical['rail_stations_processes'],data_strategic['pax_assigned_to_itineraries_options'],iteration,parameter_name,config)
