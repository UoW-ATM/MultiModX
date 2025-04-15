from pathlib import Path
import os
import shutil
import argparse
import pandas as pd
import tomli
import numpy as np
import matplotlib.pyplot as plt
from kpi_lib_strategic import strategic_total_journey_time, diversity_of_destinations, modal_share, pax_time_efficiency, demand_served, load_factor, resilience_alternatives, catchment_area, cost_per_user,co2_emissions, buffer_in_itineraries, seamless_of_travel, pax_processes_time, resilience_replanned, pax_resilience_replanned
from kpi_lib_tactical import flight_arrival_delay, kerb2gate_time, total_journey_time, variability

def read_strategic_output(path_to_strategic_output,preprocessed_version):
	pax_assigned_to_itineraries_options = pd.read_csv(Path(path_to_strategic_output) / ('pax_assigned_to_itineraries_options_'+preprocessed_version+'.csv'))
	possible_itineraries_clustered_pareto_filtered = pd.read_csv(Path(path_to_strategic_output) / ('possible_itineraries_clustered_pareto_filtered_'+preprocessed_version+'.csv'))
	demand = pd.read_csv(Path(path_to_strategic_output) / '..' / '..' / '..' / 'demand'/'demand_ES_MD_intra_v0.4.csv')
	pax_assigned_seats_max_target = pd.read_csv(Path(path_to_strategic_output) / ('pax_assigned_seats_max_target_'+preprocessed_version+'.csv'))
	pax_assigned_tactical = pd.read_csv(Path(path_to_strategic_output) / ('pax_assigned_tactical_'+preprocessed_version+'.csv'))
	pax_assigned_tactical_not_supported = pd.read_csv(Path(path_to_strategic_output) / ('pax_assigned_tactical_not_supported_'+preprocessed_version+'.csv'))
	rail_timetable_proc = pd.read_csv(Path(path_to_strategic_output) / '..' / ('rail_timetable_proc_'+preprocessed_version+'.csv'))
	flight_schedules_proc = pd.read_csv(Path(path_to_strategic_output) / '..' / ('flight_schedules_proc_'+preprocessed_version+'.csv'))
	nuts_regional_archetype_info = pd.read_csv(Path(path_to_strategic_output) / '..' / '..' /'..' / 'nuts_regional_archetype_info_v0.2.csv')

	data = {'pax_assigned_to_itineraries_options':pax_assigned_to_itineraries_options, 'possible_itineraries_clustered_pareto_filtered':possible_itineraries_clustered_pareto_filtered, 'demand':demand, 'pax_assigned_seats_max_target':pax_assigned_seats_max_target,'pax_assigned_tactical':pax_assigned_tactical,'pax_assigned_tactical_not_supported':pax_assigned_tactical_not_supported, 'rail_timetable_proc':rail_timetable_proc,'flight_schedules_proc':flight_schedules_proc, 'nuts_regional_archetype_info':nuts_regional_archetype_info}
	return data

def read_tactical_data(path_to_tactical_output,path_to_tactical_input):

	df_pax = pd.read_csv(Path(path_to_tactical_output) / 'output_pax.csv.gz',index_col=0,low_memory=False)
	df_flights = pd.read_csv(Path(path_to_tactical_output) / 'output_flights.csv.gz',index_col=0,low_memory=False)
	airport_processes = pd.read_parquet(Path(path_to_tactical_input) / 'data' / 'airports' / 'airport_processes.parquet')
	#input_pax = pd.read_parquet(Path(path_to_tactical_input) / 'case_studies' / 'case_study=0' / 'data' / 'pax' / 'pax_assigned_tactical_0.parquet')
	rail_stations_processes = pd.read_parquet(Path(path_to_tactical_input) / 'case_studies' / 'case_study=0' / 'data' / 'ground_mobility' / 'rail_stations_processes_v0.1.parquet')

	data = {'pax':df_pax,'flights':df_flights,'airport_processes':airport_processes,'rail_stations_processes':rail_stations_processes}
	return data

def recreate_output_folder(folder_path: Path):
    """
    Check if a folder exists, delete it if it does, and recreate it as an empty folder.

    Args:
        folder_path (Path): The path to the folder.
    """
    if folder_path.exists():

        shutil.rmtree(folder_path)

    folder_path.mkdir(parents=True, exist_ok=True)


def read_config(toml_path):

	with open(Path(toml_path), mode="rb") as fp:
		toml_config = tomli.load(fp)

	return toml_config

def save_results(results):
	res_list = []
	#print(results)
	for indicator, variants in results.items():
		for variant in variants:
			#print(variant)
			if isinstance(variant['val'], pd.DataFrame):
				variant['val'].to_csv(Path(config['output']['path_to_output']) / (indicator+'__'+variant['name']+'.csv'),index=False)
			if np.isscalar(variant['val']):
				#print('x',variant['val'])
				res_list.append({'indicator':indicator,'variant':variant['name'],'value':variant['val']})
	if len(res_list)>0:
		pd.DataFrame(res_list).to_csv(Path(config['output']['path_to_output']) / ('indicators.csv'),index=False)

def read_results(paths,config):
	plot_column = 'strategic_total_journey_time__sum'
	results = pd.DataFrame()
	for i,path in enumerate(paths):
		indicator_path = Path(config['output']['path_to_output']) / path / 'indicators' / 'indicators.csv'
		if not indicator_path.exists():
			print('No indicators.csv for ',path)
			continue
		df = pd.read_csv(indicator_path)
		df['pi'] = df['indicator']+'__'+df['variant']
		df = df.set_index('pi').drop(columns=['indicator','variant']).rename({'value': path}, axis=1).transpose().reset_index().rename({'index': 'experiment'}, axis=1)
		#print(df)
		results = pd.concat([results,df])

	print(results)
	if len(results)>0:
		results.to_csv(Path(config['output']['path_to_output']) / 'comparison.csv')
		ax = results.plot.bar(x='experiment', y=plot_column, rot=0)
		#plt.show()
		#print(results.loc[(results['indicator']=='strategic_total_journey_time')&(results['variant']=='avg'),].columns[2:])

def read_results_replanned(paths,config):

	data_list = []
	for i,path in enumerate(paths):
		data = {}
		preprocessed_version = config['input']['preprocessed_version'][i]
		flights_path = Path(config['output']['path_to_output']) / path / ('flight_schedules_proc_'+preprocessed_version+'.csv')
		rail_path = Path(config['output']['path_to_output']) / path / ('rail_timetable_proc_'+preprocessed_version+'.csv')
		flight_schedules_proc = pd.read_csv(flights_path, parse_dates=['sobt','sibt'])
		rail_timetable_proc = pd.read_csv(rail_path, parse_dates=['arrival_time','departure_time'])
		rail_timetable_proc['departure_time'] = pd.to_datetime(rail_timetable_proc['departure_time'], utc=True)
		rail_timetable_proc['arrival_time'] = pd.to_datetime(rail_timetable_proc['arrival_time'], utc=True)

		pax_path0 = Path(config['output']['path_to_output']) / path / 'pax_replanned' / ('0.pax_assigned_to_itineraries_options_status_replanned_'+preprocessed_version+'.csv')
		pax_path3 = Path(config['output']['path_to_output']) / path / 'pax_replanned' / ('3.pax_reassigned_to_itineraries_'+preprocessed_version+'.csv')
		pax_path4 = Path(config['output']['path_to_output']) / path / 'pax_replanned' / ('4.pax_assigned_to_itineraries_replanned_stranded_'+preprocessed_version+'.csv')
		pax_path5 = Path(config['output']['path_to_output']) / path / 'pax_replanned' / ('5.pax_demand_assigned_summary_'+preprocessed_version+'.csv')
		if pax_path0.exists():
			pax_assigned_to_itineraries_options_status_replanned = pd.read_csv(pax_path0)
			data['pax_assigned_to_itineraries_options_status_replanned']=pax_assigned_to_itineraries_options_status_replanned
		if pax_path3.exists():
			pax_reassigned_to_itineraries = pd.read_csv(pax_path3)
			data['pax_reassigned_to_itineraries']=pax_reassigned_to_itineraries
		if pax_path4.exists():
			pax_assigned_to_itineraries_replanned_stranded = pd.read_csv(pax_path4)
			data['pax_assigned_to_itineraries_replanned_stranded']=pax_assigned_to_itineraries_replanned_stranded
		if pax_path5.exists():
			pax_demand_assigned_summary = pd.read_csv(pax_path5)
			data['pax_demand_assigned_summary']=pax_demand_assigned_summary

		data.update({'flight_schedules_proc':flight_schedules_proc,'rail_timetable_proc':rail_timetable_proc})
		data_list.append(data)

	return data_list

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='mmx_kpis', add_help=True)
	parser.add_argument('-tf', '--toml_file', help='TOML defining the indicator configuration', required=False, default='mmx_kpis.toml')
	parser.add_argument('-ex', '--experiment', help='Folder with the experiment', required=False)
	parser.add_argument('-c','--compare', nargs='+', help='Compare experiments', required=False)
	parser.add_argument('-ppv', '--preprocessed_version', nargs='+', help='Preprocessed version of schedules to use', required=False, default=['0'])

	# Examples of usage
	#python3 mmx_kpis.py -ex processed_cs10.pp00.so00_c1
	#python3 mmx_kpis.py -c processed_cs10.pp00.so00_c1 processed_cs10.pp10.so00_c1
	#python3 mmx_kpis.py -c processed_cs10.pp00.so00_c2 processed_c1_replan -ppv 0 1


	# Parse parameters
	args = parser.parse_args()
	if args.toml_file is not None:
		config = read_config(args.toml_file)

	if args.experiment is not None:
		config['input']['path_to_strategic_output'] = Path(config['input']['path_to_strategic_output']) / args.experiment / 'paths_itineraries'
		config['output']['path_to_output'] = Path(config['output']['path_to_output']) / args.experiment / 'indicators'
		config['input']['preprocessed_version'] = args.preprocessed_version[0]

	print ('KPI calculation... ')
	print(args)
	print(config)

	if args.experiment is not None:
		recreate_output_folder(Path(config['output']['path_to_output']))
		data_strategic = read_strategic_output(config['input']['path_to_strategic_output'],config['input']['preprocessed_version'])


		data_tactical = read_tactical_data(config['input']['path_to_tactical_output'],config['input']['path_to_tactical_input'])
		results = {}

		for indicator, vals in config['indicators']['strategic'].items():
			results[indicator] = []
			for variant in vals:
				if variant['variant'] is False:
					continue

				if 'name' not in variant:
					variant['name'] = variant['variant']
				if indicator == 'strategic_total_journey_time':
					val = strategic_total_journey_time(data_strategic,config,variant,variant=variant['variant'])
				if indicator == 'diversity_of_destinations':
					val = diversity_of_destinations(data_strategic,config,variant,variant=variant['variant'])
				if indicator == 'modal_share':
					val = modal_share(data_strategic,config,variant,variant=variant['variant'])
				if indicator == 'pax_time_efficiency':
					val = pax_time_efficiency(data_strategic,config,variant,variant=variant['variant'])
				if indicator == 'demand_served':
					val = demand_served(data_strategic,config,variant,variant=variant['variant'])
				if indicator == 'load_factor':
					val = load_factor(data_strategic,config,variant,variant=variant['variant'])
				if indicator == 'resilience_alternatives':
					val = resilience_alternatives(data_strategic,config,variant,variant=variant['variant'])
				if indicator == 'buffer_in_itineraries':
					val = buffer_in_itineraries(data_strategic,config,variant,variant=variant['variant'])
				if indicator == 'catchment_area':
					val = catchment_area(data_strategic,config,variant,variant=variant['variant'])
				if indicator == 'cost_per_user':
					val = cost_per_user(data_strategic,config,variant,variant=variant['variant'])
				if indicator == 'co2_emissions':
					val = co2_emissions(data_strategic,config,variant,variant=variant['variant'])
				if indicator == 'seamless_of_travel':
					val = seamless_of_travel(data_strategic,config,variant,variant=variant['variant'])
				if indicator == 'pax_processes_time':
					val = pax_processes_time(data_strategic,config,variant,variant=variant['variant'])
				results[indicator].append({'name':variant['name'],'val':val})

			save_results(results)

	if args.compare is not None:
		print(args.compare)
		config['input']['preprocessed_version'] = args.preprocessed_version
		if len(args.preprocessed_version) < len(args.compare):
			config['input']['preprocessed_version'] = ['0']*len(args.compare)
		read_results(args.compare,config)

		#replanned indicators
		data_replanned = read_results_replanned(args.compare,config)
		results = {}

		for indicator, vals in config['indicators']['replanned'].items():
			results[indicator] = []
			for variant in vals:
				if variant['variant'] is False:
					continue

				if 'name' not in variant:
					variant['name'] = variant['variant']
				if indicator == 'resilience_replanned':
					val = resilience_replanned(data_replanned,config,variant,variant=variant['variant'])
				if indicator == 'pax_resilience_replanned':
					val = pax_resilience_replanned(data_replanned,config,variant,variant=variant['variant'])
				results[indicator].append({'name':variant['name'],'val':val})

			save_results(results)

	#flight_arrival_delay(data2['flights'])
	#kerb2gate_time(data['pax'],data['airport_processes'])
	#total_journey_time(data2['pax'],data2['airport_processes'],data['pax_assigned_tactical'])
	#variability(data2['pax'])


