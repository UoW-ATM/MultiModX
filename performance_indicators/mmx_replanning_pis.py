import argparse
import shutil
from pathlib import Path
import random

from kpi_lib_replanned import (read_results_replanning, compute_status_pax,
							   plot_status_pax, plot_pax_affected_w_options_delays,
							   plot_pax_affected_w_options_delays_weighted, compute_weighted_stats,
							   plot_pax_stranded_reasons, aggreagate_num_pax_type,
							   plot_affected_vs_stranded, plot_extra_services,
							   plot_heatmap_from_df, plot_normalised_heatmap,
							   create_dataframe_serive_origin_destination,
							   draw_map_replanned_paths, add_path_replanned,
							   transform_paths_into_coordinates, read_rail_cancelled,
							   read_flights_cancelled, read_flights_added,
							   read_flights_replanned, process_replanned_compute_delay,
							   plot_map_services, group_pax_status, plot_pax_status_grouped)


def recreate_output_folder(folder_path: Path):
	"""
	Check if a folder exists, delete it if it does, and recreate it as an empty folder.

	Args:
		folder_path (Path): The path to the folder.
	"""
	if folder_path.exists():

		shutil.rmtree(folder_path)

	folder_path.mkdir(parents=True, exist_ok=True)



def compute_indicators_replanned(path_experiment, path_output, cs, nd, pp, so, dp, dm, pa):
	str_id_experiment = ('cs' + cs + '.' + 'pp' + pp + '.' + 'nd' + nd + '.'+ 'so' + so + '.' +
						 'dp' + dp + '.' + 'dm' + dm + '.' + 'pa' + pa)

	print("Computing indicators "+str_id_experiment)


	(df_pax_assigned, df_pax_kept, df_pax_need_replanning,
	 df_pax_affected, df_pax_unnafected, df_pax_reassigned,
	 df_pax_stranded, df_pax_summary, df_pax_affected_w_options) = read_results_replanning(path_experiment)

	# Compute pax status numbers
	df_pax_status = compute_status_pax(df_pax_unnafected, df_pax_stranded, df_pax_affected_w_options)
	df_pax_status.to_csv(path_output / 'pax_status_numbers.csv')
	plot_status_pax(df_pax_status, (path_output / 'figures' / (str_id_experiment+'_pax_affected_status.png')))

	# Plot pax affected with options delays
	if len(df_pax_affected_w_options) > 0:
		plot_pax_affected_w_options_delays(df_pax_affected_w_options,
										   (path_output / 'figures' / (str_id_experiment+'_delay_pax_w_options.png')))

		plot_pax_affected_w_options_delays_weighted(df_pax_affected_w_options,
										   (path_output / 'figures' / (str_id_experiment+'_delay_pax_w_options_weighted.png')))


		# Delay columns to compute stats on
		delay_cols = ['delay_departure_home', 'delay_arrival_home', 'delay_total_travel_time']

		# Apply groupby with weighted stats
		summary_df_delays_affected_w_options = df_pax_affected_w_options.groupby('pax_status_replanned').apply(
			compute_weighted_stats, delay_cols).reset_index()

		summary_df_delays_affected_w_options.T.to_csv((path_output / 'summary_df_delays_affected_w_options.csv'))


	plot_pax_stranded_reasons(df_pax_stranded,
										   (path_output / 'figures' / (str_id_experiment+'_pax_stranded_reasons.png')))

	df_pax_stranded.groupby(['pax_status_replanned', 'stranded_type'])['pax_stranded'].sum().reset_index().to_csv((path_output / 'pax_stranded_reasons.csv'), index=False)


	df_aggreagated_unaffected_affected_stranded = aggreagate_num_pax_type(df_pax_unnafected, df_pax_affected_w_options,
																		  df_pax_stranded)


	df_aggreagated_unaffected_affected_stranded.to_csv((path_output / 'pax_unnafected_affected_stranded_values.csv'))


	plot_affected_vs_stranded(df_aggreagated_unaffected_affected_stranded,
							  (path_output / 'figures' / (str_id_experiment+'_pax_affected_vs_stranded.png')))

	if len(df_pax_reassigned) > 0:
		plot_extra_services(df_pax_reassigned,
							(path_output / 'figures' / (str_id_experiment + '_extra_services_used.png')))


	if len(df_pax_need_replanning)>0:
		plot_heatmap_from_df(df_pax_need_replanning, 'origin', 'destination', 'pax',
							 save_path=(path_output / 'figures' / (str_id_experiment + '_pax_need_replanning.png')))

	if len(df_pax_reassigned)>0:
		plot_heatmap_from_df(df_pax_reassigned, 'origin', 'destination', 'pax_assigned',
							 save_path=(path_output / 'figures' / (str_id_experiment + '_pax_reassigned.png')))

	if len(df_pax_stranded)>0:
		plot_heatmap_from_df(df_pax_stranded, 'origin', 'destination', 'pax_stranded',
							 save_path=(path_output / 'figures' / (str_id_experiment + '_pax_stranded.png')))

	if len(df_pax_stranded) > 0:
		plot_normalised_heatmap(
			numerator_df=df_pax_stranded,
			denominator_df=df_pax_need_replanning,
			origin_col='origin',
			destination_col='destination',
			numerator_value_col='pax_stranded',
			denominator_value_col='pax',
			save_path=(path_output / 'figures' / f'{str_id_experiment}_pax_stranded_frac.png')
		)

	if len(df_pax_reassigned) > 0:
		plot_normalised_heatmap(
			numerator_df=df_pax_reassigned,
			denominator_df=df_pax_need_replanning,
			origin_col='origin',
			destination_col='destination',
			numerator_value_col='pax_assigned',
			denominator_value_col='pax',
			save_path=(path_output / 'figures' / f'{str_id_experiment}_pax_reassigned_frac.png')
		)

	if len(df_pax_stranded) > 0:
		# Compute paths with coordinates
		df_fts_od = create_dataframe_serive_origin_destination(path_experiment)
		df_pax_reassigned = add_path_replanned(df_pax_reassigned,df_fts_od)
		df_pax_reassigned = transform_paths_into_coordinates(df_pax_reassigned, cs)
		# Do a few plots
		df_diff_modes = df_pax_reassigned[~df_pax_reassigned.same_modes]
		df_extra_serv = df_pax_reassigned[df_pax_reassigned.extra_services>0]
		df_fewer_serv = df_pax_reassigned[df_pax_reassigned.extra_services<0]
		df_diff_i_f_node = df_pax_reassigned[~(df_pax_reassigned.same_initial_node) |
											 ~(df_pax_reassigned.same_final_node)]
		n = 5  # top n flows
		df_max_pax = df_pax_reassigned.nlargest(n, 'pax_assigned')

		def draw_n_maps_replanned_paths(df, path_output, nplots=1, ending_plot_name='', extra_folder=''):
			if len(df) > 0:
				if len(df) <= nplots:
					indices = list(range(len(df)))
				else:
					indices = random.sample(range(len(df)), nplots)

				for i in indices:
					draw_map_replanned_paths(df, i,
											 path_savefig=(
														 path_output / 'figures' / extra_folder /
														 f'{str_id_experiment}{ending_plot_name}_example_itinerary_{i}.png'))

		draw_n_maps_replanned_paths(df_diff_modes, path_output, 5, 'diff_modes', 'maps')
		draw_n_maps_replanned_paths(df_extra_serv, path_output, 5, '_more_services', 'maps')
		draw_n_maps_replanned_paths(df_fewer_serv, path_output, 5, '_fewer_services', 'maps')
		draw_n_maps_replanned_paths(df_diff_i_f_node, path_output, 5, '_diff_nodes', 'maps')
		draw_n_maps_replanned_paths(df_max_pax, path_output, 5, '_max_flows', 'maps')


def describe_replanning(path_output, vcs, cs, nd, pp, so, dp, dm, pa):
	str_id_experiment = ('cs' + cs + '.' + 'pp' + pp + '.' + 'nd' + nd + '.' + 'so' + so + '.' +
							 'dp' + dp + '.' + 'dm' + dm + '.' + 'pa' + pa)

	print("Describing replanning " + str_id_experiment)

	# Cancelled trains
	df_rail_orig_cancelled = read_rail_cancelled(vcs, cs, nd, pp, so, dp, dm, pa)
	if df_rail_orig_cancelled is not None:
		# We have trains cancelled
		plot_map_services(df_rail_orig_cancelled,type_data='gtfs',
						  legend_str=f'Cancelled rail ({len(df_rail_orig_cancelled.trip_id.drop_duplicates())} rail services)',
						  path_savefig=(path_output / 'figures' / (str_id_experiment + '_rail_cancelled.png')))

	df_flights_cancelled =  read_flights_cancelled(vcs, cs, nd, pp, so, dp, dm, pa)
	if df_flights_cancelled is not None:
		# We have flights cancelled
		plot_map_services(df_flights_cancelled, type_data='services',
						  legend_str=f'Cancelled flights ({len(df_flights_cancelled)} flights)',
						  path_savefig=(path_output / 'figures' / (str_id_experiment + '_flights_cancelled.png')))

	df_flights_added = read_flights_added(vcs, cs, nd, pp, so, dp, dm, pa)
	if df_flights_added is not None:
		# We have flights added
		plot_map_services(df_flights_added, type_data='services', colour='blue',
						  legend_str=f'Added flights ({len(df_flights_added)} flights)',
						  path_savefig=(path_output / 'figures' / (str_id_experiment + '_flights_added.png')))

	df_flights_replanned = read_flights_replanned(vcs, cs, nd, pp, so, dp, dm, pa)
	if df_flights_replanned is not None:
		# We have delayed flights
		plot_map_services(df_flights_replanned, type_data='services', colour='orange',
						  legend_str=f'Replanned flights ({len(df_flights_replanned)} flights)',
						  path_savefig=(path_output / 'figures' / (str_id_experiment + '_flights_replanned.png')))

		# Process replanned to compute delay
		df_description_delay = process_replanned_compute_delay(df_flights_replanned, vcs, cs, pp, nd, so,
															   path_savefig=(path_output / 'figures' / (
																		   str_id_experiment + '_flights_replanned_delays.png')))

		df_description_delay.to_csv(path_output / 'description_delay_replanned_flights.csv')


	# TODO: Missing replanned and added rail





if __name__ == '__main__':

	"""
	Configuration is in mmx_kpis.toml: Paths to data, where to save the results, which indicators to compute

	-ex: name of the folder with processed data  (after running strategic pipeline)

	-c: if we want to compare 2 experiments (need to have computed the indicators first for each experiment individually)

	-ppv: post-processing version (default 0): defines the number that is in file names, e.g. possible_itineraries_1.csv

	The results are saved into a specified folder 'indicators' (path in .toml) as csv or plots.
		# Examples of usage
		python3 mmx_kpis.py -ex processed_cs10.pp00.so00_c1
		python3 mmx_kpis.py -c processed_cs10.pp00.so00_c1 processed_cs10.pp10.so00_c1
		python3 mmx_kpis.py -c processed_cs10.pp00.so00_c2 processed_c1_replan -ppv 0 1
	"""

	parser = argparse.ArgumentParser(description='mmx_kpis', add_help=True)
	parser.add_argument('-pr', '--path_replanned', help='Folder with the replanned results', required=False, default='../data/')
	parser.add_argument('-v', '--version_cs', help='Version of the case study', required=False, default='0.16')
	parser.add_argument('-cs', '--case_study', help='Which case study to do', required=False, default='10')
	parser.add_argument('-nd', '--network_definition', help='Which network definition parameters to do', required=False, default='00')
	parser.add_argument('-pp', '--policy_package', help='Which policy package to do', required=False, default='00')
	parser.add_argument('-so', '--schedule_optimiser', help='Which scheduler optimiser to do', required=False, default='00.00')
	parser.add_argument('-dp', '--disruption_package', help='Which disruption package to do', required=False, default='100')
	parser.add_argument('-dm', '--disruption_management', help='Which disruption management to do', required=False,
						default='00')
	parser.add_argument('-pa', '--passenger_assigment', help='Which version of pax assigment to do', required=False, default='01')
	parser.add_argument('-pamin', '--passenger_assigment_minimum', help='If provided instead of computing indicators '
																		'compute aggregation accross PAs', required=False)
	parser.add_argument('-pamax', '--passenger_assigment_maximum', help='If provided instead of computing indicators '
																		'compute aggregation accross PAs',
						required=False)

	# Parse parameters
	args = parser.parse_args()

	if (args.passenger_assigment_minimum is not None) or (args.passenger_assigment_maximum is not None):
		if (args.passenger_assigment_minimum is not None) and (args.passenger_assigment_maximum is not None):
			#doing max min instead of a given experiment
			path_experiment_wo_ppdmpa = (Path(args.path_replanned) /
										 ('CS'+args.case_study) /
										 ('v='+args.version_cs) /
										 'replanned_disruptions' /
										 ('DP'+args.disruption_package))
			path_output = path_experiment_wo_ppdmpa
			df_pax_status_grouped = group_pax_status(path_experiment_wo_ppdmpa,
													 args.policy_package, args.disruption_management,
													 int(args.passenger_assigment_minimum),
													 int(args.passenger_assigment_maximum))

			if df_pax_status_grouped is not None:
				df_pax_status_grouped.to_csv(path_output /
											 ('pax_status_grouped_DP'+args.disruption_package+
											 '.PP'+args.policy_package+
											  '.DM'+args.disruption_management+
											 '_'+args.passenger_assigment_minimum+'-'+args.passenger_assigment_maximum+
											  '.csv'))

				plot_pax_status_grouped(df_pax_status_grouped, (path_output /
																('pax_status_grouped_DP'+args.disruption_package+
																 '.PP' + args.policy_package +
																 '.DM'+args.disruption_management+
																 '_'+args.passenger_assigment_minimum+
																 '-'+args.passenger_assigment_maximum+'.png')))


	else:
		# Doing a given experiment
		path_experiment = (Path(args.path_replanned) /
						   ('CS'+args.case_study) /
						   ('v='+args.version_cs) /
						   'replanned_disruptions' /
						   ('DP'+args.disruption_package) /
						   ('PP'+args.policy_package+'.'+'DM'+args.disruption_management+'.'+'PA'+args.passenger_assigment))

		# Recreate output folders
		recreate_output_folder((path_experiment / 'indicators'))
		recreate_output_folder((path_experiment/'indicators'/'figures'))
		recreate_output_folder((path_experiment / 'indicators' / 'figures' / 'maps'))

		describe_replanning((path_experiment / 'indicators'),
									 args.version_cs,
									 args.case_study, args.network_definition,
									 args.policy_package, args.schedule_optimiser,
									 args.disruption_package, args.disruption_management,
									 args.passenger_assigment)

		compute_indicators_replanned(path_experiment, (path_experiment / 'indicators'),
									 args.case_study, args.network_definition,
									 args.policy_package, args.schedule_optimiser,
									 args.disruption_package, args.disruption_management,
									 args.passenger_assigment)

