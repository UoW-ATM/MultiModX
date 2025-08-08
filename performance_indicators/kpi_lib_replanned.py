import re
import ast
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import geopandas as gpd
from matplotlib.patches import FancyArrowPatch
from pathlib import Path


from kpi_lib_strategic import plot_heatmap_from_df


def create_dataframe_serive_origin_destination(path_experiment):
	df_fs_replanned = pd.read_csv(path_experiment / 'processed' / 'flight_schedules_proc_1.csv')
	df_ts_replanned = pd.read_csv(path_experiment /  'processed' / 'rail_timetable_proc_1_used_internally.csv',
		dtype={'origin': 'str', 'destination': 'str'})

	df_fs_od = df_fs_replanned[['service_id', 'origin', 'destination']].copy()
	df_fs_od['od'] = df_fs_od['origin'] + ',' + df_fs_od['destination']
	df_ts_replanned = df_ts_replanned[['service_id', 'origin', 'destination']].copy()
	df_ts_replanned['od'] = df_ts_replanned['origin'] + ',' + df_ts_replanned['destination']
	df_fts_od = pd.concat([df_ts_replanned, df_fs_od])
	return df_fts_od


def read_results_replanning(path_experiment):
	df_pax_assigned = pd.read_csv(path_experiment / 'pax_replanned' / '0.pax_assigned_to_itineraries_options_status_replanned_1.csv')
	df_pax_kept = pd.read_csv(path_experiment / 'pax_replanned' / '1.pax_assigned_to_itineraries_options_kept_1.csv')
	df_pax_need_replanning = pd.read_csv(path_experiment / 'pax_replanned' / '2.pax_assigned_need_replanning_1.csv')
	df_pax_affected = pd.read_csv(path_experiment / 'pax_replanned' / '1.2.pax_assigned_kept_affected_delayed_or_connections_kept_1.csv')
	df_pax_unnafected = pd.read_csv(path_experiment / 'pax_replanned' / '1.1.pax_assigned_kept_unnafected_1.csv')
	df_pax_reassigned = pd.read_csv(path_experiment / 'pax_replanned' / '3.pax_reassigned_to_itineraries_1.csv')
	df_pax_stranded = pd.read_csv(path_experiment / 'pax_replanned' / '4.pax_assigned_to_itineraries_replanned_stranded_1.csv')
	df_pax_summary = pd.read_csv(path_experiment / 'pax_replanned' / '5.pax_demand_assigned_summary_1.csv')
	df_pax_affected_w_options = pd.read_csv(path_experiment / 'pax_replanned' / '3.1.pax_affected_all_1.csv')

	return (df_pax_assigned, df_pax_kept, df_pax_need_replanning,
			df_pax_affected, df_pax_unnafected, df_pax_reassigned,
			df_pax_stranded, df_pax_summary, df_pax_affected_w_options)


def read_flights_cancelled(vcs, cs, nd, pp, so, dp, dm, pa):
	# TODO: review paths
	path_cancelled_flights = (Path('../data/') / ("CS" + cs + '/v=' + vcs + '/replanned_disruptions/DP' + dp +
											   '/PP' + pp + '.DM' + dm + '.PA' + pa +
											   '/replanned_actions/flight_cancelled_1.csv'))

	if not os.path.exists(path_cancelled_flights):
		return None
	else:
		df_flight_cancelled = pd.read_csv(path_cancelled_flights)
		df_fs_orig = pd.read_csv(('../data/CS' + cs + '/v=' + vcs + '/output/processed_cs' +
								  cs.lower() + '.pp' + pp.lower() + '.nd' + nd.lower() + '.so' + so.lower() +
								  '/processed/flight_schedules_proc_0.csv'))

		df_airport_coord = pd.read_csv(
			'../data/CS' + cs + '/v=' + vcs + '/infrastructure/airports_info/airports_coordinates_v1.1.csv')

		df_fs_orig_cancelled = df_fs_orig[df_fs_orig.service_id.isin(list(df_flight_cancelled.service_id))]

		df_fs_orig_cancelled = df_fs_orig_cancelled.merge(df_airport_coord, left_on='origin',
														  right_on='icao_id').rename(
			columns={'lat': 'lat_orig', 'lon': 'lon_orig'})
		df_fs_orig_cancelled = df_fs_orig_cancelled.merge(df_airport_coord, left_on='destination',
														  right_on='icao_id').rename(
			columns={'lat': 'lat_dest', 'lon': 'lon_dest'})

		return df_fs_orig_cancelled


def read_flights_replanned(vcs, cs, nd, pp, so, dp, dm, pa):
	# TODO: review paths
	path_replanned_flights = (Path('../data/') / ("CS" + cs + '/v=' + vcs + '/replanned_disruptions/DP' + dp +
											   '/PP' + pp + '.DM' + dm + '.PA' + pa +
											   '/replanned_actions/flight_replanned_proc_1.csv'))
	if not os.path.exists(path_replanned_flights):
		return None
	else:
		df_flights_replanned = pd.read_csv(path_replanned_flights)
		df_airport_coord = pd.read_csv(
			'../data/CS' + cs + '/v=' + vcs + '/infrastructure/airports_info/airports_coordinates_v1.1.csv')

		df_flights_replanned = df_flights_replanned.merge(df_airport_coord, left_on='origin',
														  right_on='icao_id').rename(
			columns={'lat': 'lat_orig', 'lon': 'lon_orig'})
		df_flights_replanned = df_flights_replanned.merge(df_airport_coord, left_on='destination',
														  right_on='icao_id').rename(
			columns={'lat': 'lat_dest', 'lon': 'lon_dest'})

		return df_flights_replanned


def process_replanned_compute_delay(df_flights_replanned, vcs, cs, pp, nd, so, path_savefig=None):
	# Read original fs
	# TODO: review paths
	df_fs_orig = pd.read_csv(('../data/CS' + cs + '/v=' + vcs + '/output/processed_cs' +
							  cs.lower() + '.pp' + pp.lower() + '.nd' + nd.lower() + '.so' + so.lower() +
							  '/processed/flight_schedules_proc_0.csv'))

	df_flight_replanned = df_flights_replanned[['service_id', 'sobt']].merge(df_fs_orig[['service_id', 'sobt']],
																			on='service_id',
																			suffixes=('_replanned', '_orig'))

	df_flight_replanned['sobt_replanned'] = pd.to_datetime(df_flight_replanned['sobt_replanned'])
	df_flight_replanned['sobt_orig'] = pd.to_datetime(df_flight_replanned['sobt_orig'])

	df_flight_replanned['delay'] = (df_flight_replanned['sobt_replanned'] - df_flight_replanned[
		'sobt_orig']).dt.total_seconds() / 60

	# Set style
	plt.figure(figsize=(8, 5))
	sns.set(style="whitegrid")

	# Plot histogram
	ax = sns.histplot(
		data=df_flight_replanned,
		x='delay',
		bins=30,  # adjust bin count if needed
		color='skyblue',
		edgecolor='black'
	)

	# Titles and labels
	plt.xlabel('Flight Delay (minutes)', fontsize=12)
	plt.ylabel('Number of Flights', fontsize=12)
	plt.title('Distribution of Flight Delays', fontsize=14)

	# Optional: clean look
	sns.despine()
	plt.tight_layout()
	if path_savefig is not None:
		plt.savefig(path_savefig)
	else:
		plt.show()
	plt.close()

	return df_flight_replanned['delay'].describe().to_frame()


def read_flights_added(vcs, cs, nd, pp, so, dp, dm, pa):
	# TODO: review paths
	path_added_flights = (Path('../data/') / ("CS" + cs + '/v=' + vcs + '/replanned_disruptions/DP' + dp +
											   '/PP' + pp + '.DM' + dm + '.PA' + pa +
											   '/replanned_actions/flight_added_schedules_proc_1.csv'))
	if not os.path.exists(path_added_flights):
		return None
	else:
		df_flights_added = pd.read_csv(path_added_flights)
		df_airport_coord = pd.read_csv(
			'../data/CS' + cs + '/v=' + vcs + '/infrastructure/airports_info/airports_coordinates_v1.1.csv')

		df_flights_added = df_flights_added.merge(df_airport_coord, left_on='origin',
														  right_on='icao_id').rename(
			columns={'lat': 'lat_orig', 'lon': 'lon_orig'})
		df_flights_added = df_flights_added.merge(df_airport_coord, left_on='destination',
														  right_on='icao_id').rename(
			columns={'lat': 'lat_dest', 'lon': 'lon_dest'})

		return df_flights_added



def read_rail_cancelled(vcs, cs, nd, pp, so, dp, dm, pa):
	# TODO: review paths
	path_cancelled_rail = (Path('../data/') / ("CS"+cs + '/v='+vcs+'/replanned_disruptions/DP' + dp +
											   '/PP' + pp + '.DM' + dm + '.PA' + pa +
											   '/replanned_actions/rail_cancelled_1.csv'))

	if not os.path.exists(path_cancelled_rail):
		return None
	else:
		df_rail_cancelled = pd.read_csv(path_cancelled_rail)
		df_rail_orig_gtfs = pd.read_csv(('../data/CS' + cs + '/v='+vcs+'/output/processed_cs' +
										 cs.lower() + '.pp' + pp.lower() + '.nd' + nd.lower() + '.so' + so.lower() +
										 '/processed/rail_timetable_proc_gtfs_0.csv'), dtype={'stop_id': 'str'})

		df_rail_cancelled = df_rail_cancelled.drop_duplicates()
		if 'from' not in df_rail_cancelled.columns:
			df_rail_cancelled['from'] = None
		if 'to' not in df_rail_cancelled.columns:
			df_rail_cancelled['to'] = None

		df_rail_orig_cancelled = df_rail_orig_gtfs.merge(df_rail_cancelled, left_on='trip_id', right_on='service_id')

		df_rail_orig_cancelled = df_rail_orig_cancelled[
			(
					df_rail_orig_cancelled['from'].isna() |
					(df_rail_orig_cancelled['stop_sequence'] >= df_rail_orig_cancelled['from'].fillna(float('inf')))
			) &
			(
					df_rail_orig_cancelled['to'].isna() |
					(df_rail_orig_cancelled['stop_sequence'] <= df_rail_orig_cancelled['to'].fillna(float('-inf')))
			)
			]

		df_stops = pd.read_csv('../data/CS' + cs + '/v='+vcs+'/gtfs_es_UIC_v2.3/stops.txt', dtype={'stop_id': str})

		df_rail_orig_cancelled = df_rail_orig_cancelled.merge(df_stops, on='stop_id')

		return df_rail_orig_cancelled


def compute_status_pax(df_pax_unnafected, df_pax_stranded, df_pax_affected_w_options):

	pax_rebooked = df_pax_affected_w_options[
		((df_pax_affected_w_options.pax_status_replanned == 'replanned_no_doable') |
		 (df_pax_affected_w_options.pax_status_replanned == 'cancelled'))].copy()

	pax_same_mode = pax_rebooked[pax_rebooked.mode_combined_orig == pax_rebooked.mode_combined_replanned]
	pax_air_to_rail = pax_rebooked[
		(pax_rebooked.mode_combined_orig == 'air') & (pax_rebooked.mode_combined_replanned == 'rail')]
	pax_air_to_multi = pax_rebooked[
		(pax_rebooked.mode_combined_orig == 'air') & (pax_rebooked.mode_combined_replanned == 'multimodal')]
	pax_rail_to_air = pax_rebooked[
		(pax_rebooked.mode_combined_orig == 'rail') & (pax_rebooked.mode_combined_replanned == 'air')]
	pax_rail_to_multi = pax_rebooked[
		(pax_rebooked.mode_combined_orig == 'rail') & (pax_rebooked.mode_combined_replanned == 'multimodal')]
	pax_multi_to_air = pax_rebooked[
		(pax_rebooked.mode_combined_orig == 'multimodal') & (pax_rebooked.mode_combined_replanned == 'air')]
	pax_multi_to_rail = pax_rebooked[
		(pax_rebooked.mode_combined_orig == 'multimodal') & (pax_rebooked.mode_combined_replanned == 'rail')]

	pax_on_time = df_pax_affected_w_options[(df_pax_affected_w_options.pax_status_replanned == 'replanned_doable') & (
				df_pax_affected_w_options.delay_arrival_home == 0)]
	pax_delayed = df_pax_affected_w_options[(df_pax_affected_w_options.pax_status_replanned == 'delayed')]
	pax_doable_delayed = df_pax_affected_w_options[
		(df_pax_affected_w_options.pax_status_replanned == 'replanned_doable') & (
					df_pax_affected_w_options.delay_arrival_home != 0)]

	dict_statistics_pax_status = {
		'pax_unnafected': int(df_pax_unnafected.pax.sum()),  # Unnafected
		'pax_stranded': int(df_pax_stranded.pax_stranded.sum()),  # Stranded
		'pax_on_time': int(pax_on_time.pax_assigned.sum()),  # Affected but connections ok
		'pax_delayed': int(pax_delayed.pax_assigned.sum() + pax_doable_delayed.pax_assigned.sum()),
		'pax_rebooked_same_mode': int(pax_same_mode.pax_assigned.sum()),
		'pax_rebooked_air_2_rail': int(pax_air_to_rail.pax_assigned.sum()),
		'pax_rebooked_air_2_multi': int(pax_air_to_multi.pax_assigned.sum()),
		'pax_rebooked_rail_2_air': int(pax_rail_to_air.pax_assigned.sum()),
		'pax_rebooked_rail_2_multi': int(pax_rail_to_multi.pax_assigned.sum()),
		'pax_rebooked_multi_2_air': int(pax_multi_to_air.pax_assigned.sum()),
		'pax_rebooked_multi_2_rail': int(pax_multi_to_rail.pax_assigned.sum()),
	}

	df_pax_status = pd.DataFrame.from_dict(dict_statistics_pax_status, orient='index', columns=['number_pax'])

	return df_pax_status


def plot_status_pax(df_pax_status, path_savefig=None):
	# --- Separate unaffected and affected ---
	pax_unnaffected_plot = df_pax_status.loc['pax_unnafected', 'number_pax']
	affected_df_plot = df_pax_status.drop(index='pax_unnafected')

	# --- Total affected ---
	affected_total = affected_df_plot['number_pax'].sum()
	affected_percent = 100 * affected_total / (pax_unnaffected_plot + affected_total)

	# --- Define plotting order (adjust as needed) ---
	plot_order = [
		'pax_stranded',
		'pax_on_time',
		'pax_delayed',
		'pax_rebooked_same_mode',
		'pax_rebooked_air_2_rail',
		'pax_rebooked_air_2_multi',
		'pax_rebooked_rail_2_air',
		'pax_rebooked_rail_2_multi',
		'pax_rebooked_multi_2_air',
		'pax_rebooked_multi_2_rail'
	]

	# --- Plot settings ---
	fig, ax = plt.subplots(figsize=(8, 10))

	# Base y for stacking
	y_offset = 0

	# Color and hatch settings
	colors = {
		'pax_stranded': '#d73027',
		'pax_on_time': '#1a9850',
		'pax_delayed': '#fee08b',
		'rebooked': '#4575b4'
	}

	hatches = ['', '///', '+++', '**', '...', '|||', '\\\\\\']  # for rebooked categories

	# --- Plot stack ---
	handles = []
	labels = []

	hu = 0
	for idx, label in enumerate(plot_order):
		val = df_pax_status.loc[label, 'number_pax']
		if val == 0:
			continue

		# Determine if it's a rebooked category
		if label.startswith('pax_rebooked'):
			color = colors['rebooked']
			hatch = hatches[hu]  # shift for non-rebooked entries
			hu += 1
		else:
			color = colors.get(label, 'gray')
			hatch = ''

		bar = ax.bar(0, val, bottom=y_offset, color=color, hatch=hatch, edgecolor='black', label=label)
		handles.append(bar)
		labels.append(label.replace('pax_', '').replace('_', ' '))
		y_offset += val

	# --- Annotate affected total and percentage ---
	ax.text(0, y_offset + 50, f"Affected: {affected_total:,} ({affected_percent:.1f}%) of total pax",
			ha='center', va='bottom', fontsize=12, fontweight='bold')

	# --- Final plot formatting ---
	ax.set_xticks([])
	ax.set_ylabel("Number of passengers")
	# a#x.set_title("Passenger Outcomes (excluding unaffected)")

	# Custom legend
	ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', title="Category")

	plt.tight_layout()
	if path_savefig is not None:
		plt.savefig(path_savefig)
	else:
		plt.show()
	plt.close()



def weighted_quantile(values, weights, quantiles):
	"""
	Compute weighted quantiles. Similar to np.percentile but for weights.
	"""
	sorter = np.argsort(values)
	values = np.array(values)[sorter]
	weights = np.array(weights)[sorter]
	weighted_cumsum = np.cumsum(weights)
	total_weight = weighted_cumsum[-1]
	return np.interp(np.array(quantiles), weighted_cumsum / total_weight, values)


# Function to compute weighted stats for a group
def compute_weighted_stats(group, delay_cols):
	weights = group['pax_assigned']
	results = {}
	for col in delay_cols:
		vals = group[col]
		wmean = np.average(vals, weights=weights)
		wmin = vals.min()
		wmax = vals.max()
		wq25, wmed, wq75 = weighted_quantile(vals, weights, [0.25, 0.5, 0.75])

		results.update({
			f'{col}_mean': wmean,
			f'{col}_median': wmed,
			f'{col}_min': wmin,
			f'{col}_max': wmax,
			f'{col}_q25': wq25,
			f'{col}_q75': wq75,
		})
	return pd.Series(results)



def plot_pax_affected_w_options_delays(df_pax_affected_w_options, path_savefig):
	# Select relevant columns
	selected_columns = ['delay_departure_home', 'delay_arrival_home', 'delay_total_travel_time']
	df_melted = df_pax_affected_w_options.melt(
		id_vars='pax_status_replanned',  # this keeps the status column for grouping
		value_vars=selected_columns,
		var_name='Variable',
		value_name='Value'
	)


	rename_dict = {
		'delay_departure_home': 'Departure Delay',
		'delay_arrival_home': 'Arrival Delay',
		'delay_total_travel_time': 'Total Travel Delay'
	}

	status_rename = {
		'replanned_doable': 'Affected but kept itinerary',
		'delayed': 'Delayed',
		'cancelled': 'Cancelled',
		'replanned_no_doable': 'Affected missed connections'
	}

	df_melted['Delay Type'] = df_melted['Variable'].replace(rename_dict)
	df_melted['pax_status_replanned'] = df_melted['pax_status_replanned'].replace(status_rename)

	plt.figure(figsize=(12, 6))
	sns.violinplot(x='Delay Type', y='Value', hue='pax_status_replanned', data=df_melted, cut=0)  # , split=True)
	sns.boxplot(
		data=df_melted, x='Delay Type', y='Value',
		hue='pax_status_replanned',
		dodge=True, showcaps=True, boxprops={'facecolor': 'none'},
		whiskerprops={'linewidth': 1.5}, fliersize=0,
		legend=False
	)
	plt.legend(title='Replanned Status')
	plt.xlabel('Delay Type')
	plt.ylabel('Minutes')



	if path_savefig is not None:
		plt.savefig(path_savefig)
	else:
		plt.show()

	plt.close()


def plot_pax_affected_w_options_delays_weighted(df_pax_affected_w_options, path_savefig=None, max_weight_replication=10000):
	"""
	Plots violin plots of delay variables, accounting for weighting by 'pax_assigned'.
	Values are repeated proportionally to represent per-pax distribution, capped by max_weight_replication.

	Parameters:
	- df_pax_affected_w_options: DataFrame with delay columns and 'pax_assigned'
	- path_savefig: where to save the figure
	- max_weight_replication: cap on total data expansion to avoid memory issues
	"""

	selected_columns = ['delay_departure_home', 'delay_arrival_home', 'delay_total_travel_time']
	df_melted = df_pax_affected_w_options.melt(
		id_vars=['pax_status_replanned', 'pax_assigned'],
		value_vars=selected_columns,
		var_name='Variable',
		value_name='Value'
	)

	rename_dict = {
		'delay_departure_home': 'Departure Delay',
		'delay_arrival_home': 'Arrival Delay',
		'delay_total_travel_time': 'Total Travel Delay'
	}

	status_rename = {
		'replanned_doable': 'Affected but kept itinerary',
		'delayed': 'Delayed',
		'cancelled': 'Cancelled',
		'replanned_no_doable': 'Affected missed connections'
	}

	df_melted['Delay Type'] = df_melted['Variable'].replace(rename_dict)
	df_melted['pax_status_replanned'] = df_melted['pax_status_replanned'].replace(status_rename)

	# Normalize weights to avoid memory explosion
	total_weight = df_melted['pax_assigned'].sum()
	scale_factor = min(1.0, max_weight_replication / total_weight)
	df_melted['replication'] = (df_melted['pax_assigned'] * scale_factor).round().astype(int)

	# Expand the data according to weights
	df_expanded = df_melted.loc[df_melted.index.repeat(df_melted['replication'])]

	# Plot
	plt.figure(figsize=(12, 6))
	sns.violinplot(
		x='Delay Type',
		y='Value',
		hue='pax_status_replanned',
		data=df_expanded,
		cut=0,
		density_norm='count'
	)
	sns.boxplot(
		data=df_expanded, x='Delay Type', y='Value',
		hue='pax_status_replanned',
		dodge=True, showcaps=True, boxprops={'facecolor': 'none'},
		whiskerprops={'linewidth': 1.5}, fliersize=0,
		legend=False
	)
	plt.legend(title='Replanned Status')
	plt.xlabel('Delay Type')
	plt.ylabel('Minutes')

	if path_savefig is not None:
		plt.savefig(path_savefig, bbox_inches='tight')
	else:
		plt.show()

	plt.close()


def plot_pax_stranded_reasons(df_pax_stranded, path_savefig=None):
	grouped = df_pax_stranded.groupby(['pax_status_replanned', 'stranded_type'])['pax_stranded'].sum().reset_index()

	# Pivot and clean
	pivot_df = grouped.pivot(index='pax_status_replanned', columns='stranded_type', values='pax_stranded').fillna(0)

	# Optional: rename for aesthetics
	status_labels = {
		'cancelled': 'Cancelled',
		'replanned_no_doable': 'Connection not doable'
	}
	type_labels = {
		'no_capacity': 'No capacity',
		'no_option': 'No option'
	}

	pivot_df.index = pivot_df.index.map(status_labels.get)
	pivot_df.columns = [type_labels.get(col, col) for col in pivot_df.columns]

	# Calculate custom labels
	totals = pivot_df.sum(axis=1)
	if (len(pivot_df.values) > 1) and (len(pivot_df.columns) > 1):
		custom_labels = [
			f"{int(total)} [{int(values[0])}/{int(values[1])}]"
			for total, values in zip(totals, pivot_df.values)
		]
	else:
		custom_labels = [
			f"{int(total)}"
			for total, values in zip(totals, pivot_df.values)
		]

	# Plot
	ax = pivot_df.plot(
		kind='bar',
		stacked=True,
		figsize=(10, 6),
		colormap='Set2',
		edgecolor='black'
	)

	# Annotate the total on top of each bar
	for idx, label in enumerate(custom_labels):
		# X position: use the first container to get the x coordinate
		bar = ax.containers[0][idx]
		x = bar.get_x() + bar.get_width() / 2

		# Y position: sum the heights from each container for the same idx
		y = sum(container[idx].get_height() for container in ax.containers)

		ax.text(x, y + 5, label, ha='center', va='bottom', fontsize=10)

	# Layout
	plt.xlabel('Passenger Replanning Status')
	plt.ylabel('Number of Stranded Pax')
	plt.legend(title='Stranded Type')
	plt.xticks(rotation=0)
	plt.tight_layout()
	if path_savefig is not None:
		plt.savefig(path_savefig)
	else:
		plt.show()

	plt.close()


def aggreagate_num_pax_type(df_pax_unnafected, df_pax_affected_w_options, df_pax_stranded):
	# --- Aggregate the data ---
	unaffected = df_pax_unnafected.groupby('pax_status_replanned')['pax'].sum()
	affected = df_pax_affected_w_options.groupby('pax_status_replanned')['pax_assigned'].sum()
	stranded = df_pax_stranded.groupby('stranded_type')['pax_stranded'].sum()

	# --- Combine all into one DataFrame ---
	df_combined = pd.DataFrame({
		'Unaffected': unaffected,
		'Affected': affected,
		'Stranded': stranded
	}).fillna(0)

	return df_combined


def plot_affected_vs_stranded(df_combined, path_savefig=None):
	if 'Unaffected' in df_combined.columns:
		df_combined = df_combined.drop('Unaffected', axis=1)

	df_combined = df_combined.T  # Transpose for easier plotting

	if 'unnafected' in df_combined.columns:
		df_combined = df_combined.drop('unnafected', axis=1)

	df_combined.rename(columns={'cancelled': 'Cancelled', 'delayed': 'Delayed', 'no_capacity': 'No capacity',
								'no_option': 'No option', 'replanned_doable': 'Replanned maintain connection',
								'replanned_no_doable': 'Replanned missed connection'}, inplace=True)

	# --- Plot ---
	colors = plt.cm.tab20.colors
	fig, ax = plt.subplots(figsize=(10, 6))

	bottom = [0] * df_combined.shape[0]
	for i, column in enumerate(df_combined.columns):
		ax.bar(df_combined.index, df_combined[column], bottom=bottom, label=column, color=colors[i % len(colors)])
		bottom = [bottom[j] + df_combined[column].iloc[j] for j in range(len(bottom))]

	# --- Labels & legend ---
	ax.set_ylabel('Number of Passengers')
	ax.legend(title='Subcategory', bbox_to_anchor=(1.05, 1), loc='upper left')
	plt.tight_layout()
	if path_savefig is not None:
		plt.savefig(path_savefig)
	else:
		plt.show()
	plt.close()


def plot_extra_services(df_pax_reassigned, path_savefig=None):	# Group by extra_services and sum pax_assigned
	grouped = df_pax_reassigned.groupby('extra_services')['pax_assigned'].sum().reset_index()

	# Create labels like '0 services', '1 service', etc.
	grouped['label'] = grouped['extra_services'].apply(
		lambda x: f"{x} service" if x == 1 else f"{x} services"
	)

	# Calculate percentages
	total = grouped['pax_assigned'].sum()
	grouped['percentage'] = grouped['pax_assigned'] / total * 100

	# Sort by extra_services just in case
	grouped = grouped.sort_values('extra_services')

	# Plot
	plt.figure(figsize=(8, 5))
	sns.set(style="whitegrid")
	ax = sns.barplot(data=grouped, x='label', y='pax_assigned', color='lightblue', edgecolor='black')

	# Add labels with number and percentage
	for i, row in grouped.iterrows():
		ax.text(i, row['pax_assigned'] + max(grouped['pax_assigned']) * 0.01,
				f"{int(row['pax_assigned'])}\n({row['percentage']:.1f}%)",
				ha='center', va='bottom', fontsize=10)

	# Titles and axis labels
	# plt.title('Passengers Reassigned by Number of Extra Services', fontsize=14)
	plt.xlabel('Extra Services', fontsize=12)
	plt.ylabel('Number of Reassigned Passengers', fontsize=12)
	plt.xticks(rotation=0)

	sns.despine()
	plt.tight_layout()
	if path_savefig is not None:
		plt.savefig(path_savefig)
	else:
		plt.show()
	plt.close()


def plot_normalised_heatmap(
		numerator_df,
		denominator_df,
		origin_col,
		destination_col,
		numerator_value_col,
		denominator_value_col,
		save_path=None,
		vmin=0,
		vmax=1
):
	"""
	Plot a heatmap of (numerator / denominator) per origin-destination pair.
	Missing numerator values are treated as zero.

	Parameters:
		numerator_df (pd.DataFrame): e.g. pax_stranded.
		denominator_df (pd.DataFrame): e.g. pax_need_replanning.
		origin_col (str): Origin column name.
		destination_col (str): Destination column name.
		numerator_value_col (str): Value column for numerator.
		denominator_value_col (str): Value column for denominator.
		save_path (str): Path to save figure.
		vmin, vmax: Color scale bounds.
	"""

	# Ensure all OD pairs from denominator are retained, fill missing numerators with 0
	merged = pd.merge(
		denominator_df[[origin_col, destination_col, denominator_value_col]],
		numerator_df[[origin_col, destination_col, numerator_value_col]],
		on=[origin_col, destination_col],
		how='left'
	)

	# Fill NaNs in numerator with 0
	merged[numerator_value_col] = merged[numerator_value_col].fillna(0)

	# Avoid division by zero just in case (filter out or assign NaN)
	merged['fraction'] = merged[numerator_value_col] / merged[denominator_value_col]
	merged['fraction'] = merged['fraction'].replace([np.inf, -np.inf], np.nan)

	# Plot
	plot_heatmap_from_df(
		merged,
		origin_col=origin_col,
		destination_col=destination_col,
		value_col='fraction',
		vmin=vmin,
		vmax=vmax,
		save_path=save_path
	)


def add_path_replanned(df_pax_reassigned,df_fts_od):
	# Create a lookup dictionary from df_fts_od
	od_lookup = df_fts_od.set_index('service_id')['od'].to_dict()

	# Identify all service_id columns (e.g. service_id_0, service_id_1, ...)
	service_id_cols = [col for col in df_pax_reassigned.columns if col.startswith('service_id_')]
	service_id_cols = [s for s in service_id_cols if 'pax' not in s]

	# Function to build path for each row
	def build_path(row):
		service_ids = [sid for sid in row[service_id_cols] if pd.notna(sid)]
		ods = [od_lookup.get(sid, '') for sid in service_ids]
		return ','.join(ods)

	# Apply function to create path column
	df_pax_reassigned['path_replanned'] = df_pax_reassigned.apply(build_path, axis=1)

	df_pax_reassigned['path_replanned'] = df_pax_reassigned.path_replanned.apply(lambda x: '[' + x + ']')

	def parse_path_string(s):
		return re.findall(r'[^,\[\]\s]+', s)

	# Example usage on a DataFrame column
	df_pax_reassigned['path_replanned'] = df_pax_reassigned['path_replanned'].apply(parse_path_string)

	def remove_consecutive_duplicates(lst):
		return [x for i, x in enumerate(lst) if i == 0 or x != lst[i - 1]]

	df_pax_reassigned['path_replanned'] = df_pax_reassigned['path_replanned'].apply(remove_consecutive_duplicates)

	# Convert string representation of list to actual list for original paths
	df_pax_reassigned["path"] = df_pax_reassigned["path"].apply(ast.literal_eval)

	return df_pax_reassigned

def transform_paths_into_coordinates(df_pax_reassigned, cs):
	# TODO: path coordinates better
	# Transform paths into coordinates

	df_airport_coord = pd.read_csv(
		'../data/CS' + cs + '/v=0.16/infrastructure/airports_info/airports_coordinates_v1.1.csv')
	df_stops = pd.read_csv('../data/CS' + cs + '/v=0.16/gtfs_es_UIC_v2.3/stops.txt', dtype={'stop_id': str})

	# 1. Build lookup dictionary for coordinates

	# Airport dictionary
	airport_coords = {
		row["icao_id"]: (row["lat"], row["lon"])
		for _, row in df_airport_coord.iterrows()
	}

	# Rail stop dictionary
	rail_coords = {
		str(row["stop_id"]): (row["stop_lat"], row["stop_lon"])
		for _, row in df_stops.iterrows()
	}

	# Combine both into one lookup dictionary
	coord_lookup = {**airport_coords, **rail_coords}

	# 2. Map each path to its list of coordinates

	def path_to_coords(path):
		coords = []
		for node in path:
			if node in coord_lookup:
				coords.append(coord_lookup[node])
			else:
				# Optionally log or flag missing nodes
				print(f" Missing coordinates for: {node}")
				coords.append(None)  # Or skip with: continue
		return coords

	# Assuming your path column is named 'path'
	df_pax_reassigned["path_coords"] = df_pax_reassigned["path"].apply(path_to_coords)
	df_pax_reassigned["path_coords"] = df_pax_reassigned["path_coords"].apply(
		lambda path: [(lon, lat) for (lat, lon) in path]
	)

	# Assuming your path column is named 'path'
	df_pax_reassigned["path_replanned_coords"] = df_pax_reassigned["path_replanned"].apply(path_to_coords)
	df_pax_reassigned["path_replanned_coords"] = df_pax_reassigned["path_replanned_coords"].apply(
		lambda path: [(lon, lat) for (lat, lon) in path]
	)

	return df_pax_reassigned


def draw_path_with_arrows(ax, coords, color, arrow_size=0.0015):
	for start, end in zip(coords[:-1], coords[1:]):
		arrow = FancyArrowPatch(
			start, end,
			arrowstyle='-|>',
			color=color,
			linewidth=2.0,
			mutation_scale=10,
			zorder=4,
		)
		ax.add_patch(arrow)
	# Also plot crosses at each point
	xs, ys = zip(*coords)
	ax.scatter(xs, ys, marker='x', color=color, s=30, zorder=5)


def draw_map_replanned_paths(df_pax_reassigned, i=0, path_savefig=None):
	# Load NUTS shapefile (filtered to level 3 and Spain)
	# TODO: path shp better
	nuts = gpd.read_file("../data/EUROSTAT/NUTS_RG_01M_2021_4326.shp")
	nuts3_spain = nuts[(nuts["LEVL_CODE"] == 3) & (nuts["CNTR_CODE"] == "ES")]

	# Optional: list of NUTS3 codes to exclude
	nuts_to_avoid = []  # ["ES613", "ES703"]  # example codes
	nuts_to_avoid = ["ES703", "ES704", "ES705", "ES706", "ES707", "ES708", "ES709"]
	nuts3_filtered = nuts3_spain[~nuts3_spain["NUTS_ID"].isin(nuts_to_avoid)]

	# Start a plot
	fig, ax = plt.subplots(figsize=(10, 10))

	# Plot NUTS3 in grey
	nuts3_filtered.plot(ax=ax, color="lightgrey", edgecolor="white")

	# Plot origin destination NUTS3
	nut_orig = nuts3_spain[nuts3_spain["NUTS_ID"] == df_pax_reassigned.iloc[i].origin]
	nut_orig.plot(ax=ax, color="lightblue", edgecolor="white")

	nut_dest = nuts3_spain[nuts3_spain["NUTS_ID"] == df_pax_reassigned.iloc[i].destination]
	nut_dest.plot(ax=ax, color="lightgreen", edgecolor="white")

	origin_path = df_pax_reassigned.iloc[i].path_coords
	draw_path_with_arrows(ax, origin_path, color='red')

	replanned_path = df_pax_reassigned.iloc[i].path_replanned_coords
	draw_path_with_arrows(ax, replanned_path, color='blue')


	# Create dummy legend entries
	legend_elements = [
		mpatches.Patch(color='lightblue', label='Origin NUTS3:         ' + df_pax_reassigned.iloc[i].origin),
		mpatches.Patch(color='lightgreen', label='Destination NUTS3: ' + df_pax_reassigned.iloc[i].destination),
		mlines.Line2D([], [], color='red', marker='>', markersize=8, linestyle='-',
					  label='Original Path:     ' + str(df_pax_reassigned.iloc[i]['path'])),
		mlines.Line2D([], [], color='blue', marker='>', markersize=8, linestyle='-',
					  label='Replanned Path: ' + str(df_pax_reassigned.iloc[i]['path_replanned']))
	]

	# Add the legend to your ax
	ax.legend(handles=legend_elements, loc='upper right', fontsize='small', title_fontsize='small')

	# Final tweaks
	ax.set_axis_off()
	if path_savefig is not None:
		plt.savefig(path_savefig)
	else:
		plt.show()
	plt.close()



def plot_map_services(df, type_data='services', colour='red', legend_str='', path_savefig=None):

	# Load NUTS shapefile (filtered to level 3 and Spain)
	# TODO: path shp better
	nuts = gpd.read_file("../data/EUROSTAT/NUTS_RG_01M_2021_4326.shp")
	nuts3_spain = nuts[(nuts["LEVL_CODE"] == 3) & (nuts["CNTR_CODE"] == "ES")]

	# Optional: list of NUTS3 codes to exclude
	nuts_to_avoid = []  # ["ES613", "ES703"]  # example codes
	nuts_to_avoid = ["ES703", "ES704", "ES705", "ES706", "ES707", "ES708", "ES709"]
	nuts3_filtered = nuts3_spain[~nuts3_spain["NUTS_ID"].isin(nuts_to_avoid)]

	# Start a plot
	fig, ax = plt.subplots(figsize=(10, 10))

	# Plot NUTS3 in grey
	nuts3_filtered.plot(ax=ax, color="lightgrey", edgecolor="white")

	# origin_path = df_pax_reassigned.iloc[i].path_coords
	# draw_path_with_arrows(ax, origin_path, color='red')

	if type_data == 'gtfs':
		# Ensure coordinates are floats
		df['stop_lat'] = df['stop_lat'].astype(float)
		df['stop_lon'] = df['stop_lon'].astype(float)

		# Loop through each trip_id
		for trip_id, group in df.groupby('trip_id'):
			# Sort by stop_sequence to get the correct order
			group_sorted = group.sort_values('stop_sequence')

			# Build list of (lon, lat) tuples for plotting
			coords = list(zip(group_sorted['stop_lon'], group_sorted['stop_lat']))

			# Only draw if there are at least 2 coordinates
			if len(coords) > 1:
				draw_path_with_arrows(ax, coords, color=colour)

	else: # they are services
		# Ensure coordinates are floats
		df['lat_orig'] = df['lat_orig'].astype(float)
		df['lon_orig'] = df['lon_orig'].astype(float)
		df['lat_dest'] = df['lat_dest'].astype(float)
		df['lon_dest'] = df['lon_dest'].astype(float)

		# Iterate over rows and draw arrows from origin to destination
		for _, row in df.iterrows():
			coords = [(row['lon_orig'], row['lat_orig']), (row['lon_dest'], row['lat_dest'])]

			# Skip if origin and destination are identical
			if coords[0] != coords[1]:
				draw_path_with_arrows(ax, coords, color=colour)  # Use any color you like

	# Create dummy legend entries
	legend_elements = [
		mlines.Line2D([], [], color=colour, marker='>', markersize=8, linestyle='-', label=legend_str)]

	# Add the legend to your ax
	ax.legend(handles=legend_elements, loc='upper right', fontsize='small', title_fontsize='small')

	# Set longitude (x-axis) and latitude (y-axis) limits
	ax.set_xlim([-9.5, 4.5])  # min_lon, max_lon
	ax.set_ylim([34.9, 44.3])  # min_lat, max_lat

	# Final tweaks
	ax.set_axis_off()
	if path_savefig is not None:
		plt.savefig(path_savefig)
	else:
		plt.show()
	plt.close()

def group_pax_status(path_experiment_wo_ppdmpa, pp, dm, pa_min, pa_max):
	df_pax_st = None
	for pa in range(pa_min, pa_max+1):
		pa_str = f"{int(pa):02d}"
		path_experiment = (path_experiment_wo_ppdmpa /
						   ('PP'+pp+'.'+'DM'+dm+'.'+'PA'+pa_str) / 'indicators' / 'pax_status_numbers.csv')

		if os.path.exists(path_experiment):
			df_pax_st_e = pd.read_csv(path_experiment)
			df_pax_st_e.rename(columns={'number_pax': pa_str}, inplace=True)
			if df_pax_st is None:
				df_pax_st = df_pax_st_e
			else:
				df_pax_st = df_pax_st.merge(df_pax_st_e[['Unnamed: 0', pa_str]], on='Unnamed: 0')

	return df_pax_st


def plot_pax_status_grouped(df, path_savefig=None):
	df_filtered = df[df['Unnamed: 0'] != 'pax_unnafected'].set_index('Unnamed: 0')

	# df_filtered.T.plot.area(figsize=(12, 6), alpha=0.6)  # stacked area plot, semi-transparent

	# Define colors and hatches as you provided
	colors = {
		'pax_stranded': '#d73027',
		'pax_on_time': '#1a9850',
		'pax_delayed': '#fee08b',
		'rebooked': '#4575b4'
	}
	hatches = ['', '///', '+++', '**', '...', '|||', '\\\\\\']  # for rebooked categories

	# Your desired plot order (example)
	plot_order = [
		'pax_stranded',
		'pax_on_time',
		'pax_delayed',
		'pax_rebooked_same_mode',
		'pax_rebooked_air_2_rail',
		'pax_rebooked_air_2_multi',
		'pax_rebooked_rail_2_air',
		'pax_rebooked_rail_2_multi',
		'pax_rebooked_multi_2_air',
		'pax_rebooked_multi_2_rail'
	]

	# Extract x-axis values (versions)
	x = np.arange(len(df_filtered.columns))

	# Prepare stacked bottom baseline
	bottom = np.zeros(len(df_filtered.columns))

	fig, ax = plt.subplots(figsize=(12, 6))

	hatch_idx = 0

	for label in plot_order:
		if label not in df_filtered.index:
			continue
		y = df_filtered.loc[label].values.astype(float)

		# Choose color and hatch
		if label.startswith('pax_rebooked'):
			color = colors['rebooked']
			hatch = hatches[hatch_idx % len(hatches)]
			hatch_idx += 1
		else:
			color = colors.get(label, 'gray')
			hatch = None

		# Plot area
		ax.fill_between(x, bottom, bottom + y, facecolor=color, edgecolor='black', linewidth=0.5, hatch=hatch,
						label=label,
						alpha=0.6)
		bottom += y

	# Adjust x ticks and labels
	ax.set_xticks(x)
	ax.set_xticklabels(df_filtered.columns)
	ax.set_xlabel('PA')
	ax.set_ylabel('Number of Passengers')

	# Create custom legend with hatch patches
	import matplotlib.patches as mpatches

	legend_handles = []
	legend_labels = []

	hatch_idx = 0
	for label in plot_order:
		if label not in df_filtered.index:
			continue
		color = colors['rebooked'] if label.startswith('pax_rebooked') else colors.get(label, 'gray')
		if label.startswith('pax_rebooked'):
			hatch = hatches[hatch_idx % len(hatches)]
			hatch_idx += 1
		else:
			hatch = None

		patch = mpatches.Patch(facecolor=color, hatch=hatch, edgecolor='black')
		legend_handles.append(patch)
		legend_labels.append(label.replace('pax_', '').replace('_', ' '))

	# ax.legend(legend_handles, legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
	ax.legend(title='Passenger Status',
			  labels=legend_labels,
			  bbox_to_anchor=(1.05, 1),
			  loc='upper left',
			  fontsize=10,
			  title_fontsize=11,
			  handlelength=2.1,  # length of the legend handle
			  handleheight=2.1,  # height of the legend handle (in some matplotlib versions)
			  borderpad=1)

	plt.tight_layout()
	if path_savefig is not None:
		plt.savefig(path_savefig)
	else:
		plt.show()
	plt.close()