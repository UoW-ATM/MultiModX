import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
pio.kaleido.scope.mathjax = None
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
import ast
from shapely.geometry import Point
import sys
sys.path.append('..')
from strategic_evaluator.pax_reassigning_replanned_network import compute_capacities_available_services



def add_nuts(df):
	df['origin_nuts2'] = df.apply(lambda row: row['origin'][:4], axis=1)
	df['destination_nuts2'] = df.apply(lambda row: row['destination'][:4], axis=1)
	df['origin_nuts1'] = df.apply(lambda row: row['origin'][:3], axis=1)
	df['destination_nuts1'] = df.apply(lambda row: row['destination'][:3], axis=1)
	return df

def strategic_total_journey_time(data,config,pi_config,variant="sum"):
	print(' -- Strategic total journey time', pi_config['variant'])
	pax_assigned_to_itineraries_options = data['pax_assigned_to_itineraries_options']
	possible_itineraries_clustered_pareto_filtered = data['possible_itineraries_clustered_pareto_filtered']
	nuts_regional_archetype_info = data['nuts_regional_archetype_info']

	#only options that have pax assigned
	df = pax_assigned_to_itineraries_options[pax_assigned_to_itineraries_options['pax']>0].copy()

	#weigth total_time with pax
	df['weigthed_total_time'] = df['total_time']*df['pax']
	if variant == 'sum' :
		kpi = df['weigthed_total_time'].sum()
		# print('sum',kpi)
		return kpi

	if variant == 'avg':
		kpi = df['weigthed_total_time'].sum()/df['pax'].sum()
		# print('avg',kpi)
		return kpi

	if variant == 'avg_connecting_itineraries':
		df = df.dropna(subset=['nid_f2'])
		kpi = df['weigthed_total_time'].sum()/df['pax'].sum()
		# print('avg_connecting_itineraries',kpi)
		return kpi

	if variant == 'avg_by_nuts':
		#group by NUTS2/1
		#
		df['origin'] = df.apply(lambda row: row['alternative_id'].split('_')[0], axis=1)
		df['destination'] = df.apply(lambda row: row['alternative_id'].split('_')[1], axis=1)
		add_nuts(df)
		#its = df.merge(possible_itineraries_clustered_pareto_filtered,how='left',left_on=['alternative_id','option'],right_on=['alternative_id','option'])
		#print(its)
		#print(its[['origin','destination','weigthed_total_time']])
		#print(df)
		grouped = df.groupby(['origin','destination']).sum().reset_index()
		grouped['total_journey_time_per_pax'] = grouped['weigthed_total_time']/grouped['pax']
		grouped_nuts2 = df.groupby(['origin_nuts2', 'destination_nuts2']).sum().reset_index()
		grouped_nuts2['total_journey_time_per_pax'] = grouped_nuts2['weigthed_total_time']/grouped_nuts2['pax']
		#add_nuts(grouped))

		if pi_config['plot'] == True:
			add_nuts(grouped)
			plot_nuts(grouped,config,title='Avg total_journey_time_per_pax from ES51')

		if pi_config.get('plot_matrix', False):
			plot_heatmap_from_df(grouped, origin_col='origin', destination_col='destination',
								 value_col="total_journey_time_per_pax",
								 vmin=pi_config.get('vmin_matrix'),
								 vmax=pi_config.get('vmax_matrix'),
								 save_path=Path(config['output']['path_to_output_figs']) / ('avg_by_nuts_matrix_plot'+
																					   config.get('sufix_fig')+'.png'))

			plot_heatmap_from_df(grouped_nuts2, origin_col='origin_nuts2', destination_col='destination_nuts2',
								 value_col="total_journey_time_per_pax",
								 vmin=pi_config.get('vmin_matrix_nuts2'),
								 vmax=pi_config.get('vmax_matrix_nuts2'),
								 save_path=Path(config['output']['path_to_output_figs']) / ('avg_by_nuts_2_matrix_plot'+
																					   config.get('sufix_fig')+'.png'))


		return grouped[['origin','destination','total_journey_time_per_pax']]

	if variant == 'sum_per_region_archetype':

		df['origin'] = df.apply(lambda row: row['alternative_id'].split('_')[0], axis=1)
		df['destination'] = df.apply(lambda row: row['alternative_id'].split('_')[1], axis=1)
		df = df.merge(nuts_regional_archetype_info ,how='left',left_on='origin',right_on='origin')
		grouped = df[['weigthed_total_time','pax','regional_archetype']].groupby(['regional_archetype']).sum().reset_index()

		return grouped[['regional_archetype','weigthed_total_time']]

	if variant == 'avg_per_region_archetype':

		df['origin'] = df.apply(lambda row: row['alternative_id'].split('_')[0], axis=1)
		df['destination'] = df.apply(lambda row: row['alternative_id'].split('_')[1], axis=1)
		df = df.merge(nuts_regional_archetype_info ,how='left',left_on='origin',right_on='origin')
		grouped = df[['weigthed_total_time','pax','regional_archetype']].groupby(['regional_archetype']).sum().reset_index()
		grouped['total_journey_time_per_pax'] = grouped['weigthed_total_time']/grouped['pax']
		return grouped[['regional_archetype','total_journey_time_per_pax']]


def plot_nuts(its,config,title=''):
	nuts_data_path = config['input']['nuts_data_path']
	nuts_data = gpd.read_file(nuts_data_path)
	nuts_data.crs = 'EPSG:4326'
	es_nuts = nuts_data[(nuts_data['LEVL_CODE']==3) & (nuts_data['CNTR_CODE']=='ES')]
	df = its[its['origin_nuts2']=='ES51'].groupby(['destination'])['total_journey_time_per_pax'].mean().reset_index()
	df = df.rename({'destination': 'NUTS_ID'}, axis=1)
	#print(df)
	es_nuts = es_nuts.merge(df,how='left',on='NUTS_ID')
	#print(es_nuts)
	fig = px.choropleth(es_nuts, geojson=es_nuts.geometry, locations=es_nuts.index, color="total_journey_time_per_pax", range_color=[200,700])
	fig.update_geos(fitbounds="locations", visible=False)

	fig.update_layout(title_text = title)
	#fig.show()
	fig.write_html(Path(config['output']['path_to_output_figs']) / 'avg_by_nuts_plot.html', )


def plot_heatmap_from_df(
		df,
		origin_col,
		destination_col,
		value_col,
		vmin=None,
		vmax=None,
		save_path=None
):
	"""
	Plot a heatmap from a DataFrame using specified origin, destination, and value columns.

	Parameters:
		df (pd.DataFrame): Input DataFrame.
		origin_col (str): Name of the column representing origins.
		destination_col (str): Name of the column representing destinations.
		value_col (str): Name of the column representing the heatmap values.
		vmin (float, optional): Minimum value for color scale.
		vmax (float, optional): Maximum value for color scale.
		save_path (str, optional): If provided, saves the plot to this path using a tight bounding box.
	"""
	# Create pivot table
	pivot = df.pivot_table(index=origin_col, columns=destination_col, values=value_col)

	# Sort rows and columns alphabetically
	pivot = pivot.sort_index(axis=0).sort_index(axis=1)

	# Plot heatmap
	plt.figure(figsize=(15, 10))
	ax = sns.heatmap(
		pivot,
		annot=False,
		cmap="viridis",
		vmin=vmin,
		vmax=vmax,
		linewidths=0.5,
		linecolor='gray'
	)
	# Set ticks and labels to show all
	ax.set_xticks([i + 0.5 for i in range(len(pivot.columns))])
	ax.set_xticklabels(pivot.columns, rotation=90, ha='center')
	ax.set_yticks([i + 0.5 for i in range(len(pivot.index))])
	ax.set_yticklabels(pivot.index, rotation=0, va='center')

	#ax.set_title(f"Heatmap of {value_col}", fontsize=14)
	ax.set_xlabel(destination_col)
	ax.set_ylabel(origin_col)

	# Save or show
	if save_path:
		plt.savefig(save_path, bbox_inches='tight')
	else:
		plt.show()


def plot_top_od_stacked_bars(
    df,
    origin_col="origin_nuts2",
    destination_col="destination_nuts2",
    journey_type_col="journey_type",
    value_col="pax",
    top_n=10,
	od_totals=None,
    percentage=False,
    save_path=None
):
	"""
	Plot stacked bar chart of journey_type distribution for top N OD pairs by value_col.

	Parameters:
		df (pd.DataFrame): Input data
		origin_col (str): Column name for origin
		destination_col (str): Column name for destination
		journey_type_col (str): Column name for journey type
		value_col (str): Column name for values to plot ('pax' or 'percentage')
		top_n (int): Number of top OD pairs to show
		od_totals (pd.DataFrame): If provided total pax per OD to use for ordering columns
		percentage (bool): Normalize values to percentages per OD
		save_path (str or None): If provided, save the plot to this path
	"""
	# Copy and create OD label
	df = df.copy()
	df["OD"] = df[origin_col] + " → " + df[destination_col]

	# Get top OD pairs by total pax (regardless of value_col)
	if od_totals is not None:
		od_totals["OD"] = od_totals[origin_col] + " → "  + od_totals[destination_col]
		od_totals = od_totals.groupby("OD")[value_col].sum()
		# print(od_totals)

	else:
		od_totals = df.groupby("OD")[value_col].sum()

	top_od = od_totals.sort_values(ascending=False).head(top_n).index

	# Filter and pivot
	df_top = df[df["OD"].isin(top_od)]
	pivot = df_top.pivot_table(index="OD", columns=journey_type_col, values=value_col, fill_value=0)

	# Reorder by top_od directly
	pivot = pivot.reindex(index=top_od)

	if percentage:
		pivot = pivot.div(pivot.sum(axis=1), axis=0)

	# Plot
	ax = pivot.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="viridis")

	ylabel = "Share of Passengers" if percentage else "Number of Passengers"
	plt.ylabel(ylabel)
	plt.title(f"{'Percentage' if percentage else 'Total'} Pax by Journey Type for Top {top_n} OD Pairs")
	plt.xticks(rotation=45, ha="right")
	plt.tight_layout()

	if save_path:
		plt.savefig(save_path, bbox_inches='tight')
	else:
		plt.show()


def plot_stops_and_nuts_heatmap(
		nuts3,
		grouped_nuts,
		grouped_stops,
		airport_lat,
		airport_lon,
		label='from',
		exclude_nuts=None,
		airport_name='Airport',
		vmin=None,
		vmax=None,
		fig_map_name=None,
		fig_heat_map_name=None,
		topleft=None, bottomright=None,
		scenario="",
		show_plot=False
):
	"""
	Plot NUTS3 heatmap + rail stops + airport location.

	Parameters:
	- nuts3: GeoDataFrame with NUTS3 regions (must include 'NUTS_ID', 'geometry', 'CNTR_CODE')
	- grouped_nuts: DataFrame with ['nuts3_<label>', 'total_pax']
	- grouped_stops: DataFrame with ['stop_lat_<label>', 'stop_lon_<label>', ...]
	- airport_lat, airport_lon: float, coordinates of the airport
	- label: str, 'from' or 'to'
	- exclude_nuts: list of NUTS3 codes to exclude from the plot
	"""

	# Filter to Spain
	nuts3_spain = nuts3[nuts3['CNTR_CODE'] == 'ES'].copy()

	# Exclude selected NUTS3 if any
	if exclude_nuts:
		nuts3_spain = nuts3_spain[~nuts3_spain['NUTS_ID'].isin(exclude_nuts)]

	# Merge total_pax into NUTS
	nuts_col = f'nuts3_{label}'
	nuts3_merged = nuts3_spain.merge(grouped_nuts, left_on='NUTS_ID', right_on=nuts_col, how='left')

	# Set missing values to zero for plotting and flag them separately for grey fill
	nuts3_merged["has_data"] = nuts3_merged["total_pax"].notna()
	nuts3_merged["total_pax"] = nuts3_merged["total_pax"].fillna(0)

	if grouped_stops is not None:
		# Build GeoDataFrame of stops
		gdf_stops = gpd.GeoDataFrame(
			grouped_stops,
			geometry=gpd.points_from_xy(grouped_stops[f'stop_lon_{label}'], grouped_stops[f'stop_lat_{label}']),
			crs="EPSG:4326"
		)

	# Min and Max values for color bar
	if vmin is None:
		vmin = nuts3_merged["total_pax"].min()
	if vmax is None:
		vmax = nuts3_merged["total_pax"].max()

	# Plot
	fig, ax = plt.subplots(figsize=(10, 10))

	# First plot NUTS without data (in light grey)
	nuts3_merged[~nuts3_merged["has_data"]].plot(
		ax=ax, color="#eeeeee", edgecolor="grey", linewidth=0.5
	)

	# Then plot NUTS with data using a heatmap
	nuts3_merged[nuts3_merged["has_data"]].plot(
		ax=ax,
		column="total_pax",
		cmap="OrRd",
		edgecolor="grey",
		linewidth=0.5,
		legend=False,  # To not plot the heatmap bar
		# legend_kwds={"label": "Total Pax"},
		vmin=vmin,
		vmax=vmax
	)

	if grouped_stops is not None:
		# Plot rail stops
		gdf_stops.plot(ax=ax, color="black", markersize=30, alpha=0.8, label=f'Rail Station ({label}) {scenario}')

	if airport_lon is not None:
		# Plot airport
		ax.scatter(airport_lon, airport_lat, color="red", marker="x", s=100, label=airport_name)

	# Set zoom if coordinates provided
	if topleft and bottomright:
		# top-left: (lat1, lon1), bottom-right: (lat2, lon2)
		lat1, lon1 = topleft
		lat2, lon2 = bottomright
		ax.set_xlim(lon1, lon2)
		ax.set_ylim(lat2, lat1)

	# ax.set_title(f"Passenger Heatmap by NUTS3 ({label})", fontsize=14)
	ax.set_axis_off()
	if (airport_lon is not None) or (grouped_stops is not None):
		ax.legend(fontsize=14)
	plt.tight_layout()
	if fig_map_name is not None:
		plt.savefig(fig_map_name, bbox_inches='tight')
	if show_plot:
		plt.show()

	plt.close()

	# Plot 2: Horizontal Heatmap Bar
	# Plot the heatmap bar
	# Create the plot
	fig, ax = plt.subplots(figsize=(20, 3))

	# Plot the heatmap bar only, but do not plot the regions
	# We are just creating the colorbar here.
	sm = plt.cm.ScalarMappable(cmap="OrRd", norm=plt.Normalize(vmin=vmin, vmax=vmax))
	sm.set_array([])  # Empty array, just for colorbar

	# Add the colorbar (heatmap legend)
	cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", shrink=0.8, pad=0.1)
	# cbar.set_label("Total Pax", fontsize=18)
	cbar.set_ticks(MaxNLocator(integer=True, prune='lower'))
	cbar.ax.tick_params(labelsize=18)

	# Set title and layout adjustments
	# ax.set_title(f"Heatmap Bar for {label} (Pax Distribution)", fontsize=12)
	ax.set_axis_off()  # Hide the axes

	# Display the plot
	plt.tight_layout()
	if fig_heat_map_name is not None:
		plt.savefig(fig_heat_map_name, bbox_inches='tight')
	if show_plot:
		plt.show()

	plt.close()


def diversity_of_destinations(data,config,pi_config,variant='nuts'):
	print(' -- Diversity of destinations', pi_config['variant'])
	possible_itineraries_clustered_pareto_filtered = data['possible_itineraries_clustered_pareto_filtered']
	results = {}
	if variant == 'nuts':
		df = possible_itineraries_clustered_pareto_filtered.groupby(['origin']).nunique().reset_index()[['origin','destination']]
		if pi_config.get('plot', False):
			fig = px.bar(df, x='origin', y='destination')
			#fig.show()
			fig.write_html(Path(config['output']['path_to_output_figs']) / 'diversity_of_destinations_nuts.html', )

		return df

	if variant == 'hubs':
		its = possible_itineraries_clustered_pareto_filtered.copy()
		df = pd.DataFrame()
		leg0 = its[['origin_0','service_id_0','destination_0','mode_0']].rename({'origin_0': 'hub', 'service_id_0': 'leg_out','destination_0':'destination_out','mode_0':'mode_out'}, axis=1)
		df = pd.concat([df,leg0])
		max_legs = its['nservices'].max()
		for i in range(max_legs-1):
			leg1 = its[['destination_'+str(i),'service_id_'+str(i),'service_id_'+str(i+1),'origin_'+str(i),'destination_'+str(i+1),'mode_'+str(i+1)]].rename({'destination_'+str(i): 'hub', 'service_id_'+str(i): 'leg_in','service_id_'+str(i+1): 'leg_out','origin_'+str(i):'origin_in','destination_'+str(i+1):'destination_out','mode_'+str(i+1):'mode_out'}, axis=1)
			leg2 = its[['origin_'+str(i+1),'service_id_'+str(i),'service_id_'+str(i+1),'origin_'+str(i),'destination_'+str(i+1),'mode_'+str(i+1)]].rename({'origin_'+str(i+1): 'hub', 'service_id_'+str(i): 'leg_in','service_id_'+str(i+1): 'leg_out','origin_'+str(i):'origin_in','destination_'+str(i+1):'destination_out','mode_'+str(i+1):'mode_out'}, axis=1)
			df = pd.concat([df,leg1,leg2])

		leg7 = its[['destination_'+str(max_legs-1),'service_id_'+str(max_legs-1),'origin_'+str(max_legs-1)]].rename({'destination_'+str(max_legs-1): 'hub', 'service_id_'+str(max_legs-1): 'leg_in','origin_'+str(max_legs-1):'origin_in'}, axis=1)
		df = pd.concat([df,leg7])
		df = df.dropna(how='all')
		df['es_airport'] = df["hub"].apply(lambda x: True if str(x)[:2] in ['LE','GC'] else False)
		df = df[df['es_airport']==True]
		df_out = df.drop_duplicates(subset=['hub','leg_out','destination_out','mode_out'])[['hub','leg_out','destination_out','mode_out']]
		#con_out = df_out.groupby(["hub",'mode_out']).count().reset_index().sort_values(by='leg_out')
		con_out2 = df_out.drop_duplicates(subset=['hub','destination_out','mode_out'])[['hub','destination_out','mode_out']].groupby(['hub',
																																	  'mode_out']).count().reset_index().sort_values(by='destination_out')
		#print(con_out2)
		if pi_config.get('plot', False):
			fig = px.bar(con_out2, x='hub', y='destination_out',color='mode_out', barmode='relative')
			fig.update_layout(xaxis={'categoryorder':'total ascending'})
			#fig.show()
			fig.write_html(Path(config['output']['path_to_output_figs']) / 'diversity_of_destinations_hubs.html', )
		return con_out2

def modal_share(data,config,pi_config,variant='total'):
	print(' -- Modal share', pi_config['variant'])
	pax_assigned_to_itineraries_options = data['pax_assigned_to_itineraries_options']
	possible_itineraries_clustered_pareto_filtered = data['possible_itineraries_clustered_pareto_filtered']
	nuts_regional_archetype_info = data['nuts_regional_archetype_info']

	df = pd.concat([pax_assigned_to_itineraries_options,possible_itineraries_clustered_pareto_filtered[['journey_type']]],axis=1)
	df = df[df['pax']>0][['pax','journey_type','origin']]
	grouped = df.groupby(['journey_type']).sum().reset_index()
	grouped['percentage'] = grouped['pax']/grouped['pax'].sum()
	# print('grouped',grouped[['journey_type','pax','percentage']])
	if variant == 'total':
		return grouped[['journey_type','pax','percentage']]
	if variant == 'by_nuts':
		#grouping by nuts
		total = df.groupby(['origin'])['pax'].sum().reset_index()
		grouped = df.groupby(['origin','journey_type']).sum().reset_index()
		grouped['percentage'] = grouped.apply(lambda row: row['pax']/total[total['origin']==row['origin']]['pax'].iloc[0], axis=1)
		return  grouped[['origin','journey_type','pax','percentage']]
	if variant == 'between_nuts':
		#grouping by nuts but pair or nuts
		df = pd.concat(
			[pax_assigned_to_itineraries_options, possible_itineraries_clustered_pareto_filtered[['journey_type']]],
			axis=1)
		df = df[df['pax'] > 0][['pax', 'journey_type', 'origin', 'destination']]
		grouped = df.groupby(['origin', 'destination', 'journey_type']).sum().reset_index()
		grouped['percentage'] = grouped.apply(
			lambda row: row['pax'] / grouped[((grouped['origin'] == row['origin']) &
											  (grouped['destination'] == row['destination']))]['pax'].sum(), axis=1)
		grouped = grouped[['origin', 'destination', 'journey_type', 'pax', 'percentage']]

		if pi_config.get('plot', False):
			demand = data['demand'].copy()
			demand.rename(columns={'trips': 'pax'}, inplace=True)
			demand_agg_nuts = demand.groupby(['origin', 'destination'])['pax'].sum().reset_index()
			demand_agg_nuts.to_csv('./demand_nuts.csv')


			plot_top_od_stacked_bars(
				grouped,
				origin_col="origin",
				destination_col="destination",
				journey_type_col="journey_type",
				value_col="pax",
				top_n=pi_config.get('plot_top', 10),
				percentage=False,
				save_path=Path(config['output']['path_to_output_figs']) / ('mode_share_between_nuts_pax'+
																					   config.get('sufix_fig')+'.png'))
				

			plot_top_od_stacked_bars(
				grouped,
				origin_col="origin",
				destination_col="destination",
				journey_type_col="journey_type",
				value_col="pax",
				top_n=pi_config.get('plot_top', 10),
				od_totals=demand_agg_nuts,
				percentage=False,
				save_path=Path(config['output']['path_to_output_figs']) / ('mode_share_between_nuts_pax_per_demand'+
																					   config.get('sufix_fig')+'.png'))

			plot_top_od_stacked_bars(
				grouped,
				origin_col="origin",
				destination_col="destination",
				journey_type_col="journey_type",
				value_col="pax",
				top_n=pi_config.get('plot_top', 10),
				percentage=True,
				save_path=Path(config['output']['path_to_output_figs']) / ('mode_share_between_nuts_share'+
																					   config.get('sufix_fig')+'.png'))

			plot_top_od_stacked_bars(
				grouped,
				origin_col="origin",
				destination_col="destination",
				journey_type_col="journey_type",
				value_col="pax",
				top_n=pi_config.get('plot_top', 10),
				od_totals=demand_agg_nuts,
				percentage=True,
				save_path=Path(config['output']['path_to_output_figs']) / ('mode_share_between_nuts_share_per_demand'+
																					   config.get('sufix_fig')+'.png'))

		return grouped

	if variant == 'between_nuts_level2':
		#grouping by nuts but pair or nuts at level2
		df = pd.concat(
			[pax_assigned_to_itineraries_options, possible_itineraries_clustered_pareto_filtered[['journey_type']]],
			axis=1)
		df = df[df['pax'] > 0][['pax', 'journey_type', 'origin', 'destination']]
		add_nuts(df)
		grouped_nuts2 = df.groupby(['origin_nuts2', 'destination_nuts2', 'journey_type']).sum().reset_index()
		grouped_nuts2['percentage'] = grouped_nuts2.apply(
			lambda row: row['pax'] / grouped_nuts2[((grouped_nuts2['origin_nuts2'] == row['origin_nuts2']) &
											  (grouped_nuts2['destination_nuts2'] == row['destination_nuts2']))]['pax'].sum(), axis=1)
		grouped_nuts2 = grouped_nuts2[['origin_nuts2', 'destination_nuts2', 'journey_type', 'pax', 'percentage']]
		if pi_config.get('plot', False):
			# doing plot
			demand = data['demand'].copy()
			add_nuts(demand)
			demand.rename(columns={'trips': 'pax'}, inplace=True)
			demand_agg_nuts = demand.groupby(['origin_nuts2', 'destination_nuts2'])['pax'].sum().reset_index()

			plot_top_od_stacked_bars(
				grouped_nuts2,
				origin_col="origin_nuts2",
				destination_col="destination_nuts2",
				journey_type_col="journey_type",
				value_col="pax",
				top_n=pi_config.get('plot_top', 10),
				percentage=False,
				save_path=Path(config['output']['path_to_output_figs']) / ('mode_share_between_nuts2_pax'+
																					   config.get('sufix_fig')+'.png'))

			plot_top_od_stacked_bars(
				grouped_nuts2,
				origin_col="origin_nuts2",
				destination_col="destination_nuts2",
				journey_type_col="journey_type",
				value_col="pax",
				top_n=pi_config.get('plot_top', 10),
				od_totals=demand_agg_nuts,
				percentage=False,
				save_path=Path(config['output']['path_to_output_figs']) / ('mode_share_between_nuts2_pax_per_demand'+
																					   config.get('sufix_fig')+'.png'))

			plot_top_od_stacked_bars(
				grouped_nuts2,
				origin_col="origin_nuts2",
				destination_col="destination_nuts2",
				journey_type_col="journey_type",
				value_col="pax",
				top_n=pi_config.get('plot_top', 10),
				percentage=True,
				save_path=Path(config['output']['path_to_output_figs']) / ('mode_share_between_nuts2_share'+
																					   config.get('sufix_fig')+'.png'))

			plot_top_od_stacked_bars(
				grouped_nuts2,
				origin_col="origin_nuts2",
				destination_col="destination_nuts2",
				journey_type_col="journey_type",
				value_col="pax",
				top_n=pi_config.get('plot_top', 10),
				od_totals=demand_agg_nuts,
				percentage=True,
				save_path=Path(config['output']['path_to_output_figs']) / ('mode_share_between_nuts2_share_per_demand'+
																					   config.get('sufix_fig')+'.png'))

		return grouped_nuts2

	if variant == 'by_regional_archetype':
		#grouping by nuts
		df = df.merge(nuts_regional_archetype_info ,how='left',left_on='origin',right_on='origin')
		total = df.groupby(['regional_archetype'])['pax'].sum().reset_index()
		# print(total)
		grouped = df.groupby(['regional_archetype','journey_type']).sum().reset_index()
		grouped['percentage'] = grouped.apply(lambda row: row['pax']/total[total['regional_archetype']==row['regional_archetype']]['pax'].iloc[0], axis=1)
		# print(grouped[['regional_archetype','journey_type','pax','percentage']])
		return  grouped[['regional_archetype','journey_type','pax','percentage']]


def pax_time_efficiency(data,config,pi_config,variant='total'):
	print(' -- Pax time efficiency', pi_config['variant'])
	pax_assigned_to_itineraries_options = data['pax_assigned_to_itineraries_options']

	pax_assigned_to_itineraries_options['origin'] = pax_assigned_to_itineraries_options.apply(lambda row: row['alternative_id'].split('_')[0], axis=1)
	pax_assigned_to_itineraries_options['destination'] = pax_assigned_to_itineraries_options.apply(lambda row: row['alternative_id'].split('_')[1], axis=1)

	best = pax_assigned_to_itineraries_options.groupby(['origin','destination'])['total_time'].min()

	df = pax_assigned_to_itineraries_options[pax_assigned_to_itineraries_options['pax']>0].copy()
	df['efficiency'] = df.apply(lambda row: best.loc[row['origin'],row['destination']]/row['total_time'], axis=1)
	df['weigthed_efficiency'] = df['efficiency']*df['pax']
	if variant == 'total':
		return df['weigthed_efficiency'].sum()/df['pax'].sum()


def demand_served(data,config,pi_config,variant='total'):
	print(' -- Demand served', pi_config['variant'])
	pax_assigned_to_itineraries_options = data['pax_assigned_to_itineraries_options']
	demand = data['demand']
	nuts_regional_archetype_info = data['nuts_regional_archetype_info']

	pax_assigned_to_itineraries_options['origin'] = pax_assigned_to_itineraries_options.apply(lambda row: row['alternative_id'].split('_')[0], axis=1)
	pax_assigned_to_itineraries_options['destination'] = pax_assigned_to_itineraries_options.apply(lambda row: row['alternative_id'].split('_')[1], axis=1)

	total = pax_assigned_to_itineraries_options['pax'].sum()/demand['trips'].sum()
	if variant == 'total':
		return total

	if variant == 'total_connecting_itineraries':
		df = pax_assigned_to_itineraries_options.dropna(subset=['nid_f2']).copy()
		total = df['pax'].sum()/demand['trips'].sum()
		return total

	#grouping by nuts
	demand['trips'] = np.ceil(demand['trips'])
	demand_nuts = demand.groupby(['origin','destination'])['trips'].sum().reset_index()

	df = pax_assigned_to_itineraries_options[pax_assigned_to_itineraries_options['pax']>0]
	df_orig = df.copy()
	grouped = df.groupby(['origin','destination'])['pax'].sum().reset_index()
	grouped = grouped.merge(demand_nuts,how='right',on=['origin','destination'])
	grouped['pax'] = grouped['pax'].fillna(0)
	grouped['perc'] = np.minimum(grouped['pax'] / grouped['trips'], 1) # Could be >1 due to logit model assigning .x pax and ceil

	if variant == 'by_nuts':
		add_nuts(grouped)
		grouped_n2 = grouped.groupby(['origin_nuts2','destination_nuts2'])[['pax', 'trips']].sum().reset_index()
		grouped_n2['perc'] = np.minimum(grouped_n2['pax'] / grouped_n2['trips'], 1)
		if pi_config.get('plot', False):
			# plots
			plot_heatmap_from_df(grouped, origin_col='origin', destination_col='destination',
								 value_col="perc",
								 vmin=pi_config.get('vmin_matrix'),
								 vmax=pi_config.get('vmax_matrix'),
								 save_path=Path(config['output']['path_to_output_figs']) / ('demand_served_nuts3_' +
																					   config.get(
																						   'sufix_fig') + '.png'))

			plot_heatmap_from_df(grouped_n2, origin_col='origin_nuts2', destination_col='destination_nuts2',
								 value_col="perc",
								 vmin=pi_config.get('vmin_matrix'),
								 vmax=pi_config.get('vmax_matrix'),
								 save_path=Path(config['output']['path_to_output_figs']) / ('demand_served_nuts2_' +
																					   config.get(
																						   'sufix_fig') + '.png'))

		return {'_n3': grouped[['origin','destination','pax','trips','perc']],
				'_n2': grouped_n2[['origin_nuts2', 'destination_nuts2', 'pax', 'trips', 'perc']],}
	if variant == 'by_regional_archetype':
		grouped = grouped.merge(nuts_regional_archetype_info ,how='left',left_on='origin',right_on='origin')
		grouped.rename(columns={'regional_archetype': 'origin_regional_archetype'}, inplace=True)
		grouped = grouped.merge(nuts_regional_archetype_info, how='left', left_on='destination', right_on='origin')
		grouped.rename(columns={'regional_archetype': 'destination_regional_archetype'}, inplace=True)
		grouped2 = grouped.groupby(['origin_regional_archetype', 'destination_regional_archetype'])[['pax', 'trips']].sum().reset_index()
		grouped2['perc'] = np.minimum(grouped2['pax'] / grouped2['trips'], 1)
		if pi_config.get('plot', False):
			# plots
			plot_heatmap_from_df(grouped2, origin_col='origin_regional_archetype', destination_col='destination_regional_archetype',
								 value_col="perc",
								 vmin=pi_config.get('vmin_matrix'),
								 vmax=pi_config.get('vmax_matrix'),
								 save_path=Path(config['output']['path_to_output_figs']) / ('demand_served_regional_arch_' +
																					   config.get(
																						   'sufix_fig') + '.png'))

		return grouped2[['origin_regional_archetype', 'destination_regional_archetype', 'pax', 'trips', 'perc']]

	if variant == 'by_od':
		df = df_orig.copy()
		grouped_w_mode = df.groupby(['origin', 'destination', 'type'])['pax'].sum().reset_index()
		grouped_w_mode = grouped_w_mode.merge(demand_nuts, how='right', on=['origin', 'destination'])
		grouped_w_mode['pax'] = grouped_w_mode['pax'].fillna(0)

		def classify_mode(t):
			if pd.isna(t):
				return 'none'
			parts = t.split('_')
			unique_modes = set(parts)
			if unique_modes == {'rail'}:
				return 'rail'
			elif unique_modes == {'flight'}:
				return 'flight'
			else:
				return 'multimodal'

		grouped_w_mode['mode'] = grouped_w_mode['type'].apply(classify_mode)

		od_totals_gwm = grouped_w_mode.groupby(['origin', 'destination'])[['pax']].sum().rename(
			columns={'pax': 'total_pax'})
		grouped_w_mode = grouped_w_mode.merge(od_totals_gwm, on=['origin', 'destination'])
		grouped_w_mode['perc_demand_accross_modes'] = grouped_w_mode['total_pax'] / grouped_w_mode['trips']

		grouped_w_mode['od_pair'] = grouped_w_mode['origin'] + ' → ' + grouped_w_mode['destination']

		# Compute unused pax per OD
		grouped_w_mode['unserved_pax_total'] = grouped_w_mode['trips'] - grouped_w_mode['total_pax']

		id_od = []
		for od in pi_config['od']:
			i_od_demand = grouped_w_mode[(grouped_w_mode.origin==od[0]) & (grouped_w_mode.destination==od[1])].index
			if len(i_od_demand)>0:
				id_od.extend(i_od_demand)

		df = grouped_w_mode.loc[id_od].copy()
		if len(df)>0:
			# To ensure only one bar per OD, create a plotting dataframe
			plot_df = df.pivot_table(index='od_pair', columns='mode', values='pax', aggfunc='sum').fillna(0)
			# Add the "unused" part to reach total pax
			plot_df['unserved'] = grouped_w_mode.groupby('od_pair')['unserved_pax_total'].first()

			# Add label values
			labels = df.groupby('od_pair')['perc_demand_accross_modes'].first().apply(lambda x: f"{x * 100:.2f}%")

			# Plot setup

			# Plot
			fig, ax = plt.subplots(figsize=(10, 6))
			ax = plot_df.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab20c')

			# Y-axis in thousands
			ax.set_ylabel('Capacity Served (in \'000s of seats)')
			#ax.set_yticklabels([f'{int(y / 1000)}K' for y in ax.get_yticks()])
			ax.set_xlabel('Origin → Destination')
			plt.xticks(rotation=45, ha='right')

			# Add percentage labels on top
			for idx, (bar_label, total_height) in enumerate(zip(labels, plot_df.sum(axis=1))):
				ax.text(idx, total_height + 20,  # a bit above bar
						f"{total_height / 1000:.1f}K\n({bar_label})",
						ha='center', va='bottom', fontsize=9)

			plt.tight_layout()

			fe = pi_config.get('figure_ending','')

			plt.savefig(Path(config['output']['path_to_output_figs']) / ('demand_served_between_od' + fe +
																		 config.get(
																			 'sufix_fig') + '.png'))

		return grouped_w_mode


def compute_load_factor(df_pax_per_service, dict_seats_service):
	# Divide dataframe between flights and rail services
	df_pax_per_service_flight = df_pax_per_service[df_pax_per_service.type=='flight'].copy()
	df_pax_per_service_rail = df_pax_per_service[df_pax_per_service.type=='rail'].copy()
	df_final = []

	if len(df_pax_per_service_flight) > 0:
		# For flights straight forward computation of load factor
		df_pax_per_service_flight['max_pax_in_service'] = df_pax_per_service_flight['pax']
		df_pax_per_service_flight['max_seats_service'] = df_pax_per_service_flight['service_id'].apply(
			lambda x: dict_seats_service[x])
		df_pax_per_service_flight['load_factor'] = df_pax_per_service_flight['max_pax_in_service'] / df_pax_per_service_flight['max_seats_service']
		df_final = [df_pax_per_service_flight]

	if len(df_pax_per_service_rail) > 0:
		# For rail a bit more complex
		# First get capacity of rail services without the stops (only service_id)
		df_pax_per_service_rail['rail_service_id'] = df_pax_per_service_rail['service_id'].apply(lambda x: x.split('_')[0])
		df_pax_per_service_rail['stop_orig'] = df_pax_per_service_rail['service_id'].apply(lambda x: int(x.split('_')[1]))
		df_pax_per_service_rail['stop_dest'] = df_pax_per_service_rail['service_id'].apply(lambda x: int(x.split('_')[2]))

		def get_min_capacity_services(services_id):
			# Minimum capacity of all services
			return min([dict_seats_service[x] for x in services_id])

		dict_capacities_rail = df_pax_per_service_rail.groupby('rail_service_id')['service_id'].apply(get_min_capacity_services).to_dict()

		df_pax_per_service_rail['max_seats_service'] = df_pax_per_service_rail['rail_service_id'].apply(lambda x: dict_capacities_rail[x])

		# For each service compute the number of pax per stop
		def compute_pax_per_consecutive_segment(stops_pax):
			stops = set(stops_pax['stop_orig'])
			stops.update(stops_pax['stop_dest'])
			stops = sorted(list(stops))
			rail_service_id = stops_pax['rail_service_id'].iloc[0]

			pax_between_segments = []
			for i in range(len(stops)-1):
				pax_in_between_stops = int(stops_pax[~((stops_pax['stop_dest']<=stops[i])|(stops_pax['stop_orig']>=stops[i+1]))]['pax'].sum())
				pax_between_segments += [{'rail_service': rail_service_id,
										  'total_pax_in_service': pax_in_between_stops,
										  'stop_orig': stops[i],
										  'stop_destination': stops[i + 1]}]

			return pax_between_segments

		# Compute how many pax are per consecutive segment for each rail service
		capacity_services_consecutive_segments = df_pax_per_service_rail.groupby('rail_service_id')[['rail_service_id',
																									 'stop_orig',
																									 'stop_dest',
																									 'pax']].apply(compute_pax_per_consecutive_segment)
		capacity_services_consecutive_segments = capacity_services_consecutive_segments.reset_index()

		capacity_services_consecutive_segments = capacity_services_consecutive_segments.explode(0, ignore_index=True)
		capacity_services_consecutive_segments = pd.json_normalize(capacity_services_consecutive_segments[0])

		# Now for each rail service (railservice_stoporig_stopdest) compute load factor and seats used/available
		def get_max_pax_in_segment(service_id, cap_serv_cnsive_segments):
			rail_service =service_id.split('_')[0]
			stop_orig = int(service_id.split('_')[1])
			stop_dest = int(service_id.split('_')[2])
			x = cap_serv_cnsive_segments[((cap_serv_cnsive_segments['rail_service'] == rail_service) &
										  ~((cap_serv_cnsive_segments['stop_destination'] <= stop_orig) |
											(cap_serv_cnsive_segments['stop_orig'] >= stop_dest)))]
			return max(x['total_pax_in_service'])


		df_pax_per_service_rail['max_pax_in_service'] = df_pax_per_service_rail['service_id'].apply(lambda x: get_max_pax_in_segment(x, capacity_services_consecutive_segments))

		df_pax_per_service_rail['load_factor'] = df_pax_per_service_rail['max_pax_in_service'] / \
												   df_pax_per_service_rail['max_seats_service']

		df_pax_per_service_rail.drop(columns={'rail_service_id', 'stop_orig', 'stop_dest'}, inplace=True)

		df_final += [df_pax_per_service_rail]

	df_final = pd.concat(df_final).reset_index()

	return df_final


def compute_load_factor_paxkm(df_pax_per_service, dict_seats_service,rail_timetable_proc_used_internally,flight_schedules_proc):

	def sum_segments(segment,dist):
		segments = dist[(dist['rail_service_id']==segment['rail_service_id']) & (dist['stop_orig']>=segment['stop_orig']) & (dist['stop_dest']<=segment['stop_dest'])]
		return segments['gcdistance'].sum()
	# Divide dataframe between flights and rail services
	df_pax_per_service_flight = df_pax_per_service[df_pax_per_service.type=='flight'].copy()
	df_pax_per_service_rail = df_pax_per_service[df_pax_per_service.type=='rail'].copy()
	df_final = []

	#For flights straight forward computation of load factor
	paxkm = df_pax_per_service_flight.groupby('service_id')['pax'].sum().reset_index()

	flight_schedules_proc = flight_schedules_proc.merge(paxkm[['service_id','pax']],how='left',on='service_id')
	flight_schedules_proc['load_factor'] = flight_schedules_proc['pax'] / flight_schedules_proc['seats']
	flight_schedules_proc['mode']='flight'
	df_final.append(flight_schedules_proc)
	#if len(df_pax_per_service_flight) > 0:
		##
		#df_pax_per_service_flight['max_pax_in_service'] = df_pax_per_service_flight['pax']
		#df_pax_per_service_flight['max_seats_service'] = df_pax_per_service_flight['service_id'].apply(
			#lambda x: dict_seats_service[x])
		#df_pax_per_service_flight['load_factor'] = df_pax_per_service_flight['max_pax_in_service'] / df_pax_per_service_flight['max_seats_service']
		#df_final = [df_pax_per_service_flight]

	# For rail a bit more complex
	if len(df_pax_per_service_rail) > 0:
		#df_pax_per_service_rail.to_csv('xxx.csv')
		# add some columns
		df_pax_per_service_rail['rail_service_id'] = df_pax_per_service_rail['service_id'].apply(lambda x: x.split('_')[0])
		df_pax_per_service_rail['stop_orig'] = df_pax_per_service_rail['service_id'].apply(lambda x: int(x.split('_')[1]))
		df_pax_per_service_rail['stop_dest'] = df_pax_per_service_rail['service_id'].apply(lambda x: int(x.split('_')[2]))
		#add some columns to rail_timetable_proc_used_internally
		rail_timetable_proc_used_internally['rail_service_id'] = rail_timetable_proc_used_internally['service_id'].apply(lambda x: x.split('_')[0])
		rail_timetable_proc_used_internally['stop_orig'] = rail_timetable_proc_used_internally['service_id'].apply(lambda x: int(x.split('_')[1]))
		rail_timetable_proc_used_internally['stop_dest'] = rail_timetable_proc_used_internally['service_id'].apply(lambda x: int(x.split('_')[2]))

		#build a df of successive segments for each rail_service_id, e.g. 1-2, 2-3, 3-4
		rail_timetable_proc_used_internally['cumulative_count'] = rail_timetable_proc_used_internally.groupby(['rail_service_id','stop_orig']).cumcount()
		dist = rail_timetable_proc_used_internally[rail_timetable_proc_used_internally['cumulative_count']==0]

		#dist = rail_timetable_proc_used_internally[(rail_timetable_proc_used_internally['stop_dest']-rail_timetable_proc_used_internally['stop_orig'])==1]
		#dist['cumulative_dist'] = dist.groupby(dist['rail_service_id'])['gcdistance'].cumsum()
		#print('dist', dist[dist['rail_service_id']=='1185'])

		#distance of a service is a sum of its successive segments, e.g. 1-4 = 1-2 + 2-3 + 3-4
		rail_timetable_proc_used_internally['dist'] = rail_timetable_proc_used_internally.apply(lambda row:sum_segments(row,dist),axis=1)
		#find out the total dist of a rail_service_id and seats
		capacity = rail_timetable_proc_used_internally.groupby('rail_service_id')[['dist','seats']].max().reset_index().rename(columns={'dist':'maxdist','seats':'max_seats'})
		capacity['max_paxkm'] = capacity['max_seats']*capacity['maxdist']
		#calculate paxkm = pax*dist
		df_pax_per_service_rail = df_pax_per_service_rail.merge(rail_timetable_proc_used_internally[['service_id','dist','seats']],how='left',on='service_id')
		df_pax_per_service_rail['paxkm'] = df_pax_per_service_rail['dist']*df_pax_per_service_rail['pax']

		#print('df_pax_per_service_rail',df_pax_per_service_rail)
		#print('capacity',capacity)
		#sum paxkm for each rail_service_id
		paxkm = df_pax_per_service_rail.groupby('rail_service_id')['paxkm'].sum().reset_index()

		#print('paxkm',paxkm)
		loads = capacity.merge(paxkm,how='left',on='rail_service_id')
		loads['load_factor'] = loads['paxkm']/loads['max_paxkm']
		loads = loads.fillna(value={'load_factor':0,'paxkm':0})
		loads['mode'] = 'rail'
		#loads.to_csv('loads.csv')
		#print('loads',loads)
		df_final.append(loads)
	df = pd.concat(df_final)
	#print(df)
	return df


def load_factor(data,config,pi_config,variant='total'):
	print(' -- Load factor', pi_config['variant'])

	pax_assigned_to_itineraries_options = data['pax_assigned_to_itineraries_options']
	possible_itineraries_clustered_pareto_filtered = data['possible_itineraries_clustered_pareto_filtered']
	pax_assigned_seats_max_target = data['pax_assigned_seats_max_target']
	dict_seats_service = pax_assigned_seats_max_target[['nid','max_seats']].set_index(['nid'])['max_seats'].to_dict()
	rail_timetable_proc_used_internally = data['rail_timetable_proc_used_internally']
	flight_schedules_proc = data['flight_schedules_proc']

	df = pd.concat([pax_assigned_to_itineraries_options,possible_itineraries_clustered_pareto_filtered],axis=1)
	df_pax = df[df['pax']>0].copy()

	# Get list of pax per service regardless if used in nid_f1, nid_f2...
	# Identify nid_f columns dynamically (as there could be from nid_f1 to nid_fn
	nid_cols = [col for col in df_pax.columns if col.startswith('service_id_')]

	# Split the 'type' column into multiple type_n columns
	df_type_split = df_pax['type'].str.split('_', expand=True)
	df_type_split.columns = [f'type_{i + 1}' for i in range(df_type_split.shape[1])]
	pax_kept = df_pax.join(df_type_split)

	# Concatenate nid_f columns with their respective type_n
	df_melted = pax_kept.melt(id_vars=['pax', 'type'], value_vars=nid_cols,
							  var_name='nid_col', value_name='service_id')


	# Keep only rows where service_id is not NaN
	df_melted = df_melted.dropna(subset=['service_id']).reset_index(drop=True)

	# Extract the index of the nid_f column (e.g., nid_f1 -> 1)
	df_melted['nid_index'] = df_melted['nid_col'].str.extract(r'(\d+)').astype(int)

	# Assign the corresponding type_n based on nid_index
	df_melted['type'] = df_melted.apply(lambda x: x['type'].split('_')[x['nid_index']], axis=1)

	# Select final columns
	df_pax_per_service = df_melted[['service_id', 'type', 'pax']]

	# Add all pax in each service
	#df_pax_per_service = df_pax_per_service.groupby(['service_id', 'type'])['pax'].sum().reset_index()

	# Compute load factor per service
	loads = compute_load_factor_paxkm(df_pax_per_service, dict_seats_service,rail_timetable_proc_used_internally,flight_schedules_proc)
	#loads.to_csv('loads.csv')

	#flatten itineraries
	#services = pd.concat([df[['service_id_0','pax','mode_0']],df[['service_id_1','pax','mode_1']].rename({'service_id_1': 'service_id_0','mode_1':'mode_0'}, axis=1)],axis=0)
	#services =  services.dropna(subset=['service_id_0'])
	#services['service_id'] = services.apply(lambda row: row['service_id_0'].split('_')[0] if row['mode_0']=='rail' else row['service_id_0'], axis=1)
	#services = services.groupby(['service_id','mode_0'])['pax'].sum().reset_index()
	#services = services.merge(pax_assigned_seats_max_target,how='left',left_on='service_id',right_on='nid')
	#services['load_factor'] = services['pax']/services['max_seats']
	#print(services)
	#print('total',services['load_factor'].mean())
	if variant == 'total':

		return loads['load_factor'].mean()

	modes = loads.groupby(['mode'])['load_factor'].mean().reset_index()
	if variant == 'modes':
		return modes


def resilience_alternatives(data,config,pi_config,variant='by_nuts'):
	print(' -- Resilience alternatives', pi_config['variant'])

	possible_itineraries_clustered_pareto_filtered = data['possible_itineraries_clustered_pareto_filtered'].copy()
	nuts_regional_archetype_info = data['nuts_regional_archetype_info'].copy()

	df = possible_itineraries_clustered_pareto_filtered.groupby(['origin','destination', 'journey_type'])['option'].count().reset_index()

	if variant == 'by_nuts':
		dftotal = possible_itineraries_clustered_pareto_filtered.groupby(['origin', 'destination'])[
			'option'].count().reset_index()
		add_nuts(possible_itineraries_clustered_pareto_filtered)
		dfn2 = \
		possible_itineraries_clustered_pareto_filtered.groupby(['origin_nuts2', 'destination_nuts2', 'journey_type'])[
			'option'].count().reset_index()
		dftotaln2 = \
			possible_itineraries_clustered_pareto_filtered.groupby(['origin_nuts2', 'destination_nuts2'])[
				'option'].count().reset_index()

		if pi_config.get('plot', False):
			# plots
			plot_heatmap_from_df(dftotal, origin_col='origin', destination_col='destination',
								 value_col="option",
								 vmin=pi_config.get('vmin_matrix'),
								 vmax=pi_config.get('vmax_matrix'),
								 save_path=Path(config['output']['path_to_output_figs']) / ('resilience_options_nuts'+
																					   config.get('sufix_fig')+'.png'))

			plot_heatmap_from_df(dftotaln2, origin_col='origin_nuts2', destination_col='destination_nuts2',
								 value_col="option",
								 vmin=pi_config.get('vmin_matrix_nuts2'),
								 vmax=pi_config.get('vmax_matrix_nuts2'),
								 save_path=Path(config['output']['path_to_output_figs']) /
										   ('resilience_options_nuts2'+config.get('sufix_fig')+'.png'))

			for mode in df['journey_type'].drop_duplicates():
				vmin = pi_config.get('vmin_matrix_' + mode)
				vmax = pi_config.get('vmax_matrix_' + mode)
				plot_heatmap_from_df(df[df.journey_type==mode],
									 origin_col='origin',
									 destination_col='destination',
									 value_col="option",
									 vmin=vmin,
									 vmax=vmax,
									 save_path=Path(config['output']['path_to_output_figs']) /
											   ('resilience_options_nuts_' + mode + config.get('sufix_fig')+'.png'))

				vmin = pi_config.get('vmin_matrix_nuts2_' + mode)
				vmax = pi_config.get('vmax_matrix_nuts2_' + mode)
				plot_heatmap_from_df(dfn2[dfn2.journey_type == mode],
									 origin_col='origin_nuts2',
									 destination_col='destination_nuts2',
									 vmax=vmax,
									 vmin=vmin,
									 value_col="option",
									 save_path=Path(config['output']['path_to_output_figs']) / (
												 'resilience_options_nuts2_' + mode + config.get('sufix_fig')+'.png'))

		return {'_w_jt': df, '_total': dftotal,
				'_w_jt_n2': dfn2, '_total_n2': dftotaln2}



	if variant == 'by_regional_archetype':
		nuts_regional_archetype_info.rename(columns={'origin':'region'}, inplace=True)

		df = df.merge(nuts_regional_archetype_info ,how='left',left_on='origin',right_on='region')
		df.rename(columns={'regional_archetype':'regional_archetype_origin'}, inplace=True)
		df.drop('region', axis=1, inplace=True)
		df = df.merge(nuts_regional_archetype_info, how='left', left_on='destination', right_on='region')
		df.rename(columns={'regional_archetype': 'regional_archetype_destination'}, inplace=True)
		df.drop('region', axis=1, inplace=True)
		grouped_all = df.groupby(['regional_archetype_origin', 'regional_archetype_destination'])['option'].sum().reset_index()
		grouped_type = df.groupby(['regional_archetype_origin', 'regional_archetype_destination', 'journey_type'])[
			'option'].sum().reset_index()

		if pi_config.get('plot', False):
			# plots
			plot_heatmap_from_df(grouped_all, origin_col='regional_archetype_origin', destination_col='regional_archetype_destination',
								 value_col="option",
								 vmin=pi_config.get('vmin_matrix'),
								 vmax=pi_config.get('vmax_matrix'),
								 save_path=Path(config['output']['path_to_output_figs']) / ('resilience_options_reg_arch'+
																					   config.get('sufix_fig')+'.png'))

			for mode in grouped_type['journey_type'].drop_duplicates():
				vmin = pi_config.get('vmin_matrix_'+mode)
				vmax = pi_config.get('vmax_matrix_'+mode)
				plot_heatmap_from_df(grouped_type[grouped_type.journey_type==mode],
									 origin_col='regional_archetype_origin',
									 destination_col='regional_archetype_destination',
									 vmin=vmin,
									 vmax=vmax,
									 value_col="option",
									 save_path=Path(config['output']['path_to_output_figs']) / ('resilience_options_reg_arch_'+mode+
																					   config.get('sufix_fig')+'.png'))

		return  {'_total': grouped_all, '_w_jt': grouped_type}


def buffer_in_itineraries(data,config,pi_config,variant='sum'):
	print(' -- Buffer in itineraries', pi_config['variant'])
	pax_assigned_to_itineraries_options = data['pax_assigned_to_itineraries_options']
	possible_itineraries_clustered_pareto_filtered = data['possible_itineraries_clustered_pareto_filtered']
	df = pd.concat([pax_assigned_to_itineraries_options,possible_itineraries_clustered_pareto_filtered[['nservices','nmodes']]],axis=1)

	#only options that have pax assigned and buffer is existing (buffer is 0 for one leg itineraries)
	df = df[(df['pax']>0) & (df['nservices']>1)].copy()

	#weigth total_waiting_time with pax
	df['weigthed_total_waiting_time'] = df['total_waiting_time']*df['pax']
	kpi = df['weigthed_total_waiting_time'].sum()
	# print(kpi)
	if pi_config.get('plot', False):
		fig, ax = plt.subplots(figsize=(10, 10))
		plt.hist(df[df['nmodes']>1]['total_waiting_time'], weights=df[df['nmodes']>1]['pax'],bins=[0,5,10,15,20,25,30,35,40])  # arguments are passed to np.histogram
		plt.title("Histogram of buffers (multimodal pax)")
		plt.xlabel("Buffer (min.)")
		plt.ylabel("Number of passengers")
		plt.savefig(Path(config['output']['path_to_output_figs']) / 'buffers_hist.png')
		plt.close()

		fig, ax = plt.subplots(figsize=(10, 10))
		plt.hist(df[df['type']=='flight_rail']['total_waiting_time'], weights=df[df['type']=='flight_rail']['pax'],bins=[0,1,2,3,4,5,10,15,20,25,30,35,40])  # arguments are passed to np.histogram
		plt.title("Histogram of buffers (flight rail pax)")
		plt.savefig(Path(config['output']['path_to_output_figs']) / 'buffers_hist_fr.png')
		plt.close()

		fig, ax = plt.subplots(figsize=(10, 10))
		plt.hist(df[df['type']=='rail_flight']['total_waiting_time'], weights=df[df['type']=='rail_flight']['pax'],bins=[0,5,10,15,20,25,30,35,40])  # arguments are passed to np.histogram
		plt.title("Histogram of buffers (rail flight pax)")
		plt.savefig(Path(config['output']['path_to_output_figs']) / 'buffers_hist_rf.png')
		plt.close()

		fig, ax = plt.subplots(figsize=(10, 10))
		plt.hist(df[df['type']=='flight_flight']['total_waiting_time'], weights=df[df['type']=='flight_flight']['pax'],bins=[0,5,10,15,20,25,30,35,40])  # arguments are passed to np.histogram
		plt.title("Histogram of buffers (flight flight pax)")
		plt.savefig(Path(config['output']['path_to_output_figs']) / 'buffers_hist_ff.png')
		plt.close()


	if variant == 'sum':
		return kpi
	if variant == 'avg':
		return df['weigthed_total_waiting_time'].sum()/df['pax'].sum()


def from_to_stops(df, airport, df_stops, nuts3):
    # Filter rows that contain the airport in the path
    df_filtered = df[df['path'].apply(lambda p: airport in p)].copy()

    def find_stops(path, airport):
        if airport not in path:
            return None, None

        idx = path.index(airport)

        # Search backwards for contiguous digit-only block
        from_stop = None
        for i in range(idx - 1, -1, -1):
            if path[i].isdigit():
                from_stop = path[i]
            else:
                break

        # Search forward for contiguous digit-only block
        to_stop = None
        for i in range(idx + 1, len(path)):
            if path[i].isdigit():
                to_stop = path[i]
            else:
                break

        return from_stop, to_stop

    # Extract from/to stops
    df_filtered[["from_stop", "to_stop"]] = df_filtered["path"].apply(
        lambda p: pd.Series(find_stops(p, airport))
    )

    # Merge for from_stop
    df_merged = df_filtered.merge(
        df_stops[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']].add_suffix('_from'),
        how='left',
        left_on='from_stop',
        right_on='stop_id_from'
    )

    # Merge for to_stop
    df_merged = df_merged.merge(
        df_stops[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']].add_suffix('_to'),
        how='left',
        left_on='to_stop',
        right_on='stop_id_to'
    )

    ### --- FROM BLOCK ---
    df_from = df_merged[df_merged["from_stop"].notna()].copy()
    df_from["geometry"] = df_from.apply(
        lambda row: Point(row["stop_lon_from"], row["stop_lat_from"]), axis=1
    )
    gdf_from = gpd.GeoDataFrame(df_from, geometry="geometry", crs="EPSG:4326")
    gdf_from = gpd.sjoin(gdf_from, nuts3[["NUTS_ID", "geometry"]], how="left", predicate="within")
    gdf_from = gdf_from.rename(columns={"NUTS_ID": "nuts3_from"}).drop(columns=["index_right", "geometry"])

    grouped_from = (
        gdf_from.groupby(["stop_id_from", "stop_name_from", "stop_lat_from", "stop_lon_from"], as_index=False)
        .agg(total_pax=("pax", "sum"))
    )

    grouped_nuts_from = (
        gdf_from.groupby("nuts3_from", as_index=False)
        .agg(total_pax=("pax", "sum"))
    )

    ### --- TO BLOCK ---
    df_to = df_merged[df_merged["to_stop"].notna()].copy()
    df_to["geometry"] = df_to.apply(
        lambda row: Point(row["stop_lon_to"], row["stop_lat_to"]), axis=1
    )
    gdf_to = gpd.GeoDataFrame(df_to, geometry="geometry", crs="EPSG:4326")
    gdf_to = gpd.sjoin(gdf_to, nuts3[["NUTS_ID", "geometry"]], how="left", predicate="within")
    gdf_to = gdf_to.rename(columns={"NUTS_ID": "nuts3_to"}).drop(columns=["index_right", "geometry"])

    grouped_to = (
        gdf_to.groupby(["stop_id_to", "stop_name_to", "stop_lat_to", "stop_lon_to"], as_index=False)
        .agg(total_pax=("pax", "sum"))
    )

    grouped_nuts_to = (
        gdf_to.groupby("nuts3_to", as_index=False)
        .agg(total_pax=("pax", "sum"))
    )

    return gdf_from, grouped_from, grouped_nuts_from, gdf_to, grouped_to, grouped_nuts_to


def catchment_area(data,config,pi_config,variant='hubs'):
	print(' -- Catchment area', pi_config['variant'])

	pax_assigned_to_itineraries_options = data['pax_assigned_to_itineraries_options']
	pax_assigned_to_itineraries_options = pax_assigned_to_itineraries_options[pax_assigned_to_itineraries_options.pax>0].copy()

	if type(pax_assigned_to_itineraries_options['path'].iloc[0]) is str:
		pax_assigned_to_itineraries_options['path'] = pax_assigned_to_itineraries_options['path'].apply(ast.literal_eval)

	possible_itineraries_clustered_pareto_filtered = data['possible_itineraries_clustered_pareto_filtered']

	its = pd.concat([pax_assigned_to_itineraries_options,possible_itineraries_clustered_pareto_filtered],axis=1)
	its = its[its['pax']>0]


	df = pd.DataFrame()
	leg0 = its[['origin_0','service_id_0','destination_0','mode_0','travel_time_0']].rename({'origin_0': 'hub', 'service_id_0': 'leg_out','destination_0':'destination_out','mode_0':'mode_out','travel_time_0':'travel_time_out'}, axis=1)
	df = pd.concat([df,leg0])
	max_legs = its['nservices'].max()
	for i in range(max_legs-1):
		leg1 = its[['destination_'+str(i),'service_id_'+str(i),'service_id_'+str(i+1),'origin_'+str(i),'destination_'+str(i+1),'mode_'+str(i),'mode_'+str(i+1),'travel_time_'+str(i),'travel_time_'+str(i+1)]].rename({'destination_'+str(i): 'hub', 'service_id_'+str(i): 'leg_in','service_id_'+str(i+1): 'leg_out','origin_'+str(i):'origin_in','destination_'+str(i+1):'destination_out','mode_'+str(i):'mode_in','mode_'+str(i+1):'mode_out','travel_time_'+str(i): 'travel_time_in','travel_time_'+str(i+1): 'travel_time_out'}, axis=1)
		leg2 = its[['origin_'+str(i+1),'service_id_'+str(i),'service_id_'+str(i+1),'origin_'+str(i),'destination_'+str(i+1),'mode_'+str(i),'mode_'+str(i+1),'travel_time_'+str(i),'travel_time_'+str(i+1)]].rename({'origin_'+str(i+1): 'hub', 'service_id_'+str(i): 'leg_in','service_id_'+str(i+1): 'leg_out','origin_'+str(i):'origin_in','destination_'+str(i+1):'destination_out','mode_'+str(i):'mode_in','mode_'+str(i+1):'mode_out','travel_time_'+str(i): 'travel_time_in','travel_time_'+str(i+1): 'travel_time_out'}, axis=1)
		df = pd.concat([df,leg1,leg2])

	leg7 = its[['destination_'+str(max_legs-1),'service_id_'+str(max_legs-1),'origin_'+str(max_legs-1),'travel_time_'+str(max_legs-1),'mode_'+str(max_legs-1)]].rename({'destination_'+str(max_legs-1): 'hub', 'service_id_'+str(max_legs-1): 'leg_in','origin_'+str(max_legs-1):'origin_in','travel_time_'+str(max_legs-1):'travel_time_in','mode_'+str(max_legs-1):'mode_in'}, axis=1)
	df = pd.concat([df,leg7])
	df = df.dropna(how='all')
	df['es_airport'] = df["hub"].apply(lambda x: True if str(x)[:2] in ['LE','GC'] else False)
	df = df[df['es_airport']==True]

	df_out = pd.concat([df[['hub','mode_out','travel_time_out']].rename({'mode_out':'mode','travel_time_out':'travel_time'},axis=1),df[['hub','mode_in','travel_time_in']].rename({'mode_in':'mode','travel_time_in':'travel_time'},axis=1)])
	df_out = df_out[df_out['mode']=='rail']
	#df_out = df.drop_duplicates(subset=['hub','leg_out','destination_out','mode_out'])[['hub','leg_out','destination_out','mode_out']]
	con_out = df_out.groupby(["hub"])['travel_time'].max().reset_index()
	#con_out2 = df_out.drop_duplicates(subset=['hub','destination_out','mode_out'])[["hub",'destination_out','mode_out']].groupby(["hub",'mode_out']).count().reset_index().sort_values(by='destination_out')
	if pi_config.get('plot', False):
		fig = px.bar(con_out, x='hub', y='travel_time')
		fig.update_layout(xaxis={'categoryorder':'total ascending'})
		#fig.show()
		fig.write_html(Path(config['output']['path_to_output_figs']) / 'catchment_area_hubs.html', )

	if variant == 'hubs_rail_time':
		return con_out

	nuts_data_path = config['input']['nuts_data_path']
	nuts_data = gpd.read_file(nuts_data_path)
	# Filter to only NUTS level 3
	nuts3 = nuts_data[nuts_data["LEVL_CODE"] == 3]

	if variant in ['access_egress', 'rail_stop_pax', 'access_egress_rail_stop']:
		dict_output = {}

		exploded_path = pax_assigned_to_itineraries_options['path'].explode()

		# Filter: keep only strings of exactly 4 letters (alphabetic)
		airports = exploded_path[exploded_path.str.fullmatch(r'[A-Za-z]{4}')]

		# Get distinct entries
		airports = airports.unique()

		if variant in ['access_egress', 'access_egress_rail_stop']:
			lae = []
			# Compute access_egress
			for airport in airports:
				# Filter itineraries where the airport is the first or last
				pax_assigned_to_itineraries_options['first_node'] = pax_assigned_to_itineraries_options['path'].apply(lambda x: x[0])
				pax_assigned_to_itineraries_options['last_node'] = pax_assigned_to_itineraries_options['path'].apply(
					lambda x: x[-1])

				first_airport = pax_assigned_to_itineraries_options[pax_assigned_to_itineraries_options.first_node==airport]
				last_airport = pax_assigned_to_itineraries_options[pax_assigned_to_itineraries_options.last_node==airport]
				first_airport_nuts = first_airport[['origin', 'pax']].groupby(['origin'])['pax'].sum().reset_index()
				first_airport_nuts['type'] = 'from'
				first_airport_nuts.rename(columns={'origin': 'nuts3'}, inplace=True)
				last_airport_nuts = last_airport[['destination', 'pax']].groupby(['destination'])['pax'].sum().reset_index()
				last_airport_nuts['type'] = 'to'
				last_airport_nuts.rename(columns={'destination': 'nuts3'}, inplace=True)
				access_egress_catchment = pd.concat([first_airport_nuts, last_airport_nuts])
				access_egress_catchment['airport'] = airport

				lae += [access_egress_catchment]

			access_egress_catchment = pd.concat(lae)

			if variant == 'access_egress':
				dict_output['_access_egress_airports'] = access_egress_catchment
				if pi_config.get('plot', False):
					# Plot of access egress
					for a in pi_config['plot_airports']:
						aec = access_egress_catchment[access_egress_catchment.airport==a['airport']].copy()
						if len(aec) > 0:
							# We have data for the airport
							aec_from = aec[aec.type=='from'].copy()
							aec_to = aec[aec.type == 'to'].copy()
							aec_from.rename(columns={'pax': 'total_pax', 'nuts3': 'nuts3_from'}, inplace=True)
							aec_to.rename(columns={'pax': 'total_pax', 'nuts3': 'nuts3_to'}, inplace=True)

							topleft = None
							bottomright = None
							if a.get('topleft') is not None:
								topleft = (a['topleft'][0], a['topleft'][1])
							if a.get('bottomright') is not None:
								bottomright = (a['bottomright'][0], a['bottomright'][1])

							ac = data['airport_coords'][data['airport_coords']['icao_id']==a['airport']].iloc[0]

							plot_stops_and_nuts_heatmap(nuts3=nuts3,
														grouped_nuts=aec_from,
														grouped_stops=None,
														airport_lat=ac.lat,
														airport_lon=ac.lon,
														label='from',
														exclude_nuts=a.get('exclude_nuts', pi_config.get('exclude_nuts')),
														airport_name=a['airport'],
														vmin=a.get('vmin', pi_config.get('vmin')),
														vmax=a.get('vmax', pi_config.get('vmax')),
														fig_map_name=Path(config['output']['path_to_output_figs']) / (a['airport']+'_access'+
																					   config.get('sufix_fig')+'.png'),
														fig_heat_map_name=Path(config['output']['path_to_output_figs']) / (a['airport']+'_access_hm'+
																					   config.get('sufix_fig')+'.png'),
														topleft=topleft,
														bottomright=bottomright
														)

							plot_stops_and_nuts_heatmap(nuts3=nuts3,
														grouped_nuts=aec_to,
														grouped_stops=None,
														airport_lat=ac.lat,
														airport_lon=ac.lon,
														label='to',
														exclude_nuts=a.get('exclude_nuts',
																		   pi_config.get('exclude_nuts')),
														airport_name=a['airport'],
														vmin=a.get('vmin', pi_config.get('vmin')),
														vmax=a.get('vmax', pi_config.get('vmax')),
														fig_map_name=Path(config['output']['path_to_output_figs']) / (
																	a['airport'] + '_egress'+
																					   config.get('sufix_fig')+'.png'),
														fig_heat_map_name=Path(config['output']['path_to_output_figs']) / (
																	a['airport'] + '_egress_hm'+
																					   config.get('sufix_fig')+'.png'),
														topleft=topleft,
														bottomright=bottomright
														)
				return dict_output


		if variant in ['access_egress_rail_stop', 'rail_stop_pax']:
			lgdf_from = []
			lgdf_to = []
			lgrouped_from_to = []
			lnuts_from_to = []

			for airport in airports:
				gdf_from, grouped_from, nuts_from, gdf_to, grouped_to, nuts_to = from_to_stops(pax_assigned_to_itineraries_options,
																							   airport,
																							   data['df_stops_rail'],
																							   nuts3)

				grouped_from['type'] = 'from'
				grouped_from.rename(columns={'stop_id_from': 'stop_id', 'stop_name_from': 'stop_name',
											'stop_lat_from': 'stop_lat', 'stop_lon_from': 'stop_lon'}, inplace=True)
				grouped_to['type'] = 'to'
				grouped_to.rename(columns={'stop_id_to': 'stop_id', 'stop_name_to': 'stop_name',
											 'stop_lat_to': 'stop_lat', 'stop_lon_to': 'stop_lon'}, inplace=True)
				grouped_from_to = pd.concat([grouped_from, grouped_to])

				nuts_from['type'] = 'from'
				nuts_from.rename(columns={'nuts3_from': 'nuts3'}, inplace=True)
				nuts_to['type'] = 'to'
				nuts_to.rename(columns={'nuts3_to': 'nuts3'}, inplace=True)
				nuts_from_to = pd.concat([nuts_from, nuts_to])

				gdf_from['airport'] = airport
				gdf_to['airport'] = airport
				grouped_from_to['airport'] = airport
				nuts_from_to['airport'] = airport

				lgdf_from += [gdf_from]
				lgdf_to += [gdf_to]
				lnuts_from_to += [nuts_from_to]
				lgrouped_from_to += [grouped_from_to]

			gdf_from = pd.concat(lgdf_from)
			gdf_to = pd.concat(lgdf_to)
			nuts_from_to = pd.concat(lnuts_from_to)
			grouped_from_to = pd.concat(lgrouped_from_to)

			if variant == 'rail_stop_pax':
				dict_output['_gdf_from'] = gdf_from
				dict_output['_gdf_to'] = gdf_to
				dict_output['_stops_to_from'] = grouped_from_to
				dict_output['_nuts_from_to'] = nuts_from_to
				if pi_config.get('plot', False):
					# Plot of catchment due to rail multimodal
					for a in pi_config['plot_airports']:
						nft = nuts_from_to[nuts_from_to.airport==a['airport']].copy()
						rstops = grouped_from_to[grouped_from_to.airport==a['airport']].copy()
						if len(nft) > 0:
							# We have data for the airport
							nft_from = nft[nft.type=='from'].copy()
							nft_to = nft[nft.type == 'to'].copy()
							rail_stops_from = rstops[rstops.type=='from'].copy()
							rail_stops_to = rstops[rstops.type == 'to'].copy()

							nft_from.rename(columns={'pax_total': 'total_pax', 'nuts3': 'nuts3_from'}, inplace=True)
							nft_to.rename(columns={'pax_total': 'total_pax', 'nuts3': 'nuts3_to'}, inplace=True)
							rail_stops_from.rename(columns={'stop_lon': 'stop_lon_from', 'stop_lat': 'stop_lat_from'}, inplace=True)
							rail_stops_to.rename(columns={'stop_lon': 'stop_lon_to', 'stop_lat': 'stop_lat_to'},
												   inplace=True)

							topleft = None
							bottomright = None
							if a.get('topleft') is not None:
								topleft = (a['topleft'][0], a['topleft'][1])
							if a.get('bottomright') is not None:
								bottomright = (a['bottomright'][0], a['bottomright'][1])

							ac = data['airport_coords'][data['airport_coords']['icao_id']==a['airport']].iloc[0]

							plot_stops_and_nuts_heatmap(nuts3=nuts3,
														grouped_nuts=nft_from,
														grouped_stops=rail_stops_from,
														airport_lat=ac.lat,
														airport_lon=ac.lon,
														label='from',
														exclude_nuts=a.get('exclude_nuts', pi_config.get('exclude_nuts')),
														airport_name=a['airport'],
														vmin=a.get('vmin', pi_config.get('vmin')),
														vmax=a.get('vmax', pi_config.get('vmax')),
														fig_map_name=Path(config['output']['path_to_output_figs']) / (a['airport']+'_access_multim_rail'+
																					   config.get('sufix_fig')+'.png'),
														fig_heat_map_name=Path(config['output']['path_to_output_figs']) / (a['airport']+'_access_multim_rail_hm'+
																					   config.get('sufix_fig')+'.png'),
														topleft=topleft,
														bottomright=bottomright
														)

							plot_stops_and_nuts_heatmap(nuts3=nuts3,
														grouped_nuts=nft_to,
														grouped_stops=rail_stops_to,
														airport_lat=ac.lat,
														airport_lon=ac.lon,
														label='to',
														exclude_nuts=a.get('exclude_nuts',
																		   pi_config.get('exclude_nuts')),
														airport_name=a['airport'],
														vmin=a.get('vmin', pi_config.get('vmin')),
														vmax=a.get('vmax', pi_config.get('vmax')),
														fig_map_name=Path(config['output']['path_to_output_figs']) / (
																	a['airport'] + '_egress_multim_rail' +
																	config.get('sufix_fig') + '.png'),
														fig_heat_map_name=Path(config['output']['path_to_output_figs']) / (
																	a['airport'] + '_egress_,multim_rail_hm' +
																	config.get('sufix_fig') + '.png'),
														topleft=topleft,
														bottomright=bottomright
														)


			if variant == 'access_egress_rail_stop':
				# We have access_egress_catchment and nuts_from_to
				access_egress_w_rail_catchment = access_egress_catchment.merge(nuts_from_to, how='outer', on=['nuts3', 'type', 'airport'])
				access_egress_w_rail_catchment['pax'] = access_egress_w_rail_catchment['pax'].fillna(0)
				access_egress_w_rail_catchment['total_pax'] = access_egress_w_rail_catchment['total_pax'].fillna(0)
				access_egress_w_rail_catchment['pax_total'] = (access_egress_w_rail_catchment['pax'] +
																		   access_egress_w_rail_catchment['total_pax'])
				access_egress_w_rail_catchment.rename(columns={'pax':'pax_access_egress', 'total_pax':'pax_rail_multimodal'}, inplace=True)
				dict_output['_access_egress_w_rail'] = access_egress_w_rail_catchment

				if pi_config.get('plot', False):
					# Plot of access egress including rail
					for a in pi_config['plot_airports']:
						aec = access_egress_w_rail_catchment[access_egress_w_rail_catchment.airport==a['airport']].copy()
						rstops = grouped_from_to[grouped_from_to.airport==a['airport']].copy()
						if len(aec) > 0:
							# We have data for the airport
							aec_from = aec[aec.type=='from'].copy()
							aec_to = aec[aec.type == 'to'].copy()
							rail_stops_from = rstops[rstops.type=='from'].copy()
							rail_stops_to = rstops[rstops.type == 'to'].copy()

							aec_from.rename(columns={'pax_total': 'total_pax', 'nuts3': 'nuts3_from'}, inplace=True)
							aec_to.rename(columns={'pax_total': 'total_pax', 'nuts3': 'nuts3_to'}, inplace=True)
							rail_stops_from.rename(columns={'stop_lon': 'stop_lon_from', 'stop_lat': 'stop_lat_from'}, inplace=True)
							rail_stops_to.rename(columns={'stop_lon': 'stop_lon_to', 'stop_lat': 'stop_lat_to'},
												   inplace=True)

							topleft = None
							bottomright = None
							if a.get('topleft') is not None:
								topleft = (a['topleft'][0], a['topleft'][1])
							if a.get('bottomright') is not None:
								bottomright = (a['bottomright'][0], a['bottomright'][1])

							ac = data['airport_coords'][data['airport_coords']['icao_id']==a['airport']].iloc[0]

							plot_stops_and_nuts_heatmap(nuts3=nuts3,
														grouped_nuts=aec_from,
														grouped_stops=rail_stops_from,
														airport_lat=ac.lat,
														airport_lon=ac.lon,
														label='from',
														exclude_nuts=a.get('exclude_nuts', pi_config.get('exclude_nuts')),
														airport_name=a['airport'],
														vmin=a.get('vmin', pi_config.get('vmin')),
														vmax=a.get('vmax', pi_config.get('vmax')),
														fig_map_name=Path(config['output']['path_to_output_figs']) / (a['airport']+'_access_w_rail'+
																					   config.get('sufix_fig')+'.png'),
														fig_heat_map_name=Path(config['output']['path_to_output_figs']) / (a['airport']+'_access_w_rail_hm'+
																					   config.get('sufix_fig')+'.png'),
														topleft=topleft,
														bottomright=bottomright
														)

							plot_stops_and_nuts_heatmap(nuts3=nuts3,
														grouped_nuts=aec_to,
														grouped_stops=rail_stops_to,
														airport_lat=ac.lat,
														airport_lon=ac.lon,
														label='to',
														exclude_nuts=a.get('exclude_nuts',
																		   pi_config.get('exclude_nuts')),
														airport_name=a['airport'],
														vmin=a.get('vmin', pi_config.get('vmin')),
														vmax=a.get('vmax', pi_config.get('vmax')),
														fig_map_name=Path(config['output']['path_to_output_figs']) / (
																	a['airport'] + '_egress_w_rail' +
																	config.get('sufix_fig') + '.png'),
														fig_heat_map_name=Path(config['output']['path_to_output_figs']) / (
																	a['airport'] + '_egress_w_rail_hm' +
																	config.get('sufix_fig') + '.png'),
														topleft=topleft,
														bottomright=bottomright
														)


		return dict_output


def cost_per_user(data,config,pi_config,variant='avg'):
	print(' -- Cost per user', pi_config['variant'])

	pax_assigned_to_itineraries_options = data['pax_assigned_to_itineraries_options']
	#only options that have pax assigned
	df = pax_assigned_to_itineraries_options[pax_assigned_to_itineraries_options['pax']>0].copy()

	#weigth total_time with pax
	df['weigthed_cost'] = df['fare']*df['pax']
	kpi = df['weigthed_cost'].sum()/df['pax'].sum()
	# print(kpi)
	if variant == 'avg':
		return kpi

def co2_emissions(data,config,pi_config,variant='avg'):
	print(' -- CO2 emissions', pi_config['variant'])

	pax_assigned_to_itineraries_options = data['pax_assigned_to_itineraries_options']
	possible_itineraries_clustered_pareto_filtered = data['possible_itineraries_clustered_pareto_filtered']
	#only options that have pax assigned
	its = pd.concat([pax_assigned_to_itineraries_options,possible_itineraries_clustered_pareto_filtered],axis=1)
	its = its[its['pax']>0]

	#weigth co2 with pax
	its['weigthed_co2'] = its['total_emissions']*its['pax']
	kpi = its['weigthed_co2'].sum()/its['pax'].sum()
	# print(kpi)
	if variant == 'avg':
		return kpi

def seamless_of_travel(data,config,pi_config,variant='avg'):
	print(' -- Seamless of travel', pi_config['variant'])

	pax_assigned_to_itineraries_options = data['pax_assigned_to_itineraries_options']
	possible_itineraries_clustered_pareto_filtered = data['possible_itineraries_clustered_pareto_filtered']

	df = pd.concat([pax_assigned_to_itineraries_options,possible_itineraries_clustered_pareto_filtered],axis=1)
	df = df[df['pax']>0]

	#sum connecting times (connecting time is difference between times of successive services)
	connecting_times = df.filter(regex="connecting_time_")
	df['total_connecting_time'] = connecting_times.sum(axis=1,min_count=1)
	df = df.dropna(subset=['total_connecting_time'])
	#print('connecting_times',df)
	df['weigthed_tct'] = df['total_connecting_time']*df['pax']

	if variant == 'avg':
		kpi = df['weigthed_tct'].sum()/df['pax'].sum()
		return kpi


def pax_processes_time(data,config,pi_config,variant='avg'):
	print(' -- Pax process time', pi_config['variant'])

	pax_assigned_to_itineraries_options = data['pax_assigned_to_itineraries_options']
	possible_itineraries_clustered_pareto_filtered = data['possible_itineraries_clustered_pareto_filtered']

	df = pd.concat([pax_assigned_to_itineraries_options,possible_itineraries_clustered_pareto_filtered],axis=1)
	df = df.loc[:,~df.columns.duplicated()].copy()
	df = df[df['pax']>0]

	#pax_processes_time =
	ground_mobility_times = df.filter(regex="ground_mobility_time_")
	travel_times = df.filter(regex="travel_time_")

	df['total_ground_mobility_time'] = ground_mobility_times.sum(axis=1,min_count=0)
	df['sum_travel'] = travel_times.sum(axis=1,min_count=0)
	#df = df.dropna(subset=['total_ground_mobility_time'])

	#print(df)

	df['pax_processes_time'] = df['total_travel_time'] - df['d2i_time'] - df['i2d_time'] - df['total_waiting_time'] - df['total_ground_mobility_time'] - df['sum_travel']
	#print('connecting_times',df)
	#df.to_csv('xxx.csv')
	df['weigthed_ppt'] = df['pax_processes_time']*df['pax']

	if variant == 'avg':
		kpi = df['weigthed_ppt'].sum()/df['pax'].sum()
		return kpi

def resilience_replanned(data,config,pi_config,variant='total'):

	rows = []
	flight_schedules_proc0 = data[0]['flight_schedules_proc']
	flight_schedules_proc1 = data[1]['flight_schedules_proc']
	rail_timetable_proc_used_internally0 = data[0]['rail_timetable_proc_used_internally']#.drop_duplicates(subset=['trip_id'],keep='first')
	rail_timetable_proc_used_internally1 = data[1]['rail_timetable_proc_used_internally']#.drop_duplicates(subset=['trip_id'],keep='first')
	rail_timetable_proc_used_internally0['rail_service_id'] = rail_timetable_proc_used_internally0['service_id'].apply(lambda x: x.split('_')[0])
	rail_timetable_proc_used_internally1['rail_service_id'] = rail_timetable_proc_used_internally1['service_id'].apply(lambda x: x.split('_')[0])
	#rail_timetable_proc_used_internally0 = rail_timetable_proc_used_internally0.drop_duplicates(subset=['rail_service_id'],keep='first')
	#rail_timetable_proc_used_internally1 = rail_timetable_proc_used_internally1.drop_duplicates(subset=['rail_service_id'],keep='first')

	# print(rail_timetable_proc_used_internally0.dtypes)

	flights = flight_schedules_proc0.merge(flight_schedules_proc1,how='outer',on=['service_id'],indicator='dataframe')
	new_flights = flights[flights['dataframe']=='right_only']
	cancelled_flights = flights[flights['dataframe']=='left_only']
	same_flights = flights[flights['dataframe']=='both'].copy()
	same_flights['difference'] = abs((same_flights['sobt_x'] - same_flights['sobt_y']).dt.total_seconds()/60)
	rescheduled_flights = same_flights[same_flights['difference']>0]

	rail = rail_timetable_proc_used_internally0.merge(rail_timetable_proc_used_internally1,how='outer',on=['service_id'],indicator='dataframe')
	new_rail = rail[rail['dataframe']=='right_only']
	cancelled_rail = rail[rail['dataframe']=='left_only']
	same_rail = rail[rail['dataframe']=='both'].copy()
	same_rail['difference'] = abs((same_rail['departure_time_utc_x'] - same_rail['departure_time_utc_y']).dt.total_seconds()/60)
	rescheduled_rail = same_rail[same_rail['difference']>0].drop_duplicates(subset=['rail_service_id_x','origin_x'],keep='first')

	#print(flights,new_flights,cancelled_flights)
	#print(same_flights.dtypes)
	# print(new_rail)
	rescheduled_rail.to_csv('rescheduled_rail.csv')
	rows.append({'field':'new_flights','value':len(new_flights)})
	rows.append({'field':'cancelled_flights','value':len(cancelled_flights)})
	rows.append({'field':'rescheduled_flights','value':len(rescheduled_flights)})
	rows.append({'field':'flights_timetable_diff_sum','value':rescheduled_flights['difference'].sum()})
	rows.append({'field':'flights_timetable_diff_mean','value':rescheduled_flights['difference'].mean()})

	rows.append({'field':'new_rail','value':new_rail['rail_service_id_y'].nunique()})
	rows.append({'field':'cancelled_rail','value':cancelled_rail['rail_service_id_x'].nunique()})
	rows.append({'field':'rescheduled_rail','value':rescheduled_rail['rail_service_id_x'].nunique()})
	rows.append({'field':'rescheduled_rail_stops','value':len(rescheduled_rail)})
	rows.append({'field':'rail_timetable_diff_sum','value':rescheduled_rail['difference'].sum()})
	rows.append({'field':'rail_timetable_diff_mean','value':rescheduled_rail['difference'].mean()})

	if variant == 'total':
		df = pd.DataFrame(rows)
		# print(df)
		return df

def pax_resilience_replanned(data,config,pi_config,variant='total'):

	rows = []
	if 'pax_assigned_to_itineraries_options_status_replanned' in data[0]:
		i = 0
	elif 'pax_assigned_to_itineraries_options_status_replanned' in data[1]:
		i = 1
	else:
		raise Exception('pax_assigned_to_itineraries_options_status_replanned does not exist')

	pax_assigned_to_itineraries_options_status_replanned = data[i]['pax_assigned_to_itineraries_options_status_replanned']
	pax_reassigned_to_itineraries = data[i]['pax_reassigned_to_itineraries']
	pax_assigned_to_itineraries_replanned_stranded = data[i]['pax_assigned_to_itineraries_replanned_stranded']
	pax_demand_assigned_summary = data[i]['pax_demand_assigned_summary']

	pax_reassigned_to_itineraries['weigthed_delay'] = pax_reassigned_to_itineraries['delay_total_travel_time']*pax_reassigned_to_itineraries['pax_assigned']
	pax_reassigned_to_itineraries['weigthed_delay_arrival_home'] = pax_reassigned_to_itineraries['delay_arrival_home']*pax_reassigned_to_itineraries['pax_assigned']

	if variant == 'total':
		rows.append({'field':'total_pax','value':pax_assigned_to_itineraries_options_status_replanned['pax'].sum()})
		rows.append({'field':'pax_unnafected','value':pax_assigned_to_itineraries_options_status_replanned[pax_assigned_to_itineraries_options_status_replanned['pax_status_replanned']=='unnafected']['pax'].sum()})
		rows.append({'field':'pax_cancelled','value':pax_assigned_to_itineraries_options_status_replanned[pax_assigned_to_itineraries_options_status_replanned['pax_status_replanned']=='cancelled']['pax'].sum()})

		rows.append({'field':'sum_demand_to_replan','value':pax_demand_assigned_summary['demand_to_assign'].sum()})
		rows.append({'field':'sum_pax_assigned','value':pax_demand_assigned_summary['pax_assigned'].sum()})
		rows.append({'field':'sum_demand_unfulfilled','value':pax_demand_assigned_summary['unfulfilled'].sum()})
		rows.append({'field':'perc_demand_unfulfilled','value':pax_demand_assigned_summary['unfulfilled'].sum()/pax_assigned_to_itineraries_options_status_replanned['pax'].sum()})

		rows.append({'field':'sum_delay_travel_time','value':pax_reassigned_to_itineraries['weigthed_delay'].sum()})
		rows.append({'field':'avg_delay_travel_time','value':pax_reassigned_to_itineraries['weigthed_delay'].sum()/pax_reassigned_to_itineraries['pax_assigned'].sum()})
		rows.append({'field':'sum_delay_arrival_home','value':pax_reassigned_to_itineraries['weigthed_delay_arrival_home'].sum()})
		rows.append({'field':'avg_delay_arrival_home','value':pax_reassigned_to_itineraries['weigthed_delay_arrival_home'].sum()/pax_reassigned_to_itineraries['pax_assigned'].sum()})
		rows.append({'field':'same_modes_percent','value':(pax_reassigned_to_itineraries['same_modes']*pax_reassigned_to_itineraries['pax_assigned']).sum()/pax_reassigned_to_itineraries['pax_assigned'].sum()})

		rows.append({'field':'pax_no_option','value':pax_assigned_to_itineraries_replanned_stranded[pax_assigned_to_itineraries_replanned_stranded['stranded_type']=='no_option']['pax_stranded'].sum()})
		rows.append({'field':'pax_no_capacity','value':pax_assigned_to_itineraries_replanned_stranded[pax_assigned_to_itineraries_replanned_stranded['stranded_type']=='no_capacity']['pax_stranded'].sum()})

		df = pd.DataFrame(rows)
		# print(df)
		return df


def capacity_available(data,config,pi_config,variant='all'):
	print(' -- Capacity available', pi_config['variant'])

	demand = data['pax_assigned_to_itineraries_options'].copy()
	dict_services_w_capacity = {
		mode: dict(group[["nid", "max_seats"]].values)
		for mode, group in data['pax_assigned_seats_max_target'].groupby("mode_transport")
	}

	services_w_capacity, services_wo_capacity = compute_capacities_available_services(demand=demand,
																					  dict_services_w_capacity=dict_services_w_capacity)

	services_w_capacity['pax_possible'] = services_w_capacity['max_seats_service'] -services_w_capacity['max_pax_in_service']

	possible_itineraries = data['possible_itineraries_clustered_pareto_filtered'].copy()

	# Step 1: Build lookup sets and dict
	wo_set = set(services_wo_capacity['service_id'])
	w_dict = dict(zip(services_w_capacity['service_id'], services_w_capacity['pax_possible']))

	# Step 2: Identify service columns
	service_cols = [col for col in possible_itineraries.columns if col.startswith("service_id_")]

	# Step 3: Define function to process each row
	def compute_capacity_and_flag(row):
		services = [row[col] for col in service_cols if pd.notna(row[col])]

		# Flag if any service is missing in both tables
		unknown_services = [s for s in services if s not in w_dict and s not in wo_set]

		# If any service is in wo_capacity, total capacity is 0
		if any(s in wo_set for s in services):
			return pd.Series({'max_pax': 0, 'flag_unknown_service': bool(unknown_services)})

		# Otherwise, get min capacity of known services
		capacities = [w_dict[s] for s in services if s in w_dict]

		# If all services are known, return min capacity
		if len(capacities) == len(services):
			return pd.Series({'max_pax': min(capacities), 'flag_unknown_service': False})
		else:
			return pd.Series({'max_pax': None, 'flag_unknown_service': True})

	# Step 4: Apply the function
	df_seats_available = possible_itineraries.copy()
	df_seats_available[['max_pax', 'flag_unknown_service']] = possible_itineraries.apply(compute_capacity_and_flag, axis=1)
	df_seats_available = df_seats_available[(['origin', 'destination', 'cluster_id', 'option', 'nservices', 'journey_type',
											  'path'] + service_cols + ['max_pax', 'flag_unknown_service'])]

	add_nuts(df_seats_available)
	cap_available_nuts2_journey_type = df_seats_available.groupby(['origin_nuts2', 'destination_nuts2', 'journey_type'])['max_pax'].sum().reset_index()
	cap_available_nuts3_journey_type = df_seats_available.groupby(['origin', 'destination', 'journey_type'])[
		'max_pax'].sum().reset_index()

	total_capacity_available = df_seats_available.max_pax.sum()
	total_cap_availabe_mode = df_seats_available.groupby('journey_type').max_pax.sum().reset_index()
	total_capacity = total_cap_availabe_mode['max_pax'].sum()
	total_cap_availabe_mode['percentage'] = (total_cap_availabe_mode['max_pax'] / total_capacity) * 100

	cap_available_nuts2 = df_seats_available.groupby(['origin_nuts2', 'destination_nuts2'])['max_pax'].sum().reset_index()
	cap_available_nuts3 = df_seats_available.groupby(['origin', 'destination'])['max_pax'].sum().reset_index()

	if pi_config.get('plot', False):
		# Generate plots

		def plot_bars_values_capacity(df, savefigpath):

			# Plot total capacity available per mode
			fig, ax = plt.subplots(figsize=(8, 5))
			bars = ax.bar(df['journey_type'], df['max_pax'], color='skyblue')

			# Format y-axis in '000s
			ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x / 1000)}K"))
			ax.set_ylabel('Capacity Available (\'000 of seats)')
			ax.set_xlabel('Journey Type')

			# Add custom text on bars
			if 'percentage' not in df.columns:
				total_capacity = df['max_pax'].sum()
				df['percentage'] = (df['max_pax'] / total_capacity) * 100

			for bar, val, pct in zip(bars, df['max_pax'], df['percentage']):
				height = bar.get_height()
				ax.text(
					bar.get_x() + bar.get_width() / 2,
					height,
					f"{val / 1000:.1f}K\n({pct:.0f}% total capacity available)",
					ha='center',
					va='bottom',
					fontsize=9,
					linespacing=1.5
				)

			plt.tight_layout()
			plt.savefig(savefigpath)

		plot_bars_values_capacity(total_cap_availabe_mode, Path(config['output']['path_to_output_figs']) / ('capacity_available_mode_total' +
																		   config.get('sufix_fig') + '.png'))

		# Plot matrices
		plot_heatmap_from_df(
			cap_available_nuts2,
			origin_col='origin_nuts2', destination_col='destination_nuts2',
			value_col="max_pax",
			vmin=pi_config.get('vmin_nuts2'),
			vmax=pi_config.get('vmax_nuts2'),
			save_path=Path(config['output']['path_to_output_figs']) / ('capacity_available_total_nuts2' +
																	   config.get('sufix_fig') + '.png'))

		plot_heatmap_from_df(
			cap_available_nuts3,
			origin_col = 'origin', destination_col = 'destination',
			value_col = "max_pax",
			vmin = pi_config.get('vmin_nuts3'),
			vmax = pi_config.get('vmax_nuts3'),
			save_path = Path(config['output']['path_to_output_figs']) / ('capacity_available_total_nuts3' +
																		 config.get('sufix_fig') + '.png'))

		if pi_config.get('od_capacity_modes_nuts2') is not None:
			# Plot between NUTS2
			for od in pi_config['od_capacity_modes_nuts2']:
				capacities = cap_available_nuts2_journey_type[((cap_available_nuts2_journey_type.origin_nuts2==od[0]) &
												  			  (cap_available_nuts2_journey_type.destination_nuts2 == od[1]))].copy()
				if len(capacities) > 0:
					plot_bars_values_capacity(capacities, Path(config['output']['path_to_output_figs']) /
											  								('capacity_available_mode_'
																			 +od[0]+'-'+od[1]+
																			 config.get('sufix_fig') + '.png'))

		if pi_config.get('od_capacity_modes_nuts3') is not None:
			# Plot between NUTS3
			for od in pi_config['od_capacity_modes_nuts3']:
				capacities = cap_available_nuts3_journey_type[((cap_available_nuts3_journey_type.origin==od[0]) &
												  			  (cap_available_nuts3_journey_type.destination == od[1]))].copy()
				if len(capacities) > 0:
					plot_bars_values_capacity(capacities, Path(config['output']['path_to_output_figs']) /
											  								('capacity_available_mode_'
																			 +od[0]+'-'+od[1]+
																			 config.get('sufix_fig') + '.png'))



	return {'_total': total_capacity_available,
			'_total_mode': total_cap_availabe_mode,
			'_nuts2_journey_type': cap_available_nuts2_journey_type,
			'_nuts3_journey_type': cap_available_nuts3_journey_type,
			'_nuts2': cap_available_nuts2,
			'_nuts3': cap_available_nuts3,}
