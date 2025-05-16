import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
pio.kaleido.scope.mathjax = None
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt


def add_nuts(df):
	df['origin_nuts2'] = df.apply(lambda row: row['origin'][:4], axis=1)
	df['destination_nuts2'] = df.apply(lambda row: row['destination'][:4], axis=1)
	df['origin_nuts1'] = df.apply(lambda row: row['origin'][:3], axis=1)
	df['destination_nuts1'] = df.apply(lambda row: row['destination'][:3], axis=1)
	return df

def strategic_total_journey_time(data,config,pi_config,variant="sum"):
	print('strategic_total_journey_time')
	pax_assigned_to_itineraries_options = data['pax_assigned_to_itineraries_options']
	possible_itineraries_clustered_pareto_filtered = data['possible_itineraries_clustered_pareto_filtered']
	nuts_regional_archetype_info = data['nuts_regional_archetype_info']

	#only options that have pax assigned
	df = pax_assigned_to_itineraries_options[pax_assigned_to_itineraries_options['pax']>0].copy()

	#weigth total_time with pax
	df['weigthed_total_time'] = df['total_time']*df['pax']
	if variant == 'sum' :
		kpi = df['weigthed_total_time'].sum()
		print('sum',kpi)
		return kpi

	if variant == 'avg':
		kpi = df['weigthed_total_time'].sum()/df['pax'].sum()
		print('avg',kpi)
		return kpi

	if variant == 'avg_connecting_itineraries':
		df = df.dropna(subset=['nid_f2'])
		kpi = df['weigthed_total_time'].sum()/df['pax'].sum()
		print('avg_connecting_itineraries',kpi)
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
		print(df)
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
								 save_path=Path(config['output']['path_to_output']) / ('avg_by_nuts_matrix_plot'+
																					   config.get('sufix_fig')+'.png'))

			plot_heatmap_from_df(grouped_nuts2, origin_col='origin_nuts2', destination_col='destination_nuts2',
								 value_col="total_journey_time_per_pax",
								 vmin=pi_config.get('vmin_matrix_nuts2'),
								 vmax=pi_config.get('vmax_matrix_nuts2'),
								 save_path=Path(config['output']['path_to_output']) / ('avg_by_nuts_2_matrix_plot'+
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
	import geopandas as gpd
	nuts_data_path = config['input']['nuts_data_path']
	nuts_data = gpd.read_file(nuts_data_path)
	nuts_data.crs = 'EPSG:4326'
	es_nuts = nuts_data[(nuts_data['LEVL_CODE']==3) & (nuts_data['CNTR_CODE']=='ES')]
	df = its[its['origin_nuts2']=='ES51'].groupby(['destination'])['total_journey_time_per_pax'].mean().reset_index()
	df = df.rename({'destination': 'NUTS_ID'}, axis=1)
	print(df)
	es_nuts = es_nuts.merge(df,how='left',on='NUTS_ID')
	print(es_nuts)
	fig = px.choropleth(es_nuts, geojson=es_nuts.geometry, locations=es_nuts.index, color="total_journey_time_per_pax", range_color=[200,700])
	fig.update_geos(fitbounds="locations", visible=False)

	fig.update_layout(title_text = title)
	#fig.show()
	fig.write_html(Path(config['output']['path_to_output']) / 'avg_by_nuts_plot.html', )


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
		print(od_totals)

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


def diversity_of_destinations(data,config,pi_config,variant='nuts'):
	possible_itineraries_clustered_pareto_filtered = data['possible_itineraries_clustered_pareto_filtered']
	results = {}
	if variant == 'nuts':
		df = possible_itineraries_clustered_pareto_filtered.groupby(['origin']).nunique().reset_index()[['origin','destination']]
		if pi_config.get('plot', False):
			fig = px.bar(df, x='origin', y='destination')
			#fig.show()
			fig.write_html(Path(config['output']['path_to_output']) / 'diversity_of_destinations_nuts.html', )

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
			fig.write_html(Path(config['output']['path_to_output']) / 'diversity_of_destinations_hubs.html', )
		return con_out2

def modal_share(data,config,pi_config,variant='total'):
	pax_assigned_to_itineraries_options = data['pax_assigned_to_itineraries_options']
	possible_itineraries_clustered_pareto_filtered = data['possible_itineraries_clustered_pareto_filtered']
	nuts_regional_archetype_info = data['nuts_regional_archetype_info']

	df = pd.concat([pax_assigned_to_itineraries_options,possible_itineraries_clustered_pareto_filtered[['journey_type']]],axis=1)
	df = df[df['pax']>0][['pax','journey_type','origin']]
	grouped = df.groupby(['journey_type']).sum().reset_index()
	grouped['percentage'] = grouped['pax']/grouped['pax'].sum()
	print('grouped',grouped[['journey_type','pax','percentage']])
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
				save_path=Path(config['output']['path_to_output']) / ('mode_share_between_nuts_pax'+
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
				save_path=Path(config['output']['path_to_output']) / ('mode_share_between_nuts_pax_per_demand'+
																					   config.get('sufix_fig')+'.png'))

			plot_top_od_stacked_bars(
				grouped,
				origin_col="origin",
				destination_col="destination",
				journey_type_col="journey_type",
				value_col="pax",
				top_n=pi_config.get('plot_top', 10),
				percentage=True,
				save_path=Path(config['output']['path_to_output']) / ('mode_share_between_nuts_share'+
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
				save_path=Path(config['output']['path_to_output']) / ('mode_share_between_nuts_share_per_demand'+
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
				save_path=Path(config['output']['path_to_output']) / ('mode_share_between_nuts2_pax'+
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
				save_path=Path(config['output']['path_to_output']) / ('mode_share_between_nuts2_pax_per_demand'+
																					   config.get('sufix_fig')+'.png'))

			plot_top_od_stacked_bars(
				grouped_nuts2,
				origin_col="origin_nuts2",
				destination_col="destination_nuts2",
				journey_type_col="journey_type",
				value_col="pax",
				top_n=pi_config.get('plot_top', 10),
				percentage=True,
				save_path=Path(config['output']['path_to_output']) / ('mode_share_between_nuts2_share'+
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
				save_path=Path(config['output']['path_to_output']) / ('mode_share_between_nuts2_share_per_demand'+
																					   config.get('sufix_fig')+'.png'))

		return grouped_nuts2

	if variant == 'by_regional_archetype':
		#grouping by nuts
		df = df.merge(nuts_regional_archetype_info ,how='left',left_on='origin',right_on='origin')
		total = df.groupby(['regional_archetype'])['pax'].sum().reset_index()
		print(total)
		grouped = df.groupby(['regional_archetype','journey_type']).sum().reset_index()
		grouped['percentage'] = grouped.apply(lambda row: row['pax']/total[total['regional_archetype']==row['regional_archetype']]['pax'].iloc[0], axis=1)
		print(grouped[['regional_archetype','journey_type','pax','percentage']])
		return  grouped[['regional_archetype','journey_type','pax','percentage']]

def pax_time_efficiency(data,config,pi_config,variant='total'):
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
								 save_path=Path(config['output']['path_to_output']) / ('demand_served_nuts3_' +
																					   config.get(
																						   'sufix_fig') + '.png'))

			plot_heatmap_from_df(grouped_n2, origin_col='origin_nuts2', destination_col='destination_nuts2',
								 value_col="perc",
								 vmin=pi_config.get('vmin_matrix'),
								 vmax=pi_config.get('vmax_matrix'),
								 save_path=Path(config['output']['path_to_output']) / ('demand_served_nuts2_' +
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
								 save_path=Path(config['output']['path_to_output']) / ('demand_served_regional_arch_' +
																					   config.get(
																						   'sufix_fig') + '.png'))

		return grouped2[['origin_regional_archetype', 'destination_regional_archetype', 'pax', 'trips', 'perc']]


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
								 save_path=Path(config['output']['path_to_output']) / ('resilience_options_nuts'+
																					   config.get('sufix_fig')+'.png'))

			plot_heatmap_from_df(dftotaln2, origin_col='origin_nuts2', destination_col='destination_nuts2',
								 value_col="option",
								 vmin=pi_config.get('vmin_matrix_nuts2'),
								 vmax=pi_config.get('vmax_matrix_nuts2'),
								 save_path=Path(config['output']['path_to_output']) /
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
									 save_path=Path(config['output']['path_to_output']) /
											   ('resilience_options_nuts_' + mode + config.get('sufix_fig')+'.png'))

				vmin = pi_config.get('vmin_matrix_nuts2_' + mode)
				vmax = pi_config.get('vmax_matrix_nuts2_' + mode)
				plot_heatmap_from_df(dfn2[dfn2.journey_type == mode],
									 origin_col='origin_nuts2',
									 destination_col='destination_nuts2',
									 vmax=vmax,
									 vmin=vmin,
									 value_col="option",
									 save_path=Path(config['output']['path_to_output']) / (
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
								 save_path=Path(config['output']['path_to_output']) / ('resilience_options_reg_arch'+
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
									 save_path=Path(config['output']['path_to_output']) / ('resilience_options_reg_arch_'+mode+
																					   config.get('sufix_fig')+'.png'))

		return  {'_total': grouped_all, '_w_jt': grouped_type}


def buffer_in_itineraries(data,config,pi_config,variant='sum'):

	pax_assigned_to_itineraries_options = data['pax_assigned_to_itineraries_options']
	possible_itineraries_clustered_pareto_filtered = data['possible_itineraries_clustered_pareto_filtered']
	df = pd.concat([pax_assigned_to_itineraries_options,possible_itineraries_clustered_pareto_filtered[['nservices']]],axis=1)

	#only options that have pax assigned and buffer is existing (buffer is 0 for one leg itineraries)
	df = df[(df['pax']>0) & (df['nservices']>1)].copy()

	#weigth total_waiting_time with pax
	df['weigthed_total_waiting_time'] = df['total_waiting_time']*df['pax']
	kpi = df['weigthed_total_waiting_time'].sum()
	print(kpi)
	if variant == 'sum':
		return kpi
	if variant == 'avg':
		return df['weigthed_total_waiting_time'].sum()/df['pax'].sum()

def catchment_area(data,config,pi_config,variant='hubs'):

	pax_assigned_to_itineraries_options = data['pax_assigned_to_itineraries_options']
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
	print(con_out)
	if pi_config['plot'] == True:
		fig = px.bar(con_out, x='hub', y='travel_time')
		fig.update_layout(xaxis={'categoryorder':'total ascending'})
		#fig.show()
		fig.write_html(Path(config['output']['path_to_output']) / 'catchment_area_hubs.html', )
	if variant == 'hubs':
		return con_out

def cost_per_user(data,config,pi_config,variant='avg'):

	pax_assigned_to_itineraries_options = data['pax_assigned_to_itineraries_options']
	#only options that have pax assigned
	df = pax_assigned_to_itineraries_options[pax_assigned_to_itineraries_options['pax']>0].copy()

	#weigth total_time with pax
	df['weigthed_cost'] = df['fare']*df['pax']
	kpi = df['weigthed_cost'].sum()/df['pax'].sum()
	print(kpi)
	if variant == 'avg':
		return kpi

def co2_emissions(data,config,pi_config,variant='avg'):

	pax_assigned_to_itineraries_options = data['pax_assigned_to_itineraries_options']
	possible_itineraries_clustered_pareto_filtered = data['possible_itineraries_clustered_pareto_filtered']
	#only options that have pax assigned
	its = pd.concat([pax_assigned_to_itineraries_options,possible_itineraries_clustered_pareto_filtered],axis=1)
	its = its[its['pax']>0]

	#weigth co2 with pax
	its['weigthed_co2'] = its['total_emissions']*its['pax']
	kpi = its['weigthed_co2'].sum()/its['pax'].sum()
	print(kpi)
	if variant == 'avg':
		return kpi

def seamless_of_travel(data,config,pi_config,variant='avg'):

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

	print(rail_timetable_proc_used_internally0.dtypes)

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
	print(new_rail)
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
		print(df)
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
		print(df)
		return df
