import pandas as pd
import datetime
import numpy as np
import plotly.express as px
import plotly.io as pio
pio.kaleido.scope.mathjax = None

def add_nuts(df):
	df['origin_nuts2'] = df.apply(lambda row: row['origin'][:4], axis=1)
	df['destination_nuts2'] = df.apply(lambda row: row['destination'][:4], axis=1)
	df['origin_nuts1'] = df.apply(lambda row: row['origin'][:3], axis=1)
	df['destination_nuts1'] = df.apply(lambda row: row['destination'][:3], axis=1)
	return df

def strategic_total_journey_time(pax_assigned_to_itineraries_options,possible_itineraries_clustered_pareto_filtered):
	print('strategic_total_journey_time')
	#only options that have pax assigned
	df = pax_assigned_to_itineraries_options[pax_assigned_to_itineraries_options['pax']>0].copy()

	#weigth total_time with pax
	df['weigthed_total_time'] = df['total_time']*df['pax']
	kpi = df['weigthed_total_time'].sum()
	print(kpi)

	#group by NUTS2/1
	#
	df['origin'] = df.apply(lambda row: row['alternative_id'].split('_')[0], axis=1)
	df['destination'] = df.apply(lambda row: row['alternative_id'].split('_')[1], axis=1)
	#its = df.merge(possible_itineraries_clustered_pareto_filtered,how='left',left_on=['alternative_id','option'],right_on=['alternative_id','option'])
	#print(its)
	#print(its[['origin','destination','weigthed_total_time']])
	print(df)
	grouped = df.groupby(['origin','destination']).sum().reset_index()
	grouped['total_journey_time_per_pax'] = grouped['weigthed_total_time']/grouped['pax']
	print(grouped)

	add_nuts(grouped)
	plot_nuts(grouped,title='Avg total_journey_time_per_pax from ES11')

def plot_nuts(its,title=''):
	import geopandas as gpd
	nuts_data_path = '../demand/data/NUTS_RG_01M_2021_4326_shp.shp'
	nuts_data = gpd.read_file(nuts_data_path)
	nuts_data.crs = 'EPSG:4326'
	es_nuts = nuts_data[(nuts_data['LEVL_CODE']==3) & (nuts_data['CNTR_CODE']=='ES')]
	df = its[its['origin_nuts2']=='ES11'].groupby(['destination'])['total_journey_time_per_pax'].mean().reset_index()
	df = df.rename({'destination': 'NUTS_ID'}, axis=1)
	print(df)
	es_nuts = es_nuts.merge(df,how='left',on='NUTS_ID')
	print(es_nuts)
	fig = px.choropleth(es_nuts, geojson=es_nuts.geometry, locations=es_nuts.index, color="total_journey_time_per_pax", range_color=[200,700])
	fig.update_geos(fitbounds="locations", visible=False)

	fig.update_layout(title_text = title)
	#fig.show()

def diversity_of_destinations(possible_itineraries_clustered_pareto_filtered):

	df = possible_itineraries_clustered_pareto_filtered.groupby(['origin']).nunique().reset_index()
	print(df)
	fig = px.bar(df, x='origin', y='destination')
	fig.show()

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
	con_out = df_out.groupby(["hub",'mode_out']).count().reset_index().sort_values(by='leg_out')
	con_out2 = df_out.drop_duplicates(subset=['hub','destination_out','mode_out'])[["hub",'destination_out','mode_out']].groupby(["hub",'mode_out']).count().reset_index().sort_values(by='destination_out')
	#print(con_out2)
	fig = px.bar(con_out2, x='hub', y='destination_out',color='mode_out', barmode='relative')
	fig.update_layout(xaxis={'categoryorder':'total ascending'})
	fig.show()

def modal_share(pax_assigned_to_itineraries_options,possible_itineraries_clustered_pareto_filtered):

	df = pd.concat([pax_assigned_to_itineraries_options,possible_itineraries_clustered_pareto_filtered],axis=1)
	df = df[df['pax']>0]
	grouped = df.groupby(['journey_type']).sum().reset_index()
	grouped['percentage'] = grouped['pax']/grouped['pax'].sum()
	print(grouped[['journey_type','pax','percentage']])

	#grouping by nuts
	total = df.groupby(['origin'])['pax'].sum().reset_index()
	print(total)
	grouped = df.groupby(['origin','journey_type']).sum().reset_index()
	grouped['percentage'] = grouped.apply(lambda row: row['pax']/total[total['origin']==row['origin']]['pax'].iloc[0], axis=1)
	print(grouped[['origin','journey_type','pax','percentage']])

def pax_time_efficiency(pax_assigned_to_itineraries_options):

	pax_assigned_to_itineraries_options['origin'] = pax_assigned_to_itineraries_options.apply(lambda row: row['alternative_id'].split('_')[0], axis=1)
	pax_assigned_to_itineraries_options['destination'] = pax_assigned_to_itineraries_options.apply(lambda row: row['alternative_id'].split('_')[1], axis=1)

	best = pax_assigned_to_itineraries_options.groupby(['origin','destination'])['total_time'].min()

	df = pax_assigned_to_itineraries_options[pax_assigned_to_itineraries_options['pax']>0].copy()
	df['efficiency'] = df.apply(lambda row: best.loc[row['origin'],row['destination']]/row['total_time'], axis=1)
	df['weigthed_efficiency'] = df['efficiency']*df['pax']
	print(df)
	print('total',df['weigthed_efficiency'].sum()/df['pax'].sum())

def demand_served(pax_assigned_to_itineraries_options,demand):

	pax_assigned_to_itineraries_options['origin'] = pax_assigned_to_itineraries_options.apply(lambda row: row['alternative_id'].split('_')[0], axis=1)
	pax_assigned_to_itineraries_options['destination'] = pax_assigned_to_itineraries_options.apply(lambda row: row['alternative_id'].split('_')[1], axis=1)

	total = pax_assigned_to_itineraries_options['pax'].sum()/demand['trips'].sum()
	print('total',total)

	#grouping by nuts
	demand_nuts = demand.groupby(['origin','destination'])['trips'].sum().reset_index()
	demand_nuts['demand'] = np.ceil(demand_nuts['trips'])

	df = pax_assigned_to_itineraries_options[pax_assigned_to_itineraries_options['pax']>0]
	grouped = df.groupby(['origin','destination'])['pax'].sum().reset_index()
	grouped = grouped.merge(demand_nuts,how='left',on=['origin','destination'])
	grouped['perc'] = grouped['pax']/grouped['demand']
	print(grouped[['origin','destination','pax','demand','perc']])

def load_factor(pax_assigned_to_itineraries_options,possible_itineraries_clustered_pareto_filtered,pax_assigned_seats_max_target):

	df = pd.concat([pax_assigned_to_itineraries_options,possible_itineraries_clustered_pareto_filtered],axis=1)
	df = df[df['pax']>0]
	#flatten itineraries
	services = pd.concat([df[['service_id_0','pax','mode_0']],df[['service_id_1','pax','mode_1']].rename({'service_id_1': 'service_id_0','mode_1':'mode_0'}, axis=1)],axis=0)
	services =  services.dropna(subset=['service_id_0'])
	services['service_id'] = services.apply(lambda row: row['service_id_0'].split('_')[0] if row['mode_0']=='rail' else row['service_id_0'], axis=1)
	services = services.groupby(['service_id','mode_0'])['pax'].sum().reset_index()
	services = services.merge(pax_assigned_seats_max_target,how='left',left_on='service_id',right_on='nid')
	services['load_factor'] = services['pax']/services['max_seats']
	print(services)
	print('total',services['load_factor'].mean())

	modes = services.groupby(['mode_0'])['load_factor'].mean()
	print(modes)

def resilience_alternatives(possible_itineraries_clustered_pareto_filtered):

	df = possible_itineraries_clustered_pareto_filtered.groupby(['origin','destination'])['option'].count().reset_index()
	print(df)

def buffer_in_itineraries(pax_assigned_to_itineraries_options):

	#only options that have pax assigned
	df = pax_assigned_to_itineraries_options[pax_assigned_to_itineraries_options['pax']>0].copy()

	#weigth total_waiting_time with pax
	df['weigthed_total_waiting_time'] = df['total_waiting_time']*df['pax']
	kpi = df['weigthed_total_waiting_time'].sum()
	print(kpi)

def catchment_area(pax_assigned_to_itineraries_options,possible_itineraries_clustered_pareto_filtered):

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
	fig = px.bar(con_out, x='hub', y='travel_time')
	fig.update_layout(xaxis={'categoryorder':'total ascending'})
	fig.show()

def cost_per_user(pax_assigned_to_itineraries_options):

	#only options that have pax assigned
	df = pax_assigned_to_itineraries_options[pax_assigned_to_itineraries_options['pax']>0].copy()

	#weigth total_time with pax
	df['weigthed_cost'] = df['fare']*df['pax']
	kpi = df['weigthed_cost'].sum()/df['pax'].sum()
	print(kpi)

def co2_emissions(pax_assigned_to_itineraries_options,possible_itineraries_clustered_pareto_filtered):

	#only options that have pax assigned
	its = pd.concat([pax_assigned_to_itineraries_options,possible_itineraries_clustered_pareto_filtered],axis=1)
	its = its[its['pax']>0]

	#weigth co2 with pax
	its['weigthed_co2'] = its['total_emissions']*its['pax']
	kpi = its['weigthed_co2'].sum()/its['pax'].sum()
	print(kpi)

