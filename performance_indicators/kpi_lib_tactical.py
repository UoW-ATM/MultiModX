import pandas as pd
import datetime
import numpy as np
import plotly.express as px
import plotly.io as pio
from scipy.stats import norm, lognorm, expon
import ast
import random

def flight_arrival_delay(df_flights):

	thresholds = [0,15,30,45,60]
	results = {}
	#drop cancelled flights
	df_flights_filtered = df_flights.dropna(subset=['arrival_delay_min'])
	for threshold in thresholds:
		df = df_flights_filtered[df_flights_filtered['arrival_delay_min']>=threshold]
		kpi = df['arrival_delay_min'].count()/df_flights['arrival_delay_min'].count()
		results[threshold] = kpi

	print(results)
	return results

def kerb2gate_time(df_pax,airport_processes):
	print('kerb2gate_time')

	# generate kerb2gate_time for air pax
	air_pax = df_pax[(df_pax['kerb2gate_time']==0) & (~pd.isnull(df_pax['airport1']))].copy()
	air_pax = airport_process_time_generator(air_pax,airport_processes,process='k2g')
	#only options that have time defined
	df = df_pax[df_pax['kerb2gate_time']>0].copy()
	#merge
	df = pd.concat([df,air_pax])

	#weigth total_time with pax
	df['weigthed_time'] = df['kerb2gate_time']*df['n_pax']
	kpi = df['weigthed_time'].sum()/df['n_pax'].sum()

	print('kerb2gate_time',kpi)

def gate2kerb_time(df_pax,airport_processes):
	print('gate2kerb_time')

	# generate gate2kerb_time for air pax
	air_pax = df_pax[df_pax['gate2kerb_time']==0].copy()
	air_pax = airport_process_time_generator(air_pax,airport_processes,process='g2k')
	#only options that have time defined
	df = df_pax[df_pax['gate2kerb_time']>0].copy()
	#merge
	df = pd.concat([df,air_pax])
	#weigth total_time with pax
	df['weigthed_time'] = df['gate2kerb_time']*df['n_pax']
	kpi = df['weigthed_time'].sum()/df['n_pax'].sum()
	print('gate2kerb_time',kpi)

def total_arrival_delay(df_pax):
	print('tot_arrival_delay')
	df = df_pax[df_pax['tot_arrival_delay']<10000].copy()

	#weigth total_time with pax
	df['weigthed_time'] = df['tot_arrival_delay']*df['n_pax']
	missed_connection = df[df['modified_itinerary']==True]

	kpi = df['weigthed_time'].sum()/df['n_pax'].sum()
	print(kpi)

def stranded_pax(df_pax):
	print('stranded_pax')
	df = df_pax.copy()

	#weigth total_time with pax

	stranded = df[df['final_destination_reached']==False]

	kpi = stranded['n_pax'].sum()/df['n_pax'].sum()
	print(kpi)

def ratio_stranded_pax(df_pax):
	print('ratio_stranded_pax')
	df = df_pax.copy()

	#weigth total_time with pax

	stranded = df[df['final_destination_reached']==False]
	missed_connection = df[df['modified_itinerary']==True]

	kpi = stranded['n_pax'].sum()/missed_connection['n_pax'].sum()
	print(kpi)

def missed_connections(df_pax):
	print('missed_connections')
	df = df_pax.copy()


	missed_connection = df[df['modified_itinerary']==True]

	kpi = missed_connection['n_pax'].sum()/df['n_pax'].sum()
	print(kpi)

def total_journey_time(df_pax,airport_processes,pax_assigned_tactical):


	df_pax['weigthed_time'] = df_pax['total_time']*df_pax['n_pax']
	print(df_pax)
	df_pax.to_csv('xxx.csv')




def ground_mobility(df_pax):
	print('ground_mobility')
	df = df_pax[df_pax['ground_mobility_time']>0].copy()

	#weigth total_time with pax
	df['weigthed_time'] = df['ground_mobility_time']*df['n_pax']


	kpi = df['weigthed_time'].sum()/df['n_pax'].sum()
	print(kpi)

def variability(df_pax):

	thresholds = [0,15,30,45,60]
	results = {}
	#drop stranded pax
	df = df_pax[df_pax['tot_arrival_delay']<10000].copy()
	for threshold in thresholds:
		df_t = df[df['tot_arrival_delay']>=threshold]
		kpi = df_t['n_pax'].sum()/df_pax['n_pax'].sum()
		results[threshold] = kpi

	print(results)
	return results

def tactical_pax_time_efficiency(pax_assigned_to_itineraries_options,df_pax):

	#get min times from strategic itineraries
	pax_assigned_to_itineraries_options['origin'] = pax_assigned_to_itineraries_options.apply(lambda row: row['alternative_id'].split('_')[0], axis=1)
	pax_assigned_to_itineraries_options['destination'] = pax_assigned_to_itineraries_options.apply(lambda row: row['alternative_id'].split('_')[1], axis=1)

	best = pax_assigned_to_itineraries_options.groupby(['origin','destination'])['total_time'].min()


	df_pax['efficiency'] = df_pax.apply(lambda row: best.loc[row['origin'],row['destination']]/row['total_time'], axis=1)
	df_pax['weigthed_efficiency'] = df_pax['efficiency']*df_pax['pax']
	print(df)
	print('total',df['weigthed_efficiency'].sum()/df['pax'].sum())


