import pandas as pd
import datetime
import numpy as np
import plotly.express as px
import plotly.io as pio

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
