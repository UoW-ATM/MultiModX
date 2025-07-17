#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

def origin_before(row):
	# print(row)
	if row['Origin Country Name']=='SPAIN':
		orig = row['Origin Airport']
	elif row['Connecting Country1']=='ES':
		orig = row['Origin Airport']
	elif row['Connecting Country2']=='ES':
		orig = row['Connecting Airport1']
	elif row['Connecting Country3']=='ES':
		orig = row['Connecting Airport2']
	else:
		if  not pd.isnull(row['Connecting Airport3']):
			orig = row['Connecting Airport3']
		elif  not pd.isnull(row['Connecting Airport2']):
			orig = row['Connecting Airport2']
		elif  not pd.isnull(row['Connecting Airport1']):
			orig = row['Connecting Airport1']
		else:
			orig = row['Origin Airport']
	return orig

def destination_after(row):

	if row['Destination Country Name']=='SPAIN':
		dest = row['Destination Airport']
	elif row['Connecting Country3']=='ES':
		dest = row['Destination Airport']
	elif row['Connecting Country2']=='ES':
		if not pd.isnull(row['Connecting Country3']):
			dest = row['Connecting Airport3']
		else:
			dest = row['Destination Airport']
	elif row['Connecting Country1']=='ES':
		if  not pd.isnull(row['Connecting Country2']):
			dest = row['Connecting Airport2']
		else:
			dest = row['Destination Airport']
	elif row['Origin Country Name']=='SPAIN':
		if  not pd.isnull(row['Connecting Country1']):
			dest = row['Connecting Airport1']
		else:
			dest = row['Destination Airport']
	else:
		dest = row['Destination Airport']
	return dest

def connections_outside_spain(row):
	# print(row)
	connections = [x for x in (row['Connecting Country1'],row['Connecting Country2'],row['Connecting Country3']) if ((x != 'ES') and (not pd.isnull(x)))]

	return len(connections)

def hub_outside_spain(row):
	# print(row)
	hub=np.nan
	if row['Origin Country Name']=='SPAIN':
		if row['Connecting Country1']!='ES':
			hub = row['Connecting Airport1']
		elif row['Connecting Country2']!='ES':
			hub = row['Connecting Airport2']
		elif row['Connecting Country3']!='ES':
			hub = row['Connecting Airport3']
	else:
		if row['Connecting Country3']!='ES' and not pd.isnull(row['Connecting Country3']):
			hub = row['Connecting Airport3']
		elif row['Connecting Country2']!='ES' and not pd.isnull(row['Connecting Country2']):
			hub = row['Connecting Airport2']
		elif row['Connecting Country1']!='ES' and not pd.isnull(row['Connecting Country1']):
			hub = row['Connecting Airport1']
	return hub


def departing_last_airport(row):
	# print(row)
	orig = np.nan
	if row['Origin Country Name']=='SPAIN':
		orig = row['origin']
		if row['Connecting Country3']=='ES':
			orig = row['Connecting Airport3']
		elif row['Connecting Country2']=='ES':
			orig = row['Connecting Airport2']
		elif row['Connecting Country1']=='ES':
			orig = row['Connecting Airport1']
	return orig

def arriving_first_airport(row):
	# print(row)
	orig = np.nan
	if row['Destination Country Name']=='SPAIN':
		orig = row['destination']
		if row['Connecting Country1']=='ES':
			orig = row['Connecting Airport1_icao']
		elif row['Connecting Country2']=='ES':
			orig = row['Connecting Airport2_icao']
		elif row['Connecting Country3']=='ES':
			orig = row['Connecting Airport3_icao']
	return orig
def read_aw():
	aw_od = pd.read_excel('../data/Revised Files from AWN (ASM)/O&D (Europe to world each way and world via Europe).xlsx')




	aw_spain = aw_od[((aw_od['Connecting Country1']=='ES') | (aw_od['Connecting Country2']=='ES') | (aw_od['Connecting Country3']=='ES')) | (((aw_od['Origin Country Name']=='SPAIN') & (aw_od['Destination Country Name']!='SPAIN')) | ((aw_od['Destination Country Name']=='SPAIN') & (aw_od['Origin Country Name']!='SPAIN')))]
	aw_spain = aw_spain[~((aw_spain['Destination Country Name']=='SPAIN') & (aw_spain['Origin Country Name']=='SPAIN'))]
	#aw_spain.to_csv('../data/data_analysis/aw_spain_itineraries_raw.csv')


	# In[ ]:


	aw_spain = aw_spain[(aw_spain['Origin Airport Name']!='FERRY PORT') & (aw_spain['Origin Airport Name']!='BERNE RAILWAY SERVICE') &
	(aw_spain['Origin City Name']!='RAILWAY - GERMANY') & (aw_spain['Origin Airport Name']!='COLOGNE HBF RAIL STATION') &
	(aw_spain['Origin Airport Name']!='SANTS RAILWAY STATION') & (aw_spain['Origin Airport Name']!='ATOCHA RAILWAY STATION') &
	(aw_spain['Origin Airport Name']!='LILLE EUROPE RAIL SERVICE') & (aw_spain['Origin City Name']!='ALGECIRAS') &
	(aw_spain['Origin Airport Name']!='SBB RAILWAY SERVICE') & (aw_spain['Origin Airport Name']!='STRASBOURG BUS STATION') &
	(aw_spain['Origin Airport Name']!='DELICIAS RAILWAY STATION') & (aw_spain['Origin Airport Name']!='RAILWAY STATION') &
	(aw_spain['Origin Airport Name']!='MIDI RAILWAY STATION') & (aw_spain['Origin Airport Name']!='HBF RAILWAY STATION') &
	(aw_spain['Origin Airport Name']!='VALENCIA RAILWAY STATION') & (aw_spain['Origin Airport Name']!='CHAMARTIN RAILWAY STATION') &
	(aw_spain['Origin Airport Name']!='EK BUS STATION') & (aw_spain['Origin Airport Name']!='TRAVEL MALL EY BUS STATION') &
	(aw_spain['Origin Airport Name']!='OTTAWA RAIL STATION') & (aw_spain['Origin Airport']!='RZG') & (aw_spain['Origin Airport']!='BQC') &
	(aw_spain['Origin Airport']!='XJN') & (aw_spain['Origin Airport Name']!='ARNHEM BUS STATION') &
	(aw_spain['Origin Airport Name']!='LEON RAILWAY STATION') & (aw_spain['Origin Airport']!='XJR') &
	(aw_spain['Origin Airport Name']!='LINZ HAUPTBAHNHOF RAIL STATION') & (aw_spain['Origin Airport']!='GBX') &
	(aw_spain['Origin Airport']!='EMU') & (aw_spain['Origin Airport Name']!='SANTA JUSTA RAILWAY STATION') &
	(aw_spain['Origin Airport Name']!='MAASTRICHT RAILWAY SERVICE') & (aw_spain['Origin Airport']!='LZB') &
	(aw_spain['Origin Airport Name']!='GARE DE RENNES RAIL STATION') & (aw_spain['Origin Airport']!='PFE') &
	(aw_spain['Origin Airport Name']!='ABU DHABI EK BUS STATION') & (aw_spain['Origin Airport']!='EXC') &
	(aw_spain['Origin Airport Name']!='ST-PIERRE-CORPS RAIL STATION') & (aw_spain['Origin Airport Name']!='KINGSTON RAIL STATION') &
	(aw_spain['Origin Airport Name']!='CENTRAL RAILWAY STATION') & (aw_spain['Origin Airport Name']!='NOTTINGHAM BUS STATION') &
	(aw_spain['Origin Airport Name']!='CADIZ RAILWAY STATION') & (aw_spain['Origin Airport Name']!='KARLSRUHE HBF RAILWAY STATION') &
	(aw_spain['Origin Airport']!='QYE') & (aw_spain['Origin Airport Name']!='LE MANS RAIL STATION') &
	(aw_spain['Origin Airport Name']!='NANTES RAILWAY SERVICE') & (aw_spain['Origin Airport Name']!='LORRAINE TGV RAIL STATION') &
	(aw_spain['Origin Airport']!='FBV') & (aw_spain['Origin Airport']!='ALV') & (aw_spain['Origin Airport Name']!='PART-DIEU RAILWAY STATION') &
	(aw_spain['Origin Airport Name']!='CENTRAAL RAILWAY STATION') & (aw_spain['Origin Airport']!='XZN') &
	(aw_spain['Origin Airport Name']!='SAINT PANCRAS INTL RAIL STATION') & (aw_spain['Origin Airport Name']!='TARRAGONA/CAMP RAIL STATION') &
	(aw_spain['Origin Airport Name']!='BUS STA}TION') & (aw_spain['Origin Airport Name']!='SABTCO BUS STATION') &
	(aw_spain['Origin Airport Name']!='OXFORD RAILWAY STATION') & (aw_spain['Origin Airport']!='CLG') &
	(aw_spain['Origin Airport']!='CSL') & (aw_spain['Origin Airport']!='VTI') & (aw_spain['Origin Airport']!='SFU') &
	(aw_spain['Origin Airport']!='XRJ') & (aw_spain['Origin Airport']!='LCZ') & (aw_spain['Origin Airport']!='FDD') &
	(aw_spain['Origin Airport']!='MIL') & (aw_spain['Origin Airport Name']!='AMSTERDAM RAILWAY SERVICE') & (aw_spain['Origin Airport']!='UER') &
	(aw_spain['Origin Airport']!='FES') & (aw_spain['Origin Airport']!='ZEG')
	& (aw_spain['Origin Airport']!='XHJ')
	& (aw_spain['Origin Airport']!='NCM')
	& (aw_spain['Origin Airport']!='XJU')
	#Germany DE exclusions
	& (~aw_spain['Origin Airport'].isin(['QYG','RMS','QFP','ZCW']))
	]


	# In[ ]:


	aw_spain = aw_spain[(aw_spain['Destination Airport Name']!='FERRY PORT') & (aw_spain['Destination Airport Name']!='BERNE RAILWAY SERVICE') &
	(aw_spain['Destination City Name']!='RAILWAY - GERMANY') & (aw_spain['Destination Airport Name']!='COLOGNE HBF RAIL STATION') &
	(aw_spain['Destination Airport Name']!='SANTS RAILWAY STATION') & (aw_spain['Destination Airport Name']!='ATOCHA RAILWAY STATION') &
	(aw_spain['Destination Airport Name']!='LILLE EUROPE RAIL SERVICE') & (aw_spain['Destination City Name']!='ALGECIRAS') &
	(aw_spain['Destination Airport Name']!='SBB RAILWAY SERVICE') & (aw_spain['Destination Airport Name']!='STRASBOURG BUS STATION') &
	(aw_spain['Destination Airport Name']!='DELICIAS RAILWAY STATION') & (aw_spain['Destination Airport Name']!='RAILWAY STATION') &
	(aw_spain['Destination Airport Name']!='MIDI RAILWAY STATION') & (aw_spain['Destination Airport Name']!='HBF RAILWAY STATION') &
	(aw_spain['Destination Airport Name']!='VALENCIA RAILWAY STATION') & (aw_spain['Destination Airport Name']!='CHAMARTIN RAILWAY STATION') &
	(aw_spain['Destination Airport Name']!='EK BUS STATION') & (aw_spain['Destination Airport Name']!='TRAVEL MALL EY BUS STATION') &
	(aw_spain['Destination Airport Name']!='OTTAWA RAIL STATION') & (aw_spain['Destination Airport']!='RZG') & (aw_spain['Destination Airport']!='BQC') &
	(aw_spain['Destination Airport']!='XJN') & (aw_spain['Destination Airport Name']!='ARNHEM BUS STATION') &
	(aw_spain['Destination Airport Name']!='LEON RAILWAY STATION') & (aw_spain['Destination Airport']!='XJR') &
	(aw_spain['Destination Airport Name']!='LINZ HAUPTBAHNHOF RAIL STATION') & (aw_spain['Destination Airport']!='GBX') &
	(aw_spain['Destination Airport']!='EMU') & (aw_spain['Destination Airport Name']!='SANTA JUSTA RAILWAY STATION') &
	(aw_spain['Destination Airport Name']!='MAASTRICHT RAILWAY SERVICE') & (aw_spain['Destination Airport']!='LZB') &
	(aw_spain['Destination Airport Name']!='GARE DE RENNES RAIL STATION') & (aw_spain['Destination Airport']!='PFE') &
	(aw_spain['Destination Airport Name']!='ABU DHABI EK BUS STATION') & (aw_spain['Destination Airport']!='EXC') &
	(aw_spain['Destination Airport Name']!='ST-PIERRE-CORPS RAIL STATION') & (aw_spain['Destination Airport Name']!='KINGSTON RAIL STATION') &
	(aw_spain['Destination Airport Name']!='CENTRAL RAILWAY STATION') & (aw_spain['Destination Airport Name']!='NOTTINGHAM BUS STATION') &
	(aw_spain['Destination Airport Name']!='CADIZ RAILWAY STATION') & (aw_spain['Destination Airport Name']!='KARLSRUHE HBF RAILWAY STATION') &
	(aw_spain['Destination Airport']!='QYE') & (aw_spain['Destination Airport Name']!='LE MANS RAIL STATION') &
	(aw_spain['Destination Airport Name']!='NANTES RAILWAY SERVICE') & (aw_spain['Destination Airport Name']!='LORRAINE TGV RAIL STATION') &
	(aw_spain['Destination Airport']!='FBV') & (aw_spain['Destination Airport']!='ALV') & (aw_spain['Destination Airport Name']!='PART-DIEU RAILWAY STATION') &
	(aw_spain['Destination Airport Name']!='CENTRAAL RAILWAY STATION') & (aw_spain['Destination Airport']!='XZN') &
	(aw_spain['Destination Airport Name']!='SAINT PANCRAS INTL RAIL STATION') & (aw_spain['Destination Airport Name']!='TARRAGONA/CAMP RAIL STATION') &
	(aw_spain['Destination Airport Name']!='BUS STA}TION') & (aw_spain['Destination Airport Name']!='SABTCO BUS STATION') &
	(aw_spain['Destination Airport Name']!='OXFORD RAILWAY STATION') & (aw_spain['Destination Airport']!='CLG') &
	(aw_spain['Destination Airport']!='CSL') & (aw_spain['Destination Airport']!='VTI') & (aw_spain['Destination Airport']!='SFU') &
	(aw_spain['Destination Airport']!='XRJ') & (aw_spain['Destination Airport']!='LCZ') & (aw_spain['Destination Airport']!='FDD') &
	(aw_spain['Destination Airport']!='MIL') & (aw_spain['Destination Airport Name']!='AMSTERDAM RAILWAY SERVICE') & (aw_spain['Destination Airport']!='UER') &
	(aw_spain['Destination Airport']!='FES') & (aw_spain['Destination Airport']!='ZEG')
	& (aw_spain['Destination Airport']!='XHJ')
	& (aw_spain['Destination Airport']!='NCM')
	& (aw_spain['Destination Airport']!='XJU')
	& (aw_spain['Destination Airport']!='HSK')
	& (aw_spain['Destination Airport']!='PTE')
	& (aw_spain['Destination Airport']!='QYP')
	& (aw_spain['Destination Airport']!='JJU')
	& (aw_spain['Destination Airport']!='XJJ')
	& (aw_spain['Destination Airport']!='GJB')
	& (aw_spain['Destination Airport']!='QCE')
	& (aw_spain['Destination Airport']!='NAD')
	& (aw_spain['Destination Airport']!='DHS')
	& (aw_spain['Destination Airport']!='XIU')
	& (aw_spain['Destination Airport']!='OCD')
	& (aw_spain['Destination Airport']!='LGE')
	& (aw_spain['Destination Airport']!='NIZ')
	& (aw_spain['Destination Airport']!='NKK')
	& (aw_spain['Destination Airport']!='QWU')
	& (aw_spain['Destination Airport']!='TGL')
	& (aw_spain['Destination Airport']!='CLR')
	& (aw_spain['Destination Airport']!='GGY')
	& (aw_spain['Destination Airport']!='YJG')
	& (aw_spain['Destination Airport Name']!='INNSBRUCK RAILWAY STATION')
	& (aw_spain['Destination Airport Name']!='ST. THOMAS RAILWAY STATION')
	#Germany exclusions
	& (~aw_spain['Destination Airport'].isin(['QYG','RMS','QFP','ZCW']))
	]

	aw_spain['orig'] = aw_spain.apply(lambda row: origin_before(row), axis=1)
	aw_spain['dest'] = aw_spain.apply(lambda row: destination_after(row), axis=1)
	# print(aw_spain)
	aw_spain.to_csv('../output/data_analysis/cs11.csv', index=False)
# In[ ]:

def demand():
	aw_spain = pd.read_csv('../output/data_analysis/cs11.csv')
	print(aw_spain)

	aw_spain_demand = aw_spain.groupby(['orig', 'dest'])['Passengers'].sum().reset_index().sort_values(by='Passengers', ascending=False)
	aw_spain_demand = aw_spain_demand[aw_spain_demand.Passengers>=100]
	aw_spain_demand['Passengers'] = aw_spain_demand['Passengers']/30
	#aw_spain_demand.to_csv('../data/output/data_analysis/aw_demand.csv', index=False)


	# In[ ]:


	df_iata_icao = pd.read_csv('../data/airports/IATA_ICAO_Airport_codes_v0.3.csv')
	# Create dictionary with IATA as keys and ICAO as values
	iata_icao_dict = dict(zip(df_iata_icao['IATA'], df_iata_icao['ICAO']))

	aw_spain_demand['origin'] = aw_spain_demand['orig'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')
	aw_spain_demand['destination'] = aw_spain_demand['dest'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')


	# In[ ]:


	# aw_spain[aw_spain['Origin Airport'].isin(list(aw_spain_demand[aw_spain_demand['origin']=='NO'][['Origin Airport']].drop_duplicates()['Origin Airport']))][['Origin Airport', 'Origin Airport Name', 'Origin City Name', 'Origin Country Name']].drop_duplicates()


	# In[ ]:


	# aw_spain[aw_spain['Destination Airport'].isin(list(aw_spain_demand[aw_spain_demand['destination']=='NO'][['Destination Airport']].drop_duplicates()['Destination Airport']))][['Destination Airport', 'Destination Airport Name', 'Destination City Name', 'Destination Country Name']].drop_duplicates()


	# In[ ]:


	#Regroup by origin destination as come IATA code might be now the same (generic ariports)
	aw_spain_demand_icao = aw_spain_demand.groupby(['origin', 'destination'])['Passengers'].sum().reset_index().sort_values(by='Passengers', ascending=False)
	# aw_spain_demand_icao.to_csv('../output/data_analysis/demand_AW_icao.csv', index=False)


	# In[ ]:


	aw_spain_demand_icao.Passengers.sum()


	# In[ ]:


	aw_spain['origin'] = aw_spain['orig'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')
	aw_spain['destination'] = aw_spain['dest'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')
	aw_spain['Connecting Airport1_icao'] = aw_spain['Connecting Airport1'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')
	aw_spain['Connecting Airport2_icao'] = aw_spain['Connecting Airport2'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')
	aw_spain['Connecting Airport3_icao'] = aw_spain['Connecting Airport3'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')


	# In[ ]:


	aw_spain.to_csv('../output/data_analysis/cs11_aw_itineraries.csv')


	# In[ ]:





	# In[ ]:





	# In[ ]:


	# Create demand
	#df_regions_access = pd.read_csv('../data/data_analysis/v=0.1/infrastructure/regions_access_all.csv')

	#df_regions_access = df_regions_access[(df_regions_access.layer=='air') & (df_regions_access.access_type=='access')]
	#df_regions_access['airport_icao'] = df_regions_access['station'].apply(lambda x: iata_icao_dict.get(x,x))
	#df_regions_access.sort_values(by='airport_icao').to_csv('../data/data_analysis/v=0.1/airports_regions.csv')
	# Manually edit the ariport and regions


	# In[ ]:


	# df_airport_regions1 = pd.read_csv('../output/data_analysis/airport_regions_DE.csv')
	df_airport_regions = pd.read_csv('../output/data_analysis/airport_regions_ES__.csv')
	# df_airport_regions = pd.concat([df_airport_regions1,df_airport_regions2])

	station_region_dict = dict(zip(df_airport_regions['airport_icao'], df_airport_regions['region']))


	# In[ ]:


	aw_spain_demand['origin_ICAO'] = aw_spain_demand['orig'].apply(lambda x: iata_icao_dict.get(x,x) if x!='BJS' else 'PEK')
	aw_spain_demand['destination_ICAO'] = aw_spain_demand['dest'].apply(lambda x: iata_icao_dict.get(x,x) if x!='BJS' else 'PEK')
	aw_spain_demand['origin'] = aw_spain_demand['origin_ICAO'].apply(lambda x: station_region_dict.get(x,x))
	aw_spain_demand['destination'] = aw_spain_demand['destination_ICAO'].apply(lambda x: station_region_dict.get(x,x))
	aw_spain_demand = aw_spain_demand.groupby(['origin', 'destination'])['Passengers'].sum().reset_index().sort_values(by='Passengers', ascending=False)

	aw_spain_demand['a0'] = 'archetype_0'
	aw_spain_demand['a1'] = 'archetype_1'
	aw_spain_demand['a2'] = 'archetype_2'
	aw_spain_demand['a3'] = 'archetype_3'
	aw_spain_demand['a4'] = 'archetype_4'
	aw_spain_demand['a5'] = 'archetype_5'

	aw_spain_demand['a0v'] = aw_spain_demand['Passengers']*0.8
	aw_spain_demand['a1v'] = aw_spain_demand['Passengers']*0.04
	aw_spain_demand['a2v'] = aw_spain_demand['Passengers']*0.04
	aw_spain_demand['a3v'] = aw_spain_demand['Passengers']*0.04
	aw_spain_demand['a4v'] = aw_spain_demand['Passengers']*0.04
	aw_spain_demand['a5v'] = aw_spain_demand['Passengers']*0.04

	dfs = []
	dfs.append(aw_spain_demand[['origin','destination','a0','a0v']].rename(columns={'a0':'archetype','a0v':'trips'}))
	dfs.append(aw_spain_demand[['origin','destination','a1','a1v']].rename(columns={'a1':'archetype','a1v':'trips'}))
	dfs.append(aw_spain_demand[['origin','destination','a2','a2v']].rename(columns={'a2':'archetype','a2v':'trips'}))
	dfs.append(aw_spain_demand[['origin','destination','a3','a3v']].rename(columns={'a3':'archetype','a3v':'trips'}))
	dfs.append(aw_spain_demand[['origin','destination','a4','a4v']].rename(columns={'a4':'archetype','a4v':'trips'}))
	dfs.append(aw_spain_demand[['origin','destination','a5','a5v']].rename(columns={'a5':'archetype','a5v':'trips'}))
	# print(dfs)
	intra = pd.read_csv('/home/michal/Documents/westminster/multimodx/data/strategic/data/CS11/v=0.2/demand/demand_ES_MD_intra_v0.4.csv')
	dfs.append(intra)
	awd = pd.concat(dfs)

	awd.to_csv('../output/data_analysis/demand_AW_cs11_.csv', index=False)
	#aw_spain_demand[aw_spain_demand.Passengers>=100][['origin','destination','Passengers']].to_csv('../data/data_analysis/v=0.1/demand_from_AW_full_at_least_100pax.csv', index=False)

def add_path_f(row):
	seq = [row['Connecting Airport1_icao'],row['Connecting Airport2_icao'],row['Connecting Airport3_icao']]
	seq = [x for x in seq if x != 'NO']
	start = 0
	end = len(seq)
	if row['origin'] in seq:
		start = seq.index(row['origin'])+1
	if row['destination'] in seq:
		end = seq.index(row['destination'])

	return str([row['origin']] + seq[start:end] + [row['destination']])

	return 0
def add_path():
	aw_spain = pd.read_csv('../output/data_analysis/cs11_aw_itineraries.csv')
	aw_spain['path'] = aw_spain.apply(lambda row: add_path_f(row), axis=1)
	aw_spain.to_csv('../output/data_analysis/cs11_aw_itineraries_.csv')

def compare():
	df_airport_regions = pd.read_csv('../output/data_analysis/airport_regions_ES__.csv')


	station_region_dict = dict(zip(df_airport_regions['airport_icao'], df_airport_regions['region']))

	aw_spain = pd.read_csv('../output/data_analysis/cs11_aw_itineraries_.csv')
	demand = pd.read_csv('../output/data_analysis/demand_AW_cs11.csv')
	aw_spain['origin_nuts'] = aw_spain['origin'].apply(lambda x: station_region_dict.get(x,x))
	aw_spain['destination_nuts'] = aw_spain['destination'].apply(lambda x: station_region_dict.get(x,x))
	aw_spain.to_csv('../output/data_analysis/cs11_aw_itineraries__.csv')
	its = pd.read_csv('/home/michal/Documents/westminster/multimodx/data/strategic/data/CS11/v=0.2/output/processed_cs11.pp00.so00/paths_itineraries/potential_paths_0.csv')
	# its2 = its.merge(aw_spain[['origin_nuts','destination_nuts','path']], how='left', left_on=['origin','destination'], right_on=['origin_nuts','destination_nuts'],indicator=True)
	its2 = its.copy()
	its2['indicator'] = its2['path'].apply(lambda row: True if row in aw_spain['path'].values else False)
	its2.to_csv('../output/data_analysis/cs11_aw_itineraries2.csv')
	grouped = its2.groupby(['origin', 'destination'])['indicator'].sum().reset_index()
	grouped = grouped.merge(demand,on=['origin','destination'])
	grouped.to_csv('../output/data_analysis/xxx.csv')

	its3 = its2[its2['path'].isin(aw_spain['path'].values)]
	print(its3)

def analyse_aw():
	aw_spain = pd.read_csv('../output/data_analysis/cs11.csv')

	aw_spain['Passengers'] = aw_spain['Passengers']/30
	all_pax = aw_spain['Passengers'].sum()
	print('all_pax',all_pax)
	print('Origin/Destination Spain ',aw_spain[((aw_spain['Destination Country Name']=='SPAIN') | (aw_spain['Origin Country Name']=='SPAIN'))]['Passengers'].sum())
	print('Connecting in Spain ',aw_spain[~((aw_spain['Destination Country Name']=='SPAIN') | (aw_spain['Origin Country Name']=='SPAIN'))]['Passengers'].sum())

	aw_spain_demand = aw_spain.groupby(['orig', 'dest'])['Passengers'].sum().reset_index().sort_values(by='Passengers', ascending=False)
	df_iata_icao = pd.read_csv('../data/airports/IATA_ICAO_Airport_codes_v0.3.csv')
	# Create dictionary with IATA as keys and ICAO as values
	iata_icao_dict = dict(zip(df_iata_icao['IATA'], df_iata_icao['ICAO']))



	aw_spain['connections_nr'] = aw_spain.apply(lambda row: connections_outside_spain(row), axis=1)
	aw_spain['pax_perc'] = aw_spain['Passengers']/aw_spain['Passengers'].sum()
	# aw_spain.to_csv('../output/data_analysis/xxx_.csv')
	grouped = aw_spain.groupby('connections_nr')['pax_perc'].sum()
	print(grouped)

	grouped = aw_spain.groupby('Destination City Name')['pax_perc'].sum().reset_index().sort_values(by='pax_perc', ascending=False)
	print(grouped)

	cities = aw_spain.groupby('Destination City Name')['Destination Airport Name'].nunique().reset_index().sort_values(by='Destination Airport Name', ascending=False)
	print(cities[cities['Destination Airport Name']>1])

	aw_spain['hub'] = aw_spain.apply(lambda row: hub_outside_spain(row), axis=1)
	aw_spain['hub'] = aw_spain['hub'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')

	pax_hub =  aw_spain.dropna(subset=['hub']).groupby('hub')['Passengers'].sum().sum()
	aw_spain['pax_perc_hub'] = aw_spain['Passengers']/pax_hub
	grouped = aw_spain.groupby('hub')[['pax_perc_hub','Passengers']].sum().reset_index().sort_values(by='pax_perc_hub', ascending=False)
	print(grouped)
	print(grouped['pax_perc_hub'].sum())
	print('pax_hub',pax_hub)
	grouped = aw_spain.groupby(['hub','Destination Region Name'])[['Passengers']].sum().reset_index().sort_values(by='Passengers', ascending=False)
	print(grouped)

	arriving = aw_spain[(aw_spain['Destination Country Name']=='SPAIN')]
	departing = aw_spain[(aw_spain['Origin Country Name']=='SPAIN')]
	grouped = arriving.groupby('Origin Region Name')['pax_perc'].sum().reset_index().sort_values(by='pax_perc', ascending=False)
	print(grouped)
	grouped = departing.groupby('Destination Region Name')['pax_perc'].sum().reset_index().sort_values(by='pax_perc', ascending=False)
	print(grouped)

	grouped = arriving.groupby('Origin City Name')['pax_perc'].sum().reset_index().sort_values(by='pax_perc', ascending=False)
	grouped['cumsum'] = grouped['pax_perc'].cumsum()
	print(grouped[grouped['cumsum']<=(0.5*0.8)])
	grouped = departing.groupby('Destination City Name')['pax_perc'].sum().reset_index().sort_values(by='pax_perc', ascending=False)
	grouped['cumsum'] = grouped['pax_perc'].cumsum()
	print(grouped[grouped['cumsum']<=(0.5*0.8)])

	departing = departing[departing['hub']!='NO']
	region_sum = departing.groupby(['Destination Region Name'])['Passengers'].sum()
	departing['perc'] = departing.apply(lambda x: x['Passengers']/region_sum.loc[x['Destination Region Name']] if not pd.isna(x['Destination Region Name']) else 0 ,axis=1)
	grouped = departing.groupby(['Destination Region Name','hub'])[['Passengers','perc']].sum().reset_index()
	grouped.to_csv('../output/data_analysis/xxx.csv')
	print(grouped)

def arriving_departing_flows():
	aw_spain = pd.read_csv('../output/data_analysis/cs11.csv')

	aw_spain['Passengers'] = aw_spain['Passengers']/30

	df_iata_icao = pd.read_csv('../data/airports/IATA_ICAO_Airport_codes_v0.3.csv')
	# Create dictionary with IATA as keys and ICAO as values
	iata_icao_dict = dict(zip(df_iata_icao['IATA'], df_iata_icao['ICAO']))

	aw_spain['origin'] = aw_spain['orig'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')
	aw_spain['destination'] = aw_spain['dest'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')
	aw_spain['Connecting Airport1_icao'] = aw_spain['Connecting Airport1'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')
	aw_spain['Connecting Airport2_icao'] = aw_spain['Connecting Airport2'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')
	aw_spain['Connecting Airport3_icao'] = aw_spain['Connecting Airport3'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')

	aw_spain['Connecting Airport1'] = aw_spain['Connecting Airport1'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')
	aw_spain['Connecting Airport2'] = aw_spain['Connecting Airport2'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')
	aw_spain['Connecting Airport3'] = aw_spain['Connecting Airport3'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')

	aw_spain['path'] = aw_spain.apply(lambda row: add_path_f(row), axis=1)

	arriving = aw_spain[(aw_spain['Destination Country Name']=='SPAIN')]
	departing = aw_spain[(aw_spain['Origin Country Name']=='SPAIN')]

	arriving['entry'] = arriving.apply(lambda row: arriving_first_airport(row), axis=1)
	departing['exit'] = departing.apply(lambda row: departing_last_airport(row), axis=1)
	# arriving['entry'] = arriving['entry'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')
	# departing['exit'] = departing['exit'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')
	# arriving.to_csv('../output/data_analysis/xxx.csv')

	grouped_arriving = arriving.groupby('entry')['Passengers'].sum()
	grouped_departing = departing.groupby('exit')['Passengers'].sum()
	print(grouped_arriving,grouped_departing)

	print(arriving[['entry']])
	arriving['perc'] = arriving.apply(lambda x: x['Passengers']/grouped_arriving.loc[x['entry']] if not pd.isnull(grouped_arriving.loc[x['entry']]) else 0 ,axis=1)
	grouped = arriving.groupby(['entry','path'])['perc'].sum().reset_index()
	# print(grouped[grouped['entry']=='LEMD'])
	grouped.to_csv('../output/data_analysis/arriving_paths.csv')
	departing['perc'] = departing.apply(lambda x: x['Passengers']/grouped_departing.loc[x['exit']] if not pd.isnull(grouped_departing.loc[x['exit']]) else 0 ,axis=1)
	grouped = departing.groupby(['exit','path'])['perc'].sum().reset_index()
	# print(grouped[grouped['entry']=='LEMD'])
	grouped.to_csv('../output/data_analysis/departing_paths.csv')

	arriving_pax = arriving['Passengers'].sum()
	print(arriving[arriving['orig']!=arriving['Origin Airport']]['Passengers'].sum()/arriving_pax)
	departing_pax = departing['Passengers'].sum()
	print(departing[departing['dest']!=departing['Destination Airport']]['Passengers'].sum()/departing_pax)

def demand_cs11():
	aw_spain = pd.read_csv('../output/data_analysis/cs11.csv')

	aw_spain['Passengers'] = aw_spain['Passengers']/30
	aw_spain = aw_spain[aw_spain['Passengers']>10]

	df_iata_icao = pd.read_csv('../data/airports/IATA_ICAO_Airport_codes_v0.3.csv')
	# Create dictionary with IATA as keys and ICAO as values
	iata_icao_dict = dict(zip(df_iata_icao['IATA'], df_iata_icao['ICAO']))
	iata_icao_dict['BJS'] = 'ZBAA'

	aw_spain['origin'] = aw_spain['orig'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')
	aw_spain['destination'] = aw_spain['dest'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')
	aw_spain['Connecting Airport1_icao'] = aw_spain['Connecting Airport1'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')
	aw_spain['Connecting Airport2_icao'] = aw_spain['Connecting Airport2'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')
	aw_spain['Connecting Airport3_icao'] = aw_spain['Connecting Airport3'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')

	aw_spain['Connecting Airport1'] = aw_spain['Connecting Airport1'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')
	aw_spain['Connecting Airport2'] = aw_spain['Connecting Airport2'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')
	aw_spain['Connecting Airport3'] = aw_spain['Connecting Airport3'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')
	aw_spain['origin_airport'] = aw_spain['Origin Airport'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')
	aw_spain['destination_airport'] = aw_spain['Destination Airport'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')

	cities = aw_spain[['Destination Airport','Destination City Name','Destination Country Name']].copy().drop_duplicates()
	city_count = cities.groupby(['Destination City Name']).count().reset_index().rename(columns={'Destination Airport':'count'})
	cities = cities.merge(city_count,how='left',on='Destination City Name')
	cities = cities[cities['count']>1]


	cities_dict = dict(zip(cities['Destination Airport'],cities['Destination City Name']))
	cities_dict = {iata_icao_dict.get(x,'NO'):cities_dict[x] for x in cities_dict if x in iata_icao_dict}
	cities_dict = {x:cities_dict[x] for x in cities_dict if x[:2] not in ('LE','GC')}
	print(cities_dict)

	# aw_spain = aw_spain.replace(cities_dict)
	aw_spain['path'] = aw_spain.apply(lambda row: add_path_f(row), axis=1)
	aw_spain['origin_city'] = aw_spain['origin'].replace(cities_dict)
	aw_spain['destination_city'] = aw_spain['destination'].replace(cities_dict)

	costs = aw_spain.groupby(['origin_airport', 'destination_airport'])
	costs = costs.agg({
		'Avg. Total Fare(USD)': 'mean',  # Compute mean for nservices
		'Distance (km)': 'mean',  # Take the first journey_type
		# 'Passengers':'sum',
		#'options_in_cluster': lambda x: list(set(x))  # Keep unique options in cluster
	}).reset_index()
	# costs['origin_airport'] = costs['origin_airport'].replace(cities_dict)
	# costs['destination_airport'] = costs['destination_airport'].replace(cities_dict)
	costs.to_csv('../output/data_analysis/costs.csv')
	grouped = aw_spain.groupby(['origin', 'destination','origin_city', 'destination_city', 'path'])
	kpis = ['total_avg_travel_time', 'total_avg_cost', 'total_avg_emissions', 'total_avg_waiting_time', 'nservices']
	df_grouped = grouped.agg({
		# 'Avg. Total Fare(USD)': 'mean',  # Compute mean for nservices
		# 'Distance (km)': 'mean',  # Take the first journey_type
		'Passengers':'sum',
		#'options_in_cluster': lambda x: list(set(x))  # Keep unique options in cluster
	}).reset_index()
	df_grouped = df_grouped.merge(costs,how='left',left_on=['origin','destination'],right_on=['origin_airport','destination_airport'])

	df_grouped.to_csv('../output/data_analysis/aw_paths_costs.csv')
	arriving = aw_spain[(aw_spain['Destination Country Name']=='SPAIN')]
	departing = aw_spain[(aw_spain['Origin Country Name']=='SPAIN')]

	arriving['entry'] = arriving.apply(lambda row: arriving_first_airport(row), axis=1)
	departing['exit'] = departing.apply(lambda row: departing_last_airport(row), axis=1)

	grouped_arriving = arriving.groupby(['origin','entry'])['Passengers'].sum().reset_index()
	grouped_departing = departing.groupby(['destination','exit'])['Passengers'].sum().reset_index()
	coeff_arriving = pd.read_csv('/home/michal/Documents/westminster/multimodx/data/strategic/MultiModX/output/data_analysis/coefficients_incoming_trips_to_spain_v0.3.csv')
	coeff_departing = pd.read_csv('/home/michal/Documents/westminster/multimodx/data/strategic/MultiModX/output/data_analysis/coefficients_outgoing_trips_from_spain_v0.2.csv')
	demand_arriving = grouped_arriving.merge(coeff_arriving,how='left',left_on='entry',right_on='entry_point')
	demand_departing = grouped_departing.merge(coeff_departing,how='left',left_on='exit',right_on='exit_point')
	print(demand_arriving[demand_arriving['entry']=='LECO'])

	# arriving['origin'] = arriving['origin'].replace({})
	# cities = aw_spain.groupby('Destination City Name')['Destination Airport'].unique()
	# print(cities.reset_index())

def pax_distribution(coeff):

	probabilities = {}
	options = {}
	if 'last_airport' in coeff.columns:
		airport = 'last_airport'
	if 'first_airport' in coeff.columns:
		airport = 'first_airport'

	for airport_name in coeff[airport].unique():
		probabilities[airport_name] = coeff[coeff[airport]==airport_name]['coeff'].to_list()
		options[airport_name] = coeff[coeff[airport]==airport_name]['option'].to_list()
	return probabilities,options

def distribute_pax(num_people,probabilities,categories):
	np.random.seed(42)
	# Generate a single sample from the multinomial distribution
	counts = np.random.multinomial(n=num_people, pvals=probabilities)

	# Map the counts back to the category names
	distribution = dict(zip(categories, counts))
	return distribution

def pax_row(row,probabilities,categories,airport='last_airport'):
	# print(row)
	if row[airport] not in probabilities:
		return row['Passengers']
	# print('options',categories['LEBB'])
	distribution = distribute_pax(row['Passengers'],probabilities[row[airport]],categories[row[airport]])
	# print(distribution)
	return distribution[row['option']]
def demand_cs13():
	aw_spain = pd.read_csv('../output/data_analysis/cs11.csv')
	df_airport_regions = pd.read_csv('../output/data_analysis/airport_regions_ES__.csv')
	# df_airport_regions = pd.concat([df_airport_regions1,df_airport_regions2])

	df_iata_icao = pd.read_csv('../data/airports/IATA_ICAO_Airport_codes_v0.3.csv')
	# Create dictionary with IATA as keys and ICAO as values
	iata_icao_dict = dict(zip(df_iata_icao['IATA'], df_iata_icao['ICAO']))
	iata_icao_dict['BJS'] = 'ZBAA'

	station_region_dict = dict(zip(df_airport_regions['airport_icao'], df_airport_regions['region']))
	cities = aw_spain[['Destination Airport','Destination City Name','Destination Country Name']].copy().drop_duplicates()
	city_count = cities.groupby(['Destination City Name']).count().reset_index().rename(columns={'Destination Airport':'count'})
	cities = cities.merge(city_count,how='left',on='Destination City Name')
	cities = cities[cities['count']>1]


	cities_dict = dict(zip(cities['Destination Airport'],cities['Destination City Name']))
	cities_dict = {iata_icao_dict.get(x,'NO'):cities_dict[x] for x in cities_dict if x in iata_icao_dict}
	cities_dict = {x:cities_dict[x] for x in cities_dict if x[:2] not in ('LE','GC')}
	for c in cities_dict:
		print(cities_dict[c],',',c)

	aw_spain['Passengers'] = aw_spain['Passengers']/30
	aw_spain = aw_spain[aw_spain['Passengers']>10]
	print('pax', aw_spain['Passengers'].sum())
	df_iata_icao = pd.read_csv('../data/airports/IATA_ICAO_Airport_codes_v0.3.csv')
	# Create dictionary with IATA as keys and ICAO as values
	iata_icao_dict = dict(zip(df_iata_icao['IATA'], df_iata_icao['ICAO']))
	iata_icao_dict['BJS'] = 'ZBAA'

	aw_spain['origin'] = aw_spain['orig'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')
	aw_spain['destination'] = aw_spain['dest'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')
	aw_spain['Connecting Airport1_icao'] = aw_spain['Connecting Airport1'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')
	aw_spain['Connecting Airport2_icao'] = aw_spain['Connecting Airport2'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')
	aw_spain['Connecting Airport3_icao'] = aw_spain['Connecting Airport3'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')

	aw_spain['Connecting Airport1'] = aw_spain['Connecting Airport1'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')
	aw_spain['Connecting Airport2'] = aw_spain['Connecting Airport2'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')
	aw_spain['Connecting Airport3'] = aw_spain['Connecting Airport3'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')
	# aw_spain['origin_airport'] = aw_spain['Origin Airport'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')
	# aw_spain['destination_airport'] = aw_spain['Destination Airport'].apply(lambda x: iata_icao_dict.get(x,'NO') if x!='BJS' else 'PEK')



	aw_spain['path'] = aw_spain.apply(lambda row: add_path_f(row), axis=1)
	aw_spain['origin_city'] = aw_spain['origin'].replace(cities_dict)
	aw_spain['destination_city'] = aw_spain['destination'].replace(cities_dict)

	arriving = aw_spain[(aw_spain['Destination Country Name']=='SPAIN')]
	departing = aw_spain[(aw_spain['Origin Country Name']=='SPAIN')]
	print('pax', arriving['Passengers'].sum(), departing['Passengers'].sum())
	coeff_arriving = pd.read_csv('/home/michal/Documents/westminster/multimodx/data/strategic/data/CS13/v=0.2/demand/coefficients_incoming_trips_to_spain_v0.5.csv')
	coeff_departing = pd.read_csv('/home/michal/Documents/westminster/multimodx/data/strategic/data/CS13/v=0.2/demand/coefficients_outgoing_trips_from_spain_v0.5.csv')
	coeff_arriving['option'] = coeff_arriving.index
	coeff_departing['option'] = coeff_departing.index
	coeff_arriving = coeff_arriving.rename(columns={'destination':'destination_nuts'})
	coeff_departing = coeff_departing.rename(columns={'origin':'origin_nuts'})
	# print('coeff_arriving',coeff_arriving)
	demand_arriving = arriving.merge(coeff_arriving,how='left',left_on='destination',right_on='last_airport')
	demand_departing = departing.merge(coeff_departing,how='left',left_on='origin',right_on='first_airport')
	# print(demand_arriving[demand_arriving['last_airport']=='LECO'])
	prob_arriving,options_arriving = pax_distribution(coeff_arriving)
	prob_departing,options_departing = pax_distribution(coeff_departing)
	# print('prob_arriving',prob_arriving,options_arriving)
	# pax_row(prob_arriving,options_arriving)
	print('demand_arriving',demand_arriving[['node_sequence_ground']])
	demand_arriving['pax'] = demand_arriving.apply(lambda row: pax_row(row,prob_arriving,options_arriving,airport='last_airport'), axis=1)
	demand_arriving['path'] = demand_arriving.apply(lambda row: row['path'][:-1]+', '+row['node_sequence_ground'][1:] if (row['node_sequence_ground']!='egress' and not pd.isnull(row['node_sequence_ground'])) else row['path'], axis=1)
	# print(demand_arriving[demand_arriving['last_airport']=='LECO'][['origin', 'destination']])
	# demand_arriving.to_csv('../output/data_analysis/xxx.csv')

	demand_departing['pax'] = demand_departing.apply(lambda row: pax_row(row,prob_departing,options_departing,airport='first_airport'), axis=1)
	demand_departing['path'] = demand_departing.apply(lambda row: row['node_sequence_ground'][:-1]+', '+row['path'][1:] if (row['node_sequence_ground']!='access' and not pd.isnull(row['node_sequence_ground'])) else row['path'], axis=1)

	demand_arriving['destination_nuts'] = demand_arriving.apply(lambda row: station_region_dict.get(row['destination'],row['destination']) if pd.isna(row['destination_nuts']) else row['destination_nuts'], axis=1)
	demand_arriving = demand_arriving.drop(columns=['destination'])
	demand_arriving = demand_arriving.rename(columns={'destination_nuts':'destination'})

	print(demand_arriving[['origin', 'destination']])
	demand_departing['origin_nuts'] = demand_departing.apply(lambda row: station_region_dict.get(row['origin'],row['origin']) if pd.isna(row['origin_nuts']) else row['origin_nuts'], axis=1)
	demand_departing = demand_departing.drop(columns=['origin'])
	demand_departing = demand_departing.rename(columns={'origin_nuts':'origin'})
	demand = pd.concat([demand_arriving,demand_departing])
	demand['origin'] = demand['origin'].replace(cities_dict)
	demand['destination'] = demand['destination'].replace(cities_dict)
	demand = demand[demand['pax']>0]
	demand.to_csv('../output/data_analysis/paths_cs13.csv')
	grouped = demand.groupby(['origin', 'destination'])['pax'].sum().reset_index()
	# grouped = grouped.rename(columns={'origin_city':'origin','destination_city':'destination'})
	print(grouped)
	grouped = grouped.rename(columns={'pax':'trips'})
	grouped['archetype'] = 'archetype_0'
	grouped['date'] = 20220923
	intra = pd.read_csv('/home/michal/Documents/westminster/multimodx/data/strategic/data/CS13/v=0.2/demand/demand_ES_MD_intra_v0.4.csv')
	demand_total = pd.concat([grouped,intra])
	demand_total.to_csv('../output/data_analysis/demand_cs13.csv')

if __name__ == '__main__':
	print ('aw data... ')
	# read_aw()
	# demand()
	# add_path()
	# compare()
	# analyse_aw()
	# arriving_departing_flows()
	# demand_cs11()
	demand_cs13()
