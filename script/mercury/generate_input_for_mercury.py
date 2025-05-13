import pandas as pd
from pathlib import Path
import shutil
import numpy as np

def find_similar_fp(fp_pool,fp_pool_point_m,missing,schedules,path_to_mercury_input,airlines_static):

	fps = fp_pool.merge(fp_pool_point_m[fp_pool_point_m['name']=='landing'][['fp_pool_id','time_min']], left_on=['id'], right_on=['fp_pool_id'], how='left')
	fps['domestic'] = fps.apply(lambda row: ((row['icao_orig'][:2] in ['GC','LE']) or (row['icao_dest']=='GEML')) and ((row['icao_dest'][:2] in ['GC','LE']) or (row['icao_dest']=='GEML')), axis=1)

	seats = schedules[['bada_code_ac_model','max_seats']].groupby('bada_code_ac_model').max().to_dict()['max_seats']
	seats['B789']=333
	print('seats',seats)


	compatible_ac = {ac:[] for ac in seats}
	for ac in seats:
		for ac2 in seats:
			if seats[ac2]>seats[ac]:
				compatible_ac[ac].append(ac2)
	for ac in compatible_ac:
		compatible_ac[ac].sort(key= lambda x:seats[x])
	print('compatible_ac',compatible_ac)
	selecteds = []
	for i,row in missing.iterrows():
		#print(row)
		if row['aircraft_type']=='B789':
			row['bada_code_ac_model']='B789'
			print('xb789')
		#print(fps)
		t = (row['sibt']-row['sobt']).seconds/60.0
		fps['gcdistance'] = row['gcdistance']
		fps['diff'] = (1.852*fps['fp_distance_nm'] - row['gcdistance'])
		fps['perc'] = abs(fps['diff']/fps['gcdistance'])
		fps['ref_time'] = t
		similar2 = fps[(fps['icao_orig']==row['origin'])&(fps['icao_dest']==row['destination'])]

		similar = fps[(fps['bada_code_ac_model']==row['bada_code_ac_model'])&(abs(fps['perc'])<0.1)&(fps['domestic']==True)&(fps['ref_time']>fps['time_min'])]
		if len(similar)<1:
			similar = fps[(fps['bada_code_ac_model']==row['bada_code_ac_model'])&(abs(fps['perc'])<0.1)&(fps['ref_time']>fps['time_min'])]
		if len(similar)<1:
			similar = fps[(fps['bada_code_ac_model'].isin(compatible_ac[row['bada_code_ac_model']]))&(abs(fps['perc'])<0.1)&(fps['ref_time']>fps['time_min'])]
		#print(similar)
		similar = similar.sort_values(by=['perc'])

		print(row['bada_code_ac_model'],row['origin'],row['destination'],len(similar2))
		#print('similar',similar.head())
		selected = similar.iloc[[0]].copy()

		#case1 change origin/destination of fp
		selected['icao_orig'] = row['origin']
		selected['icao_dest'] = row['destination']
		selected['trajectory_version'] = 3
		#print(selected[['bada_code_ac_model']],bada[bada['bada_code_ac_model'].isin(selected['bada_code_ac_model'])].iloc[0]['ac_icao'])
		#print(pd.DataFrame(selected))
		#case2 change aircraft_type in the schedule
		schedules.loc[schedules['nid']==row['nid'],['aircraft_type']] = bada[bada['bada_code_ac_model'].isin(selected['bada_code_ac_model'])]['ac_icao'].iloc[0]
		selecteds.append(selected)

	#schedules['nid'] = schedules.index
	#replace airlines which are not in airlines_static, e.g. some regional airlines
	schedules['airline'] = schedules['airline'].replace('UGA','IBB')
	schedules['airline'] = schedules['airline'].replace('SKZ','IBB')
	schedules['airline'] = schedules['airline'].replace('IBK','IBB')
	schedules['airline'] = schedules['airline'].replace('CDN','IBB')
	#print(schedules)
	schedules.to_csv(path_to_mercury_input+'schedules.csv')
	schedules.to_parquet(path_to_mercury_input+'flight_schedules_tactical_1.parquet')
	# new_fp_pool = pd.concat([fp_pool]+selecteds)
	# new_fp_pool.to_parquet(path_to_mercury_input+'new_fp_pool.parquet')
	# new_fp_pool.to_csv(path_to_mercury_input+'new_fp_pool.csv')

def recreate_output_folder(folder_path: Path):
    """
    Check if a folder exists, delete it if it does, and recreate it as an empty folder.

    Args:
        folder_path (Path): The path to the folder.
    """
    if folder_path.exists():

        shutil.rmtree(folder_path)

    folder_path.mkdir(parents=True, exist_ok=True)

experiment = 'processed_cs10.pp20.nd02.so10.01'
path_to_data = '../../../data/CS10/v=0.16/output/'+experiment+'/paths_itineraries/'
path_to_mercury_input = path_to_data+'../mercury_input/'
recreate_output_folder(Path(path_to_mercury_input))

pax = pd.read_csv(path_to_data+'pax_assigned_tactical_1.csv')
if 'leg3' not in pax.columns:
	pax['leg3'] = np.nan
	#print(pax[['leg1','leg2']])
pax['gtfs_pre'] = 'gtfs_es_UIC_v2.3.zip'
pax['gtfs_post'] = 'gtfs_es_UIC_v2.3.zip'
pax['rail_pre'] = pax['rail_pre'].apply(lambda x: x.split('_')[0] if '_' in str(x) else x)
pax['rail_post'] = pax['rail_post'].apply(lambda x: x.split('_')[0] if '_' in str(x) else x)
pax = pax.rename(columns={'nid_x':'nid'})
pax.to_parquet(path_to_mercury_input+'pax_assigned_tactical_1.parquet')
pax.to_csv(path_to_mercury_input+'pax_assigned_tactical_1.csv')




filename = 'flight_schedules_tactical_1'
f = pd.read_csv(path_to_data+filename+'.csv', parse_dates=['sobt','sibt'])
f_filtered = f[(f['nid'].isin(pax['leg1'])) | (f['nid'].isin(pax['leg2'])) | (f['nid'].isin(pax['leg3']))]
#f_filtered.to_parquet(path_to_mercury_input+filename+'_intra.parquet')
#f_filtered.to_csv(path_to_data+filename+'_filtered.csv')
f_filtered['domestic'] = f_filtered.apply(lambda row: ((row['origin'][:2] in ['GC','LE']) or (row['origin']=='GEML')) and ((row['destination'][:2] in ['GC','LE']) or (row['origin']=='GEML')), axis=1)
intra = f_filtered[f_filtered['domestic']==True]
#print('flight_schedules_tactical_1',f_filtered,f_filtered.dtypes)
#print(intra)

route_pool = pd.read_parquet('/home/michal/Documents/westminster/multimodx/input/scenario=1/data/flight_plans/routes/route_pool_new.parquet')
fp_pool = pd.read_parquet('/home/michal/Documents/westminster/multimodx/input/scenario=1/data/flight_plans/flight_plans_pool/fp_pool_m_w_tv.parquet')
fp_pool_point_m = pd.read_parquet('/home/michal/Documents/westminster/multimodx/input/scenario=1/data/flight_plans/flight_plans_pool/fp_pool_point_m.parquet')
airlines_static = pd.read_parquet('/home/michal/Documents/westminster/multimodx/input/scenario=1/data/airlines/airline_static_old1409.parquet')
#airlines_static.to_csv(path_to_mercury_input+'airlines_static.csv')
eaman = pd.read_parquet('/home/michal/Documents/westminster/multimodx/input/scenario=1/data/eaman/eaman_definition_old1409.parquet')
row = ['D','GCLP',20,12]
row_df = pd.DataFrame(columns=eaman.columns, data=[row])
# new_eaman = eaman.iloc[[0]]
new_eaman = pd.concat([eaman,row_df])
new_eaman.to_parquet(path_to_mercury_input+'eaman.parquet')
new_eaman.to_csv(path_to_mercury_input+'eaman.csv')

bada=pd.read_parquet('/home/michal/Documents/westminster/multimodx/input/scenario=1/data/ac_performance/bada/ac_eq_badacomputed_static.parquet')
f_filtered = f_filtered.merge(bada[['bada_code_ac_model','ac_icao']], left_on=['aircraft_type'], right_on=['ac_icao'], how='left')
df_all = f_filtered.merge(route_pool.drop_duplicates(), left_on=['origin','destination'], right_on=['icao_orig','icao_dest'], how='left', indicator=True)
df_fp = f_filtered.merge(fp_pool, left_on=['origin','destination','bada_code_ac_model'], right_on=['icao_orig','icao_dest','bada_code_ac_model'], how='left', indicator=True)
missing = df_fp[df_fp['_merge'] == 'left_only']
#missing = missing.drop_duplicates(subset=['origin','destination'])
print(missing)
missing.to_csv(path_to_mercury_input+'missing.csv')
find_similar_fp(fp_pool,fp_pool_point_m,missing,f_filtered,path_to_mercury_input,airlines_static)


connecting_times = pd.read_csv(path_to_data+'../processed/transition_layer_connecting_times.csv')

a_to_b = pd.DataFrame()
a_to_b['origin'] = connecting_times['origin']
a_to_b['destination'] = connecting_times['destination']
a_to_b['mean'] = connecting_times['avg_travel_time']+connecting_times['extra_avg_travel_time']
a_to_b['std'] = 5
a_to_b['pax_type'] = ''
a_to_b['estimation_scale'] = 0
ct=a_to_b.copy()
#b_to_a = pd.DataFrame()
#b_to_a['origin'] = connecting_times['destination_station']
#b_to_a['destination'] = connecting_times['origin_station']
#b_to_a['mean'] = connecting_times['avg_travel_b_a']
#b_to_a['std'] = 5
#b_to_a['pax_type'] = ''
#b_to_a['estimation_scale'] = 0

#ct = pd.concat([a_to_b,b_to_a])
#print(ct)
#ct.to_parquet(path_to_mercury_input+'../connecting_times.parquet')
ct.to_csv(path_to_mercury_input+'connecting_times.csv')
ct = pd.read_csv(path_to_mercury_input+'connecting_times.csv')
ct.to_parquet(path_to_mercury_input+'connecting_times.parquet')

print(connecting_times)
station_airports = connecting_times[(connecting_times['layer_id_origin']=='air') & (connecting_times['layer_id_destination']=='rail')]
sa = pd.DataFrame()
sa['stop_id'] = connecting_times['destination']
sa['icao_id'] = connecting_times['origin']
sa.to_csv(path_to_mercury_input+'airport_stations.csv')
#sa = pd.read_csv(path_to_data+'../airport_stations.csv')
sa.to_parquet(path_to_mercury_input+'airport_stations.parquet')

airports = pd.read_parquet('/home/michal/Documents/westminster/multimodx/input/scenario=9/data/airports/airport_info_static_old1409.parquet')
airports['kerb2gate_mean'] = 90.0
airports['kerb2gate_std'] = 0.0
airports['gate2kerb_mean'] = 30.0
airports['gate2kerb_std'] = 0.0
airports.loc[airports['icao_id']=='LEMD','MCT_standard'] = 78
airports.loc[airports['icao_id']=='LEMD','MCT_domestic'] = 47
airports.loc[airports['icao_id']=='LEMD','MCT_international'] = 79
airports.loc[airports['icao_id']=='LFPG','MCT_standard'] = 59
airports.loc[airports['icao_id']=='LFPG','MCT_domestic'] = 45
airports.loc[airports['icao_id']=='LFPG','MCT_international'] = 61
airports.to_parquet(path_to_mercury_input+'airport_info_static.parquet')

#bada=pd.read_parquet('/home/michal/Documents/westminster/multimodx/input/scenario=1/data/ac_performance/bada/ac_eq_badacomputed_static.parquet')
#print(bada)
#bada.to_csv('bada.csv')

airport_processes = pd.read_csv(path_to_data+'../../../infrastructure/pax_processes/airport_processes_v0.1.csv')
airport_codes = pd.read_csv(path_to_data+'../../../infrastructure/airports_info/IATA_ICAO_Airport_codes_v1.3.csv')
airport_processes = airport_processes.merge(airport_codes[['IATA','ICAO']],how='left', left_on='airport',right_on='IATA').rename(columns={"ICAO": "icao_id", }).drop(['IATA','airport'], axis=1)

df = pd.DataFrame()
df = airports[~airports['icao_id'].isin(airport_processes['icao_id'])].rename(columns={"kerb2gate_mean": "k2g", 'kerb2gate_std':'k2g_std',"gate2kerb_mean": "g2k", 'gate2kerb_std':'g2k_std'}).copy()
df['k2g_multimodal'] = df['k2g']
df['g2k_multimodal'] = df['g2k']
df['pax_type'] = 'all'
airport_processes = pd.concat([airport_processes,df[['pax_type','k2g','g2k','k2g_multimodal','g2k_multimodal','icao_id','k2g_std','g2k_std']]],axis=0)

airport_processes['k2g_std'] = 0.0
airport_processes['g2k_std'] = 0.0
airport_processes.to_csv(path_to_mercury_input+'airport_processes.csv')
airport_processes.to_parquet(path_to_mercury_input+'airport_processes.parquet')

rail_stations_processes = pd.read_csv(path_to_data+'../../../infrastructure/pax_processes/rail_stations_processes_v0.1.csv')
rail_stations_processes.to_parquet(path_to_mercury_input+'rail_stations_processes_v0.1.parquet')
