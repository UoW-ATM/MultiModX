import pandas as pd
import libs.airac_database_functions as adf
import libs.load_ddr_functions as ldf
import numpy as np
from decimal import *

def process_all_ft(allft_path,verbose=True):

	airac = 20230922
	if verbose:
		print("Before reading")

	if allft_path.endswith('.gz') or allft_path.endswith('.ALL_FT+'):
		data, ddr_version = ldf.read_all_ft(allft_path,airac)
	else:
		print("Format of ALL_FT file not recognised")
		sys.exit(0)

	data = ldf.format(data,ddr_version)
	print(data)
	#data.head().to_csv('xxx.csv')
	#data = ldf.set_airport_reg_ids(data,airports_dict,regs_dict)

	if verbose:
		print("DDR version",ddr_version)

	#if remove_duplicates_ifps_id:
		#if verbose:
			#print("Removind duplicates ifps_id, keeping one with aobt")
			#print(len(data))

		#data = data.sort_values(by=['ifps_id','aobt'])
		#data['prev_ifps_id']=data['ifps_id'].shift(1)
		#data = data[data['ifps_id']!=data['prev_ifps_id']]
		#data = data.drop({'prev_ifps_id'},axis=1)
		#data = data.sort_index().reset_index(drop=True)

		#if verbose:
			#print("After removing duplicates there are ",len(data))

	#if verbose:
		#print("Before adding in DB")

	#data['individual_day'] = individual_full_day


	#with mysql_connection(profile=credentials,database=database) as conn:
		#coord_geo_dict, max_coord_id, fids_dict = adf.add_flight_details_in_db(conn['engine'],data,airac=airac,verbose=verbose,
			#airports_geo_dict=airports_geo_dict,airports_dict=airports_dict,
			#wpt_geo_dict=wpt_geo_dict,
			#airsp_dict=airsp_dict,sect_dict=sect_dict,coord_geo_dict=coord_geo_dict,max_coord_id=max_coord_id,
			#excluding=excluding, ddr_version=ddr_version, if_conflict=if_conflict_replace, fids_dict=fids_dict,
			#load_ftfm=load_ftfm, load_rtfm=load_rtfm, load_ctfm=load_ctfm, load_scr=load_scr,
			#load_srr=load_srr, load_sur=load_sur, load_dct=load_dct, load_cpf=load_cpf)

	return data, ddr_version

def haversine(lon1, lat1, lon2, lat2):
	"""
	Calculate the great circle distance between two points
	on the earth (specified in decimal degrees)
	----------
	Parameters
	----------
	lon1, lat1 : coordinates point 1 in degrees
	lon2, lat2 : coordinates point 2 in degrees
	-------
	Return
	------
	Distance between points in km
	"""
	# convert decimal degrees to radians
	lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
	# haversine formula
	dlon = lon2 - lon1
	dlat = lat2 - lat1
	km = 2. * 6373. * np.arcsin(np.sqrt(np.sin(dlat/2.)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.)**2))
	return km

def zip_with_scalar_divide_point(l, o):
	return ((o, i.split(":")) for i in l)
def extract_lat(x):
		index_lat=max(x.find('N'), x.find('S'))

		deg=Decimal(x[0:2])
		if index_lat>=4:
			minutes=Decimal(x[2:4])
		else:
			minutes=Decimal(0)
		if index_lat>=6:
			seconds=Decimal(x[4:6])
		else:
			seconds=Decimal(0)

		lat=deg+minutes/60+seconds/60/60
		if x.find('S')>0:
			lat=-(deg+minutes/60+seconds/60/60)

		return float(lat)

def extract_lon(x):
	index_lat=max(x.find('N'), x.find('S'))
	deg=Decimal(x[index_lat+1:index_lat+4])
	if index_lat+5<len(x):
		minutes=Decimal(x[index_lat+4:index_lat+6])
	else:
		minutes=Decimal(0)

	if index_lat+7<len(x):
		seconds=Decimal(x[index_lat+6:index_lat+8])
	else:
		seconds=Decimal(0)

	lon=deg+minutes/60+seconds/60/60
	if x.find('W')>0:
		lon=-(deg+minutes/60+seconds/60/60)

	return float(lon)

def dataframe_airspaceProfile(data, airspaceName, engine, airsp_dict, sect_dict, dict_trajectory, coord_geo_dict, max_coord_id, ddr_version=3,dict_date={}):
	#For each airspace field in each flight divide it into a dataframe identified by filght_id and allAirsp information

	read_points_on_demand = (len(coord_geo_dict)==0)#read points on demand

	if len(data.loc[(~pd.isnull(data[airspaceName])),:])>0:

		d=(data.apply(lambda x: zip_with_scalar_divide_point(x[airspaceName].split(" "),x['ifps_id']) if
						  not pd.isnull(x[airspaceName]) else np.nan, axis=1))

		d=[ x for x in d if not pd.isnull(x)]

		f=pd.DataFrame([item for sublist in [list(gen) for gen in d] for item in sublist])

		f.columns = ['ifps_id','allAirsp']

		#divide the information of the ALL_FT+ into fields
		g=f[['ifps_id']].copy()
		g['trajectory_id']=g['ifps_id'].apply(lambda x: dict_trajectory.get(x))
		g['time_entry_orig']=f['allAirsp'].apply(lambda x: x[0])
		g['airspace']=f['allAirsp'].apply(lambda x: x[1])
		g['time_exit_orig']=f['allAirsp'].apply(lambda x: x[2])
		g['airspace_type']=f['allAirsp'].apply(lambda x: x[3])
		g['latlon_entry']=f['allAirsp'].apply(lambda x: x[4])
		g['latlon_exit']=f['allAirsp'].apply(lambda x: x[5])
		g['fl_entry']=f['allAirsp'].apply(lambda x: x[6] if x[6] != '' else np.nan)
		g['fl_exit']=f['allAirsp'].apply(lambda x: x[7] if x[7] != '' else np.nan)
		g['distance_entry']=f['allAirsp'].apply(lambda x: x[8] if x[8] != '' else np.nan)
		g['distance_exit']=f['allAirsp'].apply(lambda x: x[9] if x[9] != '' else np.nan)
		g['entry_point_lat']=g['latlon_entry'].apply(lambda x: extract_lat(x))
		g['entry_point_lon']=g['latlon_entry'].apply(lambda x: extract_lon(x))
		g['exit_point_lat']=g['latlon_exit'].apply(lambda x: extract_lat(x))
		g['exit_point_lon']=g['latlon_exit'].apply(lambda x: extract_lon(x))

		g['airspace_id']=g.apply(lambda x: airsp_dict.get(x['airspace']) if (x['airspace_type']=="AREA" or
																			x['airspace_type']=="NAS" or
																			x['airspace_type']=="AUA" or
																			x['airspace_type']=="CS" or
																			x['airspace_type']=="CRSA" or
																			x['airspace_type']=="CLUS")
																		 else np.NaN, axis=1)

		g['sector_id']=g.apply(lambda x: sect_dict.get(x['airspace']) if (x['airspace_type']=="NS" or
																			x['airspace_type']=="FIR" or
																			x['airspace_type']=="AOI" or
																			x['airspace_type']=="AOP" or
																			x['airspace_type']=="ES" or
																			x['airspace_type']=="ERSA")
																		 else np.NaN, axis=1)

		g['geopoint_entry_id']=g['latlon_entry'].apply(lambda x: coord_geo_dict.get(x)['geo_id'] if not pd.isnull(coord_geo_dict.get(x,np.nan))
													   else np.nan)



		g['geopoint_exit_id']=g['latlon_exit'].apply(lambda x: coord_geo_dict.get(x)['geo_id'] if not pd.isnull(coord_geo_dict.get(x,np.nan))
													   else np.nan)


		missing_geo_points=pd.DataFrame(g.loc[pd.isnull(g['geopoint_entry_id']),['latlon_entry']].latlon_entry.unique())
		missing_geo_points.columns=['sid']

		indexes_missing_entry=pd.isnull(g['geopoint_entry_id'])

		missing_geo_points_exit=pd.DataFrame(g.loc[pd.isnull(g['geopoint_exit_id']),['latlon_exit']].latlon_exit.unique())
		missing_geo_points_exit.columns=['sid']

		indexes_missing_exit=pd.isnull(g['geopoint_exit_id'])

		missing_geo_points = pd.merge(missing_geo_points, missing_geo_points_exit, on='sid', how='outer')

		missing_geo_points.drop_duplicates(inplace=True)

		missing_geo_points=missing_geo_points.loc[missing_geo_points['sid']!='']


		#if read_points_on_demand:
		  #dict_coord_points_extra = read_coordpoints_with_geopoints(engine,sid=list(missing_geo_points['sid']))
		  #coord_geo_dict.update(dict_coord_points_extra)

		  #g['geopoint_entry_id']=g['latlon_entry'].apply(lambda x: coord_geo_dict.get(x)['geo_id'] if not pd.isnull(coord_geo_dict.get(x,np.nan))
													   #else np.nan)


		  #g['geopoint_exit_id']=g['latlon_exit'].apply(lambda x: coord_geo_dict.get(x)['geo_id'] if not pd.isnull(coord_geo_dict.get(x,np.nan))
														 #else np.nan)


		  #missing_geo_points=pd.DataFrame(g.loc[pd.isnull(g['geopoint_entry_id']),['latlon_entry']].latlon_entry.unique())
		  #missing_geo_points.columns=['sid']

		  #indexes_missing_entry=pd.isnull(g['geopoint_entry_id'])

		  #missing_geo_points_exit=pd.DataFrame(g.loc[pd.isnull(g['geopoint_exit_id']),['latlon_exit']].latlon_exit.unique())
		  #missing_geo_points_exit.columns=['sid']

		  #indexes_missing_exit=pd.isnull(g['geopoint_exit_id'])

		  #missing_geo_points = pd.merge(missing_geo_points, missing_geo_points_exit, on='sid', how='outer')

		  #missing_geo_points.drop_duplicates(inplace=True)

		  #missing_geo_points=missing_geo_points.loc[missing_geo_points['sid']!='']




		#if not missing_geo_points.empty:
			##There are geopoints missing
			#max_coord_id=read_maxid_coordpoints(engine)
			#add_missing_coordinate_geopoints(engine,missing_geo_points)

			#coord_geo_dict_extra=read_coordpoints_with_geopoints(engine,max_coord_id)
			#coord_geo_dict.update(coord_geo_dict_extra)
			#max_coord_id=read_maxid_coordpoints(engine)

			##g['geopoint_entry_id']=g['latlon_entry'].apply(lambda x: coord_geo_dict.get(x)['geo_id'] if not pd.isnull(coord_geo_dict.get(x,np.nan))
			##									   else np.nan)
			##g['geopoint_exit_id']=g['latlon_exit'].apply(lambda x: coord_geo_dict.get(x)['geo_id'] if not pd.isnull(coord_geo_dict.get(x,np.nan))
			##									   else np.nan)

			#g.loc[indexes_missing_entry,['geopoint_entry_id']]=g.loc[indexes_missing_exit]['latlon_entry']\
									#.apply(lambda x: coord_geo_dict.get(x)['geo_id']
									 #if not pd.isnull(coord_geo_dict.get(x,np.nan))
									 #else np.nan)

			#g.loc[indexes_missing_exit,['geopoint_exit_id']]=g.loc[indexes_missing_exit]['latlon_exit']\
									#.apply(lambda x: coord_geo_dict.get(x)['geo_id']
									 #if not pd.isnull(coord_geo_dict.get(x,np.nan))
									 #else np.nan)



		if ddr_version >= 3:
			g['time_entry']=g['time_entry_orig'].apply(lambda x: x[0:4]+"-"+x[4:6]+"-"+x[6:8]+" "+
													   x[8:10]+":"+x[10:12]+":"+x[12:14])
			g['time_exit']=g['time_exit_orig'].apply(lambda x: x[0:4]+"-"+x[4:6]+"-"+x[6:8]+" "+
													 x[8:10]+":"+x[10:12]+":"+x[12:14])

		else:
			print(f['ifps_id'])
			g['date_ini']=f['ifps_id'].apply(lambda x: datetime.strptime(dict_date.get(x)[0:10], '%Y-%m-%d'))
			g['hour_ini']=g['ifps_id'].apply(lambda x: int(dict_date.get(x)[11:13]))
			g['time_entry_orig_formated']=g['time_entry_orig'].apply(lambda x: np.nan if x==''
																	 else x[0:2]+":"+x[2:4]+":"+x[4:6])
			g['time_exit_orig_formated']=g['time_exit_orig'].apply(lambda x: np.nan if x==''
																	 else x[0:2]+":"+x[2:4]+":"+x[4:6])
			#in some cases the timeover is "" like in flight AA34925529 / flihgt_id=39429 which starts at ZZZZ

			g['time_entry']=g.apply(lambda x:
								  np.nan if x['time_entry_orig']==''
								  else
								  x['date_ini'].strftime('%Y-%m-%d')+" "+x['time_entry_orig_formated']
								  if int(x['time_entry_orig'][0:2])>=x['hour_ini'] else
								  (x['date_ini']+timedelta(days=1)).strftime('%Y-%m-%d')+" "+x['time_entry_orig_formated']
								  , axis=1)

			g['change_day_entry']=g.apply(lambda x:
								  np.nan if x['time_entry_orig']==''
								  else
								  0
								  if int(x['time_entry_orig'][0:2])>=x['hour_ini'] else
								  1
								  , axis=1)



			g['time_exit']=g.apply(lambda x:
								  np.nan if x['time_exit_orig']==''
								  else
								  x['date_ini'].strftime('%Y-%m-%d')+" "+x['time_exit_orig_formated']
								  if int(x['time_exit_orig'][0:2])>=x['hour_ini'] and int(x['time_entry_orig'][0:2])>=x['hour_ini'] else
								  (x['date_ini']+timedelta(days=1)).strftime('%Y-%m-%d')+" "+x['time_exit_orig_formated']
								  , axis=1)



			g['change_day_exit']=g.apply(lambda x:
								  np.nan if x['time_exit_orig']==''
								  else
								  0
								  if int(x['time_exit_orig'][0:2])>=x['hour_ini'] else
								  1
								  , axis=1)



		g=g[['ifps_id','trajectory_id', 'sector_id', 'airspace_id', 'geopoint_entry_id', 'geopoint_exit_id', 'fl_entry', 'fl_exit','distance_entry','distance_exit','time_entry','time_exit','airspace','airspace_type', 'latlon_entry','latlon_exit', 'entry_point_lat', 'entry_point_lon', 'exit_point_lat', 'exit_point_lon']]





	else:
		g=None


	if read_points_on_demand:
	  coord_geo_dict = {}

	return g, coord_geo_dict, max_coord_id

if __name__ == "__main__":
	data, ddr_version = process_all_ft('/home/michal/Documents/westminster/multimodx/data/strategic/data/routes_ddr/20190906.ALL_FT+')
	missing = pd.read_csv('/home/michal/Documents/westminster/multimodx/Mercury_old_private/output/routes_missing.csv')
	airspace_static = pd.read_parquet('/home/michal/Documents/westminster/multimodx/input/scenario=1/data/flight_plans/routes/airspace_static.parquet')
	route_pool_has_airspace = pd.read_parquet('/home/michal/Documents/westminster/multimodx/input/scenario=1/data/flight_plans/routes/route_pool_has_airspace.parquet')
	route_pool_ = pd.read_parquet('/home/michal/Documents/westminster/multimodx/input/scenario=1/data/flight_plans/routes/route_pool.parquet')

	max_route_id = route_pool_['id'].max()+1
	df = data.merge(missing,how='inner',left_on=['origin','destination'],right_on=['origin','destination'])
	print(df)
	g, coord_geo_dict, max_coord_id = dataframe_airspaceProfile(df, 'ftfmAirspProfile', "", {}, {'FIR':'FIR'}, {}, {}, 0, ddr_version=ddr_version,dict_date={})
	g = g[g['airspace_type'].isin(['NAS'])]
	g = g.merge(data[['origin','destination','ifps_id']],how='left',on='ifps_id')
	g['nid'] = g['ifps_id'].apply(lambda x: g['ifps_id'].unique().tolist().index(x)+max_route_id)
	g.insert(0, 'order', (g.groupby(['ifps_id']).cumcount()+1))
	g = g.merge(airspace_static[['sid','id']],how='left',left_on=['airspace'],right_on=['sid'])
	print(g)
	#g.to_csv('xxx.csv')
	#route_pool_id	airspace_id	sequence	entry_point	exit_point	distance_entry	distance_exit	gcd_km	airspace_orig_sid	exit_point_lat	exit_point_lon	entry_point_lat	entry_point_lon
	route_pool_airspace = g.drop(columns=['trajectory_id','sector_id','airspace_id','geopoint_entry_id','geopoint_exit_id','fl_entry','fl_exit','time_entry','time_exit', 'airspace_type','latlon_entry','latlon_exit','ifps_id']).copy()
	route_pool_airspace = route_pool_airspace.rename(columns={'order':'sequence','nid':'route_pool_id','id':'airspace_id','airspace':'airspace_orig_sid',})
	route_pool_airspace['gcd_km'] = route_pool_airspace.apply(lambda row: haversine(row['entry_point_lon'],row['entry_point_lat'],row['exit_point_lon'],row['exit_point_lat']),axis=1)
	route_pool_airspace['distance_entry'] = pd.to_numeric(route_pool_airspace['distance_entry'])
	route_pool_airspace['distance_exit'] = pd.to_numeric(route_pool_airspace['distance_exit'])
	print(route_pool_airspace)
	route_pool = route_pool_airspace.drop_duplicates(subset=['route_pool_id'],keep='last').copy()
	#id	based_route_pool_static_id	based_route_pool_o_d_generated	tact_id	f_airac_id	icao_orig	icao_dest	fp_distance_km	fp_distance_km_orig	f_database	type
	route_pool['type'] = 'historic'
	route_pool['f_database'] = 'ddr_1909'
	route_pool = route_pool.rename(columns={'route_pool_id':'id','distance_exit':'fp_distance_km','origin':'icao_orig','destination':'icao_dest'})
	route_pool['fp_distance_km'] = pd.to_numeric(route_pool['fp_distance_km'])
	route_pool['fp_distance_km_orig'] = route_pool['fp_distance_km']
	route_pool['based_route_pool_static_id'] = route_pool['id']
	route_pool = route_pool[['id','based_route_pool_static_id','icao_orig','icao_dest','fp_distance_km','fp_distance_km_orig','f_database','type']]
	print(route_pool)
	#print(route_pool.dtypes)
	#route_pool.to_parquet('rp.parquet')

	#concat old and new files
	route_pool_new = pd.concat([route_pool_,route_pool])
	route_pool_has_airspace_new = pd.concat([route_pool_has_airspace,route_pool_airspace.drop(columns=['origin','destination','sid'])])
	route_pool_new.to_parquet('/home/michal/Documents/westminster/multimodx/input/scenario=1/data/flight_plans/routes/route_pool_new.parquet')
	route_pool_has_airspace_new.to_parquet('/home/michal/Documents/westminster/multimodx/input/scenario=1/data/flight_plans/routes/route_pool_has_airspace_new.parquet')
	route_pool_new.to_csv('route_pool_new.csv')
	route_pool_has_airspace_new.to_csv('route_pool_has_airspace_new.csv')
