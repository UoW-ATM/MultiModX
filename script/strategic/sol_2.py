#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import itertools
from datetime import timedelta
import random


def load_data(path_train_schedule, path_flight_schedule, path_possible_itineraries):
    # load data and reform the data structure
    flight_schedule = pd.read_csv(path_flight_schedule)
    train_schedule = pd.read_csv(path_train_schedule)
    itrys = pd.read_csv(path_possible_itineraries)
    itrys.loc[itrys.nservices[itrys.nservices==3].index,'nservices'] = 2 
    itrys = itrys[['nservices', 'nmodes', 'service_id_0', 'origin_0', 'destination_0',
       'mode_0', 'service_id_1', 'origin_1', 'destination_1', 'mode_1', 'demand']]
    train_schedule = train_schedule[['trip_id','stop_id','stop_sequence','arrival_time','departure_time']]
    services = train_schedule.rename(columns={'trip_id':'service_id'})
    services['product']='rail'
    
    # rail timetable time formating
    services.arrival_time = '2014-09-12 '+services.arrival_time 
    services.departure_time = '2014-09-12 '+services.departure_time 
    for ind in services.arrival_time.index:
        i = services.arrival_time.loc[ind]
        if int(i[11:-6])>=24:
            i = i[0:8]+str(int(i[8:10])+1)+' '+str(int(i[11:-6])-24)+i[-6::]
            services.loc[ind,'arrival_time']=i
        i = services.departure_time.loc[ind]
        if int(i[11:-6])>=24:
            i = i[0:8]+str(int(i[8:10])+1)+' '+str(int(i[11:-6])-24)+i[-6::]
            services.loc[ind,'departure_time']=i
    
    # transform flight schedule to GTFS format
    tmp = []
    for index, row in flight_schedule.iterrows():
        tmp.append([row.service_id,row.origin,'1',row.sobt,row.sobt,'air'])
        tmp.append([row.service_id,row.destination,'2',row.sibt,row.sibt,'air'])
    services = services.append(pd.DataFrame(tmp,columns=services.columns),ignore_index = True)
    
    services = services.astype('string')
    services.stop_sequence = services.stop_sequence.astype('int')
    services.arrival_time = pd.to_datetime(services.arrival_time, format = '%Y-%m-%d %H:%M:%S')
    services.departure_time = pd.to_datetime(services.departure_time, format = '%Y-%m-%d %H:%M:%S')
    return itrys, services



def timetable_sync(itrys,services):
    # generate new timetable(services) with an additional column 'change_type' to identify timetable change
    # the new timetable is generated randomly for now 
    # service change type: fixed service-0, new service-1, cancelled service-2(completely canceled or partially), timetable shifted-3 (fixed route with new timetable), rerouted services-4 (partially fixed and partially new)
    services_new = services.copy()
    services_id = services.service_id.drop_duplicates()
    services_new['change_type'] = 0
    # new service
    for i in range(random.randint(1,5)):
        tmp_id = services_id.iloc[random.randint(0,len(services_id))]
        tmp = services[services.service_id == tmp_id].copy()
        if tmp['product'].iloc[0] == 'rail':
            new_service_id = str(random.randint(1000,9999))+'2014-09-12'
        else:
            new_service_id = str(random.randint(10000,99999))
        time_change = timedelta(minutes = random.randint(-30,30))
        tmp.arrival_time = tmp.arrival_time+time_change
        tmp.departure_time = tmp.departure_time+time_change
        tmp['change_type'] = 1
        tmp['service_id'] = new_service_id
        services_new = pd.concat([services_new,tmp],ignore_index = True)

    # cancelled servcie
    for i in range(random.randint(1,5)):
        tmp_id = services_id.iloc[random.randint(0,len(services_id))]
        inds = services_new[services_new.service_id == tmp_id].index
        services_new.loc[inds,'change_type'] = 2
        stop_cancelled = inds[random.randint(0,len(inds)-1)::]
        services_new.loc[stop_cancelled,'departure_time'] = np.nan
        services_new.loc[stop_cancelled,'arrival_time'] = np.nan

    # timetable-shfited
    for i in range(random.randint(1,5)):
        tmp_id = services_id.iloc[random.randint(0,len(services_id))]
        tmp = services[services.service_id == tmp_id].copy()
        time_change = timedelta(minutes = random.randint(-30,30))
        tmp.arrival_time = tmp.arrival_time+time_change
        tmp.departure_time = tmp.departure_time+time_change
        tmp['change_type'] = 3
        services_new = services_new[services_new.service_id != tmp_id]
        services_new = pd.concat([services_new,tmp],ignore_index = True)

    #rerouted service
    for i in range(random.randint(1,5)):
        tmp_id = services_id.iloc[random.randint(0,len(services_id))]
        tmp = services[services.service_id == tmp_id].copy()
        tmp['change_type'] = 4
        services_new = services_new[services_new.service_id != tmp_id]
        services_new = pd.concat([services_new,tmp],ignore_index = True)
    
    return services_new


if __name__ == '__main__':
    path_train_schedule = 'input\\rail_timetable_proc_gtfs_0.csv'
    path_flight_schedule = 'input\\flight_schedules_proc_0.csv'
    path_possible_itineraries = 'input\\possible_itineraries_with_demand.csv'
    path_new_schedule = 'output\\new_schedule.csv'
    itrys,services = load_data(path_train_schedule, path_flight_schedule, path_possible_itineraries)
    services_new = timetable_sync(itrys,services)
    services_new.to_csv(path_new_schedule)






