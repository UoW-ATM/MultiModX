import pandas as pd

def replan_rail_timetable(rs_planned, rail_replanned=None, rail_cancelled=None, rail_added=None):
    ## Remove duplicates rows in rail planned timetable
    rs_planned = rs_planned.drop_duplicates()
    rs_planned['status'] = 'planned'

    # Remove cancelled services
    if rail_cancelled is not None:
        # Remove trains cancelled
        rs_planned = rs_planned.merge(rail_cancelled, left_on='trip_id', right_on='service_id', how='left')

        # keep rs that are not cancelled
        rs_planned = rs_planned[~(((~rs_planned['from'].isna()) & (rs_planned['to'].isna()) & (
                rs_planned['stop_sequence'] >= rs_planned['from'])) |
                                  ((~rs_planned['from'].isna()) & (~rs_planned['to'].isna()) & (
                                          rs_planned['stop_sequence'] >= rs_planned['from']) & (
                                           rs_planned['stop_sequence'] <= rs_planned['to'])) |
                                  ((rs_planned['from'].isna()) & (~rs_planned['to'].isna()) & (
                                          rs_planned['stop_sequence'] <= rs_planned['to']))
                                  )]
        rs_planned = rs_planned.drop(['service_id', 'from', 'to'], axis=1)

    # Remove trains replanned (as they'll be added as replanned)
    if rail_replanned is not None:
        rs_planned = rs_planned[~rs_planned.trip_id.isin(rail_replanned.trip_id)]

        # Add new trains replanned
        rail_replanned['status'] = 'replanned'
        rs_planned = pd.concat([rs_planned, rail_replanned])

    if rail_added is not None:
        # Add additional new rail services
        rail_added['status'] = 'added'
        rs_planned = pd.concat([rs_planned, rail_added])

    # Remove duplicates (in case)
    rs_planned = rs_planned.drop_duplicates()

    return rs_planned


def replan_flight_schedules(fs_planned, fs_replanned=None, fs_cancelled=None, fs_added=None):
    fs_planned['status'] = 'planned'

    if fs_cancelled is not None:
        fs_planned = fs_planned[~fs_planned.service_id.isin(fs_cancelled.service_id)]

    if fs_replanned is not None:
        fs_planned = fs_planned[~fs_planned.service_id.isin(fs_replanned.service_id)]
        fs_replanned['status'] = 'replanned'
        fs_planned = pd.concat([fs_planned, fs_replanned])

    if fs_added is not None:
        fs_added['status'] = 'added'
        fs_planned = pd.concat([fs_planned, fs_added])

    return fs_planned

