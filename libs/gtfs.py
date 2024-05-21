import pandas as pd

def get_stop_times_on_date(date, calendar, calendar_dates, trips, stop_times, drop_same_trip_short_name=True):

    day_name_of_week = date.strftime('%A').lower()

    active_service_ids = set(calendar[(calendar['start_date'] <= date) & (calendar['end_date'] >= date) &
                                      (calendar[day_name_of_week] == 1)]['service_id'])

    # Adjust Service IDs Using Calendar Dates
    exception_dates = calendar_dates[calendar_dates['date'] == date]
    active_service_ids = active_service_ids | set(exception_dates[exception_dates['exception_type'] == 1]['service_id'])
    excluded = set(exception_dates[exception_dates['exception_type'] == 2]['service_id'])
    active_service_ids = active_service_ids - excluded

    #  Filter Trips
    trips_on_given_day = trips[trips['service_id'].isin(active_service_ids)]

    if drop_same_trip_short_name:
        trips_on_given_day = trips_on_given_day.drop_duplicates(subset=['trip_short_name'], keep='first')

    # Filter Stop Times
    stop_times_on_given_day = stop_times[stop_times['trip_id'].isin(trips_on_given_day['trip_id'])]

    # Output the filtered stop times
    return stop_times_on_given_day

def add_date_and_handle_overflow(time_str, date):
    hour, minute, second = map(int, time_str.split(':'))
    if hour >= 24:
        # If time exceeds 24 hours, add a day to the date component and adjust the time
        date_adjusted = date + pd.Timedelta(days=1)
        hour %= 24
    else:
        date_adjusted = date
    return date_adjusted + pd.Timedelta(hours=hour, minutes=minute, seconds=second)