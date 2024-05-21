import pandas as pd
import numpy as np
from libs.uow_tool_belt.general_tools import haversine
from libs.gtfs import add_date_and_handle_overflow


def process_list_distance_ranges(distance_ranges_list, max_distance=None):
    distance_ranges = []
    for de in distance_ranges_list:
        if de[2] is None:
            distance_ranges.append((de[0], de[1]))
        else:
            if (max_distance is not None) and (max_distance < de[1]):
                distance_ranges.extend([(i, i + de[2]) for i in range(de[0], int(np.ceil(max_distance)), de[2])])
                break
            else:
                distance_ranges.extend([(i, i + de[2]) for i in range(de[0], de[1], de[2])])

    return distance_ranges


def generate_heuristic_rail(gtfs_path, rail_stations_considered, date, country=None, interval_range=50):
    # Read GTFS
    # Load rail data
    parent_stop_dict = {}
    stops = pd.read_csv(gtfs_path / 'stops.txt')
    # Create a dictionary of parent stations
    # Check notebook to read more than one country and TODO check parent station works Germany
    parent_stop_dict = {row['stop_id']: f"{int(row['parent_station'])}"
                        for index, row in
                        stops[~stops.parent_station.isnull()][['stop_id', 'parent_station']].iterrows()}

    # Drop stops that have a parent
    stops = stops[stops.parent_station.isnull()]

    # Read stop_times
    stop_times = pd.read_csv(gtfs_path / 'stop_times.txt')
    stop_times['stop_id'] = stop_times['stop_id'].apply(lambda x: parent_stop_dict.get(x, x))

    if country is not None:
        stops['stop_id'] = stops['stop_id'].apply(lambda x: country+str(x))
        stop_times['stop_id'] = stop_times['stop_id'].apply(lambda x: country+str(x))

    st_filt = stop_times[stop_times.stop_id.isin(rail_stations_considered)].merge(stops[['stop_id', 'stop_lat',
                                                                                         'stop_lon']],
                                                                                  left_on='stop_id', right_on='stop_id')


    st_filt['arrival_time'] = st_filt['arrival_time'].apply(lambda x: add_date_and_handle_overflow(x, date))
    st_filt['departure_time'] = st_filt['departure_time'].apply(lambda x: add_date_and_handle_overflow(x, date))

    # Prepare an empty list to collect results
    results = []

    # Group by trip_id and sort by stop_sequence
    grouped = st_filt.groupby('trip_id')
    for trip_id, group in grouped:
        group = group.sort_values(by='stop_sequence')
        n = len(group)
        for i in range(n):
            for j in range(i + 1, n):
                lat1, lon1 = group.iloc[i][['stop_lat', 'stop_lon']]
                lat2, lon2 = group.iloc[j][['stop_lat', 'stop_lon']]
                distance = haversine(lon1, lat1, lon2, lat2)

                # Calculate time difference in seconds
                time1 = group.iloc[i]['departure_time']
                time2 = group.iloc[j]['arrival_time']
                time_diff = (time2 - time1).total_seconds() / 60

                results.append((trip_id, distance, time_diff))

    # Create a new DataFrame from the results
    result_df = pd.DataFrame(results, columns=['trip_id', 'distance_km', 'time_diff_min'])

    # Compute min/max per group

    # Create distance bins every 50 km
    bins = np.arange(0, result_df['distance_km'].max() + interval_range, interval_range)
    result_df['distance_bin'] = pd.cut(result_df['distance_km'], bins=bins)

    # Group by the distance bins and calculate the minimum time difference for each bin
    grouped_bins = result_df.groupby('distance_bin')['time_diff_min'].min().reset_index()

    # Initialize empty lists to store the intercalated series
    min_series = []
    max_series = []

    # Iterate through each row of the DataFrame
    for index, row in grouped_bins.iterrows():
        # Get the minimum and maximum distance for the current bin
        min_distance = row['distance_bin'].left
        max_distance = row['distance_bin'].right

        # Append the minimum distance and the corresponding value to the min_series
        min_series.append((min_distance, row['time_diff_min']))

        # Append the maximum distance and the corresponding value to the max_series
        max_series.append((max_distance, row['time_diff_min']))

        # If it's not the last bin, append the maximum distance again with the next value
        if index < len(grouped_bins) - 1:
            next_min_distance = grouped_bins.loc[index + 1, 'distance_bin'].left
            max_series.append((max_distance, row['time_diff_min']))
            max_series.append((next_min_distance, row['time_diff_min']))

    # Convert the lists to Series
    min_series = pd.Series(dict(min_series))
    max_series = pd.Series(dict(max_series))

    # Initialize an empty dictionary to store the merged values
    merged_outcome = []

    # Merge the min and max series using zip
    merged_series = list(zip(min_series.items(), max_series.items()))

    # Iterate through the merged series and add each pair of values to the list
    for (min_distance, min_value), (max_distance, max_value) in merged_series:
        merged_outcome.append((min_distance, min_value))
        merged_outcome.append((max_distance, max_value))

    # Create a Series from the merged outcome
    x_dist_rail_heur = [x[0] for x in merged_outcome]
    y_time_rail_heur = [x[1] for x in merged_outcome]

    # Initialize empty lists for the DataFrame columns
    min_dist = []
    max_dist = []
    time = []

    # Iterate through the intervals defined by x and y
    for i in range(len(x_dist_rail_heur)):
        # Append the minimum distance of the current interval
        min_dist.append(x_dist_rail_heur[i])
        # Append the maximum distance of the current interval (except for the last interval)
        if i < len(x_dist_rail_heur) - 1:
            max_dist.append(x_dist_rail_heur[i + 1])
        else:
            max_dist.append(float('inf'))  # Use infinity for the last interval
        # Append the time of the current interval
        time.append(y_time_rail_heur[i])

    # Create a DataFrame from the lists
    df = pd.DataFrame({'min_dist': min_dist, 'max_dist': max_dist, 'time': time})
    df = df[df.min_dist != df.max_dist]
    return df, result_df[['distance_km', 'time_diff_min']].rename(columns={'time_diff_min': 'duration_minutes'})


def generate_heuristic_air(fs_path, airp_static_path, distance_ranges):
    fs = pd.read_csv(fs_path)
    arp = pd.read_csv(airp_static_path)

    fs = fs.merge(arp, left_on=['origin'], right_on=['icao_id'],
                  suffixes=('', '_orig')).merge(arp, left_on=['destination'], right_on=['icao_id'],
                                                suffixes=('_orig', '_dest'))

    fs['sobt'] = pd.to_datetime(fs['sobt'])
    fs['sibt'] = pd.to_datetime(fs['sibt'])

    # Calculate flight block time in minutes
    fs['duration_minutes'] = (fs['sibt'] - fs['sobt']).dt.total_seconds() / 60

    # Calculate distance in kilometers
    fs['distance_km'] = haversine(fs['lon_orig'], fs['lat_orig'], fs['lon_dest'], fs['lat_dest'])

    distance_ranges = process_list_distance_ranges(distance_ranges, max(fs.distance_km))

    # Initialize an empty list to store the result of minimum time required per distance range
    result = []

    # Iterate over the distance ranges
    for min_dist, max_dist in distance_ranges:
        # Filter the fastest_service DataFrame for the current distance range
        bin_data = fs[(fs['distance_km'] > min_dist) & (fs['distance_km'] <= max_dist)]
        # If there's data for the current range, extract the minimum duration_minutes
        if not bin_data.empty:
            min_duration = bin_data['duration_minutes'].min()
            # Append the result to the list
            result.append((min_dist, max_dist, min_duration))

    # Convert the result to a DataFrame
    result_df = pd.DataFrame(result, columns=['min_dist', 'max_dist', 'time'])

    return result_df, fs
