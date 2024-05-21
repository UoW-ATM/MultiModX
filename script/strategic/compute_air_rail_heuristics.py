from pathlib import Path
import argparse
import tomli
import ast
import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.insert(1, '../..')

from strategic_evaluator.create_air_rail_heuristic_path_finding import generate_heuristic_air, generate_heuristic_rail


def create_air_heuristic(heuristic_config):
    df_h_air, fs = generate_heuristic_air((Path(heuristic_config['air_heuristic']['network_path']) /
                                           heuristic_config['air_heuristic']['flight_schedules']),
                                          (Path(heuristic_config['air_heuristic']['network_path']) /
                                           heuristic_config['air_heuristic']['airports_static']),
                                          ast.literal_eval(heuristic_config['air_heuristic']['dist_range']))

    df_h_air.to_csv(Path(heuristic_config['output']['output_folder']) / 'air_time_heuristics.csv', index=False)

    # Plot the result
    x_dist_air_heur = []
    y_time_air_heur = []

    # Iterate through each row of the DataFrame
    for index, row in df_h_air.iterrows():
        # Append min_dist and max_dist to x list
        x_dist_air_heur.extend([row['min_dist'], row['max_dist']])
        # Append time to y list twice per row
        y_time_air_heur.extend([row['time'], row['time']])

    plt.figure(figsize=(10, 6))
    plt.scatter(fs['distance_km'], fs['duration_minutes'], alpha=0.6, edgecolors='w', linewidth=0.5)
    plt.plot(x_dist_air_heur, y_time_air_heur, marker='o', color='r')
    plt.title('Time Difference vs Distance')
    plt.xlabel('Distance (km)')
    plt.ylabel('Time Difference (minutes)')
    plt.xlim(0, max(x_dist_air_heur))
    plt.ylim(0, max(y_time_air_heur))
    plt.grid(True)
    plt.savefig(Path(heuristic_config['output']['output_folder']) / 'air_time_heuristics.png')


def create_rail_heuristic(heuristic_config):

    country = None

    if 'rail_stations_considered_nuts' in heuristic_config['rail_heuristic'].keys():
        # Rail stations considered form filtering by NUTS
        stops_keep = pd.read_csv(Path(heuristic_config['rail_heuristic']['rail_stations_considered_nuts']))

        def convert_to_list_of_tuples(value):
            if pd.isna(value):
                return []
            return ast.literal_eval(value)

        stops_keep['rail_stations'] = stops_keep['rail_stations'].apply(convert_to_list_of_tuples)

        rail_stations_considered = []
        for rs in stops_keep.rail_stations:
            for station in rs:
                rail_stations_considered.append(station[0])

        country = heuristic_config['rail_heuristic']['country']
    else:
        rail_stations_considered = pd.read_csv(Path(heuristic_config['rail_heuristic']['rail_stations_considered']))
        rail_stations_considered = list(rail_stations_considered['stop_id'])

    date_rail = pd.to_datetime(heuristic_config['rail_heuristic']['date'], format='%Y%m%d')

    df_h_rail, rs = generate_heuristic_rail((Path(heuristic_config['rail_heuristic']['gtfs_path'])),
                                            rail_stations_considered, date_rail, country,
                                            interval_range=heuristic_config['rail_heuristic'].get('interval_range'))

    df_h_rail.to_csv(Path(heuristic_config['output']['output_folder']) / 'rail_time_heuristics.csv', index=False)

    # Plot the result
    x_dist_rail_heur = []
    y_time_rail_heur = []

    # Iterate through each row of the DataFrame
    for index, row in df_h_rail.iterrows():
        # Append min_dist and max_dist to x list
        x_dist_rail_heur.extend([row['min_dist'], row['max_dist']])
        # Append time to y list twice per row
        y_time_rail_heur.extend([row['time'], row['time']])

    plt.figure(figsize=(10, 6))
    plt.scatter(rs['distance_km'], rs['duration_minutes'], alpha=0.6, edgecolors='w', linewidth=0.5)
    plt.plot(x_dist_rail_heur, y_time_rail_heur, marker='o', color='r')
    plt.title('Time Difference vs Distance')
    plt.xlabel('Distance (km)')
    plt.ylabel('Time Difference (minutes)')
    plt.xlim(0, max(rs['distance_km']))
    plt.ylim(0, max(rs['distance_km']))
    plt.grid(True)
    plt.savefig(Path(heuristic_config['output']['output_folder']) / 'rail_time_heuristics.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Heuristic computation', add_help=True)

    parser.add_argument('-tf', '--toml_file', help='TOML defining the network', required=True)

    # Parse parameters
    args = parser.parse_args()

    with open(Path(args.toml_file), mode="rb") as fp:
        heuristic_config = tomli.load(fp)

    if 'air_heuristic' in heuristic_config.keys():
        pass
        #create_air_heuristic(heuristic_config)

    if 'rail_heuristic' in heuristic_config.keys():
        create_rail_heuristic(heuristic_config)






