import io
import os
import ast
from dash import State, Input, Output, exceptions, no_update
import pandas as pd
from config import DATA_FOLDER, VARIABLES
from utils import (get_empty_map, get_map_w_data_from_nuts, create_matrix_figure,
                   read_df_data_from_variable, gdf_to_geojson, add_paths_map,
                   create_barchar_types, add_airports_map, add_catchment_areas_map)
from data_loader import (load_nuts3_geodata, load_rail_stops, load_airports, read_pax_itineraries, read_pax_paths)


def load_initial_data():
    df_nuts = load_nuts3_geodata()
    geojson_data = gdf_to_geojson(df_nuts, "NUTS_ID")
    df_nuts = df_nuts.drop({'geometry'}, axis=1).to_json(orient='split')
    df_rail_stops = load_rail_stops().to_json(orient='split')
    df_airports = load_airports().to_json(orient='split')
    return {'df_nuts': df_nuts, 'geojson_data': geojson_data,
            'df_rail_stops': df_rail_stops, 'df_airports': df_airports}


def register_callbacks(app):
    # --- LOAD CACHE DATA ---
    @app.callback(
        Output("cache-data", "clear_data"),
        Input("reset-cache-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_cache(n):
        return True  # This triggers dcc.Store to clear its value

    # Load data when the app starts
    @app.callback(
        Output("cache-data", "data"),
        Input("cache-data", "data"),
        prevent_initial_call="initial_duplicate"
    )
    def populate_cache(existing_data):
        if existing_data is None:
            data = load_initial_data()
            return data
        raise exceptions.PreventUpdate

    # --------- CALLBACK TO COMPOSE FOLDER ---------
    @app.callback(
        Output('case-study-folder', 'data'),
        Input('cs-select', 'value'),
        Input('pp-select', 'value'),
        Input('nd-select', 'value'),
        Input('so-select', 'value')
    )
    def compose_folder(cs, pp, nd, so):
        if cs and pp and nd and so:
            return f"processed_cs{cs}.pp{pp}.nd{nd}.so{so}"
        return None

    @app.callback(
        Output('selection-summary', 'children'),
        Input('case-study-folder', 'data')
    )
    def update_selection_summary(folder_name):
        if folder_name is None:
            return "Please select all parameters."

        full_path = os.path.join(DATA_FOLDER, folder_name)
        folder_name_print = folder_name.split("_")[1]
        if os.path.exists(full_path):
            return f"{folder_name_print}"
        else:
            return f"{folder_name_print} (❌ Not valid)"

    # --------- CALLBACK FOR CLICKING UPDATE ---------
    @app.callback(
        Output('click-id-store', 'data'),
        Input('main-graph', 'clickData'),
    )
    def store_click_id(clickData):
        if clickData is None:
            return None
        return clickData

    # --------- CALLBACK FOR TABLE UPDATE WITH MAP ---------
    @app.callback(
        Output('map-data-table', 'data'),
        Output('map-table-container', 'style'),
        Output('selected-path-indices', 'data'),
        Input('main-graph', 'restyleData'),
        Input('visualisation-radio', 'value'),
        State('selected-path-indices', 'data'),
        State('cache-df_it', 'data')
    )
    def update_table_on_path_selection(restyle_data, vis_type, selected_indices, cache_df_it):
        if vis_type != 'map' or not restyle_data:
            return [], {'display': 'none'}, {'indices':[], 'od':''}

        df_it = pd.DataFrame(cache_df_it['df_it'])
        od = ''
        if len(df_it)>0:
            od = df_it.iloc[0]['origin']+'_'+df_it.iloc[0]['destination']

        if od != selected_indices['od']:
            # New origin destination
            selected_indices['od'] = od
            selected_indices['indices'] = []

        dict_trace_info = cache_df_it['trace_info']

        trace_to_idx = dict_trace_info['trace_to_index']
        offset = dict_trace_info['line_trace_offset']

        visibility_changes, trace_indices = restyle_data
        selected_indices = selected_indices['indices']
        # Start from current stored selection
        selected_set = set(selected_indices or [])

        if 'visible' in visibility_changes:
            for visible, trace_idx in zip(visibility_changes['visible'], trace_indices):
                adjusted_index = trace_idx - offset
                if 0 <= adjusted_index < len(trace_to_idx):
                    path_idx = trace_to_idx[adjusted_index]
                    if visible is True:
                        selected_set.add(path_idx)
                    else:
                        selected_set.discard(path_idx)

        if not selected_set:
            return [], {'display': 'none'}, {'indices':[], 'od':od}

        filtered_df = df_it.loc[list(selected_set)].drop_duplicates(subset='path').copy()
        filtered_df['path'] = filtered_df['path'].apply(lambda x: str(x))

        return filtered_df[[
                        "origin", "destination", "path", "total_time", "total_waiting_time", "fare",
                        "access_time", "egress_time", "d2i_time", "i2d_time", "pax"
                    ]].to_dict('records'), {'display': 'block'}, {'indices': list(selected_set), 'od':od}



    # --------- MAIN CALLBACK ---------
    @app.callback(
        Output('main-graph', 'figure'),
        Output('current-origin', 'data'),
        Output('current-destination', 'data'),
        Output('map-title', 'children'),
        Output('bar-chart', 'figure'),
        Output('bar-chart-container', 'style'),
        Output('cache-df_it', 'data'),
        Input('case-study-folder', 'data'),
        Input('visualisation-radio', 'value'),
        Input('variable-dropdown', 'value'),
        Input('click-id-store', 'data'),
        State('current-origin', 'data'),
        State('current-destination', 'data'),
        State('cache-data', 'data')
    )

    # --------- REMAINING CALLBACK ---------
    def update_output(case_study, vis_type, variable, clickData, current_origin, current_destination, cached_data):
        if vis_type == 'map' and (not variable or not case_study):
            fig = get_empty_map(cached_data)
            return fig, current_origin, current_destination, "", {}, {'display': 'none'}, None

        if not case_study or not vis_type or not variable:
            fig = get_empty_map(cached_data)
            return fig, current_origin, current_destination, "", {}, {'display': 'none'}, None

        variable_selected = variable
        variable = VARIABLES[variable_selected]

        # Check if variable selected needs origin and destination
        if variable['type'].startswith('od_'):

            # Logic to manage the selection of origin and destination
            new_nuts = None
            if clickData:
                point = clickData['points'][0]
                # Prefer 'location', fallback to 'customdata'
                new_nuts = point.get('location') or (
                    point.get('customdata')[0] if point.get('customdata') else None
                )

            if new_nuts is None:
                current_origin = None
                current_destination = None
                message = 'Click a region to select new origin'
            else:
                if current_origin == new_nuts:
                    # Clicking same origin and destination, clear
                    current_origin = None
                    message = 'Click a region to select new origin'
                if current_origin is None:
                    current_origin = new_nuts
                    message = f'Click a region to select destination\nOrigin: {current_origin}'
                elif current_destination is None:
                    current_destination = new_nuts
                    message = 'Click a region to select new origin'
                else:
                    current_origin = new_nuts
                    message = f'Click a region to select destination\nOrigin: {current_origin}'
                    current_destination = None

            if current_origin is None:
                fig = get_empty_map(cached_data, current_origin, current_destination)
                return fig, current_origin, current_destination, message, {}, {'display': 'none'},None
            elif current_destination is None:
                # We have origin but not destination
                df = None
                if variable['type']=='od_trips':
                    df_it = read_pax_itineraries(case_study, "paths_itineraries",
                                                 variable['files']['pax_assigned_to_itineraries'])
                    if df_it is None:
                        fig = get_empty_map(cached_data)
                        return fig, None, None, f"Not exists: {variable['files']['pax_assigned_to_itineraries']}", {}, {'display': 'none'},None

                    df_it = df_it[(df_it.origin == current_origin) & (df_it.pax > 0)]
                    df = df_it
                elif variable['type']=='od_paths':
                    df_p = read_pax_paths(case_study, "paths_itineraries",
                                                 variable['files'])
                    df_p = df_p[(df_p.origin == current_origin) & (df_p.num_pax > 0)]
                    df = df_p
                    if df_p is None:
                        fig = get_empty_map(cached_data)
                        return fig, None, None, f"Not exists: {variable['files']['pax_assigned_to_paths']}", {}, {'display': 'none'},None

                fig = get_empty_map(cached_data, current_origin, current_destination,
                                    df[['destination']].drop_duplicates().rename(columns={'destination':'NUTS_ID'}))
                return fig, current_origin, current_destination, message, {}, {'display': 'none'},  None
            elif (current_origin is None) and (current_destination is None):
                # We don't have origin nor destination
                fig = get_empty_map(cached_data, current_origin, current_destination)
                return fig, current_origin, current_destination, message, {}, {'display': 'none'}, None

            # We have the origin and destination
            if variable['type']=='od_trips':
                # Reading origin-destination trips info
                df_it = read_pax_itineraries(case_study, "paths_itineraries",
                                             variable['files']['pax_assigned_to_itineraries'])

                df_it = df_it[(df_it.origin==current_origin) & (df_it.destination==current_destination) &
                                  (df_it.pax>0)]
                if len(df_it) == 0:
                    fig = get_empty_map(cached_data, current_origin, current_destination)
                    return fig, current_origin, current_destination, 'No paths between selected origin and destination', {}, {'display': 'none'}, None

                # Define aggregation logic
                agg_dict = {
                    'total_waiting_time': 'mean',
                    'total_time': 'mean',
                    'fare': 'mean',
                    'access_time': 'mean',
                    'egress_time': 'mean',
                    'd2i_time': 'mean',
                    'i2d_time': 'mean',
                    'pax': 'sum'
                }

                # Group by origin, destination, and path
                df_it = df_it.groupby(['origin', 'destination', 'path', 'type'], as_index=False).agg(agg_dict)

                df_rail_stops = pd.read_json(io.StringIO(cached_data['df_rail_stops']), orient='split',
                                             dtype={'stop_id:str'})
                df_airports = pd.read_json(io.StringIO(cached_data['df_airports']), orient='split')
                dict_airport_coords = dict(zip(df_airports['icao_id'], zip(df_airports['lat'], df_airports['lon'])))
                dict_rail_coords = dict(
                    zip(df_rail_stops['stop_id'], zip(df_rail_stops['stop_lat'], df_rail_stops['stop_lon'])))
                rail_ids = set(df_rail_stops['stop_id'])

                def get_coord(node_id):
                    return dict_airport_coords.get(node_id) or dict_rail_coords.get(node_id)

                def get_path_types(path_list):
                    return ['rail' if node in rail_ids else 'airport' for node in path_list]

                df_it['path'] = df_it['path'].apply(ast.literal_eval)
                df_it['path_coords'] = df_it['path'].apply(lambda lst: [get_coord(x) for x in lst])
                df_it['path_types'] = df_it['path'].apply(get_path_types)
                df_it = df_it.sort_values('pax', ascending=False).reset_index(drop=True)

                # Read the data and plot what's needed
                fig = get_empty_map(cached_data, current_origin, current_destination, name_in_nuts=False)
                fig, trace_info = add_paths_map(fig, df_it)
                message = f'Origin: {current_origin} -- Destination: {current_destination}'

                dict_df_it = {'df_it': df_it.to_dict('records')}
                dict_df_it['trace_info'] = trace_info

                # Process percentage per type
                def classify_mode(mode_str):
                    modes = set(mode_str.split('_'))
                    if modes == {'flight'}:
                        return 'air'
                    elif modes == {'rail'}:
                        return 'rail'
                    else:
                        return 'multimodal'

                df_it['mode_type'] = df_it['type'].apply(classify_mode)
                fig_bar = create_barchar_types(df_it)

                return fig, current_origin, current_destination, message, fig_bar, {'display': 'block'}, dict_df_it

            elif variable['type']=='od_paths':
                df_p = read_pax_paths(case_study, "paths_itineraries",
                                      variable['files'])
                df_p = df_p[(df_p.origin == current_origin) & (df_p.destination == current_destination) &
                              (df_p.num_pax > 0)]
                if len(df_p) == 0:
                    fig = get_empty_map(cached_data, current_origin, current_destination)
                    return fig, current_origin, current_destination, 'No paths between selected origin and destination', {}, {'display': 'none'}, None

                df_rail_stops = pd.read_json(io.StringIO(cached_data['df_rail_stops']), orient='split',
                                             dtype={'stop_id:str'})
                df_airports = pd.read_json(io.StringIO(cached_data['df_airports']), orient='split')
                dict_airport_coords = dict(zip(df_airports['icao_id'], zip(df_airports['lat'], df_airports['lon'])))
                dict_rail_coords = dict(
                    zip(df_rail_stops['stop_id'], zip(df_rail_stops['stop_lat'], df_rail_stops['stop_lon'])))
                rail_ids = set(df_rail_stops['stop_id'])

                def get_coord(node_id):
                    return dict_airport_coords.get(node_id) or dict_rail_coords.get(node_id)

                def get_path_types(path_list):
                    return ['rail' if node in rail_ids else 'airport' for node in path_list]

                df_p['path'] = df_p['path'].apply(ast.literal_eval)
                df_p['path_coords'] = df_p['path'].apply(lambda lst: [get_coord(x) for x in lst])
                df_p['path_types'] = df_p['path'].apply(get_path_types)
                df_p = df_p.sort_values('num_pax', ascending=False).reset_index(drop=True)
                df_p.rename(columns={'num_pax':'pax', 'total_travel_time':'total_time',
                                     'total_cost':'fare'}, inplace=True)

                # Read the data and plot what's needed
                fig = get_empty_map(cached_data, current_origin, current_destination, name_in_nuts=False)
                fig, trace_info = add_paths_map(fig, df_p)
                message = f'Origin: {current_origin} -- Destination: {current_destination}'

                dict_df_p = {'df_it': df_p.to_dict('records')}
                dict_df_p['trace_info'] = trace_info

                def classify_mode_paths(paths):
                    modes = set(paths)
                    if modes == {'flight'}:
                        return 'air'
                    elif modes == {'rail'}:
                        return 'rail'
                    else:
                        return 'multimodal'

                # Process percentage per type
                df_p['mode_type'] = df_p['path_types'].apply(classify_mode_paths)
                fig_bar = create_barchar_types(df_p)

                return fig, current_origin, current_destination, message, fig_bar, {'display': 'block'}, dict_df_p

            # Read the data and plot what's needed
            fig = get_empty_map(cached_data, current_origin, current_destination)
            return fig, current_origin, current_destination, message, {}, {'display': 'none'}, cached_data


        else:
            # Load CSV data single output file
            csv_path = os.path.join(DATA_FOLDER, case_study, "indicators", variable['file'])
            if os.path.isfile(csv_path):
                df = read_df_data_from_variable(csv_path, variable['type'])
            else:
                fig = get_empty_map(cached_data)
                return fig, None, None, f"Not exists: {csv_path}", {}, {'display': 'none'}, None

            if (variable['type']=='catchment_areas') and vis_type=="matrix":
                fig = get_empty_map(cached_data)
                return fig, None, None, f"For catchment area use map", {}, {'display': 'none'}, None

            if vis_type == "matrix":
                # ------ HEATMAP MATRIX MODE ------
                fig = create_matrix_figure(df, "origin", "destination", values="value", labels=variable_selected)
                return fig, None, None, variable_selected, {}, {'display': 'none'}, None

            elif vis_type == "map":
                # ------ MAP MODE ------
                if variable['type'] == 'catchment_areas':
                    airport_clicked = None
                    if clickData and 'customdata' in clickData['points'][0]:
                        airport_clicked = clickData['points'][0]['customdata']

                    df_airports = pd.read_json(io.StringIO(cached_data['df_airports']), orient='split')
                    if airport_clicked:
                        # Filter only airport selected
                        df = df[df['airport']==airport_clicked]

                    df_airports_coords = df[['airport']].drop_duplicates().merge(df_airports, left_on='airport',
                                                                               right_on='icao_id')
                    fig = get_empty_map(cached_data)
                    fig = add_airports_map(fig, df_airports_coords)
                    if airport_clicked:
                        # Plot the catchment areas:
                        fig_to = fig
                        fig_from = get_empty_map(cached_data)
                        fig_from = add_airports_map(fig_from, df_airports_coords)
                        fig_to = add_catchment_areas_map(fig_to, df, cached_data, pax_to_from='pax_to')
                        fig_from = add_catchment_areas_map(fig_from, df, cached_data, pax_to_from='pax_from')
                        return fig_from, None, None, f"Catchment area {airport_clicked}", fig_to, {'display': 'block'}, None
                    else:
                        return fig, None, None, "Click airport for catchment area", {}, {'display': 'none'}, None
                else:
                    new_origin = None
                    if clickData and 'location' in clickData['points'][0]:
                        new_origin = clickData['points'][0]['location']

                    origin = new_origin or current_origin
                    if not origin:
                        fig = get_empty_map(cached_data)
                        return fig, None, None, "Click a region to start visualising from", {},{'display': 'none'}, None

                    df_sub = df[df["origin"] == origin]
                    df_nuts = pd.read_json(io.StringIO(cached_data['df_nuts']), orient='split')
                    nuts_name = df_nuts[df_nuts.NUTS_ID == origin].iloc[0]['NUTS_NAME']

                    fig = get_map_w_data_from_nuts(cached_data, df_sub, origin, variable_selected)

                    return fig, origin, None, f"Origin: {origin} - {nuts_name}", {}, {'display': 'none'}, None

        return {}, None, None, "", {}, {'display': 'none'}, None

