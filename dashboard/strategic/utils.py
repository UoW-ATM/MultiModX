import io
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from shapely.geometry import mapping
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

# --------- AUXILIARY FUNCTIONS ---------
def read_df_data_from_variable(csv_path, variable_type):
    df = pd.read_csv(csv_path)
    if variable_type == 'travel_time':
        df.rename(columns={'total_journey_time_per_pax': 'value'}, inplace=True)
        df['label'] = df['value'].apply(lambda x: str(round(x,1)))
    elif variable_type == 'demand_origin':
        df.rename(columns={'trips': 'value'}, inplace=True)
        df['label'] = df['value'].apply(lambda x: str(round(x,1)))
    elif variable_type == 'demand_served':
        df.rename(columns={'perc': 'value'}, inplace=True)
        df['label'] = "0%"
        df.loc[df.value>0, 'label'] = df[df.value>0].apply(lambda x: str(int(x['trips']))+ '/' +
                                                                     str(int(x['trips']/x['value']))+
                                         ' '+str(round(100*x['value'],2))+'%', axis=1)

    return df


# --------- AUXILIARY FUNCTIONS CREATE PLOTS ---------
def create_matrix_figure(df, index, columns, values, labels):
    # Pivot value and label data
    value_matrix = df.pivot(index=index, columns=columns, values=values)
    label_matrix = df.pivot(index=index, columns=columns, values="label")

    # Create custom hover text
    hover_text = value_matrix.copy().astype(str)
    for i in range(value_matrix.shape[0]):
        for j in range(value_matrix.shape[1]):
            #val = value_matrix.iloc[i, j]
            lab = label_matrix.iloc[i, j]
            #hover_text.iloc[i, j] = f"{labels}: {val}<br>Label: {lab}"
            hover_text.iloc[i, j] = f"{labels}: {lab}"

    fig = go.Figure(data=go.Heatmap(
        z=value_matrix.values,
        x=value_matrix.columns,
        y=value_matrix.index,
        colorscale="YlGnBu",
        hoverinfo="text",
        text=hover_text.values
    ))

    fig.update_layout(
        xaxis_title=columns,
        yaxis_title=index,
        coloraxis_colorbar=dict(title=labels)
    )

    return fig

# --------- GEOJSON CONVERSION ---------
def gdf_to_geojson(gdf, id_column):
    features = []
    for _, row in gdf.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
        feature = {
            "type": "Feature",
            "geometry": mapping(row.geometry),
            "properties": {id_column: row[id_column]}
        }
        features.append(feature)
    return {
        "type": "FeatureCollection",
        "features": features
    }


# --------- GENERATE AN EMPTY MAP ---------
def colour_nuts(fig, geojson_data, df_nuts, colour='#ffcccc', name=''):
    # Plot origin NUTS (fill as colour)
    fig.add_trace(go.Choropleth(
        geojson=geojson_data,
        locations=df_nuts["NUTS_ID"],
        z=[0] * len(df_nuts),  # dummy values
        featureidkey="properties.NUTS_ID",
        colorscale=[[0, colour], [1, colour]],
        showscale=False,
        marker_line_width=0.2,
        marker_line_color="black",
        name=name
    ))

    if len(df_nuts)==1:
        nuts_id = df_nuts.iloc[0]['NUTS_ID']

        # Get centroids from the NUTS GeoDataFrame
        # Extract features and build a GeoDataFrame from geojson
        gdf_from_geojson = gpd.GeoDataFrame.from_features(geojson_data["features"])
        dfg_origin_nuts = gdf_from_geojson[gdf_from_geojson['NUTS_ID'] == nuts_id].copy()

        # Reproject to a metric CRS (EPSG:3035 is good for Europe)
        gdf_from_geojson_proj = dfg_origin_nuts.set_crs(epsg=4326).to_crs(epsg=3035)

        # Compute centroids in the projected CRS
        dfg_origin_nuts["centroid"] = gdf_from_geojson_proj.geometry.centroid

        # Optionally transform centroids back to geographic CRS (EPSG:4326)
        dfg_origin_nuts["centroid"] = dfg_origin_nuts["centroid"].to_crs(epsg=4326)

        df_nuts = df_nuts.merge(
            dfg_origin_nuts[["NUTS_ID", "centroid"]],
            on="NUTS_ID",
            how="left"
        )

        df_nuts["lon"] = df_nuts["centroid"].apply(lambda point: point.x if point else None)
        df_nuts["lat"] = df_nuts["centroid"].apply(lambda point: point.y if point else None)

        # Add cross symbols at centroids
        if name !='':
            text = f"{name}<br>{nuts_id}"
        else:
            text = f"{nuts_id}"

        fig.add_trace(go.Scattergeo(
            lon=df_nuts["lon"],
            lat=df_nuts["lat"],
            mode="text",
            text=text,
            textfont=dict(color="black", size=10),
            showlegend=False
        ))

    return fig


def get_empty_map(nuts_dict, origin_nuts=None, destination_nuts=None, other_nuts=None, name_in_nuts=True):
    nuts_gdf_f =  pd.read_json(io.StringIO(nuts_dict['df_nuts']), orient='split')
    df_origin = nuts_gdf_f[nuts_gdf_f.NUTS_ID == origin_nuts].copy()
    df_destination = nuts_gdf_f[nuts_gdf_f.NUTS_ID == destination_nuts].copy()
    nuts_gdf_f = nuts_gdf_f[~(nuts_gdf_f.NUTS_ID == origin_nuts) & ~(nuts_gdf_f.NUTS_ID == destination_nuts)]

    geojson_data = nuts_dict['geojson_data'] #gdf_to_geojson(nuts_gdf, "NUTS_ID")
    nuts_gdf_f["grey_fill"] = 1

    fig = px.choropleth(
        nuts_gdf_f,
        geojson=geojson_data,
        locations=nuts_gdf_f["NUTS_ID"],
        featureidkey="properties.NUTS_ID",
        color="grey_fill",
        color_continuous_scale=[[0, "lightgrey"], [1, "lightgrey"]],  # single grey color scale
        hover_name="NUTS_ID"
    )

    if len(df_origin) > 0:
        if name_in_nuts:
            fig = colour_nuts(fig, geojson_data, df_origin, name="Origin")
        else:
            fig = colour_nuts(fig, geojson_data, df_origin)
    if len(df_destination) > 0:
        if name_in_nuts:
            fig = colour_nuts(fig, geojson_data, df_destination, colour="lightyellow", name="Destination")
        else:
            fig = colour_nuts(fig, geojson_data, df_destination, colour="lightyellow")
    if other_nuts is not None:
        if len(other_nuts) > 0:
            fig = colour_nuts(fig, geojson_data, other_nuts, colour='lightpink')

    fig.update_geos(fitbounds="locations", visible=False)
    # Hide color scale bar
    fig.update_layout(coloraxis_showscale=False, margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig


def get_map_w_data_from_nuts(nuts_dict, df_sub, origin_nuts, variable_selected=''): #nuts_dict, origin_nuts=None, destination_nuts=None):
    nuts_gdf_f = pd.read_json(io.StringIO(nuts_dict['df_nuts']), orient='split')
    df_origin = nuts_gdf_f[nuts_gdf_f.NUTS_ID == origin_nuts].copy()
    nuts_gdf_f = nuts_gdf_f[~(nuts_gdf_f.NUTS_ID == origin_nuts)]

    geojson_data = nuts_dict['geojson_data']  # gdf_to_geojson(nuts_gdf, "NUTS_ID")

    merged = nuts_gdf_f.set_index("NUTS_ID").copy()
    merged["value"] = df_sub.set_index("destination")["value"]
    merged["label"] = df_sub.set_index("destination")["label"]
    merged["color"] = merged["value"]

    merged_reset = merged.reset_index()
    unreachable = merged_reset[merged_reset["value"].isna()]

    fig = px.choropleth(
        merged_reset,
        geojson=geojson_data,
        locations="NUTS_ID",
        featureidkey="properties.NUTS_ID",
        color="color",
        color_continuous_scale="Viridis",
        range_color=(0, merged_reset["color"].max()),
        hover_name="NUTS_ID",
        hover_data={"NUTS_NAME": True,
                    "label": True},
        labels={"color": variable_selected}
    )

    # Plot unreachable NUTS(fill as grey)
    fig.add_trace(go.Choropleth(
        geojson=geojson_data,
        locations=unreachable["NUTS_ID"],
        z=[0] * len(unreachable),  # dummy values
        featureidkey="properties.NUTS_ID",
        colorscale=[[0, "lightgrey"], [1, "lightgrey"]],
        showscale=False,
        marker_line_width=0.2,
        marker_line_color="black",
        name="Unreachable"
    ))

    # Plot origin NUTS
    if len(origin_nuts) > 0:
        fig = colour_nuts(fig, geojson_data, df_origin, name='X')

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_coloraxes(colorbar_title=variable_selected)

    return fig


def add_paths_map(fig, df_it):
    trace_to_idx = []

    # --- Line thickness normalization ---
    pax_min = df_it['pax'].min()
    pax_max = df_it['pax'].max()
    range_span = pax_max - pax_min
    if range_span == 0:
        df_it['norm_pax'] = 5  # fixed if only one value
    else:
        df_it['norm_pax'] = 3 + 4 * ((df_it['pax'] - pax_min) / range_span)  # range: 3–7

    # --- Color map for pax ---
    norm = plt.Normalize(pax_min, pax_max)
    cmap = plt.get_cmap("viridis")
    df_it['line_color'] = df_it['pax'].apply(
        lambda x: f"rgb{tuple((np.array(cmap(norm(x)))[:3] * 255).astype(int))}".
                                             replace("np.int64(","").replace(")","")+")")

    # --- Plot nodes ---
    for _, row in df_it.iterrows():
        for node_id, coord, node_type in zip(row['path'], row['path_coords'], row['path_types']):
            marker_symbol = 'circle' if node_type == 'rail' else 'x'
            marker_color = 'black' if node_type == 'rail' else 'blue'

            # Text label
            fig.add_trace(go.Scattergeo(
                lat=[coord[0]],
                lon=[coord[1]],
                mode="text",
                text=node_id,
                textfont=dict(color="black", size=10),
                showlegend=False
            ))

            # Marker symbol
            fig.add_trace(go.Scattergeo(
                lat=[coord[0]],
                lon=[coord[1]],
                mode='markers',
                marker=dict(size=10, color=marker_color, symbol=marker_symbol),
                hoverinfo='text',
                text=f"Node: {node_id}",
                name=node_type,
                showlegend=False
            ))

    # --- Plot path segments ---
    start_trace_index = len(fig.data)
    for idx, row in df_it.iterrows():
        path = row['path']
        coords = row['path_coords']
        types = row['path_types']
        color = row['line_color']
        width = row['norm_pax']
        pax = row['pax']
        path_label = ' → '.join(path)

        for i in range(len(coords) - 1):
            segment_lat = [coords[i][0], coords[i + 1][0]]
            segment_lon = [coords[i][1], coords[i + 1][1]]
            segment_type = types[i]
            dash_style = 'dash' if segment_type == 'rail' else 'solid'

            fig.add_trace(go.Scattergeo(
                lat=segment_lat,
                lon=segment_lon,
                mode='lines',
                line=dict(width=width, color=color, dash=dash_style),
                hoverinfo='text',
                text=f"Pax: {pax}",
                name=path_label,
                showlegend=(i == 0),
                legendgroup=path_label,
                visible='legendonly',
                customdata=[idx] #* (len(coords) - 1)
            ))
            trace_to_idx.append(idx)

    line_trace_offset = start_trace_index



    fig.update_layout(
        legend=dict(
            y=0.85,  # Adjust this lower (1.0 is top, 0 is bottom)
            yanchor='top',
            x=1.02,  # Optional: move legend to the right of the plot
            xanchor='left',
            bgcolor='rgba(255,255,255,0.7)',  # semi-transparent background
            bordercolor='black',
            borderwidth=1
        )
    )


    return fig, {'trace_to_index': trace_to_idx, 'line_trace_offset': line_trace_offset}


def create_barchar_types(df):
    # Step 1: Group and sum pax
    mode_summary = df.groupby('mode_type')['pax'].sum().reset_index()

    # Step 2: Calculate percentage
    total_pax = mode_summary['pax'].sum()
    mode_summary['share_percent'] = (mode_summary['pax'] / total_pax) * 100

    # Step 3: Plot with Plotly
    fig = go.Figure(
        data=[
            go.Bar(
                x=mode_summary['mode_type'],
                y=mode_summary['share_percent'],
                text=mode_summary['share_percent'].apply(lambda x: f"{x:.1f}%"),
                textposition='auto',
                marker=dict(color=['#1f77b4', '#2ca02c', '#d62728'])  # optional color mapping
            )
        ]
    )

    # Step 4: Update layout
    fig.update_layout(
         font=dict(size=12),
         title="Passenger Mode Share by Type",
         xaxis_title="Mode Type",
         yaxis_title="Passenger Share (%)",
         yaxis=dict(range=[0, 100]),
         template='plotly_white',
         height=200,
         margin=dict(t=20, b=20, l=40, r=20)
    )

    return fig


def add_airports_map(fig, df_airports_coords):
    # Add the airport markers
    fig.add_trace(go.Scattergeo(
        lon=df_airports_coords['lon'],
        lat=df_airports_coords['lat'],
        text=df_airports_coords['airport'],  # label shown on hover
        mode='markers+text',
        marker=dict(
            size=8,
            color='blue',
            symbol='circle',
            line=dict(width=1, color='white')
        ),
        textposition='top center',
        name='Airports',
        hoverinfo='text',
        customdata=df_airports_coords['icao_id'],  # allows click detection later
        showlegend=False
    ))

    return fig


def add_catchment_areas_map(fig, df, nuts_dict, pax_to_from='pax_to'):
    nuts_gdf_f = pd.read_json(io.StringIO(nuts_dict['df_nuts']), orient='split')
    geojson_data = nuts_dict['geojson_data']  # gdf_to_geojson(nuts_gdf, "NUTS_ID")

    # Group by nuts3 and type, sum pax_total
    grouped = df.groupby(['nuts3', 'type'])['pax_total'].sum().reset_index()

    # Pivot so we have separate columns for 'to' and 'from'
    summary = grouped.pivot(index='nuts3', columns='type', values='pax_total').reset_index()

    # Rename columns for clarity, fill NaNs with 0
    summary_df = summary.rename(columns={'to': 'pax_to', 'from': 'pax_from'}).fillna(0)

    max_pax = max(summary_df['pax_from'].max(), summary_df['pax_to'].max())

    merged = nuts_gdf_f.set_index("NUTS_ID").copy()
    merged["value"] = summary_df.set_index("nuts3")[pax_to_from]
    merged["label"] = summary_df.set_index("nuts3")[pax_to_from]
    merged["color"] = merged["value"]
    # Get color scale range shared for both pax_from and pax_to
    merged_reset = merged.reset_index()

    merged_reset = merged_reset[merged_reset['value']>0]

    # Add 'to airport' layer
    if pax_to_from == 'pax_to':
        name = 'To Airport'
        colortit = 'Pax to Airport'
    else:
        name = 'From Airport'
        colortit = 'Pax from Airport'

    fig.add_trace(go.Choropleth(
        geojson=geojson_data,
        locations=merged_reset['NUTS_ID'],
        z=merged_reset['value'],
        featureidkey="properties.NUTS_ID",
        colorscale="YlGnBu",
        zmin=0,
        zmax=max_pax,
        colorbar_title=colortit,
        name=name,
        hovertext=merged_reset['NUTS_NAME'],
        hoverinfo='text+z',
    ))


    if pax_to_from == 'pax_to':
        df_multimodal = df[(df.type=='to') & (df.pax_rail_multimodal>0)]
    else:
        df_multimodal = df[(df.type == 'from') & (df.pax_rail_multimodal > 0)]

    df_multimodal = df_multimodal[['nuts3']].drop_duplicates()
    if len(df_multimodal) > 0:
        gdf_from_geojson = gpd.GeoDataFrame.from_features(geojson_data["features"])
        dfg_nuts = gdf_from_geojson[gdf_from_geojson['NUTS_ID'].isin(df_multimodal.nuts3)].copy()

        # Reproject to a metric CRS (EPSG:3035 is good for Europe)
        gdf_geojson_proj = dfg_nuts.set_crs(epsg=4326).to_crs(epsg=3035)

        # Compute centroids in the projected CRS
        dfg_nuts["centroid"] = gdf_geojson_proj.geometry.centroid

        # Optionally transform centroids back to geographic CRS (EPSG:4326)
        dfg_nuts["centroid"] = dfg_nuts["centroid"].to_crs(epsg=4326)


        dfg_nuts["lon"] = dfg_nuts["centroid"].apply(lambda point: point.x if point else None)
        dfg_nuts["lat"] = dfg_nuts["centroid"].apply(lambda point: point.y if point else None)

        fig.add_trace(go.Scattergeo(
            lon=dfg_nuts["lon"],
            lat=dfg_nuts["lat"],
            mode="text",
            text="o",
            textfont=dict(color="black", size=10),
            showlegend=False
        ))


    return fig
