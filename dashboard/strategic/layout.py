
from dash import html, dcc, dash_table
from data_loader import get_case_study_folders, get_components_cs, load_variable_options

def create_layout():
    components = get_components_cs(get_case_study_folders())
    variable_options = load_variable_options()

    layout = html.Div([
          # stores data per user session
        html.H1("Strategic Multimodal Transport Dashboard"),
        html.Button("Reset Cache", id="reset-cache-btn"),
        dcc.Store(id='cache-data', storage_type='session'),
        dcc.Store(id='cache-df_it', storage_type='session'),
        dcc.Store(id='click-id-store'),
        dcc.Store(id='selected-path-indices', data=[]),

        html.Div([
            html.Div([
                html.Label("Select Case Study (CS):"),
                dcc.RadioItems(
                    id='cs-select',
                    options=[{"label": f"cs{v}", "value": v} for v in sorted(components["cs"])],
                    inline=True
                )
            ]),
            html.Div([
                html.Label("Select Policy Package (PP):"),
                dcc.RadioItems(
                    id='pp-select',
                    options=[{"label": f"pp{v}", "value": v} for v in sorted(components["pp"])],
                    inline=True
                )
            ]),
            html.Div([
                html.Label("Select Network Definition (ND):"),
                dcc.RadioItems(
                    id='nd-select',
                    options=[{"label": f"nd{v}", "value": v} for v in sorted(components["nd"])],
                    inline=True
                )
            ]),
            html.Div([
                html.Label("Select Schedule Optimiser (SO):"),
                dcc.RadioItems(
                    id='so-select',
                    options=[{"label": f"so{v}", "value": v} for v in sorted(components["so"])],
                    inline=True
                )
            ])
        ], style={"marginBottom": "20px"}),

        html.Div(id='selection-summary', style={
            'marginTop': '20px',
            'marginBottom': '30px',  # Add space below
            'fontWeight': 'bold',
            'fontSize': '16px',
            'textAlign': 'center',  # Center the text
            'width': '100%'  # Ensure full width for centering
        }),

        html.Div([
            html.Label("Select visualisation type:"),
            dcc.RadioItems(
                id='visualisation-radio',
                options=[
                    {'label': 'Map', 'value': 'map'},
                    {'label': 'Matrix', 'value': 'matrix'}
                ],
                value='map'
            )
        ]),

        html.Label("Select Variable:"),
        dcc.Dropdown(
            id='variable-dropdown',
            options=variable_options,
            placeholder="Choose a variable"
        ),

        html.Div(id='map-title', style={"marginTop": "20px", "fontWeight": "bold"}),

        html.Div(id='bar-chart-container', children=[
            dcc.Graph(id='bar-chart')
        ], style={'display': 'none'}),

        html.Div(id='map-table-container', children=[
            dash_table.DataTable(
                id='map-data-table',
                columns=[
                    {"name": col, "id": col} for col in [
                        "origin", "destination", "path", "total_time", "total_waiting_time", "fare",
                        "access_time", "egress_time", "d2i_time", "i2d_time", "pax"
                    ]
                ],
                style_table={'overflowX': 'auto'},
            )
        ], style={'display': 'none'}),

        dcc.Graph(id='main-graph'),

        dcc.Store(id='current-origin'),
        dcc.Store(id='current-destination'),
        dcc.Store(id='case-study-folder')
    ])

    return layout

