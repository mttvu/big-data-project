import pandas as pd
# Dash imports
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash
# Eigen imports
import os, sys
# sys.path.append(os.getcwd() + '/..')
from covadem.data.rws_data import RWSData
from covadem.data.bodem_ligging import calculate_bedlevel
from covadem.machine_learning.kritische_zone import get_zones_waterlevel, get_zones_legenda
from covadem.machine_learning.Algoritm import TimeSeriesPrediction
from covadem.machine_learning.SnellePyTorch import thing, LSTM

# from backports.datetime_fromisoformat import MonkeyPatch
# MonkeyPatch.patch_fromisoformat()

# Plotly imports
import plotly.express as px
import plotly.graph_objects as go

# Overige imports
from datetime import datetime
from datetime import timedelta
import base64

# Torch imports
import torch

external_stylesheets = [dbc.themes.SLATE]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

rws = RWSData()


def get_meetpunten_trace(kaart, status, selected_color, normal_color, click=None):
    meetpunten_df = rws.get_coordinates_meetpunten()
    # filter op status
    if status == 'Normale waterstand':
        meetpunten_df = meetpunten_df[(meetpunten_df[kaart] == status) | (meetpunten_df[kaart] == 'Normaal')]
    else:
        meetpunten_df = meetpunten_df[meetpunten_df[kaart] == status]

    trace = go.Scattermapbox(
        lat=meetpunten_df['lat'],
        lon=meetpunten_df['lng'],
        mode='markers',
        name=status,
        marker=go.scattermapbox.Marker(
            size=14,
            color=normal_color
        ),
        selected={
            'marker':
                {
                    'color': selected_color
                }
        },
        hovertext=meetpunten_df['MEETPUNT_IDENTIFICATIE'],
        customdata=meetpunten_df['MEETPUNT_IDENTIFICATIE'],
    )
    if click:
        trace.selectedpoints = [click.get('points')[0]['pointIndex']]

    return trace


def generate_plot():
    gridline_color = '#aaa'
    background_color = '#32383e'
    fig = go.Figure(
        layout=go.Layout(
            template='plotly_dark',
            plot_bgcolor=background_color,
            paper_bgcolor=background_color,
            autosize=True,
            margin=go.layout.Margin(l=0, r=35, t=0, b=0),
            showlegend=True,
        ))

    fig.update_xaxes(linecolor=gridline_color, gridcolor=gridline_color)
    fig.update_yaxes(linecolor=gridline_color, gridcolor=gridline_color)
    return fig


def generate_graph_container(figure):
    return dcc.Graph(figure=figure, config=get_graph_config(), style={'height': '51vh'})


def get_graph_config():
    return {
        "modeBarButtonsToRemove": ['lasso2d', 'toImage', 'autoScale2d', 'toggleSpikelines']
    }


def get_covadem_logo():
    covadem_path = 'Covadem2.png'
    covadem_afbeelding = base64.b64encode(open(covadem_path, 'rb').read()).decode('ascii')
    return html.Img(src='data:image/png;base64,{}'.format(covadem_afbeelding),
                    style={'height': '30%', 'width': '30%'})


mapbox_access_token = open(".mapbox_token").read()

meetpunten_kaart = go.Figure(
    data=[
        get_meetpunten_trace('waterdepthStatus','kritisch', '#636EFA', '#EF553B'),
        get_meetpunten_trace('waterdepthStatus','normaal', '#636EFA', '#00CC96')
    ],
    layout=go.Layout(
        paper_bgcolor='#32383e',
        autosize=True,
        margin=go.layout.Margin(l=0, r=35, t=0, b=0),
        mapbox=dict(
            center=dict(lat=51.8513629, lon=5.3937509),
            style="dark",
            zoom=8,
            accesstoken=mapbox_access_token
        ),
        legend=dict(
            bgcolor='#32383e',
            font=dict(color='#ffffff')

        )
    ),
)

app.layout = html.Div(children=[
    # grafiek data
    dcc.Store(id="store"),
    # meetpunt refresh interval
    dcc.Interval(id='status-interval', interval=5 * 1000, n_intervals=0),
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col(width=4),
                # Dashboard titel
                dbc.Col(html.H1('CoVadem Analytics',
                                style={'color': '#ffffff', 'fontSize': 30, 'text-align': 'center'}),
                        width=4),
                dbc.Col(html.Div(children=get_covadem_logo()), width=4, style={'text-align': 'end'}),
            ], style={
                'height': '10vh'
            }),

            # Date range picker
            dbc.Row([
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='kaart-dropdown',
                                options=[
                                    {'label': 'Waterhoogte', 'value': 'waterlevelStatus'},
                                    {'label': 'Waterdiepte', 'value': 'waterdepthStatus'},
                                    {'label': 'Bodemligging', 'value': 'bedlevelStatus'}
                                ],
                                value='waterdepthStatus'
                            )
                ]), style={'height': '12vh'}), width=6),
                dbc.Col(dbc.Card(dbc.CardBody([
                    dbc.Row([
                        dbc.Col(dcc.DatePickerRange(
                            id='date-picker',
                            min_date_allowed=datetime(2018, 1, 1),
                            max_date_allowed=datetime(2020, 1, 1),
                            initial_visible_month=datetime(2018, 1, 1),
                        ), style={'padding-right': '0px'}),
                        dbc.Col(dcc.Dropdown(
                            id='forecast-dropdown',
                            options=[
                                {'label': 'Week', 'value': 'Week'},
                                {'label': 'Maand', 'value': 'Maand'},
                            ],
                            value='Week'
                        )
                            , style={'padding-left': '0px'})
                    ])
                ], style={'height': '12vh'})), width=6)
            ], justify="start", style={'align-items': 'start', 'padding': '10px'}),

            # plots
            dbc.Row([

                # meetpunten kaart
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            dcc.Graph(id='map-rws-meetpunten',
                                      figure=meetpunten_kaart,
                                      )])), width=6),

                dbc.Col(
                    # tabs van de grafieken
                    dbc.Card(
                        dbc.CardBody(
                            [   # title meetpunt
                                dbc.Spinner(id='loading-icon', fullscreen=True, fullscreen_style={'background-color': 'transparent'},
                                            children=html.H3(id='meetpunt-naam', children='Meetpunt')),
                                dbc.Tabs(
                                [
                                    dbc.Tab(label="Waterhoogte", tab_id="waterhoogte"),
                                    dbc.Tab(label="Waterdiepte", tab_id="waterdiepte"),
                                    dbc.Tab(label="Bodemligging", tab_id="bodemligging"),
                                    dbc.Tab(label="Accuracy", tab_id="accuracy"),
                                    dbc.Tab(label="All", tab_id="all")

                                ],
                                id="tabs",
                                active_tab="waterhoogte",
                            ),
                                # grafiek
                                dcc.Loading(id='loading-grafiek', type='default',
                                            children=html.Div(id="tab-content")),
                            ])), width=6)
            ])
        ]), color='dark'
    )
])


# handles tab clicks, displays the correct plot
@app.callback(
    dash.dependencies.Output("tab-content", "children"),
    [dash.dependencies.Input("tabs", "active_tab"),
     dash.dependencies.Input("store", "data")],
)
def render_tab_content(active_tab, data):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """
    if active_tab and data is not None:
        return generate_graph_container(figure=data[active_tab])
    return generate_graph_container(figure=generate_plot())


def get_waterlevel_plot(df, forecast):
    wl_plot = generate_plot()
    total_time = df['time']
    prediction_trace = None
    try:
        ob = thing(DATA=df, N_FORECAST=(forecast), StartingTimeDelta=timedelta(minutes=10),
                    newTimeDelta=timedelta(days=1), timeCol='time', dataCol='NUMERIEKEWAARDE', seq=30)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = LSTM(device=device)
        state = torch.load("..//machine_learning//MODELS//WaterLevel//" + str(df['MEETPUNT_IDENTIFICATIE'][0]) + "_waterLevel_V2.pt", map_location=device)
        model.load_state_dict(state)
        wh_pre = ob.predict(model)
        total_time = pd.concat([df, wh_pre]).sort_values('time').reset_index(drop=True)['time']
        prediction_trace = go.Scatter(x=wh_pre['time'], y=wh_pre['NUMERIEKEWAARDE'], mode='lines', line_color='red', name='Prediction')
    except Exception:
        pass

    wl_plot.update_xaxes(title_text='Tijd')
    wl_plot.update_yaxes(title_text='Waterhoogte t.o.v NAP (m)')

    # zones
    zones = get_zones_waterlevel(df['MEETPUNT_IDENTIFICATIE'][0])

    for i, zone in zones.iterrows():
        zone_limit = zone['to']
        if pd.isna(zone['to']):
            zone_limit = zone['from'] +1
        wl_plot.add_trace(
            go.Scatter(
                x=total_time,
                y=[zone_limit for x in range(len(total_time))],
                line_color=zone['color'],
                fill='tonexty',
                name=zone['label'],
                mode='lines'
            )
        )
    wl_plot.add_trace(go.Scatter(x=df['time'], y=df['NUMERIEKEWAARDE'], name='Waterhoogte'))
    
    if prediction_trace:
        wl_plot.add_trace(prediction_trace)

    wl_plot.update_layout(yaxis_range=[df['NUMERIEKEWAARDE'].min()-1,df['NUMERIEKEWAARDE'].max()+1])


    return wl_plot


def get_waterdepth_plot(df, forecast, meetpunt):
    wd_plot = generate_plot()
    total_time = df['time']

    try:
        ob = thing(DATA=df, N_FORECAST=(forecast), StartingTimeDelta=timedelta(minutes=10),
                    newTimeDelta=timedelta(days=1), timeCol='time', dataCol='y', seq=30)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = LSTM(device=device)
        state = torch.load("..//machine_learning//MODELS//WaterDepth//" + str(meetpunt) + "_waterDiepte_V1.pt", map_location=device)
        model.load_state_dict(state)
        wd_pre = ob.predict(model)
        total_time = pd.concat([df, wd_pre]).sort_values('time').reset_index(drop=True)['time']
        wd_plot.add_trace(go.Scatter(x=wd_pre['time'], y=wd_pre['y'], mode='markers', marker_color='red', name='Prediction'))
    except Exception:
        pass

    wd_plot.update_xaxes(title_text='Tijd')
    wd_plot.update_yaxes(title_text='Waterdiepte (m)')
    wd_plot.add_trace(go.Scatter(x=df['time'], y=df['y'], mode='markers', name='Data'))

    # Add line that indicates the critical zone limit. (2.80m)
    wd_plot.add_trace(go.Scatter(
        x=total_time,
        y=[2.8 for x in range(len(total_time))],
        line_color='rgb(139,0,0)',
        fill='tozeroy',
        name='Critical zone limit',
        mode='lines'
    ))

    return wd_plot


def get_bedlevel_plot(bedlevel_df, forecast, meetpunt):
    bl_plot = generate_plot()

    try:
        ob = thing(DATA=bedlevel_df, N_FORECAST=(forecast), StartingTimeDelta=timedelta(minutes=10),
                    newTimeDelta=timedelta(days=1), timeCol='index', dataCol='bedlevel', seq=7)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = LSTM(device=device)
        state = torch.load("..//machine_learning//MODELS//BodelLigging//" + str(meetpunt) + "_bodemligging_V1.pt", map_location=device)
        model.load_state_dict(state)
        bl_pre = ob.predict(model)
        bl_plot.add_trace(go.Scatter(x=bl_pre['time'], y=bl_pre['bedlevel'], mode='markers', marker_color='red', name='Prediction'))
    except Exception:
        pass

    bl_plot.update_xaxes(title_text='Tijd')
    bl_plot.update_yaxes(title_text='Bodemligging t.o.v NAP (m)')
    bl_plot.add_trace(go.Scatter(x=bedlevel_df.index, y=bedlevel_df['bedlevel'], name='Bedlevel', mode='markers'))

    return bl_plot


def get_waterlevel_accuracy_plot(df, forecast):
    try:
        ob = thing(DATA=df, N_FORECAST=(forecast), StartingTimeDelta=timedelta(minutes=10),
                   newTimeDelta=timedelta(days=1), timeCol='time', dataCol='NUMERIEKEWAARDE', seq=30)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = LSTM(device=device)
        state = torch.load("..//machine_learning//MODELS//WaterLevel//" + str(df['MEETPUNT_IDENTIFICATIE'][0]) + "_waterLevel_V2.pt", map_location=device)
        model.load_state_dict(state)
        train, test, pred = ob.accCheck(model, ob.train, ob.test)

        wl_plot = generate_plot()
        wl_plot.update_xaxes(title_text='Tijd')
        wl_plot.update_yaxes(title_text='Waterhoogte t.o.v NAP (m)')
        wl_plot.add_trace(go.Scatter(x=train.index, y=train['Data'], name='Train'))
        wl_plot.add_trace(go.Scatter(x=test.index, y=test['Data'], mode='lines', line_color='orange', name='Test'))
        wl_plot.add_trace(
            go.Scatter(x=test.index, y=pred['NUMERIEKEWAARDE'], mode='lines', line_color='green', name='Prediction'))
        return wl_plot
    except Exception:
        return generate_plot()


def get_all_plot(waterlevel, waterdepth, bedlevel):
    plot = generate_plot()

    plot.update_xaxes(title_text='Tijd')
    plot.update_yaxes(title_text='Hoogte (m)')
    plot.add_trace(go.Scatter(

        x=waterlevel['time'],
        y=waterlevel['NUMERIEKEWAARDE'],
        name='Waterlevel'))

    plot.add_trace(go.Scatter(

        x=waterdepth.index,
        y=waterdepth['y'],
        name='Waterdepth',
        mode='markers'

    ))

    plot.add_trace(go.Scatter(

        x=bedlevel.index,
        y=bedlevel['bedlevel'],
        name='Bedlevel',
        mode='markers'))
    return plot


# Data van de app over het meetpunt en datums
@app.callback(
    [dash.dependencies.Output('meetpunt-naam', 'children'),
     dash.dependencies.Output('store', 'data'),
     ],
    [dash.dependencies.Input('map-rws-meetpunten', 'clickData'),
     dash.dependencies.Input('date-picker', 'start_date'),
     dash.dependencies.Input('date-picker', 'end_date'),
     dash.dependencies.Input('forecast-dropdown', 'value')])
def update_meetpunt_data(clickData, start_date, end_date, value):
    if clickData and start_date and end_date:
        # Data over het aangeklikte meetpunt
        meetpunt = clickData['points'][0]['customdata']
        forecast = 7 if value == 'Week' else 30

        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)

        # Get waterlevel, waterdepth en bodemligging data
        waterlevel_df = rws.get_waterlevel(start, end, meetpunt)
        waterdepth_df = rws.get_covadem_waterdepth(start, end, meetpunt, 0.5)
        bedlevel_df = calculate_bedlevel(waterlevel_df, waterdepth_df)

        wl_plot = get_waterlevel_plot(waterlevel_df, forecast)
        wd_plot = get_waterdepth_plot(waterdepth_df, forecast, meetpunt=meetpunt)
        bl_plot = get_bedlevel_plot(bedlevel_df, forecast, meetpunt=meetpunt)
        acc_plot = get_waterlevel_accuracy_plot(waterlevel_df, forecast)
        combo_plot = get_all_plot(waterlevel_df, waterdepth_df, bedlevel_df)

        plots = {'waterhoogte': wl_plot,
                 'waterdiepte': wd_plot,
                 'bodemligging': bl_plot,
                 'accuracy': acc_plot,
                 'all': combo_plot
                 }
        return [meetpunt, plots]
    else:
        return ['Meetpunt', None]


@app.callback(
    [dash.dependencies.Output('map-rws-meetpunten', 'figure')],
    [dash.dependencies.Input('status-interval', 'n_intervals'),
     dash.dependencies.Input('map-rws-meetpunten', 'clickData'),
     dash.dependencies.Input('kaart-dropdown', 'value')],
    [dash.dependencies.State('map-rws-meetpunten', 'figure')]
)
def update_meetpunten_status(n, click, kaart, figure):
    if kaart == 'waterlevelStatus':
        figure['data'] = []
        zones = get_zones_legenda()
        for i, zone in zones.iterrows():
            figure['data'].append(get_meetpunten_trace(kaart, zone['label'], '#636EFA', zone['color'], click))
        return [figure]

    if kaart == 'bedlevelStatus':
        kaart = 'waterdepthStatus'
    figure['data'] = [
        get_meetpunten_trace(kaart, 'kritisch', '#636EFA', '#EF553B', click),
        get_meetpunten_trace(kaart, 'normaal', '#636EFA', '#00CC96', click)
    ]
    return [figure]


if __name__ == '__main__':
    # Run de app
    app.run_server(debug=True)
