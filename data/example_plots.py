# from os import remove
# import pandas as pd
# import sys
# sys.path.append("..")
# from pkg_resources import add_activation_listener
# import plotly.express as px
# import os
# import numpy as np
# #from fbprophet import Prophet
# from datetime import datetime
# #from fbprophet.plot import add_changepoints_to_plot, plot_plotly
# from data.rws_data import RWSData
# from scipy.stats import zscore
# import plotly.graph_objects as go
# from data.cleaning import groupby_frequency, remove_outliers
#
#
# def calculate_bedlevel(wl_df, wd_df):
#     # group data by hour (average per hour)
#     lobith_waterlevel = groupby_frequency(
#     df=wl_df,
#     frequency='H',
#     x='time',
#     y='NUMERIEKEWAARDE')
#
#     # necessary for merging with waterdepth later
#     lobith_waterlevel.set_index('time', inplace=True)
#
#     wd_df.columns = ['time','lat','lng','y']
#     wd_df = wd_df[['time', 'y']]
#
#     # group data by hour (average per hour)
#     wd_df = groupby_frequency(
#         df=wd_df,
#         frequency='H',
#         x='time',
#         y='y')
#
#     # necessary for merging with waterlevel later
#     wd_df.set_index('time', inplace=True)
#
#     # remove outliers using zscore
#     lobith_waterdepth = remove_outliers(
#         df=wd_df,
#         target_column='y',
#         window='7D',
#         threshold=1.5)
#
#     lobith_waterdepth.columns = ['waterdepth']
#     lobith_waterlevel.columns = ['waterlevel']
#
#     # convert cm to meters (covadem data is in meters)
#     lobith_waterlevel = lobith_waterlevel / 100
#     merged = lobith_waterlevel.join(lobith_waterdepth, how='outer')
#
#     # calculate bedlevel
#     merged['bedlevel'] = merged['waterlevel'] - merged['waterdepth']
#     return merged
#
#     # Inspiratie voor combi plot
#     # def plot_all(self):
#     #     plot = go.Figure()
#     #     plot.add_trace(go.Scatter(
#     #         x=self.lobith_df.index,
#     #         y=self.lobith_df.waterlevel,
#     #         name='Water Level'))
#     #     plot.add_trace(go.Scatter(
#     #         x=self.lobith_df.index,
#     #         y=self.lobith_df.waterdepth,
#     #         name='Water Depth',
#     #         mode='markers'))
#     #     plot.add_trace(go.Scatter(
#     #         x=self.lobith_df.index,
#     #         y=self.lobith_df.bedlevel,
#     #         name='Bed Level',
#     #         mode='markers'))
#
#     #     return plot

# Table inspiratie
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

df = pd.read_csv('https://gist.githubusercontent.com/chriddyp/c78bf172206ce24f77d6363a2d754b59/raw/c353e8ef842413cae56ae3920b8fd78468aa4cb2/usa-agricultural-exports-2011.csv')


def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H4(children='US Agriculture Exports (2011)'),
    generate_table(df)
])

if __name__ == '__main__':
    app.run_server(debug=True)
