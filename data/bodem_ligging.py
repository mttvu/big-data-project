from os import remove
import pandas as pd
import sys
sys.path.append("..")
from pkg_resources import add_activation_listener
import plotly.express as px
import os
import numpy as np
#from fbprophet import Prophet
from datetime import datetime
#from fbprophet.plot import add_changepoints_to_plot, plot_plotly
from covadem.data.rws_data import RWSData
from scipy.stats import zscore
import plotly.graph_objects as go
from covadem.data.cleaning import groupby_frequency, remove_outliers


def calculate_bedlevel(wl_df, wd_df):
    """

    :param wl_df: waterhoogte/waterlevel (RWS)
    :param wd_df: waterdiepte (Covadem)
    :return: gecombineerde dataframe van de waterhoogte(RWS) en waterdiepte(Covadem) met de berekende bodemligging
    """
    # group data by hour (average per hour)
    waterlevel = groupby_frequency(
    df=wl_df, 
    frequency='H',
    x='time',
    y='NUMERIEKEWAARDE')

    # necessary for merging with waterdepth later
    waterlevel.set_index('time', inplace=True)

    waterdepth = wd_df
    waterlevel.columns = ['waterlevel']

    # convert cm to meters (covadem data is in meters)
    merged = waterlevel.join(waterdepth, how='outer')

    # calculate bedlevel
    merged['bedlevel'] = merged['waterlevel'] - merged['y']
    return merged
