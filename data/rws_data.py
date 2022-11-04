import sys
# sys.path.append("..")
import pandas as pd
from pyproj import Proj, transform
# import datashader as ds
#import holoviews as hv
#from holoviews import opts
#from holoviews.operation.datashader import datashade, rasterize
#from holoviews.element.tiles import StamenTerrainRetina
# import datashader as ds
from datetime import datetime as dt
from datetime import date as date
import sys
# sys.path.append(r"C:\Users\N_ADi\Desktop\covadem\covadem\covadem")
from covadem.database_code.meetpunten_db import MeetpuntenDB
from covadem.database_code.mongo_db import MongoCovadem, MongoRWS
import os
from covadem.data.cleaning import groupby_frequency, remove_outliers

class RWSData:

    def __init__(self):
        # systeem rws
        self.in_proj = Proj(init='epsg:25831')
        # systeem covadem
        self.out_proj = Proj(init='epsg:4326')
        self.rws_db = MongoRWS()
        self.covadem_db = MongoCovadem()

        package_dir = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(package_dir, 'meetpunten.csv')
        self.meetpunten = pd.read_csv(file)
        self.sql_db = MeetpuntenDB(USER="ruijten4", PASS="r6ehAdkPGJ92JUwM")

    def convertCoords(self, row):
        x, y = transform(self.in_proj, self.out_proj, row['X'], row['Y'])
        return pd.Series([x, y])

    # voor een nieuwe rws csv bestand cleanen
    def process_rws_csv(self):
        path = 'C:/Users/thaom/Documents/School/Jaar 4/Themasemester Big Data/Covadem/20201113_020/20201113_020.csv'
        rws_df = pd.read_csv(path, sep=';', engine='python')
        print('done loading')

        rws_df = rws_df[
            ['MEETPUNT_IDENTIFICATIE', 'EPSG', 'X', 'Y', 'WAARNEMINGDATUM', 'WAARNEMINGTIJD', 'NUMERIEKEWAARDE']]
        # convert coordinates to lat lon
        print(rws_df['EPSG'].unique())
        # replace , with .
        rws_df['X'] = rws_df.apply(lambda row: row.X.replace(',', '.'), axis=1)
        rws_df['Y'] = rws_df.apply(lambda row: row.Y.replace(',', '.'), axis=1)
        rws_df[['lng', 'lat']] = rws_df.apply(self.convertCoords, axis=1)
        print('done converting coordinates')

        # create a datetime object from WAARNEMINGDATUM and WAARNEMINGTIJD
        rws_df['time'] = pd.to_datetime(rws_df['WAARNEMINGDATUM'] + ' ' + rws_df['WAARNEMINGTIJD'], format="%d-%m-%Y %H:%M:%S")
        rws_df['NUMERIEKEWAARDE'] = rws_df['NUMERIEKEWAARDE'].astype('int64')
        print('done cleaning')
        return rws_df

    def get_coordinates_meetpunten(self):
        return self.sql_db.get_meetpunten()

    # get waterlevel by date and meetpunt
    def get_waterlevel(self, start, end, meetpunt):
        df = self.rws_db.get_by_meetpunt(meetpunt, start, end)
        return df

    def get_covadem_waterdepth(self, start, end, meetpunt_naam, radius_km):
        meetpunt = self.meetpunten[self.meetpunten['name'] == meetpunt_naam]
        df = self.covadem_db.get_by_radius(start, end, meetpunt.iloc[0]['lat'], meetpunt.iloc[0]['lng'], radius_km)
        df.columns = ['time','lat','lng','y']
        df = df[['time', 'y']]

        # group data by hour (average per hour)
        df = groupby_frequency(
            df=df, 
            frequency='H',
            x='time',
            y='y')
        df.set_index('time', inplace=True, drop=False)

        # remove outliers using zscore
        df = remove_outliers(
            df=df, 
            target_column='y', 
            window='7D', 
            threshold=1.5)
        #print(df.head())
        return df
