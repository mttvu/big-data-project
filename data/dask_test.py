import dask.dataframe as dd
import holoviews as hv
import datashader as ds
from colorcet import fire
import datashader.transfer_functions as tf
import plotly.express as px
import geoviews as gv
import hvplot.dask  # noqa: adds hvplot method to dask objects
import datashader as ds
from holoviews import opts
from holoviews.operation.datashader import datashade, rasterize
from holoviews.element.tiles import StamenTerrainRetina
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
from shapely.ops import unary_union
from shapely.prepared import prep
from dask.diagnostics import ProgressBar
from itertools import (takewhile,repeat)

print('loading..')
path = 'C:/Users/thaom/Documents/School/Jaar 4/Themasemester Big Data/Covadem/Covadem Dataset/waterdepth-nl-area2-2018-2019-20200723/waterdepth-nl-area2-2018-2019-20200723.csv'
df = dd.read_csv(path, dtype={'gpsDistanceFore': 'float64','lpp': 'float64'})
print('done loading \o/')

#remove non water
land_shp_fname = shpreader.natural_earth(resolution='50m',
                                       category='physical', name='land')
land_geom = unary_union(list(shpreader.Reader(land_shp_fname).geometries()))
land = prep(land_geom)

def is_water(x, y):
    return ~land.contains(sgeom.Point(x, y))

df['is_water'] = df.apply(lambda x: is_water(x.lat, x.lng), axis=1)

def plot_all_points(df_points):
    # datashader plot all points
    hv.extension('bokeh')
    df_coordinates = df_points[['lng','lat']] 
    x, y = ds.utils.lnglat_to_meters(df_coordinates.lng, df_coordinates.lat)
    ddf = df_coordinates.assign(x=x, y=y).persist()

    points = hv.Points(ddf, ['x', 'y'])

    datashade(points).opts(width=700, height=500, bgcolor="lightgray")
    tiles = StamenTerrainRetina().opts(xaxis=None, yaxis=None, width=700, height=500)
    tiles * datashade(points)
    
plot_all_points(df)
plot_all_points(df[df['is_water'] == True])

# subset and convert time to datetime 
df_coordinates = df[['time','lng','lat','waterDepth']]
df_coordinates['time'] = dd.to_datetime(df_coordinates['time'])
# count data per month
df_2018 = df_coordinates[df_coordinates.time.dt.year == 2018]
df_2018.groupby([df_2018.time.dt.year, df_2018.time.dt.month]).agg("count").compute()

df_water = df[df['is_water'] == True]
df_water = df_water[['time','lng','lat','waterDepth']]
a = df_water.shape
a[0].compute(),a[1]