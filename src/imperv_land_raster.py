""" This script generates mena and median impervious land values within buffers.

    Created by Suzie Hoops
"""

import rasterio
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import shape
from rasterstats import zonal_stats
import matplotlib.pyplot as plt


# read in raster file
imperv_land_file = rasterio.open('data/remote_sensing_lab_umn/tif_base_landcover_minnesota/landcover_impervious_statewide2013_v2.tif')
imperv_land = imperv_land_file.read(1)
#imperv_land_file.crs                   # note epsg:26915
#imperv_land_file.shape                 # note size is (43742, 38127)

# plot it
fig, ax = plt.subplots()
img = ax.imshow(imperv_land)
plt.show()


##### TRAINING ####

# read in buffers (training)
buffers = pd.read_csv('out/training_assessed_lakes_2020_buffers.csv', delimiter=",", header=0)
buffers['geometry'] = buffers['geometry'].apply(wkt.loads)
buffers = gpd.GeoDataFrame(buffers, crs="epsg:26915").set_geometry('geometry')

# get the mean & median impervious land value per buffer
stats_per_buffer = zonal_stats(buffers.geometry, imperv_land, affine=imperv_land_file.transform, stats=['mean', 'median'], nodata=0.0)
stats_per_buffer = pd.DataFrame(stats_per_buffer)

# write out to csv file w/ columns: u_id, lake_id, buffer_size, imperv_land_mean, imperv_land_median
stats_pd = pd.DataFrame({'u_id': buffers.u_id, 'lake_id': buffers.lake_id, 'buffer_size': buffers.buffer_size,
    'imperv_land_mean': stats_per_buffer['mean'], 'imperv_land_median': stats_per_buffer['median']})
stats_pd.to_csv('out/imperv_land_training.csv')



##### TESTING ####

# read in buffers (training)
buffers = pd.read_csv('out/testing_assessed_lakes_2022_buffers.csv', delimiter=",", header=0)
buffers['geometry'] = buffers['geometry'].apply(wkt.loads)
buffers = gpd.GeoDataFrame(buffers, crs="epsg:26915").set_geometry('geometry')

# get the mean & median impervious land value per buffer
stats_per_buffer = zonal_stats(buffers.geometry, imperv_land, affine=imperv_land_file.transform, stats=['mean', 'median'], nodata=0.0)
stats_per_buffer = pd.DataFrame(stats_per_buffer)

# write out to csv file w/ columns: u_id, lake_id, buffer_size, imperv_land_mean, imperv_land_median
stats_pd = pd.DataFrame({'u_id': buffers.u_id, 'lake_id': buffers.lake_id, 'buffer_size': buffers.buffer_size,
    'imperv_land_mean': stats_per_buffer['mean'], 'imperv_land_median': stats_per_buffer['median']})
stats_pd.to_csv('out/imperv_land_testing.csv')


# close file when done
imperv_land_file.close()
