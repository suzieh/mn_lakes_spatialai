import os
import project_funcs
import argparse
import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely import wkt

def load_precipitation_data( filepath=r"" ):
    ''' Load the precipitation data into a geopandas dataframe. The longtitude and latitudes
    are stored as a POINT geometry and are initially epsg:4326. The POINT geometry is
    transformed into epsg:26915 format before returning. The column headers of the dataframe
    are the following:

            lat
            lon
            time
            precip

    https://psl.noaa.gov/data/gridded/tables/precipitation.html

    '''
    data = xr.open_dataset( filepath ).to_dataframe().reset_index()
    data = data.groupby( by=["lat", "lon"] ).sum().reset_index()
    data = gpd.GeoDataFrame( data,
                             crs="epsg:4326",
                             geometry=gpd.points_from_xy( data.lon,
                                                          data.lat ) )
    data.to_crs( crs="epsg:26915", inplace=True )
    return data


def main( args ):

    precipitation = load_precipitation_data( os.path.join( args.precip_path, "precip.V1.0.mon.mean.nc" ) )
    lakes = project_funcs.load_lake_assessment_data( os.path.join( args.lakes_path, args.lakes_filename ) )

    # Filter out precipitation data from outside of Minnesota
    usa = gpd.read_file( r"data/States 21basic/geo_export_c51447ec-9bc1-4f77-a3a8-f223ff618c5e.shp" )
    minnesota = usa[usa.state_name == 'Minnesota'].copy()
    minnesota.to_crs( crs="epsg:26915", inplace=True )
    precip_in_mn_idx = gpd.sjoin( minnesota, precipitation, how="inner" )
    precipitation_in_minnesota = precipitation.iloc[precip_in_mn_idx["index_right"]].copy()

    # Find the closest precipitation measurement and add create the feature vector
    # The features file header:
    #   u_id, lake_id, geom_type, geo_feature, feature_type, buffer_size, value
    closest_idx = gpd.sjoin_nearest( lakes, precipitation_in_minnesota, distance_col="distances")
    features = pd.DataFrame(columns=["lake_id", "geom_type", "geo_feature", "feature_type", "buffer_size", "value" ])
    features["lake_id"] = lakes["Assessment AUID"]
    features["geom_type"] = "point"
    features["geo_feature"] = "precipitation"
    features["feature_type"] = "total on record" 
    features["buffer_size"] = 0
    features["value"] = precipitation.iloc[closest_idx["index_right"]]["precip"].values
    features.reset_index( inplace=True )
    features.drop( [ 'index' ], inplace=True, axis=1 )
    features.index.name = 'u_id'
    #features.to_csv( os.path.join( args.lakes_path, "precipitation_features.csv" ) )


    ''' The following is used for plotting and debug. '''

    base = minnesota.plot( facecolor="none", aspect=1 )
    precipitation_in_minnesota.plot(ax=base, markersize=20, color='red')
    lakes = lakes[lakes["Assessment AUID"]!="16-0001-00"]
    lakes.plot(ax=base)
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument( '--lakes_path', type=str, default='data/training_data_2018-2020/',
                         help='The path of the assessed lakes input csv file and output features file' )
    parser.add_argument( '--lakes_filename', type=str, default='training_lake_assessments.csv',
                         help='The filename the assessed lakes input csv file' )
    parser.add_argument( '--precip_path', type=str, default='data/precipitation/',
                         help='The path of the precipitation input csv file' )
    args = parser.parse_args()

    main( args )

