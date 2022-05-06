import project_funcs
import os
import argparse
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_residence_data( filepath="" ):
    ''' Load the residence data into a geopandas dataframe. The longtitude and latitudes
    are stored as a POINT geometry and are initially epsg:4326. The POINT geometry is
    transformed into epsg:26915 format before returning. The column headers of the dataframe
    are the following:

            Latitude
            Longitude
            Address Line 1
            City
            State
            Zip
            County Name
            Length of Residence
            Dwelling Type
            geometry
    '''
    data = pd.read_csv( filepath )
    data = gpd.GeoDataFrame( data,
                             crs="epsg:4326",
                             geometry=gpd.points_from_xy( data.Longitude,
                                                          data.Latitude ) )
    data.to_crs( crs="epsg:26915", inplace=True )
    return data


def decode_length_of_residence( encoding ):
    ''' Decode the length of residence data and create three groups.
        A = 00 Years
        B = 01 Years
        C = 02 Years
        D = 03 Years
        E = 04 Years
        F = 05 Years
        G = 06 Years
        H = 07 Years
        I = 08 Years
        J = 09 Years
        K = 10 Years
        L = 11-15 Years
        M = 16-19 Years
        N = 20+ Years
    '''
    rv = ''
    if encoding in ['A','B','C','D','E','F','G','H','I','J']:
        rv = '0-9 years'
    elif encoding in ['K','L','M']:
        rv = '10-19 years'
    elif encoding in ['N']:
        rv = '>=20 years'
    return rv


def main( args ):

    # # residences = load_residence_data( 'data/mn_residential_data/residence.csv' )
    # # lakes = project_funcs.load_lake_assessment_data( args.lakes_filepath )

    # # Create a new table where the bounds of the lake polygons are used to create
    # # the various buffer sizes. These buffers are then used to do a spatial join
    # # with the points (lat/lon) of the residential data.
    # # 
    # for buffer_size in [ 0, 500, 1000, 3000 ]:
    #     print("Working on buffer %i"%(buffer_size))
    #     buffer = lakes.bounds
    #     buffer["lake_id"] = lakes["Assessment AUID"]
    #     buffer["minx"] = buffer.apply( lambda row: row["minx"] - buffer_size, axis=1 )
    #     buffer["miny"] = buffer.apply( lambda row: row["miny"] - buffer_size, axis=1 )
    #     buffer["maxx"] = buffer.apply( lambda row: row["maxx"] + buffer_size, axis=1 )
    #     buffer["maxy"] = buffer.apply( lambda row: row["maxy"] + buffer_size, axis=1 )
    #     buffer["geometry"] = buffer.apply( lambda row : Polygon( zip( [ row['minx'], row['minx'], row['maxx'], row['maxx'], row['minx'] ], 
    #                                                                   [ row['maxy'], row['miny'], row['miny'], row['maxy'], row['maxy'] ] ) ), axis=1 )
    #     buffer.drop( [ 'minx', 'miny', 'maxx', 'maxy' ], inplace=True, axis=1 )
    #     buffer = gpd.GeoDataFrame( buffer, crs="epsg:26915" )

    #     residences_in_buffer = gpd.sjoin( buffer, residences, how="inner" )

    #     # The features file header:
    #     #   u_id, lake_id, geom_type, geo_feature, feature_type, buffer_size, value
    #     features = residences_in_buffer[["lake_id"]].copy()
    #     features["geom_type"] = "point"
    #     features["geo_feature"] = "residence"
    #     features["feature_type"] = residences_in_buffer.apply( lambda row: decode_length_of_residence(row["Length of Residence"]), axis=1 )
    #     features["buffer_size"] = buffer_size
    #     features["value"] = 1
    #     features.reset_index( inplace=True )
    #     features.drop( [ 'index' ], inplace=True, axis=1 )
    #     features.index.name = 'u_id'

        # The individual buffers were saved and merged outside of this script
        # features.to_csv( r"data/training_data_2018-2020/residence_buffer_"+str(buffer_size)+".csv" )


    ''' The following is used for plotting and debug. '''

    # features = pd.read_csv( r"data/training_data_2018-2020/residence_features.csv" )
    # features["features"] = features["geo_feature"].map(str) + " " + \
    #                        features["feature_type"].map(str) + " " + \
    #                        features["buffer_size"].map(str)
    # feature_vectors = features.pivot_table( index="lake_id", columns="features", values="value", aggfunc=np.sum, fill_value=0 )

    # features[ 'lake' ] = features[ 'geometry' ].apply( wkt.loads )
    # features[ 'buffer' ] = features[ 'buffer' ].apply( wkt.loads )

    # features = gpd.GeoDataFrame( features,
    #                              geometry=gpd.points_from_xy( features.Longitude,
    #                                                           features.Latitude ) )


    # usa = gpd.read_file( r"data/States 21basic/geo_export_c51447ec-9bc1-4f77-a3a8-f223ff618c5e.shp" )
    # usa.to_crs( crs="epsg:26915", inplace=True )
    # base = usa[usa.state_name == 'Minnesota'].plot( facecolor="none", aspect=1 )
    # features.set_geometry( "lake", inplace=True )
    # features.plot( ax=base )
    # features.set_geometry( "buffer", inplace=True )
    # features.plot( ax=base, color="none", edgecolor="red", linewidth=1 )
    # features.set_geometry( "geometry", inplace=True )
    # features.set_crs(crs="epsg:4326", inplace=True)
    # features.to_crs( crs="epsg:26915", inplace=True )
    # features.plot( ax=base, markersize=1, color='black' )
    # plt.show()

    # features_4326 = features.to_crs( crs="epsg:4326" )
    # features_4326.plot(ax=base, color="none", edgecolor="red", linewidth=1)

    residences = load_residence_data( 'data/residence_example_dent.csv' )
    lakes = project_funcs.load_lake_assessment_data( 'data/lake_assessments_example_big_mcdonald.csv' )

    # residences = load_residence_data( 'data/mn_residential_data/residence.csv' )
    # lakes = project_funcs.load_lake_assessment_data( args.lakes_filepath )

    minx, miny, maxx, maxy = lakes.loc[[0],'geometry'].bounds.values[0]# + [-500, -500, 500, 500]
    box = Polygon(zip([minx, minx, maxx, maxx, minx],[maxy, miny, miny, maxy, maxy]))
    boxes = gpd.GeoDataFrame(index=[0], crs="epsg:26915", geometry=[box])
    res_mask = residences["geometry"].within( boxes.loc[0,"geometry"] )
    usa = gpd.read_file( r"data/States 21basic/geo_export_c51447ec-9bc1-4f77-a3a8-f223ff618c5e.shp" )
    usa.to_crs( crs="epsg:26915", inplace=True )
    base = usa[usa.state_name == 'Minnesota'].plot( facecolor="none", aspect=1 )
    # base = usa[usa.state_name == 'Minnesota'].plot( facecolor="none" )
    # lakes_4326 = lakes.to_crs( crs="epsg:4326" )
    # residences_4326 = residences.to_crs( crs="epsg:4326" )
    # boxes_4326 = boxes.to_crs( crs="epsg:4326" )
    lakes.plot( ax=base )
    # residences.plot( ax=base, markersize=1, color='black' )
    residences[res_mask==True].plot( ax=base, markersize=1, color='red' )
    residences[res_mask==False].plot( ax=base, markersize=1, color='black' )
    boxes.plot(ax=base, color="none", edgecolor="red", linewidth=1)
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument( '--lakes_filepath', type=str, default='data/training_data_2018-2020/training_lake_assessments.csv',
                         help='The filename and path of the assessed lakes csv file' )
    args = parser.parse_args()

    main( args )
