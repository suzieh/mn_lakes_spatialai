import os
import geopandas as gpd
import pandas as pd
from shapely import wkt
import numpy as np

def save_shp_to_csv( filepath=r"data/assessed_water_2020/", filename=r"assessed_2020_lakes"):
    ''' This function will convert a *.shp file to a *.csv file. This is handy for using
    other graphical tools to filter and process the data.
    '''
    data = gpd.read_file( os.path.join( filepath, filename,".shp"  ) )
    data.to_csv( os.path.join( filepath, filename,"_shp.csv" ) )

def load_lake_assessment_data( filepath="" ):
    ''' Load the lake assessment data into a geopandas dataframe. The lake shapes are stored
    as POLYGONS in the geometry columns using the "epsg:26915" format. The column headers of
    the dataframe are the following:
        
            Assessment AUID
            Assessment year
            Assessment type
            Water body name
            Water body type
            Year added to List
            Watershed name
            Affected designated use
            Pollutant or stressor
            PCA Watershed ID
            PCA Watershed Name
            Impaired
            geometry
    '''
    data = pd.read_csv( filepath )
    data[ 'geometry' ] = data[ 'geometry' ].apply( wkt.loads )
    data = gpd.GeoDataFrame( data, crs="epsg:26915" )
    return data

def generate_feature_vectors(filepath=""):
    ''' Combine all of the features into a single dataframe and then create the feature vectors
    for each of the lakes.
    '''
    residence_features = pd.read_csv( os.path.join( filepath, "residence_features.csv" ) )
    osm_features = pd.read_csv( os.path.join( filepath, "osm_features.csv" ) )
    precipitation_features = pd.read_csv( os.path.join( filepath, "precipitation_features.csv" ) )
    impervious_features = pd.read_csv( os.path.join( filepath, "impervious_features.csv" ) )

    features = pd.concat([residence_features, osm_features, precipitation_features, impervious_features])

    features["features"] = features["geo_feature"].map(str) + " " + \
                           features["feature_type"].map(str) + " " + \
                           features["buffer_size"].map(str)
    feature_vectors = features.pivot_table( index="lake_id", columns="features", values="value", aggfunc=np.sum, fill_value=0 )

    ''' For debug '''
    #feature_vectors.to_csv( os.path.join( filepath, "feature_vectors.csv" ) )

    return feature_vectors
