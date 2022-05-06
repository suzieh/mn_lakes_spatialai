""" This script generates mena and median impervious land values within buffers.
    We used this script to create impervious land cover values per buffer in the
    training and testing data.
"""

import rasterio
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import shape
from rasterstats import zonal_stats
import matplotlib.pyplot as plt


def calculate_raster_buffer_overlap(imperv_land_file, imperv_land, buff_file, out_file):
    # Raster file values (can uncomment if you need to see/change these)
    #imperv_land_file.crs                   # note epsg:26915
    #imperv_land_file.shape                 # note size is (43742, 38127)

    # Plot the raster file (also optional)
    fig, ax = plt.subplots()
    img = ax.imshow(imperv_land)
    plt.show()

    # Read in buffers (training or testing)
    buffers = pd.read_csv(buff_file, delimiter=",", header=0)
    buffers['geometry'] = buffers['geometry'].apply(wkt.loads)
    buffers = gpd.GeoDataFrame(buffers, crs="epsg:26915").set_geometry('geometry')

    # Mean and median values per buffer
    stats_per_buffer = zonal_stats(buffers.geometry, imperv_land, affine=imperv_land_file.transform, stats=['mean', 'median'], nodata=0.0)
    stats_per_buffer = pd.DataFrame(stats_per_buffer)

    # Write out to csv file w/ columns: u_id, lake_id, buffer_size, imperv_land_mean, imperv_land_median
    stats_pd = pd.DataFrame({'u_id': buffers.u_id, 'lake_id': buffers.lake_id, 'buffer_size': buffers.buffer_size,
        'imperv_land_mean': stats_per_buffer['mean'], 'imperv_land_median': stats_per_buffer['median']})
    stats_pd.to_csv(out_file)


def main():
    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_raster', type=str, default='data/remote_sensing_lab_umn/tif_base_landcover_minnesota/landcover_impervious_statewide2013_v2.tif',
                        help='Path to raster file input.')
    parser.add_argument('--input_buffer', type=str, default='out/training_assessed_lakes_2020_buffers.csv',
                        help='Path to training or testing buffer file.')
    parser.add_argument('--output_file', type=str, default='out/imperv_land_training.csv',
                        help='Path to output file containing values per buffer.')
    args = parser.parse_args()

    # Open the file and pass to the functions
    imperv_land_file = rasterio.open(args.input_raster)
    imperv_land = imperv_land_file.read(1)

    # Run calculation
    calculate_raster_buffer_overlap(imperv_land_file, imperv_land, args.input_buffer, args.output_file)

    # Close file when done
    imperv_land_file.close()
