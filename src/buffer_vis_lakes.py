""" This script creates visual representations of the lakes and buffers,
    as well as credating files for the buffers to be used in PostgreSQL
    to align with Open Street Map data.

    Note buffer sizes are 0m, 500m, 1000m, 3000m

    Usage: python3 buffer_vis_lakes.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR

    Created by Suzie Hoops (adapted from running in Google Colab)
"""
import os
import pandas as pd
import geopandas as gpd 
from shapely import wkt
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

def create_buffers(input_dir, output_dir):
    # Read in the CSV file containing lake geoms (in 'geometry' column)
    train_lakes = pd.read_csv(os.path.join(input_dir,'training_assessed_lakes_2020.csv'), delimiter=",", header=0, index_col=0)
    train_lakes['geometry'] = train_lakes['geometry'].apply(wkt.loads)
    test_lakes = pd.read_csv(os.path.join(input_dir,'testing_assessed_lakes_2022.csv'), delimiter=",", header=0, index_col=0)
    test_lakes = test_lakes.rename(columns={"Shape_Leng": "SHAPE_Leng", "Shape_Area": "SHAPE_Area"})
    test_lakes['geometry'] = test_lakes['geometry'].apply(wkt.loads)
    train_glakes = gpd.GeoDataFrame(train_lakes, crs="epsg:26915").set_geometry('geometry') # NAD83 code is EPSG:26915
    test_glakes = gpd.GeoDataFrame(test_lakes, crs="epsg:26915").set_geometry('geometry') # NAD83 code is EPSG:26915

    # Create dataframes per group
    ## Mark the training/testing lakes
    train_glakes['col_group'] = (train_glakes['AQR_LAST_A'] >= 2018) | (train_glakes['AQL_LAST_A'] >= 2018) | (train_glakes['AQC_LAST_A'] >= 2018)
    train_glakes['col_group'] = train_glakes.col_group.replace({True: 'forestgreen', False: 'blue'})
    test_glakes['col_group'] = (test_glakes['AQR_LAST_A'] >= 2020) | (test_glakes['AQL_LAST_A'] >= 2020) | (test_glakes['AQC_LAST_A'] >= 2020)
    test_glakes['col_group'] = test_glakes.col_group.replace({True: 'darkviolet', False: 'blue'})
    ## Refined training set
    train_glakes = train_glakes[(train_glakes['col_group'] == 'forestgreen')]
    ## Combined set
    glakes = train_glakes.append(test_glakes)
    glakes = glakes.drop_duplicates(['AUID'], keep='first')
    ## Refined testing set
    test_glakes = test_glakes[(test_glakes['col_group'] == 'darkviolet')]

    # Create Buffers - Training set
    buffers = gpd.GeoDataFrame(columns=['lake_id', 'lake_name', 'buffer_size', 'geometry'], crs='epsg:4269', geometry='geometry')
    bounds = train_glakes['geometry'].bounds
    for idx,row in train_glakes.iterrows():
        # bounding box buffer (no addition)
        minx, miny, maxx, maxy = bounds.loc[idx]
        box = gpd.GeoDataFrame({'lake_id': row['AUID'], 'lake_name': row['NAME'], 'buffer_size': 0,
                                'geometry': Polygon(zip([minx, minx, maxx, maxx, minx],[maxy, miny, miny, maxy, maxy]))},
                                crs = 'epsg:4269', geometry='geometry', index=[0])
        buffers = buffers.append(box, ignore_index=True)
        # 500m buffer
        minx, miny, maxx, maxy = bounds.loc[idx] + [-500, -500, 500, 500]
        box = gpd.GeoDataFrame({'lake_id': row['AUID'], 'lake_name': row['NAME'], 'buffer_size': 500,
                                'geometry': Polygon(zip([minx, minx, maxx, maxx, minx],[maxy, miny, miny, maxy, maxy]))},
                                crs = 'epsg:4269', geometry='geometry', index=[0])
        buffers = buffers.append(box, ignore_index=True)
        # 1000m buffer
        minx, miny, maxx, maxy = bounds.loc[idx] + [-1000, -1000, 1000, 1000]
        box = gpd.GeoDataFrame({'lake_id': row['AUID'], 'lake_name': row['NAME'], 'buffer_size': 1000,
                                'geometry': Polygon(zip([minx, minx, maxx, maxx, minx],[maxy, miny, miny, maxy, maxy]))},
                                crs = 'epsg:4269', geometry='geometry', index=[0])
        buffers = buffers.append(box, ignore_index=True)
        # 3000m buffer
        minx, miny, maxx, maxy = bounds.loc[idx] + [-3000, -3000, 3000, 3000]
        box = gpd.GeoDataFrame({'lake_id': row['AUID'], 'lake_name': row['NAME'], 'buffer_size': 3000,
                                'geometry': Polygon(zip([minx, minx, maxx, maxx, minx],[maxy, miny, miny, maxy, maxy]))},
                                crs = 'epsg:4269', geometry='geometry', index=[0])
        buffers = buffers.append(box, ignore_index=True)
    # print to CSV
    buffers['u_id'] = range(len(buffers))
    buffers_pd = pd.DataFrame(buffers)
    buffers_pd = buffers_pd[['u_id', 'lake_id', 'lake_name', 'buffer_size', 'geometry']]
    buffers_pd.to_csv(os.path.join(output_dir,'training_assessed_lakes_2020_buffers.csv'), index=False)

    # Create Buffers - Testing set
    buffers = gpd.GeoDataFrame(columns=['lake_id', 'lake_name', 'buffer_size', 'geometry'], crs='epsg:4269', geometry='geometry')
    bounds = test_glakes['geometry'].bounds
    for idx,row in test_glakes.iterrows():
        # bounding box buffer (no addition)
        minx, miny, maxx, maxy = bounds.loc[idx]
        box = gpd.GeoDataFrame({'lake_id': row['AUID'], 'lake_name': row['NAME'], 'buffer_size': 0,
                                'geometry': Polygon(zip([minx, minx, maxx, maxx, minx],[maxy, miny, miny, maxy, maxy]))},
                                crs = 'epsg:4269', geometry='geometry', index=[0])
        buffers = buffers.append(box, ignore_index=True)
        # 500m buffer
        minx, miny, maxx, maxy = bounds.loc[idx] + [-500, -500, 500, 500]
        box = gpd.GeoDataFrame({'lake_id': row['AUID'], 'lake_name': row['NAME'], 'buffer_size': 500,
                                'geometry': Polygon(zip([minx, minx, maxx, maxx, minx],[maxy, miny, miny, maxy, maxy]))},
                                crs = 'epsg:4269', geometry='geometry', index=[0])
        buffers = buffers.append(box, ignore_index=True)
        # 1000m buffer
        minx, miny, maxx, maxy = bounds.loc[idx] + [-1000, -1000, 1000, 1000]
        box = gpd.GeoDataFrame({'lake_id': row['AUID'], 'lake_name': row['NAME'], 'buffer_size': 1000,
                                'geometry': Polygon(zip([minx, minx, maxx, maxx, minx],[maxy, miny, miny, maxy, maxy]))},
                                crs = 'epsg:4269', geometry='geometry', index=[0])
        buffers = buffers.append(box, ignore_index=True)
        # 3000m buffer
        minx, miny, maxx, maxy = bounds.loc[idx] + [-3000, -3000, 3000, 3000]
        box = gpd.GeoDataFrame({'lake_id': row['AUID'], 'lake_name': row['NAME'], 'buffer_size': 3000,
                                'geometry': Polygon(zip([minx, minx, maxx, maxx, minx],[maxy, miny, miny, maxy, maxy]))},
                                crs = 'epsg:4269', geometry='geometry', index=[0])
        buffers = buffers.append(box, ignore_index=True)
    # print to CSV
    buffers['u_id'] = range(len(buffers))
    buffers_pd = pd.DataFrame(buffers)
    buffers_pd = buffers_pd[['u_id', 'lake_id', 'lake_name', 'buffer_size', 'geometry']]
    buffers_pd.to_csv(os.path.join(output_dir,'testing_assessed_lakes_2022_buffers.csv'), index=False)

    # Return the glakes geodataframe
    return glakes


def visualize_lakes(output_dir, glakes):
    # Visualize all lakes
    fig, ax = plt.subplots(figsize = (16,12))
    glakes.plot(ax=ax, color="blue", edgecolor=glakes['col_group'], aspect=1)
    plt.savefig(f"{output_dir}/mn_all_lakes.jpg")

    # Plot one lake example - lake of the isles
    ##print(glakes.loc[glakes['NAME'] == "Lake of the Isles"])
    fig, ax = plt.subplots(figsize = (10,12))
    glakes.loc[[1900],'geometry'].plot(ax=ax, color="blue", aspect=1)
    plt.savefig(f"{output_dir}/isles.jpg")

    # Print bounding boxes with the lake
    ##print(glakes.loc[glakes['NAME'] == "Lake of the Isles"])
    bounds = glakes.loc[[1900],'geometry'].bounds  # note that bounds works per row!
    minx, miny, maxx, maxy = bounds.values[0] + [-500, -500, 500, 500] # note the expansion of the bounding box here.
    box = Polygon(zip([minx, minx, maxx, maxx, minx],[maxy, miny, miny, maxy, maxy])) # interior: x_list, y_list for five points
    minx, miny, maxx, maxy = bounds.values[0]
    box2 = Polygon(zip([minx, minx, maxx, maxx, minx],[maxy, miny, miny, maxy, maxy]))
    boxes = gpd.GeoDataFrame(index=[0], crs="epsg:4269", geometry=[box])
    boxes2 = gpd.GeoDataFrame(index=[0], crs="epsg:4269", geometry=[box2])

    # Draw a box on the plot
    fig, ax = plt.subplots(figsize = (10,12))
    base = glakes.loc[[1900],'geometry'].plot(ax=ax, color='blue')
    boxes.plot(ax=base, color="none", edgecolor="red", linewidth=3, aspect=1)
    boxes2.plot(ax=base, color="none", edgecolor="red", linewidth=3, aspect=1)
    plt.savefig(f"{images_dir}/isles_bounding_box.jpg")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/',
                        help='Path to the directory containing the assessed lakes CSV files.')
    parser.add_argument('--output_dir', type=str, default='out/',
                        help='Path to the directory where the output CSV files and images will be placed')
    args = parser.parse_args()

    glakes = create_buffers(args.input_dir, args.output_dir)

    visualize_lakes(args.output_dir, glakes)

    main()
