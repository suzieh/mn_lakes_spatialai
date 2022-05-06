# mn_lakes_spatialai
Using spatial-enabled machine learning methods for model-based assessment of lakes in MN, USA.

## What our code does
The src folder contains all our code for generating buffers, determining feature vectors from buffers, and finally constructing two random forest models to predict lake impairments. A brief explanation of files:
- osm_sql : directory of python and SQL code for generating PostgreSQL tables of OpenStreetMap features and finding buffer overlap with PostGIS.
- affected_use*.py : construct & tune a random forest classifier for different aquatic impairment groups: life, recreation, consumption
- binary_classification*.py : construct & tune a random forest classifier to predict impairment status (y/n) of MN lakes in 2020-2021
- buffer_vis_lakes.py : constructing buffers (0m, 500m, 1000m, 3000m) for lakes in our dataset, optionally visualizing the buffers.
- generate_precipitation_features.py : Determine precipitation values for each lake in the dataset.
- generate_residence_features.py : Determine number of residences (categorized by most recent purchase) falling within lake buffers.
- imperv_land_raster.py : Determine impervious land cover within each lake buffer.
- project_funcs.py : Helper functions for the random forest models.
- random_forrest.py : Running and testing the random forest models.

Due to limitations on GitHub, we could not make this repo fully end-to-end due to the large size of our diverse datasets. For help running on the datasets we have obtained and are storing elsewhere, please contact the contributors.

## Approach overview
Below is a visual overview of our approach for your reference.
![alt text](https://github.com/suzieh/mn_lakes_spatialai/blob/main/approach.jpg)
