import project_funcs
import binary_classification_rf_tuning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def get_training_data():
    geo_abs_vectors = project_funcs.generate_feature_vectors( filepath = "data/training_data_2018-2020/")
    lakes = pd.read_csv( r"data/training_data_2018-2020/training_lake_assessments.csv" )
    ground_truth_labels = lakes[["Assessment AUID","Impaired"]].copy()
    ground_truth_labels.rename(columns={"Assessment AUID": "lake_id"}, inplace=True)
    data = pd.merge( ground_truth_labels, geo_abs_vectors, on="lake_id", how="inner" )
    y = data["Impaired"]
    x = data.drop( [ "lake_id", "Impaired" ], axis=1 )
    return x,y

def get_split_training_data():
    x, y = get_training_data()
    x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.33, random_state=42 )
    return x_train, x_test, y_train, y_test

def get_testing_data():
    geo_abs_vectors = project_funcs.generate_feature_vectors( filepath = "data/testing_data_2020-2022/")
    lakes = pd.read_csv( r"data/testing_data_2020-2022/testing_lake_assessments.csv" )
    ground_truth_labels = lakes[["Assessment AUID","Impaired"]].copy()
    ground_truth_labels.rename(columns={"Assessment AUID": "lake_id"}, inplace=True)
    data = pd.merge( ground_truth_labels, geo_abs_vectors, on="lake_id", how="inner" )
    y = data["Impaired"]
    x = data.drop( [ "lake_id", "Impaired" ], axis=1 )
    return x,y

def get_tuned_rf_classifier():
    return RandomForestClassifier( class_weight="balanced", max_samples=0.6, max_depth=10 )



''' Used for tuning the RF model parameters '''
# x_train, x_test, y_train, y_test = get_split_training_data()
# scaler = preprocessing.StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# binary_classification_rf_tuning.tune_max_depth( x_train, x_test, y_train, y_test )
# binary_classification_rf_tuning.tune_max_features( x_train, x_test, y_train, y_test )
# binary_classification_rf_tuning.tune_max_samples( x_train, x_test, y_train, y_test )
# binary_classification_rf_tuning.tune_min_samples_leaf( x_train, x_test, y_train, y_test )
# binary_classification_rf_tuning.tune_min_samples_split( x_train, x_test, y_train, y_test )

''' Train the tuned RF model and evalate against testing data '''
x_train, y_train = get_training_data()
x_test, y_test = get_testing_data()

x_test = x_test[ x_test.columns.intersection( x_train.columns ) ]
x_train = x_train[ x_train.columns.intersection( x_test.columns ) ]

# Scale the feature vectors
scaler = preprocessing.StandardScaler()
x_train_columns = x_train.columns
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

rf_classifier = get_tuned_rf_classifier()
rf_classifier.fit( x_train, y_train )

important_features = pd.DataFrame( [ x_train_columns, rf_classifier.feature_importances_ ] ).T
important_features.columns = ["features","importance (%)"]
important_features = important_features.sort_values("importance (%)", ascending=False ).reset_index()
print( important_features.head( 20 ) )

y_pred = rf_classifier.predict( x_test )
print( "accuracy score: %2.1f"%(100.0 * accuracy_score( y_test, y_pred ) ) )

