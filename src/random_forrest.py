import project_funcs
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score


def get_important_training_features():

    # Create the geographic abstraction vectors
    geo_abs_vectors = project_funcs.generate_feature_vectors( filepath = "data/training_data_2018-2020/")

    # Get the ground truth labels
    lakes = pd.read_csv( r"data/training_data_2018-2020/training_lake_assessments.csv" )
    ground_truth_labels = lakes[["Assessment AUID","Impaired"]].copy()
    ground_truth_labels.rename(columns={"Assessment AUID": "lake_id"}, inplace=True)

    # Combing the data and prepare it for training
    training_data = pd.merge( ground_truth_labels, geo_abs_vectors, on="lake_id", how="inner" )
    y = training_data["Impaired"]
    X = training_data.drop( [ "lake_id", "Impaired" ], axis=1 )

    # Setup the normalization for the geographic abstraction vectors and random forrest
    # classifier for training. 
    scaler = preprocessing.MinMaxScaler().fit( X )
    X_scaled = scaler.transform( X )
    rf_classifier = RandomForestClassifier( max_depth=2, random_state=0, max_samples=0.1 )

    # Train on all features
    rf_classifier.fit( X_scaled, y )

    # Select the top 20 features and use only those
    top_20_features_df = pd.DataFrame( [ X.columns, rf_classifier.feature_importances_ ] ).T
    top_20_features_df.columns = ["features","importance (%)"]
    top_20_features_df = top_20_features_df.sort_values("importance (%)", ascending=False ).reset_index()
    top_20_features_df = top_20_features_df.head( 20 )

    return top_20_features_df["features"]


def get_rf_model( important_training_features ):

    # Create the geographic abstraction vectors
    geo_abs_vectors = project_funcs.generate_feature_vectors( filepath = "data/training_data_2018-2020/")
    geo_abs_vectors = geo_abs_vectors[ geo_abs_vectors.columns.intersection( important_training_features ) ]

    # Get the ground truth labels
    lakes = pd.read_csv( r"data/training_data_2018-2020/training_lake_assessments.csv" )
    ground_truth_labels = lakes[["Assessment AUID","Impaired"]].copy()
    ground_truth_labels.rename(columns={"Assessment AUID": "lake_id"}, inplace=True)

    # Combing the data and prepare it for training
    training_data = pd.merge( ground_truth_labels, geo_abs_vectors, on="lake_id", how="inner" )
    y = training_data["Impaired"]
    X = training_data.drop( [ "lake_id", "Impaired" ], axis=1 )

    # Setup the normalization for the geographic abstraction vectors and random forrest
    # classifier for training. 
    scaler = preprocessing.MinMaxScaler().fit( X )
    X_scaled = scaler.transform( X )
    rf_classifier = RandomForestClassifier( max_depth=2, random_state=0, max_samples=0.1 )

    # Train on the most important features
    rf_classifier.fit( X_scaled, y )

    return scaler, rf_classifier



''' Testing '''
# Get the most important features and the trained random forrest model
important_training_features = get_important_training_features()
scaler, rf_classifer = get_rf_model( important_training_features )

# Create the geographic abstraction vectors
geo_abs_vectors = project_funcs.generate_feature_vectors( filepath = "data/testing_data_2020-2022/")
geo_abs_vectors = geo_abs_vectors[ geo_abs_vectors.columns.intersection( important_training_features ) ]

# Get the ground truth labels
lakes = pd.read_csv( r"data/testing_data_2020-2022/testing_lake_assessments.csv" )
ground_truth_labels = lakes[["Assessment AUID","Impaired"]].copy()
ground_truth_labels.rename(columns={"Assessment AUID": "lake_id"}, inplace=True)

# Combing the data and prepare it for testing
testing_data = pd.merge( ground_truth_labels, geo_abs_vectors, on="lake_id", how="inner" )
y_test = testing_data["Impaired"]
X_test = testing_data.drop( [ "lake_id", "Impaired" ], axis=1 )

# Evaluate the model
X_test_scaled = scaler.transform( X_test )
y_pred_test = rf_classifer.predict( X_test_scaled )
print( "accuracy score: %2.1f"%(100.0 * accuracy_score(y_test, y_pred_test)) )
