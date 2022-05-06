import project_funcs
import affected_use_classification_rf_tuning
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


def get_training_data():
    geo_abs_vectors = project_funcs.generate_feature_vectors( filepath = "data/training_data_2018-2020/")
    lakes = pd.read_csv( r"data/training_data_2018-2020/training_lake_assessments_affected_use.csv" )
    ground_truth_labels = lakes[["Assessment AUID","Affected designated use"]].copy()
    ground_truth_labels.rename(columns={"Assessment AUID": "lake_id"}, inplace=True)
    data = pd.merge( ground_truth_labels, geo_abs_vectors, on="lake_id", how="inner" )
    y = data["Affected designated use"]
    x = data.drop( [ "lake_id", "Affected designated use" ], axis=1 )
    return x,y

def get_split_training_data():
    x, y = get_training_data()
    x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.33, random_state=42 )
    return x_train, x_test, y_train, y_test

def get_testing_data():
    geo_abs_vectors = project_funcs.generate_feature_vectors( filepath = "data/testing_data_2020-2022/")
    lakes = pd.read_csv( r"data/testing_data_2020-2022/testing_lake_assessments_affected_use.csv" )
    ground_truth_labels = lakes[["Assessment AUID","Affected designated use"]].copy()
    ground_truth_labels.rename(columns={"Assessment AUID": "lake_id"}, inplace=True)
    data = pd.merge( ground_truth_labels, geo_abs_vectors, on="lake_id", how="inner" )
    y = data["Affected designated use"]
    x = data.drop( [ "lake_id", "Affected designated use" ], axis=1 )
    return x,y

def get_tuned_rf_classifier():
    return RandomForestClassifier( class_weight="balanced", n_estimators=40, max_samples=0.3, max_depth=3 )

''' Used for tuning the RF model parameters '''
# x_train, x_test, y_train, y_test =  get_split_training_data()
# scaler = preprocessing.StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# affected_use_classification_rf_tuning.tune_n_estimators( x_train, x_test, y_train, y_test )
# affected_use_classification_rf_tuning.tune_max_depth( x_train, x_test, y_train, y_test )
# affected_use_classification_rf_tuning.tune_max_features( x_train, x_test, y_train, y_test )
# affected_use_classification_rf_tuning.tune_max_samples( x_train, x_test, y_train, y_test )
# affected_use_classification_rf_tuning.tune_min_samples_leaf( x_train, x_test, y_train, y_test )
# affected_use_classification_rf_tuning.tune_min_samples_split( x_train, x_test, y_train, y_test )

''' Train the tuned RF model and evalate against testing data '''
x_train, y_train = get_training_data()
x_test, y_test = get_testing_data()

# Make sure the columns are consistent between the training and testing sets
x_test = x_test[ x_test.columns.intersection( x_train.columns ) ]
x_train = x_train[ x_train.columns.intersection( x_test.columns ) ]

# Scale the feature vectors
scaler = preprocessing.StandardScaler()
x_train_columns = x_train.columns
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Get the tuned RF classifier model fit to the training data 
rf_classifier = get_tuned_rf_classifier()
rf_classifier.fit( x_train, y_train )

important_features = pd.DataFrame( [ x_train_columns, rf_classifier.feature_importances_ ] ).T
important_features.columns = ["features","importance (%)"]
important_features = important_features.sort_values("importance (%)", ascending=False ).reset_index()
print( important_features.head( 20 ) )

# Make predictions and evaluate the model
y_pred = rf_classifier.predict( x_test )
cm = confusion_matrix( y_test, y_pred, labels=rf_classifier.classes_ )
print( "accuracy consumption: %2.1f, accuracy life: %2.1f, accuracy recreation: %2.1f" % 
( 100 * cm[0,0] / cm[0,:].sum(), 100 * cm[1,1] / cm[1,:].sum(), 100 * cm[2,2] / cm[2,:].sum() ) )
disp = ConfusionMatrixDisplay( confusion_matrix=cm, display_labels=rf_classifier.classes_ )
disp.plot()
plt.show()
