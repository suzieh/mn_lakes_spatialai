from matplotlib.legend_handler import HandlerLine2D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


''' Tuning n_estimators '''
def tune_n_estimators( x_train, x_test, y_train, y_test ):
    n_estimators = [1, 2, 4, 8, 16, 32, 64, 100]
    train_results = []
    test_results = []
    for estimator in n_estimators:
        rf = RandomForestClassifier( class_weight="balanced", n_estimators=estimator )
        rf.fit(x_train, y_train)
        train_pred = rf.predict(x_train)
        train_results.append(accuracy_score(y_train, train_pred))
        y_pred = rf.predict(x_test)
        test_results.append(accuracy_score(y_test, y_pred))
        
    line1, = plt.plot(n_estimators, train_results, 'b', label='Train Accuracy')
    line2, = plt.plot(n_estimators, test_results, 'r', label='Test Accuracy')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Accuracy')
    plt.xlabel('n_estimators')
    plt.show() 

''' Tuning max_samples '''
def tune_max_samples( x_train, x_test, y_train, y_test ):
    max_samples = np.linspace(0.1, 0.9, 9, endpoint=True) # Default is root(num_features)
    train_results = []
    test_results = []
    for max_sample in max_samples:
        rf = RandomForestClassifier( class_weight="balanced", max_samples=max_sample )
        rf.fit(x_train, y_train)
        train_pred = rf.predict(x_train)
        train_results.append(accuracy_score(y_train, train_pred))
        y_pred = rf.predict(x_test)
        test_results.append(accuracy_score(y_test, y_pred))

    line1, = plt.plot(max_samples, train_results, 'b', label='Train Accuracy')
    line2, = plt.plot(max_samples, test_results, 'r', label='Test Accuracy')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Accuracy')
    plt.xlabel('max_samples')
    plt.show()

''' Tuning max_features '''
def tune_max_features( x_train, x_test, y_train, y_test ):
    max_features = list(range(1,25)) # Default is root(num_features)
    train_results = []
    test_results = []
    for max_feature in max_features:
        rf = RandomForestClassifier( class_weight="balanced", max_features=max_feature )
        rf.fit(x_train, y_train)
        train_pred = rf.predict(x_train)
        train_results.append(accuracy_score(y_train, train_pred))
        y_pred = rf.predict(x_test)
        test_results.append(accuracy_score(y_test, y_pred))

    line1, = plt.plot(max_features, train_results, 'b', label='Train Accuracy')
    line2, = plt.plot(max_features, test_results, 'r', label='Test Accuracy')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Accuracy')
    plt.xlabel('max_feature')
    plt.show()

''' Tuning min_samples_leaf '''
def tune_min_samples_leaf( x_train, x_test, y_train, y_test ):
    min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
    train_results = []
    test_results = []
    for min_samples_leaf  in min_samples_leafs:
        rf = RandomForestClassifier( class_weight="balanced", min_samples_leaf =min_samples_leaf )
        rf.fit(x_train, y_train)
        train_pred = rf.predict(x_train)
        train_results.append(accuracy_score(y_train, train_pred))
        y_pred = rf.predict(x_test)
        test_results.append(accuracy_score(y_test, y_pred))

    line1, = plt.plot(min_samples_leafs, train_results, 'b', label='Train Accuracy')
    line2, = plt.plot(min_samples_leafs, test_results, 'r', label='Test Accuracy')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Accuracy')
    plt.xlabel('min_samples_leaf ')
    plt.show()

''' Tuning min_samples_split '''
def tune_min_samples_split( x_train, x_test, y_train, y_test ):
    min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
    train_results = []
    test_results = []
    for min_samples_split in min_samples_splits:
        rf = RandomForestClassifier( class_weight="balanced", min_samples_split=min_samples_split  )
        rf.fit(x_train, y_train)
        train_pred = rf.predict(x_train)
        train_results.append(accuracy_score(y_train, train_pred))
        y_pred = rf.predict(x_test)
        test_results.append(accuracy_score(y_test, y_pred))

    line1, = plt.plot(min_samples_splits, train_results, 'b', label='Train Accuracy')
    line2, = plt.plot(min_samples_splits, test_results, 'r', label='Test Accuracy')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Accuracy')
    plt.xlabel('min_samples_split')
    plt.show()


''' Tuning max_depth '''
def tune_max_depth( x_train, x_test, y_train, y_test ):
    max_depths = np.linspace(1, 32, 32, endpoint=True)
    train_results = []
    test_results = []
    for max_depth in max_depths:
        rf = RandomForestClassifier( class_weight="balanced", max_depth=max_depth  )
        rf.fit(x_train, y_train)
        train_pred = rf.predict(x_train)
        train_results.append(accuracy_score(y_train, train_pred))
        y_pred = rf.predict(x_test)
        test_results.append(accuracy_score(y_test, y_pred))

    line1, = plt.plot(max_depths, train_results, 'b', label='Train Accuracy')
    line2, = plt.plot(max_depths, test_results, 'r', label='Test Accuracy')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Accuracy')
    plt.xlabel('max_depth')
    plt.show()
