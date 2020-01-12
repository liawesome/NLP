import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from src.data_loading import load_data
from config import test_data_path, train_data_path, output_csv_path
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
import numpy as np
from sklearn.model_selection import KFold
import csv
from src.classifiers import decision_tree_classifier, logistic_regression, svc, lsvc
from scipy.stats import randint as randint
from scipy.stats import uniform
from sklearn.model_selection import RepeatedKFold

# 1.Pre-processing data #
# split our data into training and testing data
def split_data(data):
    X_train, X_test, Y_train, Y_test = train_test_split(data['data'], data['target'], train_size=0.8,
                                                        test_size=0.2)
    return [X_train, Y_train, X_test, Y_test]

#2. Feature engineering
#raw text data will be transformed into feature vectors
#and new features will be created using the existing dataset with the three following methods
def tfidf_pipeline(classifier):
    return Pipeline([('vect', CountVectorizer()),  #BoW
                     ('tfidf', TfidfTransformer()),
                     ('norm', Normalizer()),   #scaling individual samples to have unit norm
                     classifier])

# set binary matrix
def binary_occurrences_pipeline(classifier):
    return Pipeline([('vect', CountVectorizer(binary=True)),  # For True, all non zero counts are set to 1.
                     classifier])

# load the data from both test_data and train_data
test_data = load_data(test_data_path)
data = load_data(train_data_path)

# ================output====================
def output_prediction(Y_pred, test_data, test_index=None):
    output = []
    for i, y in enumerate(Y_pred):
        index = i if test_index is None else test_index[i]
        output.append([test_data['file_names'][index], y])

    def sortFiles(x):
        try:
            return int(x[0])
        except ValueError:
            return x[0]

    output.sort(key=sortFiles)
    with open(output_csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["index", "Labels"])
        for x in output:
            w.writerow(x)


# use our splited test set from the 4000 data sets to predict the Y test value
def experiment(pipeline, model_name, X_train, Y_train, X_test, Y_test):

    pipeline.fit(X_train, Y_train)
    Y_pred = pipeline.predict(X_test).tolist()
    #print(Y_pred)
    #print(Y_test)
    print(metrics.classification_report(Y_test, Y_pred,
                                            target_names=["label_0", "label_1"]))
    f1 = metrics.f1_score(Y_test, Y_pred, average='micro')
    return [f1, Y_pred, pipeline]

# Model validation
def experiment_basic_validation():
    [X_train, Y_train, X_test, Y_test] = split_data(data)
    #[f1, Y_pred, model] = experiment(tfidf_pipeline(('clf', MultinomialNB())), "mnb+tfidf", X_train, Y_train, X_test, Y_test)
    #[f1, Y_pred, model] = experiment(binary_occurrences_pipeline(('clf', MultinomialNB())), "mnb+binary", X_train, Y_train, X_test, Y_test)
    #[f1, Y_pred, model] = experiment(tfidf_pipeline(('dtc', decision_tree_classifier)), "mnb+tfidf", X_train, Y_train, X_test, Y_test)
    [f1, Y_pred, model] = experiment(tfidf_pipeline(('lr', lsvc)), "mnb+tfidf", X_train, Y_train, X_test, Y_test)
    #[f1, Y_pred, model] = experiment(tfidf_pipeline(('lgr', logistic_regression)), "mnb+tfidf", X_train, Y_train, X_test, Y_test)

    pred = model.predict(test_data['data'])
    output_prediction(pred, test_data)

kFold_n = 8

def experiment_k_fold():
    best_f1 = 0
    best_Y_pred = []
    best_test_index = []
    best_model = None
    result = None
    kf = KFold(n_splits=kFold_n, random_state=None, shuffle=True)
    pipeline = tfidf_pipeline(('lr', logistic_regression))
    for train_index, test_index in kf.split(data['data']):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train = [data['data'][i] for i in train_index]
        X_test = [data['data'][i] for i in test_index]
        Y_train = [data['target'][i] for i in train_index]
        Y_test = [data['target'][i] for i in test_index]
        [f1, Y_pred, model] = experiment(pipeline, "mnb+tfidf", X_train, Y_train, X_test, Y_test)
        pred = model.predict(test_data['data'])
        if (f1 > best_f1):
            best_f1 = f1
            best_Y_pred = Y_pred
            best_test_index = test_index
            best_model = model
            result = pred
    print("Best f1: ", best_f1)
    # Y_pred_test = best_model.predict(test_data['data'])
    output_prediction(pred, test_data)


experiment_basic_validation()
#experiment_k_fold()
