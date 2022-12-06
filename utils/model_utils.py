import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
import lightgbm as lgbm
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
from sklearn.metrics import plot_confusion_matrix

def split_data(path: str, test_size: float):

    """
    Retrieve and split data between train and test
    """

    features = pd.read_csv(path, sep=';',  thousands=',').drop(columns="ID")
    # Labels are the values we want to predict
    labels = np.array(features['Default (y)'])
    features= features.drop(['Default (y)', 'Pred_default (y_hat)', 'PD','Group'] , axis = 1)

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = test_size, random_state = 42, stratify=labels)

    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    return train_features, test_features, train_labels, test_labels


def oversample(method, train_features, train_labels, sampling_strategy=0.55):

    """
    Oversample the minority class.
    """

    print('Features shape:', train_features.shape)
    print('Labels shape:', train_labels.shape)

    # transform the dataset
    oversample = method(sampling_strategy=sampling_strategy)
    train_features_oversampled, train_labels_oversampled = oversample.fit_resample(train_features, train_labels)

    print('Features oversampled shape:', train_features_oversampled.shape)
    print('Labels oversampled shape:', train_labels_oversampled.shape)

    return train_features_oversampled, train_labels_oversampled


def train_model(algo,
                train_features,
                train_labels,
                best_params=False,
                oversample_method=False,
                sampling_strategy=0.55):


    """
    Train the classifier.
    """

    if oversample_method:
        train_features, train_labels = oversample(oversample_method,
                                                  train_features,
                                                  train_labels,
                                                  sampling_strategy)
    if best_params:
        model = algo(**best_params)
    else:
        model = algo()
    
    model.fit(train_features, train_labels)

    return model, train_features, train_labels