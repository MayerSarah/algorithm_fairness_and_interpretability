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


def plot_confusion_matrix(model, test_features, test_labels):

    """
    Plot the confusion matrix of the classifier.
    """

    predictions_model = model.predict(test_features)
    # Get and reshape confusion matrix data
    matrix = confusion_matrix(test_labels, predictions_model)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    print(matrix[0] * len(test_labels[test_labels == 0]))
    print(matrix[1] * len(test_labels[test_labels == 1]))

    # Build the plot
    plt.figure(figsize=(16,7))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size':2},
                cmap=plt.cm.Greens, linewidths=0.2)

    # Add labels to the plot
    class_names = ['has_not_defaulted', 'has_defaulted']
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names, rotation=25)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix for Random Forest Model')
    plt.show()


def plot_metrics(model, test_features, test_labels):

    """
    Plot metrics to evalute the performance of the classifier.
    """

    predictions_model = model.predict(test_features)
    print('Accuracy : %.3f' % accuracy_score(predictions_model, test_labels))

    print(classification_report(test_labels, predictions_model))


def plot_roc_curve(model, test_features, test_labels):

    """
    Plot the ROC curve of the classifier.

    """
    
    model_roc = RocCurveDisplay.from_estimator(model, test_features, test_labels)
    plt.show()