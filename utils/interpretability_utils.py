import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.inspection import PartialDependenceDisplay
from PyALE import ale
import lime
import lime.lime_tabular
import shap
# print the JS visualization code to the notebook


def plot_pdp(data : pd.DataFrame, model):
    """
    This function is used to display the Partial Dependence Plot.
    The Partial Dependence Plot (PDP) shows the marginal effect one
    feature has on the predicted outcome of a machine learning model (Friedman 2001).
    """
    fig, ax = plt.subplots(figsize=(14, 14))
    PartialDependenceDisplay.from_estimator(model, data, range(len(data.columns)), ax=ax)
    pass

def plot_ale(data : pd.DataFrame, model):
    """ 
    This function is used to plot the ALE (Accumulated Local Effect).
    Accumulated Local Effects (ALE) plots describe how features influence 
    the prediction of a ML model on average, while taking into account the 
    dependence between the features.
    """
    for feat in data.columns:
        ale_eff = ale(
            X=data, model=model, feature=[feat], grid_size=50, include_CI=True, C=0.95
            )
        plt.show()

def plot_ice(data : pd.DataFrame, model ):
    """ 
    This function is used to plot the ICE (Individual Conditional Expectation).
    Individual Conditional Expectation (ICE) plots display one curve per instance 
    that shows how the instance’s prediction changes when a feature changes.
    """
    fig, ax = plt.subplots(figsize=(14, 14))
    PartialDependenceDisplay.from_estimator(model, data, range(len(data.columns)), kind='individual', ax=ax)

def display_lime(data: pd.DataFrame, model, index_to_explain : int):
    """
    This function is used to display Lime. 
    LIME explains the prediction of any model by learning from an interpretable model locally around the prediction.
    """
    feature_list = list(data.columns)
    explainer = lime.lime_tabular.LimeTabularExplainer(data.astype(int).values, mode='classification',feature_names=data.columns)
    exp = explainer.explain_instance(data.loc[index_to_explain,feature_list].astype(int).values, model.predict_proba, num_features=len(feature_list))
    exp.show_in_notebook(show_table=True)

def display_shap(data: pd.DataFrame, model, index_to_explain: int):
    """ 
    This function is used to display shap. 
    The Shapley value of the feature xj is 
    the weighted average contribution of xj across all possible subsets S, 
    where S does not include xj.
    """
    
    explainer = shap.TreeExplainer(model)
    choosen_instance = data.loc[[index_to_explain]]
    shap_values = explainer.shap_values(choosen_instance)
    shap.force_plot(explainer.expected_value[1], shap_values[1], choosen_instance)


