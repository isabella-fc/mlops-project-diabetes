import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np
import pickle
import yaml
import os
import warnings
warnings.filterwarnings("ignore", category=Warning)
import mlflow
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def model_train(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.DataFrame, 
    y_test: pd.DataFrame,
    parameters: Dict[str, Any],
    best_columns: list[str]
) -> Tuple[object, list[str], dict, plt.Figure]:

    with open('conf/local/mlflow.yml') as f:
        experiment_name = yaml.load(f, Loader=yaml.loader.SafeLoader)['tracking']['experiment']['name']
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)
    logger.info('Model training started')

    try:
        with open(os.path.join(os.getcwd(), 'data', '06_models', 'champion_model.pkl'), 'rb') as f:
            classifier = pickle.load(f)
    except:
        classifier = RandomForestClassifier(**parameters['baseline_model_params'])

    results_dict = {}

    with mlflow.start_run(experiment_id=experiment_id, nested=True):
        if parameters.get("use_feature_selection", False):
            X_train = X_train[best_columns]
            X_test = X_test[best_columns]

        y_train = np.ravel(y_train)
        model = classifier.fit(X_train, y_train)

        # Metrics
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        acc_train = accuracy_score(y_train, y_train_pred)
        acc_test = accuracy_score(y_test, y_test_pred)

        results_dict['classifier'] = classifier.__class__.__name__
        results_dict['train_score'] = acc_train
        results_dict['test_score'] = acc_test

        logger.info(f"Model trained. Accuracy on test: {acc_test:.4f}")

        # SHAP summary plot
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        fig = plt.figure()
        shap.summary_plot(shap_values[1], X_train, feature_names=X_train.columns, show=False)

        return model, list(X_train.columns), results_dict, fig
