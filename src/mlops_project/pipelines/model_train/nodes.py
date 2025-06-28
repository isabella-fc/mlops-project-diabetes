import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np
import pickle
import yaml
import os
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore", category=Warning)
import mlflow
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt
from pathlib import Path
logger = logging.getLogger(__name__)
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings(action="ignore", category=DataConversionWarning)

def model_train(
    X_train: pd.DataFrame, 
    X_val: pd.DataFrame, 
    y_train: pd.DataFrame, 
    y_val: pd.DataFrame,
    parameters: Dict[str, Any],
    best_columns: list[str]
) -> Tuple[object, list[str], dict]:

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
            X_val = X_val[best_columns]

        y_train = y_train.values.ravel()
        model = classifier.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        # Metrics
        acc_train = accuracy_score(y_train, y_train_pred)
        acc_val = accuracy_score(y_val, y_val_pred)
        f1_train = f1_score(y_train, y_train_pred, average="macro")
        f1_val = f1_score(y_val, y_val_pred, average="macro")

        results_dict["classifier"] = classifier.__class__.__name__
        results_dict["train_accuracy"] = acc_train
        results_dict["val_accuracy"] = acc_val
        results_dict["train_f1_macro"] = f1_train
        results_dict["val_f1_macro"] = f1_val

        logger.info(f"Train acc: {acc_train:.4f}, Val acc: {acc_val:.4f}")
        logger.info(f"Train F1: {f1_train:.4f}, Val F1: {f1_val:.4f}")

        # Log metrics to MLflow
        mlflow.log_metric("train_accuracy", acc_train)
        mlflow.log_metric("val_accuracy", acc_val)
        mlflow.log_metric("train_f1_macro", f1_train)
        mlflow.log_metric("val_f1_macro", f1_val)

        # SHAP analysis
        explainer = shap.Explainer(model)
        shap_values = explainer(X_train)

        plot_dir = Path("data/08_reporting/shap_summary")
        plot_dir.mkdir(parents=True, exist_ok=True)

        if shap_values.values.ndim == 3 and shap_values.values.shape[2] == 3:
            for class_idx in range(3):
                fig = plt.figure()
                shap.summary_plot(
                    shap_values.values[:, :, class_idx],
                    X_train,
                    feature_names=X_train.columns,
                    show=False
                )
                plot_path = plot_dir / f"shap_summary_class_{class_idx}.png"
                plt.savefig(plot_path, bbox_inches="tight")
                plt.close(fig)
                mlflow.log_artifact(str(plot_path), artifact_path="shap_summary_plots")

        else:
            logger.warning(
                f"SHAP value shape unexpected: {shap_values.values.shape}. Expected 3D array for multiclass output."
            )

            
        return model, list(X_train.columns), results_dict, plt

