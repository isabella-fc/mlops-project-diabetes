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
    X_test: pd.DataFrame, 
    y_train: pd.DataFrame, 
    y_test: pd.DataFrame,
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
            X_test = X_test[best_columns]

        y_train = y_train.values.ravel()
        model = classifier.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Metrics
        acc_train = accuracy_score(y_train, y_train_pred)
        acc_test = accuracy_score(y_test, y_test_pred)
        f1_train = f1_score(y_train, y_train_pred, average="macro")
        f1_test = f1_score(y_test, y_test_pred, average="macro")

        results_dict["classifier"] = classifier.__class__.__name__
        results_dict["train_accuracy"] = acc_train
        results_dict["test_accuracy"] = acc_test
        results_dict["train_f1_macro"] = f1_train
        results_dict["test_f1_macro"] = f1_test

        logger.info(f"Train acc: {acc_train:.4f}, Test acc: {acc_test:.4f}")
        logger.info(f"Train F1: {f1_train:.4f}, Test F1: {f1_test:.4f}")

        # Log metrics to MLflow
        mlflow.log_metric("train_accuracy", acc_train)
        mlflow.log_metric("test_accuracy", acc_test)
        mlflow.log_metric("train_f1_macro", f1_train)
        mlflow.log_metric("test_f1_macro", f1_test)

               
        run_id = mlflow.active_run().info.run_id
        metrics_plot = {
            "Train Accuracy": acc_train,
            "Test Accuracy": acc_test,
            "Train F1": f1_train,
            "Test F1": f1_test
        }
        plot_path = plot_model_metrics_bar(metrics_plot, run_id)
        mlflow.log_artifact(str(plot_path), artifact_path="metric_plots")

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



def plot_model_metrics_bar(metrics: dict, run_id: str, output_dir: str = "data/08_reporting/metrics") -> Path:
    
    fig, ax = plt.subplots()
    ax.bar(metrics.keys(), metrics.values(), color=["blue", "blue", "green", "green"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Metrics")
    plt.xticks(rotation=20)
    plt.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plot_path = output_path / f"metrics_{run_id}.png"
    plt.savefig(plot_path)
    plt.close(fig)

    return plot_path

def plot_confusion_matrix(y_true, y_pred, labels, run_id, output_dir="data/08_reporting/confusion_matrix") -> Path:
 
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_path = output_path / f"confusion_matrix_{run_id}.png"
    plt.savefig(plot_path)
    plt.close(fig)

    return plot_path