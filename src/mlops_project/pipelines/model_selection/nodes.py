import pandas as pd
import numpy as np
import logging
import pickle
import yaml
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, f1_score
import mlflow
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def _get_or_create_experiment_id(experiment_name: str) -> str:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.info(f"Experiment '{experiment_name}' not found. Creating a new one.")
        return mlflow.create_experiment(experiment_name)
    return experiment.experiment_id


def model_selection(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    champion_dict: Dict[str, Any],
    champion_model: Any,
    parameters: Dict[str, Any]
) -> Any:
    
    import pickle
    from pathlib import Path

    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])

    
    with open("conf/local/mlflow.yml") as f:
        experiment_name = yaml.safe_load(f)['tracking']['experiment']['name']
        experiment_id = _get_or_create_experiment_id(experiment_name)
    
    logger.info("testing models")

    models_dict = {
        "RandomForestClassifier": RandomForestClassifier(),
        "GradientBoostingClassifier": GradientBoostingClassifier(),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "XGBClassifier": XGBClassifier(eval_metric="mlogloss")
    }

    results = {}

    for name, model in models_dict.items():
        with mlflow.start_run(experiment_id=experiment_id, nested=True):
            mlflow.sklearn.autolog()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="macro")
            results[name] = {"accuracy": acc, "f1_macro": f1}
            logger.info(f"{name}: Accuracy = {acc:.4f} | F1_macro = {f1:.4f}")
            mlflow.log_metric("f1_macro", f1)

    best_model_name = max(results, key=lambda name: results[name]["f1_macro"])
    best_model = models_dict[best_model_name]
    logger.info(f"Best base model: {best_model_name} (F1_macro = {results[best_model_name]['f1_macro']:.4f})")

    # hyperparameter tuning
    param_grid = parameters['hyperparameters'].get(best_model_name, {})
    with mlflow.start_run(experiment_id=experiment_id, nested=True):
        grid = GridSearchCV(best_model, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        tuned_preds = best_model.predict(X_test)

        tuned_f1 = f1_score(y_test, tuned_preds, average="macro")
        tuned_acc = accuracy_score(y_test, tuned_preds)

        logger.info(f"Tuned model scores: Accuracy = {tuned_acc:.4f}, F1_macro = {tuned_f1:.4f}")
        mlflow.log_metric("tuned_f1_macro", tuned_f1)
        mlflow.log_metric("tuned_accuracy", tuned_acc)

        if champion_dict["test_score"] < tuned_f1:
            logger.info(f"New champion! F1_macro = {tuned_f1:.4f} > {champion_dict['test_score']:.4f}")
            with open("data/07_models/champion_dict.pkl", "wb") as f:
                pickle.dump({"test_score": tuned_f1}, f)
            with open("data/07_models/champion_model.pkl", "wb") as f:
                pickle.dump(best_model, f)
            return best_model
        else:
            logger.info(f"Champion remains. F1_macro = {tuned_f1:.4f} <= {champion_dict['test_score']:.4f}")
            return champion_model if champion_model is not None else best_model  # ✅ fallback to something




def load_or_init_champion() -> Tuple[Dict[str, Any], Any]:
    try:
        with open("data/07_models/champion_dict.pkl", "rb") as f:
            champion_dict = pickle.load(f)
        with open("data/07_models/champion_model.pkl", "rb") as f:
            champion_model = pickle.load(f)
    except FileNotFoundError:
        champion_dict = {"test_score": 0.0}
        champion_model = None
    return champion_dict, champion_model