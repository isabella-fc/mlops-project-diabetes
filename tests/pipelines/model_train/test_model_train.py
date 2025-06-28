import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from mlops_project.pipelines.model_train.nodes import model_train


@patch("mlops_project.pipelines.model_train.nodes.mlflow.start_run")
@patch("mlops_project.pipelines.model_train.nodes.mlflow.get_experiment_by_name")
@patch("mlops_project.pipelines.model_train.nodes.pickle.load", side_effect=FileNotFoundError)
def test_model_train_runs_without_shap(
    mock_pickle,
    mock_get_experiment,
    mock_mlflow_run
):
    mock_get_experiment.return_value = MagicMock(experiment_id="unit-test-exp")
    mock_mlflow_run.return_value.__enter__.return_value = MagicMock()

    X = pd.DataFrame({
        "f1": np.random.rand(20),
        "f2": np.random.rand(20),
        "f3": np.random.rand(20)
    })
    y = pd.DataFrame({"target": np.random.choice([0, 1, 2], size=20)})

    selected_features = ["f1", "f2"]

    parameters = {
        "use_feature_selection": True,
        "baseline_model_params": {
            "n_estimators": 10,
            "max_depth": 3,
            "random_state": 42
        }
    }

    model, features, metrics, plt_obj = model_train(X, X, y, y, parameters, selected_features)

    assert model is not None
    assert isinstance(metrics, dict)
    assert "train_accuracy" in metrics
    assert "val_accuracy" in metrics
    assert "train_f1_macro" in metrics
    assert "val_f1_macro" in metrics
    assert features == selected_features
