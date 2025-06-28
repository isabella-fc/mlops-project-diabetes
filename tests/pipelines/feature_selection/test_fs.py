import pandas as pd
import numpy as np
import os
import tempfile
from unittest.mock import patch, MagicMock
from mlops_project.pipelines.feature_selection.nodes import feature_selection


@patch("mlops_project.pipelines.feature_selection.nodes.pickle.load")
@patch("mlops_project.pipelines.feature_selection.nodes.open")
def test_feature_selection_rfe_selects_features(mock_open, mock_pickle):

    X = pd.DataFrame({
        "f1": np.random.rand(20),
        "f2": np.random.rand(20),
        "f3": np.random.rand(20),
        "datetime": pd.date_range("2022-01-01", periods=20)  
    })
    y = pd.Series(np.random.choice([0, 1, 2], size=20))

    params = {
        "feature_selection": "rfe",
        "baseline_model_params": {
            "n_estimators": 10,
            "max_depth": 3,
            "random_state": 42
        }
    }

    # simulate no champion model
    mock_pickle.side_effect = FileNotFoundError  


    with patch("mlops_project.pipelines.feature_selection.nodes.os.getcwd", return_value=tempfile.gettempdir()), \
         patch("mlops_project.pipelines.feature_selection.nodes.pd.Series.to_csv") as mock_to_csv:

        selected = feature_selection(X, y, params)

    assert isinstance(selected, list)
    assert all(isinstance(f, str) for f in selected)
    assert "datetime" not in selected  # dropped
    assert mock_to_csv.called
