import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from mlops_project.pipelines.data_drift.nodes import data_drift_report


@patch("mlops_project.pipelines.data_drift.nodes.mlflow.log_artifact")
@patch("mlops_project.pipelines.data_drift.nodes.mlflow.start_run")
@patch("mlops_project.pipelines.data_drift.nodes.Report")
def test_data_drift_report_creates_and_logs_html(
    mock_report_class,
    mock_start_run,
    mock_log_artifact
):

    mock_report = MagicMock()
    mock_report_class.return_value = mock_report
    mock_start_run.return_value.__enter__.return_value = MagicMock()


    reference_data = pd.DataFrame({
        "feature1": np.random.rand(10),
        "feature2": np.random.rand(10)
    })

    current_data = pd.DataFrame({
        "feature1": np.random.rand(10),
        "feature2": np.random.rand(10),
        "extra_feature": np.random.rand(10)
    })

    output_path = "mock_output.html"


    data_drift_report(reference_data, current_data, output_path)


    mock_report_class.assert_called_once()
    mock_report.run.assert_called_once()
    mock_report.save_html.assert_called_once_with(output_path)
    mock_log_artifact.assert_called_once_with(output_path)
