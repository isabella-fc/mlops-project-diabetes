import pandas as pd
from unittest.mock import patch, MagicMock
from mlops_project.pipelines.data_units_tests.nodes import data_unit_tests


@patch("mlops_project.pipelines.data_units_tests.nodes.Path.exists", return_value=True)
@patch("mlops_project.pipelines.data_units_tests.nodes.load_suite")
@patch("mlops_project.pipelines.data_units_tests.nodes.PandasDataset")
def test_data_unit_tests_returns_summary(mock_pandas_dataset_class, mock_load_suite, mock_exists):

    df = pd.DataFrame({
        "age": [5, 7, 11],
        "bmi": [25.0, 30.5, 22.1],
        "genhlth": [1, 2, 3],
        "diabetes_012": [0, 1, 2]
    })

    # mocked expectation suite
    mock_expectation = MagicMock()
    mock_expectation.kwargs = {"column": "age"}
    mock_suite = MagicMock()
    mock_suite.expectations = [mock_expectation]
    mock_load_suite.return_value = mock_suite

    # mock PandasDataset.validate()
    mock_dataset = MagicMock()
    mock_dataset.validate.return_value = {
        "success": True,
        "results": [{"success": True}, {"success": True}]
    }
    mock_pandas_dataset_class.return_value = mock_dataset

    result = data_unit_tests(df, base_path="unused/path")

    for key in ["numerical", "binary", "target"]:
        assert key in result
        assert result[key]["success"] is True
        assert result[key]["total_expectations"] == 2
        assert result[key]["failed_expectations"] == 0
