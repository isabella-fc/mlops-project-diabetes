import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from src.mlops_project.pipelines.ingestion.nodes import ingest_data

@pytest.fixture
def sample_patients_df():
    return pd.DataFrame({
        "Diabetes_012": [0.0, 1.0, 2.0],
        "age": [5, 7, 11],
        "bmi": [22.3, 33.1, 28.9],
        "genhlth": [1, 2, 3],
        "menthlth": [0, 10, 5],
        "physhlth": [5, 15, 0],
        "education": [3, 4, 5],
        "income": [2, 6, 4],
        "highbp": [0.0, 1.0, 1.0],
        "smoker": [1.0, 0.0, 1.0]
    })

@pytest.fixture
def mock_parameters():
    return {"build_data_docs": False, "target_column": "Diabetes_012"}

@patch("mlops_project.pipelines.ingestion.nodes.get_context")
def test_ingest_data_passes_validation(mock_get_context, sample_patients_df, mock_parameters):
    # Mock GE validator result
    mock_context = MagicMock()
    mock_validator = MagicMock()
    mock_validator.validate.return_value.success = True
    mock_context.get_validator.return_value = mock_validator
    mock_get_context.return_value = mock_context

    output_df = ingest_data(sample_patients_df, mock_parameters)

    assert isinstance(output_df, pd.DataFrame)
    assert "datetime" in output_df.columns
    assert "index" in output_df.columns
    assert output_df.shape[0] == 3
    assert output_df["Diabetes_012"].isin([0, 1, 2]).all()
