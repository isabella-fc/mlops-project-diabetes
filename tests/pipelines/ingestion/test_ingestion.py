import pandas as pd
import pytest
from unittest.mock import patch
from mlops_project.pipelines.ingestion.nodes import ingest_data

@pytest.fixture
def sample_patients_df():
    return pd.DataFrame({
        "Diabetes_012": [0.0, 1.0, 2.0],
        "age": [5.0, 7.0, 11.0],
        "bmi": [22.3, 33.1, 28.9],
        "genhlth": [1.0, 2.0, 3.0],
        "menthlth": [0.0, 10.0, 5.0],
        "physhlth": [5.0, 15.0, 0.0],
        "education": [3.0, 4.0, 5.0],
        "income": [2.0, 6.0, 4.0],
        "highbp": [0.0, 1.0, 1.0],
        "smoker": [1.0, 0.0, 1.0]
    })

@pytest.fixture
def mock_parameters():
    return {
        "build_data_docs": False,
        "target_column": "Diabetes_012",
        "to_feature_store": False
    }

@patch("mlops_project.pipelines.ingestion.nodes.to_feature_store")  # ✅ patch the function you want to skip
def test_ingest_data_passes_validation(mock_to_feature_store, sample_patients_df, mock_parameters):
    output_df = ingest_data(sample_patients_df, mock_parameters)

    assert isinstance(output_df, pd.DataFrame)
    assert "datetime" in output_df.columns
    assert "index" in output_df.columns
    assert output_df.shape == (3, sample_patients_df.shape[1] + 2)
    assert set(output_df["Diabetes_012"]) == {0.0, 1.0, 2.0}
    mock_to_feature_store.assert_not_called()
