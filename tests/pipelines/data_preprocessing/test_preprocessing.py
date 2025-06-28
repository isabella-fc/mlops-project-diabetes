import pandas as pd
import numpy as np
from mlops_project.pipelines.data_preprocessing.nodes import preprocess_data, check_missing_flag


def test_preprocess_data_outlier_removal_and_scaling():
    # BMI outlier in sample
    X_train = pd.DataFrame({
        "BMI": [22.0, 27.5, 100.0],  
        "MentHlth": [5, 2, 1],
        "PhysHlth": [10, 3, 0],
        "Age": [5, 8, 9],
        "Other": [1, 2, 3]
    })

    y_train = pd.DataFrame({"target": [0, 1, 2]}) 

    X_val = pd.DataFrame({
        "BMI": [25.0, 30.0],
        "MentHlth": [4, 1],
        "PhysHlth": [6, 1],
        "Age": [6, 7],
        "Other": [0, 0]
    })

    X_test = pd.DataFrame({
        "BMI": [23.0],
        "MentHlth": [2],
        "PhysHlth": [2],
        "Age": [7],
        "Other": [1]
    })

    X_train_p, X_val_p, X_test_p, y_train_p = preprocess_data(X_train, X_val, X_test, y_train)

    # check outlier was removed from training set
    assert len(X_train_p) == 2
    assert not (X_train_p["BMI"] > 80).any()
    assert len(y_train_p) == 2

    # check missing flag replacement
    assert not X_train_p.isna().any().any()
    assert not X_val_p.isna().any().any()
    assert not X_test_p.isna().any().any()

    # check that scaling is applied (mean ~0, std ~1 for train set)
    assert np.allclose(X_train_p[["BMI", "MentHlth", "PhysHlth", "Age"]].mean(), 0, atol=1e-6)
    assert np.allclose(X_train_p[["BMI", "MentHlth", "PhysHlth", "Age"]].std(ddof=0), 1, atol=1e-6)

    # check untouched column remains
    assert "Other" in X_train_p.columns


def test_check_missing_flag_detects_flagged_values():
    df1 = pd.DataFrame({"a": [1, -9999], "b": [3, 4]})
    df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})

    result = check_missing_flag(df1, df2)

    assert result == {"dataset_1": True, "dataset_2": False}
