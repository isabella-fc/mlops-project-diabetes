import pandas as pd
from src.mlops_project.pipelines.split_data.nodes import split_data


def test_split_data_stratified_and_shapes():
    # create dummy dataset with 20 rows, balanced target
    df = pd.DataFrame({
        "feature_1": range(20),
        "feature_2": range(100, 120),
        "Diabetes_012": [0]*7 + [1]*7 + [2]*6  
    })

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    assert len(df) == sum(len(x) for x in [X_train, X_val, X_test])

    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)

    assert "Diabetes_012" not in X_train.columns
    assert "Diabetes_012" not in X_val.columns
    assert "Diabetes_012" not in X_test.columns

    assert sorted(y_train.unique()) == [0, 1, 2]
    assert sorted(y_val.unique()) == [0, 1, 2]
    assert sorted(y_test.unique()) == [0, 1, 2]

    #check approximate sizes (rounding due to stratified split)
    total = len(df)
    test_size = round(0.15 * total)
    val_size = round(0.15 * (total - test_size))

    assert abs(len(X_test) - test_size) <= 1
    assert abs(len(X_val) - val_size) <= 1
