import pandas as pd
from mlops_project.pipelines.feature_engineering.nodes import new_features, add_engineered_features


def test_add_engineered_features_creates_all_expected_columns():
    df = pd.DataFrame({
        "BMI": [25, 35],
        "GenHlth": [3, 4],
        "MentHlth": [10, 5],
        "PhysHlth": [5, 10],
        "DiffWalk": [0, 1],
        "PhysActivity": [1, 0],
        "Fruits": [1, 0],
        "Veggies": [1, 1],
        "HvyAlcoholConsump": [0, 1],
        "Smoker": [0, 1],
        "AnyHealthcare": [1, 0],
        "NoDocbcCost": [0, 1],
        "HighBP": [1, 0],
        "HighChol": [1, 1],
        "Age": [5, 8]
    })


    df_fe = add_engineered_features(df)

    #check new columns exist
    expected_columns = [
        "HealthIndex",
        "LifestyleScore",
        "Mental_Physical_Gap",
        "AccessIssues",
        "DiabetesRiskComposite"
    ]

    for col in expected_columns:
        assert col in df_fe.columns, f"{col} not found in engineered features"

   
    for col in expected_columns:
        assert pd.api.types.is_numeric_dtype(df_fe[col]), f"{col} is not numeric"


def test_new_features_applies_feature_engineering_to_all_sets():
    df = pd.DataFrame({
        "BMI": [25, 30],
        "GenHlth": [3, 2],
        "MentHlth": [5, 6],
        "PhysHlth": [4, 2],
        "DiffWalk": [0, 1],
        "PhysActivity": [1, 1],
        "Fruits": [1, 1],
        "Veggies": [1, 1],
        "HvyAlcoholConsump": [0, 0],
        "Smoker": [0, 1],
        "AnyHealthcare": [1, 1],
        "NoDocbcCost": [0, 0],
        "HighBP": [1, 0],
        "HighChol": [0, 1],
        "Age": [5, 6]
    })

    X_train_fe, X_val_fe, X_test_fe = new_features(df, df, df)

    assert X_train_fe.equals(X_val_fe)
    assert X_train_fe.equals(X_test_fe)

    assert "DiabetesRiskComposite" in X_train_fe.columns
    assert X_train_fe.shape[1] == df.shape[1] + 5 
