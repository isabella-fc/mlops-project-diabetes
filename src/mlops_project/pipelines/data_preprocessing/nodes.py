import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train_copy = X_train.copy()
    X_val_copy = X_val.copy()
    X_test_copy = X_test.copy()

    # Drop extreme outliers in training set only
    X_train_copy = X_train_copy[X_train_copy["BMI"] < 80]

    # Fill missing values with -9999 (used as a flag for missing data)
    X_train_copy = X_train_copy.fillna(-9999)
    X_val_copy = X_val_copy.fillna(-9999)
    X_test_copy = X_test_copy.fillna(-9999)

    # Define columns to scale
    columns_to_scale = ["BMI", "MentHlth", "PhysHlth", "Age"]

    # Fit scaler only on training data
    scaler = StandardScaler()
    X_train_copy[columns_to_scale] = scaler.fit_transform(X_train_copy[columns_to_scale])
    X_val_copy[columns_to_scale] = scaler.transform(X_val_copy[columns_to_scale])
    X_test_copy[columns_to_scale] = scaler.transform(X_test_copy[columns_to_scale])

    return X_train_copy, X_val_copy, X_test_copy


def check_missing_flag(*datasets: pd.DataFrame) -> dict:
    return {
        f"dataset_{i}": (df == -9999).any().any()
        for i, df in enumerate(datasets, start=1)
    }
