from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(
    data: pd.DataFrame,
    target_column: str = "Diabetes_012",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
):
    y = data[target_column]
    X = data.drop(columns=[target_column])

    # First split off test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
        shuffle=True
    )

    relative_val_size = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_val_size,
        stratify=y_temp,
        random_state=random_state,
        shuffle=True
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
