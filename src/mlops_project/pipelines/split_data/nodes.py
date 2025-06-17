from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(data: pd.DataFrame, target_column: str = "Diabetes_012", test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42):
    y = data[target_column]
    
    train_data, temp_data = train_test_split(
        data,
        test_size=test_size + val_size,
        stratify=y,
        random_state=random_state,
        shuffle=True
    )

    y_temp = temp_data[target_column]
    relative_val_size = val_size / (test_size + val_size)

    val_data, test_data = train_test_split(
        temp_data,
        test_size=1 - relative_val_size,
        stratify=y_temp,
        random_state=random_state,
        shuffle=True
    )

    return train_data, val_data, test_data
