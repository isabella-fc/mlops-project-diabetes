import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    # Explorar mais, testezinho para ver whatzappenings
    # Remove extreme outliers in the 'BMI' column
    df = df[df["BMI"] < 80]

    # Normalize columns using StandardScaler
    columns_to_scale = ["BMI", "MentHlth", "PhysHlth", "Age"]
    scaler = StandardScaler()
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    return df
