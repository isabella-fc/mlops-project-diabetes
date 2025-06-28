import pandas as pd

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["HealthIndex"] = (
        df["BMI"] / df["BMI"].max() +
        df["GenHlth"] / 5 +
        df["MentHlth"] / 30 +
        df["PhysHlth"] / 30 +
        df["DiffWalk"]
    )

    df["LifestyleScore"] = (
        df["PhysActivity"] +
        df["Fruits"] +
        df["Veggies"] -
        df["HvyAlcoholConsump"] -
        df["Smoker"]
    )

    df["Mental_Physical_Gap"] = df["MentHlth"] - df["PhysHlth"]

    df["AccessIssues"] = (1 - df["AnyHealthcare"]) + df["NoDocbcCost"]

    df["DiabetesRiskComposite"] = (
        df["HighBP"] +
        df["HighChol"] +
        df["BMI"].apply(lambda x: 1 if x > 30 else 0) +
        df["PhysActivity"].apply(lambda x: 0 if x == 1 else 1) +
        df["Age"].apply(lambda x: 1 if x >= 8 else 0)
    )

    return df


def new_features(
    X_train_processed: pd.DataFrame,
    X_val_processed: pd.DataFrame,
    X_test_processed: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        add_engineered_features(X_train_processed),
        add_engineered_features(X_val_processed),
        add_engineered_features(X_test_processed)
    )
