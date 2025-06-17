import pandas as pd

def new_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    # Confirmar se o Max BMI vai ser com ou sem os outliers
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
    # Idk se isto tá certo, é kinda o que estamos a prever iodshapihf
    df["DiabetesRiskComposite"] = (
        df["HighBP"] +
        df["HighChol"] +
        df["BMI"].apply(lambda x: 1 if x > 30 else 0) +
        df["PhysActivity"].apply(lambda x: 0 if x == 1 else 1) +
        df["Age"].apply(lambda x: 1 if x >= 8 else 0)
    )

    return df
