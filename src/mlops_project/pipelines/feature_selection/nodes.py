import pandas as pd
import numpy as np
import os
import logging
import pickle
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import time
name_suffix = "_" + time.strftime("%Y%m%d_%H%M%S")


def feature_selection(X_train: pd.DataFrame, y_train: pd.DataFrame, parameters: Dict[str, Any]) -> list[str]:
    log = logging.getLogger(__name__)
    log.info(f"Received {len(X_train.columns)} columns before filtering")

    # Drop datetime (created for hopswork)
    X_train = X_train.drop(columns=["datetime"], errors="ignore")
    log.info(f"Using {len(X_train.columns)} columns for feature selection")

    if parameters["feature_selection"] == "rfe":
        y_train = np.ravel(y_train)

        try:
            with open(os.path.join(os.getcwd(), 'data', '06_models', 'champion_model.pkl'), 'rb') as f:
                classifier = pickle.load(f)
        except:
            classifier = RandomForestClassifier(**parameters['baseline_model_params'])

        rfe = RFE(classifier)
        rfe = rfe.fit(X_train, y_train)
        f = rfe.get_support(1)  
        X_cols = X_train.columns[f].tolist()
    else:
        raise ValueError("Unsupported feature_selection method: only 'rfe' is currently implemented.")

    pd.Series(X_cols).to_csv("data/06_selected_features/selected_features_list.csv", index=False)

    selected_df = X_train[X_cols]
    df_path = f"data/06_selected_features/current_data.csv"
    selected_df.to_csv(df_path, index=False)

    log.info(f"Selected {len(X_cols)} best features")
    return X_cols
