import pandas as pd
import numpy as np
import os
import logging
import pickle
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE


def feature_selection(X_train: pd.DataFrame, y_train: pd.DataFrame, parameters: Dict[str, Any]) -> list[str]:
    log = logging.getLogger(__name__)
    log.info(f"We start with: {len(X_train.columns)} columns")

    if parameters["feature_selection"] == "rfe":
        y_train = np.ravel(y_train)

        try:
            with open(os.path.join(os.getcwd(), 'data', '06_models', 'champion_model.pkl'), 'rb') as f:
                classifier = pickle.load(f)
        except:
            classifier = RandomForestClassifier(**parameters['baseline_model_params'])

        rfe = RFE(classifier) 
        rfe = rfe.fit(X_train, y_train)
        f = rfe.get_support(1)  # most important features
        X_cols = X_train.columns[f].tolist()
    else:
        raise ValueError("Unsupported feature_selection method: only 'rfe' is currently implemented.")

    log.info(f"Number of best columns is: {len(X_cols)}")
    return X_cols
