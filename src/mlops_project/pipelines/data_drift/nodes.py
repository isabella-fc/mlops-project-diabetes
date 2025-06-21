import logging
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import mlflow


log = logging.getLogger(__name__)

def data_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_path: str
) -> None:
    shared_columns = list(set(reference_data.columns) & set(current_data.columns))
    reference_data = reference_data[shared_columns].copy()
    current_data = current_data[shared_columns].copy()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html(output_path)

    with mlflow.start_run(run_name="data_drift_report", nested=True):
        mlflow.log_artifact(output_path)
