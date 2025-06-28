from kedro.pipeline import node, Pipeline
from .nodes import data_drift_report

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=data_drift_report,
                inputs=dict(
                    reference_data="reference_dataset_preprocessed",
                    current_data="current_dataset",
                    output_path="params:data_drift_report_path"
                ),
                outputs=None,
                name="data_drift_report_node"
            )
        ]
    )