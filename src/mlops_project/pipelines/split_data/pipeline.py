from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_data,
            inputs="ingested_data",
            outputs=[
                "X_train", "X_val", "X_test",
                "y_train", "y_val", "y_test"
            ],
            name="split_data_node"
        )
    ])
