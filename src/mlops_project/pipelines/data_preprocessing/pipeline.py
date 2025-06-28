from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess_data,
            inputs=["X_train", "X_val", "X_test", "y_train"],
            outputs=["X_train_processed", "X_val_processed", "X_test_processed", "y_train_matched"],
            name="preprocess_data_node"
        )
    ])
