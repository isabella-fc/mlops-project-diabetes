from kedro.pipeline import Pipeline, node, pipeline
from .nodes import new_features

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=new_features,
            inputs=["X_train_processed", "X_val_processed", "X_test_processed"],
            outputs=["X_train_fe", "X_val_fe", "X_test_fe"],
            name="engineer_features_node"
        )
    ])
