from kedro.pipeline import Pipeline, node, pipeline
from .nodes import feature_selection

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=feature_selection,
            inputs=["X_train_fe", "y_train", "params:feature_selection"],
            outputs="selected_features",
            name="feature_selection_node"
        )
    ])