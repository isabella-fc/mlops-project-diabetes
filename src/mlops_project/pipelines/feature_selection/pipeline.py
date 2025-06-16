from kedro.pipeline import Pipeline, node, pipeline
from .nodes import feature_selection

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=feature_selection,
            inputs=["model_input_table", "model_input_target", "params:feature_selection"],
            outputs="selected_features",
            name="feature_selection_node"
        )
    ])