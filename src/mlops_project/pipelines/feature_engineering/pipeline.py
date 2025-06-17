from kedro.pipeline import Pipeline, node, pipeline
from .nodes import new_features

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=new_features,
            inputs="preprocessed_data",
            outputs="engineered_data",
            name="engineer_features_node"
        )
    ])
