from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess_data,
            inputs="ingested_data",
            outputs="preprocessed_data",
            name="preprocess_data_node"
        )
    ])
