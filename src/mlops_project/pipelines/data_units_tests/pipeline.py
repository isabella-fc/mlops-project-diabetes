from kedro.pipeline import Pipeline, node, pipeline
from .nodes import data_unit_tests

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=data_unit_tests,
            inputs="ingested_data",
            outputs="validation_report",
            name="data_unit_tests_node"
        )
    ])
