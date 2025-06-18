from kedro.pipeline import Pipeline, node, pipeline

from .nodes import ingest_data, build_expectation_suite


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=ingest_data,
            inputs=["patients", "params:ingestion"],
            outputs="ingested_data",
            name="ingest_data_with_validation"
        )
    ])