from kedro.pipeline import Pipeline, node, pipeline

from .nodes import ingest_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=ingest_data,
                inputs="patients",
                outputs="ingested_data",
                name="ingestion_node",
            )
        ]
    )