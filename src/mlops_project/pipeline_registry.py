"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from mlops_project.pipelines import (
    ingestion as data_ingestion,
    feature_selection,
    split_data

    )


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    ingestion_pipeline = data_ingestion.create_pipeline()
    split_pipeline = split_data.create_pipeline()


    return {
        "ingestion": ingestion_pipeline,
        "feature_selection": feature_selection.create_pipeline(),
        "split_data": split_data.create_pipeline()
        }
