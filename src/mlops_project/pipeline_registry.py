"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from mlops_project.pipelines import (
    ingestion as data_ingestion,
    feature_selection,
    split_data,
    data_preprocessing,
    feature_engineering,
    model_train,
    data_units_tests,
    data_drift,
    model_selection,
    )


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    ingestion_pipeline = data_ingestion.create_pipeline()


    return {
        "ingestion": ingestion_pipeline,
        "split_data": split_data.create_pipeline(),
        "data_preprocessing": data_preprocessing.create_pipeline(),
        "feature_engineering": feature_engineering.create_pipeline(),
        "feature_selection": feature_selection.create_pipeline(),
        "model_train": model_train.create_pipeline(),
        "data_units_tests": data_units_tests.create_pipeline(),
        "drift": data_drift.create_pipeline(),
        "model_selection": model_selection.create_pipeline(),


        "__default__": (
            ingestion_pipeline +
            data_units_tests.create_pipeline() +
            split_data.create_pipeline() +
            data_preprocessing.create_pipeline() +
            feature_engineering.create_pipeline() +
            feature_selection.create_pipeline() +
            model_selection.create_pipeline() +
            model_train.create_pipeline() +
            data_drift.create_pipeline()
        )
        }