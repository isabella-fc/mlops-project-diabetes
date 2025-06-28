

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
    
    ingestion_pipeline = data_ingestion.create_pipeline()
    feature_selection_pipeline = feature_selection.create_pipeline()
    split_pipeline = split_data.create_pipeline()
    preprocessing_pipeline = data_preprocessing.create_pipeline()
    engineering_pipeline = feature_engineering.create_pipeline()
    train_pipeline = model_train.create_pipeline()
    tests_pipeline = data_units_tests.create_pipeline()
    drift_pipeline = data_drift.create_pipeline()
    selection_pipeline = model_selection.create_pipeline()

    return {
        "ingestion": ingestion_pipeline,
        "split_data": split_pipeline,
        "data_preprocessing": preprocessing_pipeline,
        "feature_engineering": engineering_pipeline,
        "feature_selection": feature_selection_pipeline,
        "model_train": train_pipeline,
        "data_units_tests": tests_pipeline,
        "drift": drift_pipeline,
        "model_selection": selection_pipeline,

  
        "__default__": (
            ingestion_pipeline
            + tests_pipeline
            + split_pipeline
            + preprocessing_pipeline
            + engineering_pipeline
            + feature_selection_pipeline
            + selection_pipeline
            + train_pipeline
            + drift_pipeline
        ),
    }
