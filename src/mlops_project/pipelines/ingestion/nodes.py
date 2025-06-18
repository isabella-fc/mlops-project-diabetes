import pandas as pd
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
import logging
import great_expectations as ge
from typing import Dict, Any
from great_expectations.core import ExpectationSuite
import hopsworks

from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

conf_path = str(Path("") / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]

logger = logging.getLogger(__name__)


def build_expectation_suite(expectation_suite_name: str, feature_group: str) -> ExpectationSuite:

    expectation_suite = ExpectationSuite(
        expectation_suite_name=expectation_suite_name
    )


    # Range check for categorical features
    expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "Age", "min_value": 1, "max_value": 13},
            )
        )
    
    expectation_suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "GenHlth", "min_value": 1, "max_value": 5},
        )
    )

    expectation_suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "Education", "min_value": 1, "max_value": 6},
        )
    )

    expectation_suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "Income", "min_value": 1, "max_value": 8},
        )
    )
    
    for col in ['MentHlth', 'PhysHlth']:
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": col, "min_value": 0, "max_value": 30},
            )
        )


    # Numerical features expectations

    if feature_group == "numerical_features":
        for col in ['BMI', 'GenHlth','MentHlth', 'PhysHlth','Age', 'Education','Income']:
            expectation_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": col, "type_": "float64"},
                )
            )
            expectation_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_not_be_null",
                    kwargs={"column": col},
                )
            )

    # Binary features expectations

    if feature_group == "binary_features":
        binary_columns = [
            "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
            "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
            "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
            "DiffWalk", "Sex"
        ]
        for col in binary_columns:
            expectation_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_in_set",
                    kwargs={"column": col, "value_set": [0.0, 1.0]},
                )
            )
            expectation_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_not_be_null",
                    kwargs={"column": col},
                )
            )

    # Target feature expectations

    if feature_group == "target":
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_in_set",
                kwargs={"column": "Diabetes_012", "value_set": [0.0, 1.0, 2.0]},
            )
        )
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "Diabetes_012"},
            )
        )

    return expectation_suite


def to_feature_store(
    data: pd.DataFrame,
    group_name: str,
    feature_group_version: int,
    description: str,
    group_description: list,
    validation_expectation_suite: ExpectationSuite,
    credentials_input: dict
):
    project = hopsworks.login(
        api_key_value=credentials_input["FS_API_KEY"],
        project=credentials_input["FS_PROJECT_NAME"]
    )
    feature_store = project.get_feature_store()

    feature_group = feature_store.get_or_create_feature_group(
        name=group_name,
        version=feature_group_version,
        description=description,
        primary_key=["index"],
        event_time="datetime",
        online_enabled=False,
        expectation_suite=validation_expectation_suite,
    )

    feature_group.insert(
        features=data,
        overwrite=False,
        write_options={"wait_for_job": True}
    )

    for feature in group_description:
        feature_group.update_feature_description(feature["name"], feature["description"])

    feature_group.statistics_config = {
        "enabled": True,
        "histograms": True,
        "correlations": True,
    }
    feature_group.update_statistics_config()
    feature_group.compute_statistics()

    logger.info(f"Successfully saved '{group_name}' to Hopsworks Feature Store.")
    return feature_group


def ingest_data(
    patients: pd.DataFrame,
    parameters: Dict[str, Any]
) -> pd.DataFrame:

    df_full = patients.copy()
    df_full = df_full.reset_index()
    df_full["datetime"] = pd.Timestamp.now()

    logger.info(f"Loaded dataset with shape: {df_full.shape}")

    # Infer features
    numerical_features = df_full.select_dtypes(include=["float64", "int64"]).columns.tolist()
    binary_features = [col for col in numerical_features if df_full[col].dropna().isin([0.0, 1.0]).all()]
    numerical_features = [col for col in numerical_features if col not in binary_features + ["Diabetes_012", "index"]]

    logger.info(f"Detected numerical: {numerical_features}")
    logger.info(f"Detected binary: {binary_features}")

    # Build GE suites
    suite_numerical = build_expectation_suite("numerical_suite", "numerical_features")
    suite_binary = build_expectation_suite("binary_suite", "binary_features")
    suite_target = build_expectation_suite("target_suite", "target")

    # GE validation
    ge_df = ge.from_pandas(df_full)
    for suite in [suite_numerical, suite_binary, suite_target]:
        result = ge_df.validate(expectation_suite=suite)
        if not result.success:
            logger.error("GE validation failed.")
            raise ValueError("Great Expectations validation failed.")

    logger.info("GE validation passed.")

    if parameters.get("to_feature_store", False):
        credentials_input = credentials["feature_store"]

        df_numerical = df_full[["index", "datetime"] + numerical_features]
        df_binary = df_full[["index", "datetime"] + binary_features]
        df_target = df_full[["index", "datetime", "Diabetes_012"]]

        to_feature_store(
            data=df_numerical,
            group_name="numerical_features",
            feature_group_version=1,
            description="Numerical features from diabetes dataset",
            group_description=[],
            validation_expectation_suite=suite_numerical,
            credentials_input=credentials_input
        )

        to_feature_store(
            data=df_binary,
            group_name="binary_features",
            feature_group_version=1,
            description="Binary features from diabetes dataset",
            group_description=[],
            validation_expectation_suite=suite_binary,
            credentials_input=credentials_input
        )

        to_feature_store(
            data=df_target,
            group_name="target_features",
            feature_group_version=1,
            description="Target labels from diabetes dataset",
            group_description=[],
            validation_expectation_suite=suite_target,
            credentials_input=credentials_input
        )

    return df_full