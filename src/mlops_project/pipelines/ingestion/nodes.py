import pandas as pd
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
import logging
import great_expectations as ge
from typing import Dict, Any
from great_expectations.core import ExpectationSuite
import pickle
from great_expectations.core.batch import RuntimeBatchRequest
from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

conf_path = str(Path("") / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]

import pandas as pd
import logging
from typing import Dict, Any
from great_expectations.data_context import get_context
from great_expectations.core.batch import BatchRequest

logger = logging.getLogger(__name__)


def save_expectation_suite(expectation_suite: ExpectationSuite, filename: str):
    path = Path("data/01_raw") / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(expectation_suite, f)


def build_expectation_suite(expectation_suite_name: str, feature_group: str, available_columns: list[str]) -> ExpectationSuite:
    expectation_suite = ExpectationSuite(expectation_suite_name=expectation_suite_name)

    def add_if_exists(col: str, expectation: ExpectationConfiguration):
        if col in available_columns:
            expectation_suite.add_expectation(expectation)

    # Categorical-like ranges
    add_if_exists("age", ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={"column": "age", "min_value": 1, "max_value": 13},
    ))

    add_if_exists("genhlth", ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={"column": "genhlth", "min_value": 1, "max_value": 5},
    ))

    add_if_exists("education", ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={"column": "education", "min_value": 1, "max_value": 6},
    ))

    add_if_exists("income", ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={"column": "income", "min_value": 1, "max_value": 8},
    ))

    for col in ['menthlth', 'physhlth']:
        add_if_exists(col, ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": col, "min_value": 0, "max_value": 30},
        ))

    if feature_group == "numerical_features":
        for col in ['bmi', 'genhlth', 'menthlth', 'physhlth', 'age', 'education', 'income']:
            add_if_exists(col, ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_of_type",
                kwargs={"column": col, "type_": "float64"},
            ))
            add_if_exists(col, ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": col},
            ))

    if feature_group == "binary_features":
        binary_columns = [
            "highbp", "highchol", "cholcheck", "smoker", "stroke",
            "heartdiseaseorattack", "physactivity", "fruits", "veggies",
            "hvyalcoholconsump", "anyhealthcare", "nodocbccost",
            "diffwalk", "sex"
        ]
        for col in binary_columns:
            add_if_exists(col, ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_in_set",
                kwargs={"column": col, "value_set": [0.0, 1.0]},
            ))
            add_if_exists(col, ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": col},
            ))

    if feature_group == "target":
        add_if_exists("diabetes_012", ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "diabetes_012", "value_set": [0.0, 1.0, 2.0]},
        ))
        add_if_exists("diabetes_012", ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "diabetes_012"},
        ))

    return expectation_suite



def ingest_data(
    patients: pd.DataFrame,
    parameters: Dict[str, Any]
) -> pd.DataFrame:

    df_full = patients.copy()
    df_full = df_full.reset_index()
    df_full["datetime"] = pd.Timestamp.now()

    logger.info(f"Loaded dataset with shape: {df_full.shape}")


    numerical_features = df_full.select_dtypes(include=["float64", "int64"]).columns.tolist()
    binary_features = [col for col in numerical_features if df_full[col].dropna().isin([0.0, 1.0]).all()]
    numerical_features = [col for col in numerical_features if col not in binary_features + ["Diabetes_012", "index"]]

    logger.info(f"Detected numerical: {numerical_features}")
    logger.info(f"Detected binary: {binary_features}")

    # Slice feature subsets
    df_numerical = df_full[["index", "datetime"] + numerical_features]
    df_binary = df_full[["index", "datetime"] + binary_features]
    df_target = df_full[["index", "datetime", "Diabetes_012"]]


    context = get_context()

    def validate_with_suite(data: pd.DataFrame, suite_name: str, asset_name: str):
        batch_request = RuntimeBatchRequest(
            datasource_name="my_pandas_datasource",
            data_connector_name="default_runtime_data_connector_name",
            data_asset_name=asset_name,
            runtime_parameters={"batch_data": data},
            batch_identifiers={"default_identifier_name": "default"},
        )

        validator = context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=suite_name
        )

        result = validator.validate()
        if not result.success:
            raise ValueError(f"Great Expectations validation failed for suite: {suite_name}")

    # Validate each group with corresponding suite
    validate_with_suite(df_numerical, "numerical_suite", "numerical_features")
    validate_with_suite(df_binary, "binary_suite", "binary_features")
    validate_with_suite(df_target, "target_suite", "target_features")

    logger.info("✅ All Great Expectations validations passed.")

    # (Optional) Build Data Docs if desired
    if parameters.get("build_data_docs", False):
        context.build_data_docs()
        logger.info("Data Docs built successfully.")

    return df_full
    # # Store in Hopsworks if flag is set
    # if parameters.get("to_feature_store", True):
    #     from pathlib import Path
    #     from kedro.config import OmegaConfigLoader
    #     from kedro.framework.project import settings

    #     conf_path = str(Path("") / settings.CONF_SOURCE)
    #     conf_loader = OmegaConfigLoader(conf_source=conf_path)
    #     credentials_input = conf_loader["credentials"]["feature_store"]


    #     to_feature_store(
    #         data=df_numerical,
    #         group_name="numerical_features",
    #         feature_group_version=1,
    #         description="Numerical features from diabetes dataset",
    #         group_description=[],
    #         validation_expectation_suite=build_expectation_suite("numerical_suite", "numerical_features", df_numerical.columns.tolist()),
    #         credentials_input=credentials_input
    #     )

    #     to_feature_store(
    #         data=df_binary,
    #         group_name="binary_features",
    #         feature_group_version=1,
    #         description="Binary features from diabetes dataset",
    #         group_description=[],
    #         validation_expectation_suite=build_expectation_suite("binary_suite", "binary_features", df_binary.columns.tolist()),
    #         credentials_input=credentials_input
    #     )

    #     to_feature_store(
    #         data=df_target,
    #         group_name="target_features",
    #         feature_group_version=1,
    #         description="Target variable from diabetes dataset",
    #         group_description=[],
    #         validation_expectation_suite=build_expectation_suite("target_suite", "target", df_target.columns.tolist()),
    #         credentials_input=credentials_input
    #     )

