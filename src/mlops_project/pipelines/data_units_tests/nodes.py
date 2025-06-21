from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.dataset import PandasDataset
import pandas as pd

def data_unit_tests(df: pd.DataFrame, expectation_suite: ExpectationSuite) -> dict:
    dataset = PandasDataset(df.copy())

    for expectation in expectation_suite.expectations:
        dataset.append_expectation(expectation)

    validation_result = dataset.validate()

    success = validation_result["success"]
    total_expectations = len(validation_result["results"])
    failed_expectations = [
        result for result in validation_result["results"] if not result["success"]
    ]

    return {
        "success": success,
        "total_expectations": total_expectations,
        "failed_expectations": len(failed_expectations),
        "failed_details": failed_expectations
    }
