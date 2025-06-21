import pickle
from pathlib import Path
from great_expectations.dataset import PandasDataset
import pandas as pd

def load_suite(suite_path: str):
    with open(suite_path, "rb") as f:
        return pickle.load(f)

def data_unit_tests(df: pd.DataFrame, base_path: str = "data/01_raw") -> dict:
    results = {}

    suite_files = {
        "numerical": "numerical_suite.pkl",
        "binary": "binary_suite.pkl",
        "target": "target_suite.pkl"
    }

    for suite_name, suite_file in suite_files.items():
        suite_path = Path(base_path) / suite_file
        if not suite_path.exists():
            results[suite_name] = {"error": f"Missing suite file: {suite_file}"}
            continue

        suite = load_suite(suite_path)
        expected_columns = {e.kwargs["column"] for e in suite.expectations if "column" in e.kwargs}
        df_subset = df[[col for col in df.columns if col in expected_columns]].copy()

        dataset = PandasDataset(df_subset)
        for expectation in suite.expectations:
            dataset.append_expectation(expectation)

        validation_result = dataset.validate()

        failed = [r for r in validation_result["results"] if not r["success"]]
        results[suite_name] = {
            "success": validation_result["success"],
            "total_expectations": len(validation_result["results"]),
            "failed_expectations": len(failed),
            "failed_details": failed
        }

    return results
