import logging
import pandas as pd

def ingest_data(patients: pd.DataFrame) -> pd.DataFrame:
    """Load and return patient data from the raw CSV file."""
    return patients