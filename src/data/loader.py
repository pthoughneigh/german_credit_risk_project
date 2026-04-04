import pandas as pd
from src.config import RAW_DATA_FILE

def load_raw_data() -> pd.DataFrame:
    """
        Load the raw German credit dataset from CSV.

        Returns:
            pd.DataFrame: Raw dataset
        """
    df = pd.read_csv(RAW_DATA_FILE)
    return df