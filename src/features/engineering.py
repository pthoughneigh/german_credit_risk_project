import numpy as np
import pandas as pd

def create_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create basic derived features for the dataset.

    This function generates a new feature:
    - 'Credit burden per month' = Credit amount / Duration

    The computation is performed only for valid rows where:
    - 'Duration' is not missing and not equal to zero
    - 'Credit amount' is not missing

    Invalid rows are assigned NaN to avoid division errors.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing at least the following columns:
        - 'Credit amount'
        - 'Duration'

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the additional feature:
        - 'Credit burden per month' (rounded to 2 decimal places)

    Notes
    -----
    - The original DataFrame is not modified (a copy is created).
    - Division by zero is explicitly prevented.
    - Missing values are preserved as NaN where computation is not possible.
    - This is a simple feature engineering step often used to normalize
      credit amount relative to loan duration.
    """
    df = df.copy()

    valid_mask = (
        df['Duration'].notna() &
        (df['Duration'] != 0) &
        df['Credit amount'].notna()
    )

    df['Credit burden per month'] = np.nan

    df.loc[valid_mask, 'Credit burden per month'] = (
        df.loc[valid_mask, 'Credit amount'] /
        df.loc[valid_mask, 'Duration']
    )

    df['Credit burden per month'] = df['Credit burden per month'].round(2)

    return df