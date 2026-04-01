import pandas as pd

def drop_useless_columns(df) -> pd.DataFrame:
    """
    Drops useless columns from a dataframe.

    :return: a dataframe without useless columns
    """
    df = df.copy()

    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)

    return df