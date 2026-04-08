import numpy as np

def create_basic_features(df):
    df = df.copy()

    valid_mask = (
        df['Duration'].notna() &
        (df['Duration'] != 0) &
        df['Credit amount'].notna()
    )

    df['credit_burden_per_month'] = np.nan

    df.loc[valid_mask, 'credit_burden_per_month'] = (
        df.loc[valid_mask, 'Credit amount'] /
        df.loc[valid_mask, 'Duration']
    )

    df['credit_burden_per_month'] = df['credit_burden_per_month'].round(2)

    return df