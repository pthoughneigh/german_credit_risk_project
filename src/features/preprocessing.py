import pandas as pd

def encode_target(df, target_col):
    df = df.copy()
    target_map = {"good": 0, "bad": 1}

    # check unexpected values
    unique_values = df[target_col].unique()
    unexpected = set(unique_values) - set(target_map.keys())

    if unexpected:
        print(f"Warning: unexpected target values found: {unexpected}")

    # map values
    df[target_col] = df[target_col].map(target_map)

    # drop NaN (unknown values)
    df = df.dropna(subset=[target_col])

    return df

def encode_ordinal_columns(df, ordinal_cols):
    df = df.copy()
    for col in ordinal_cols:

        df[col] = df[col].fillna("unknown")
        col_map = {}
        if col == "Saving accounts":
            col_map = {"unknown": -1, "little": 0, "moderate": 1, "rich": 2, "quite rich": 3 }

        elif col == "Checking account":
            col_map = {"unknown": -1, "little": 0, "moderate": 1, "rich": 2}

        df[col] = df[col].map(col_map)

    return df

def encode_nominal_columns(df, nominal_cols):
    df = df.copy()
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=False)
    return df

def scale_numerical_columns(df, numerical_cols):
    df = df.copy()
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())

        mean = df[col].mean()
        std = df[col].std()

        if pd.isna(std) or std < 1e-12:
            df[col] = 0.0
        else:
            df[col] = (df[col] - mean) / std

    return df

def prepare_modeling_dataset(df, target_col, ordinal_cols, nominal_cols, numerical_cols):
    df = df.copy()

    df = (
        df
        .pipe(encode_target, target_col)
        .pipe(encode_ordinal_columns, ordinal_cols)
        .pipe(encode_nominal_columns, nominal_cols)
        .pipe(scale_numerical_columns, numerical_cols)
    )

    return df
