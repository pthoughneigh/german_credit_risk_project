import pandas as pd
from typing import Sequence, Dict, Tuple


def encode_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Encode the target variable into numeric format.

    This function maps categorical target labels to numeric values:
    - "good" -> 0
    - "bad" -> 1

    Any unexpected values are detected and reported. Rows with unknown
    target values are removed.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the target column.
    target_col : str
        Name of the target column to encode.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the encoded target column. Rows with invalid
        target values are dropped.

    Notes
    -----
    - Uses a fixed mapping: {"good": 0, "bad": 1}
    - Unknown values are converted to NaN and then removed
    - Does NOT modify the original DataFrame
    - Final target is suitable for binary classification models
    """
    df = df.copy()
    target_map = {"good": 0, "bad": 1}

    unique_values = df[target_col].dropna().unique()
    unexpected = set(unique_values) - set(target_map.keys())

    if unexpected:
        print(f"Warning: unexpected target values found: {unexpected}")

    df[target_col] = df[target_col].map(target_map)
    df = df.dropna(subset=[target_col])

    df[target_col] = df[target_col].astype(int)

    return df


def encode_ordinal_columns(
    df: pd.DataFrame,
    ordinal_cols: Sequence[str]
) -> pd.DataFrame:
    """
    Encode ordinal categorical columns into numeric values.

    Ordinal features have a natural order, so they are mapped to integers
    while preserving category order.

    Missing values are replaced with "unknown", which is mapped to -1.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    ordinal_cols : Sequence[str]
        List of ordinal column names to encode.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with encoded ordinal columns.

    Notes
    -----
    - Encoding preserves ordering relationships between categories
    - "unknown" values are explicitly handled and mapped to -1
    - Each column has its own mapping dictionary
    - Does NOT modify the original DataFrame
    """
    df = df.copy()

    ordinal_mappings: Dict[str, Dict[str, int]] = {
        "Saving accounts": {
            "unknown": -1,
            "little": 0,
            "moderate": 1,
            "rich": 2,
            "quite rich": 3,
        },
        "Checking account": {
            "unknown": -1,
            "little": 0,
            "moderate": 1,
            "rich": 2,
        },
    }

    for col in ordinal_cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

        if col not in ordinal_mappings:
            raise ValueError(f"No ordinal mapping defined for column '{col}'.")

        df[col] = df[col].fillna("unknown")
        df[col] = df[col].map(ordinal_mappings[col])

        unexpected = df[col].isna()
        if unexpected.any():
            raise ValueError(
                f"Column '{col}' contains unmapped values after encoding."
            )

        df[col] = df[col].astype(int)

    return df


def encode_nominal_columns(
    df: pd.DataFrame,
    nominal_cols: Sequence[str]
) -> pd.DataFrame:
    """
    Encode nominal categorical columns using one-hot encoding.

    Each category is converted into a binary indicator column.
    One category per feature is dropped to avoid multicollinearity
    in linear models.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    nominal_cols : Sequence[str]
        List of nominal column names to encode.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with one-hot encoded columns.

    Notes
    -----
    - Uses pd.get_dummies with drop_first=True
    - Avoids dummy-variable redundancy for linear models
    - Creates an implicit baseline category
    - Does NOT modify the original DataFrame
    """
    df = df.copy()
    df = pd.get_dummies(df, columns=list(nominal_cols), drop_first=True)
    return df


def fit_numerical_scaler(
    df: pd.DataFrame,
    numerical_cols: Sequence[str]
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute scaling parameters for numerical columns.

    Missing values are imputed with the training median before estimating
    scaling statistics. Standardization uses mean and standard deviation.

    Parameters
    ----------
    df : pd.DataFrame
        Training DataFrame used to estimate scaling parameters.
    numerical_cols : Sequence[str]
        Numerical columns to scale.

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        medians : pd.Series
            Median values used for missing-value imputation.
        means : pd.Series
            Mean values of imputed training columns.
        stds : pd.Series
            Standard deviations of imputed training columns, with near-zero
            values protected against division by zero.

    Notes
    -----
    This function should be fitted on the training set only.
    """
    medians = df[list(numerical_cols)].median()

    temp = df[list(numerical_cols)].copy()
    temp = temp.fillna(medians)

    means = temp.mean()
    stds = temp.std()
    stds = stds.mask(stds < 1e-12, 1.0)

    return medians, means, stds


def transform_numerical_columns(
    df: pd.DataFrame,
    numerical_cols: Sequence[str],
    medians: pd.Series,
    means: pd.Series,
    stds: pd.Series
) -> pd.DataFrame:
    """
    Apply numerical scaling using precomputed parameters.

    Each column is transformed as:
        (value - mean) / standard deviation

    Missing values are filled with precomputed medians.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to transform.
    numerical_cols : Sequence[str]
        Numerical columns to scale.
    medians : pd.Series
        Medians computed on the training set for imputation.
    means : pd.Series
        Means computed on the training set.
    stds : pd.Series
        Standard deviations computed on the training set.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with scaled numerical columns.

    Notes
    -----
    - Does NOT modify the original DataFrame
    - Uses training statistics only
    - Suitable for application to both train and test sets
    """
    df = df.copy()

    for col in numerical_cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

        df[col] = df[col].fillna(medians[col])
        df[col] = (df[col] - means[col]) / stds[col]

    return df


def finalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final cleanup of feature data types.

    Converts boolean columns into integer format (0 and 1),
    ensuring consistency across all features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with boolean columns converted to integers.

    Notes
    -----
    - Ensures all features are numeric
    - Improves compatibility with ML models and downstream analysis
    - Does NOT modify the original DataFrame
    """
    df = df.copy()

    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


def prepare_modeling_dataset(
    df: pd.DataFrame,
    target_col: str,
    ordinal_cols: Sequence[str],
    nominal_cols: Sequence[str]
) -> pd.DataFrame:
    """
    Apply non-leaking preprocessing steps for modeling.

    This function performs only transformations that do not require
    fitting statistics from the full dataset:
    1. Encode target variable
    2. Encode ordinal features
    3. Encode nominal features
    4. Finalize feature data types

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame.
    target_col : str
        Name of the target column.
    ordinal_cols : Sequence[str]
        List of ordinal columns.
    nominal_cols : Sequence[str]
        List of nominal columns.

    Returns
    -------
    pd.DataFrame
        Processed dataset ready for train-test split and later scaling.

    Notes
    -----
    Scaling is intentionally excluded from this function to avoid
    data leakage. Numerical scaling should be fitted on the training
    set only and then applied to both train and test sets.
    """
    df = df.copy()

    df = (
        df
        .pipe(encode_target, target_col)
        .pipe(encode_ordinal_columns, ordinal_cols)
        .pipe(encode_nominal_columns, nominal_cols)
        .pipe(finalize_features)
    )

    return df