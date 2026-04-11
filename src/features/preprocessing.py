import pandas as pd
from typing import Sequence


def encode_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
        Encode the target variable into numeric format.

        This function maps categorical target labels to numeric values:
        - "good" → 0
        - "bad" → 1

        Any unexpected values are detected and reported. Rows with unknown
        target values are removed.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing the target column.

        target_col : str
            Name of the target column to encode.

        Returns
        -------
        pandas.DataFrame
            A new DataFrame with the encoded target column. Rows with invalid
            target values are dropped.

        Notes
        -----
        - Uses a fixed mapping: {"good": 0, "bad": 1}
        - Unknown values are converted to NaN and then removed
        - Does NOT modify the original DataFrame
        - Final target is suitable for binary classification models

        Example
        -------
        Input:
            Risk = ["good", "bad", "good"]

        Output:
            Risk = [0, 1, 0]
        """
    df = df.copy()
    target_map = {"good": 0, "bad": 1}

    unique_values = df[target_col].unique()
    unexpected = set(unique_values) - set(target_map.keys())

    if unexpected:
        print(f"Warning: unexpected target values found: {unexpected}")

    df[target_col] = df[target_col].map(target_map)
    df = df.dropna(subset=[target_col])

    return df


def encode_ordinal_columns(
    df: pd.DataFrame,
    ordinal_cols: Sequence[str]
) -> pd.DataFrame:
    """
        Encode ordinal categorical columns into numeric values.

        Ordinal features have a natural order (e.g. "little" < "moderate" < "rich"),
        so they are mapped to integers preserving this order.

        Missing values are replaced with "unknown", which is mapped to -1.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame.

        ordinal_cols : list of str
            List of ordinal column names to encode.

        Returns
        -------
        pandas.DataFrame
            A new DataFrame with encoded ordinal columns.

        Notes
        -----
        - Encoding preserves ordering relationships between categories
        - "unknown" values are explicitly handled and mapped to -1
        - Each column has its own mapping dictionary
        - Does NOT modify the original DataFrame

        Important
        ---------
        Distances between encoded values are artificial.
        For example:
            moderate (1) → rich (2)
        does NOT necessarily mean the same increase as:
            little (0) → moderate (1)

        Example
        -------
        Input:
            Saving accounts = ["little", "moderate", None]

        Output:
            Saving accounts = [0, 1, -1]
        """
    df = df.copy()
    for col in ordinal_cols:
        df[col] = df[col].fillna("unknown")
        col_map = {}

        if col == "Saving accounts":
            col_map = {
                "unknown": -1,
                "little": 0,
                "moderate": 1,
                "rich": 2,
                "quite rich": 3,
            }
        elif col == "Checking account":
            col_map = {
                "unknown": -1,
                "little": 0,
                "moderate": 1,
                "rich": 2,
            }

        df[col] = df[col].map(col_map)

    return df


def encode_nominal_columns(
    df: pd.DataFrame,
    nominal_cols: Sequence[str]
) -> pd.DataFrame:
    """
        Encode nominal categorical columns using one-hot encoding.

        Each category is converted into a binary column (0 or 1).
        One category per feature is dropped to avoid multicollinearity
        (dummy variable trap).

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame.

        nominal_cols : list of str
            List of nominal (unordered categorical) columns to encode.

        Returns
        -------
        pandas.DataFrame
            A new DataFrame with one-hot encoded columns.

        Notes
        -----
        - Uses pd.get_dummies with drop_first=True
        - Avoids linear dependency between dummy variables
        - Creates an implicit baseline category
        - Does NOT modify the original DataFrame

        Example
        -------
        Input:
            Sex = ["male", "female"]

        Output:
            Sex_male = [1, 0]
            (female is baseline)
        """
    df = df.copy()
    df = pd.get_dummies(df, columns=list(nominal_cols), drop_first=True)
    return df


def scale_numerical_columns(
    df: pd.DataFrame,
    numerical_cols: Sequence[str]
) -> pd.DataFrame:
    """
       Standardize numerical columns using z-score normalization.

       Each column is transformed as:
           (value - mean) / standard deviation

       Missing values are filled with the median before scaling.

       Parameters
       ----------
       df : pandas.DataFrame
           Input DataFrame.

       numerical_cols : list of str
           List of numerical columns to scale.

       Returns
       -------
       pandas.DataFrame
           A new DataFrame with scaled numerical columns.

       Notes
       -----
       - Handles missing values using median imputation
       - Protects against division by zero (very small std)
       - Columns with near-zero variance are set to 0.0
       - Does NOT modify the original DataFrame

       Important
       ---------
       Scaling is important for models sensitive to feature magnitude:
       - Logistic Regression
       - SVM
       - KNN

       Example
       -------
       Input:
           Age = [20, 30, 40]

       Output:
           Age ≈ [-1.22, 0, 1.22]
       """
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


def finalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
       Final cleanup of feature data types.

       Converts boolean columns into integer format (0 and 1),
       ensuring consistency across all features.

       Parameters
       ----------
       df : pandas.DataFrame
           Input DataFrame.

       Returns
       -------
       pandas.DataFrame
           A new DataFrame with boolean columns converted to integers.

       Notes
       -----
       - Ensures all features are numeric (int/float)
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
    nominal_cols: Sequence[str],
    numerical_cols: Sequence[str]
) -> pd.DataFrame:
    """
        Complete preprocessing pipeline for modeling.

        Applies all preprocessing steps in sequence:
        1. Encode target variable
        2. Encode ordinal features
        3. Encode nominal features (one-hot encoding)
        4. Scale numerical features
        5. Finalize feature data types

        Parameters
        ----------
        df : pandas.DataFrame
            Raw input DataFrame.

        target_col : str
            Name of the target column.

        ordinal_cols : list of str
            List of ordinal columns.

        nominal_cols : list of str
            List of nominal columns.

        numerical_cols : list of str
            List of numerical columns.

        Returns
        -------
        pandas.DataFrame
            Fully processed dataset ready for machine learning.

        Notes
        -----
        - Uses pandas `.pipe()` for clean functional chaining
        - Each step operates on a copy of the DataFrame
        - Output is fully numeric and model-ready

        Pipeline intuition
        ------------------
        Raw data → Clean → Encode → Scale → Ready for model

        Important
        ---------
        This function should be applied BEFORE train-test split only if:
        - transformations do not leak information

        For strict ML pipelines:
        - scaling should be fitted only on training data

        Example
        -------
        df_processed = prepare_modeling_dataset(
            df,
            target_col="Risk",
            ordinal_cols=["Saving accounts", "Checking account"],
            nominal_cols=["Sex", "Housing", "Purpose"],
            numerical_cols=["Age", "Credit amount", "Duration"]
        )
        """
    df = df.copy()

    df = (
        df
        .pipe(encode_target, target_col)
        .pipe(encode_ordinal_columns, ordinal_cols)
        .pipe(encode_nominal_columns, nominal_cols)
        .pipe(scale_numerical_columns, numerical_cols)
        .pipe(finalize_features)
    )

    return df