import numpy as np
import pandas as pd


def split_features_target(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """
        Split a DataFrame into features (X) and target (y).

        This function separates the input DataFrame into:
        - X: all columns except the target column
        - y: the target column

        A copy of the DataFrame is created to avoid modifying the original data.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataset containing both features and target.

        target_col : str
            Name of the target column.

        Returns
        -------
        X : pandas.DataFrame
            DataFrame containing feature columns.

        y : pandas.Series
            Series containing the target values.

        Raises
        ------
        ValueError
            If the target column is not present in the DataFrame.

        Notes
        -----
        - The original DataFrame is not modified.
        - The function assumes a supervised learning setting.
        """
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    return X, y


def train_test_split_stratified_custom(X: pd.DataFrame, y: pd.Series, test_size: float=0.2, random_state: int= 42)\
        -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split features and target into stratified train and test sets.

    This function performs a stratified split, ensuring that the proportion
    of each class in the target variable (y) is preserved in both the training
    and testing sets.

    The split is performed independently for each class:
    - indices for each class are shuffled
    - a portion is assigned to the test set
    - the rest is assigned to the training set
    - final train and test indices are shuffled again

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.

    y : pandas.Series
        Target vector containing class labels.

    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
        Must be between 0 and 1.

    random_state : int, default=42
        Seed for the random number generator to ensure reproducibility.

    Returns
    -------
    X_train : pandas.DataFrame
        Training subset of features.

    X_test : pandas.DataFrame
        Testing subset of features.

    y_train : pandas.Series
        Training subset of target values.

    y_test : pandas.Series
        Testing subset of target values.

    Raises
    ------
    ValueError
        If X and y have different lengths.
        If test_size is not between 0 and 1.

    Notes
    -----
    - Stratification ensures class distribution is preserved across splits.
    - The function operates on indices to maintain correct (X, y) pairing.
    - Two levels of shuffling are used:
        1. Within each class (to randomize selection)
        2. Globally (to remove class ordering in final datasets)
    - Small class edge cases are handled to ensure both train and test
      contain at least one sample when possible.
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    rng = np.random.default_rng(random_state)

    train_indices = []
    test_indices = []

    classes = y.unique()

    for cls in classes:
        cls_indices = y[y == cls].index.to_numpy()
        rng.shuffle(cls_indices)

        n_test = int(len(cls_indices) * test_size)

        if n_test == 0 and len(cls_indices) > 1:
            n_test = 1
        if n_test == len(cls_indices):
            n_test = len(cls_indices) - 1

        cls_test_indices = cls_indices[:n_test]
        cls_train_indices = cls_indices[n_test:]

        test_indices.extend(cls_test_indices)
        train_indices.extend(cls_train_indices)

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    X_train = X.loc[train_indices].copy()
    X_test = X.loc[test_indices].copy()
    y_train = y.loc[train_indices].copy()
    y_test = y.loc[test_indices].copy()

    return X_train, X_test, y_train, y_test