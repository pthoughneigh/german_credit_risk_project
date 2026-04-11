from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind


# =========================
# BASIC OVERVIEW
# =========================
def print_shape(df: pd.DataFrame) -> None:
    """
    Print the number of rows and columns in the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    None
        This function prints the shape of the DataFrame.
    """
    print("\n=== SHAPE ===")
    rows, columns = df.shape
    print(f"rows: {rows}, columns: {columns}")


def print_categorical_columns_unique_values(
    df: pd.DataFrame,
    categorical_columns: Sequence[str]
) -> None:
    """
    Print unique non-null values for each categorical column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    categorical_columns : Sequence[str]
        List or sequence of categorical column names.

    Returns
    -------
    None
        This function prints unique values for each categorical column.
    """
    print("\n=== CATEGORICAL COLUMNS UNIQUE VALUES ===")
    for col in categorical_columns:
        print(f"\nCategorical Column: {col}")
        print(df[col].dropna().unique())


def print_missing_values(df: pd.DataFrame) -> None:
    """
    Print the number of missing values for each column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    None
        This function prints the number of missing values per column.
    """
    print("\n=== MISSING VALUES ===")
    print(df.isna().sum())


def print_target_distribution(
    df: pd.DataFrame,
    target_col: str = "Risk"
) -> None:
    """
    Print target value counts and proportions.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    target_col : str, default="Risk"
        Name of the target column.

    Returns
    -------
    None
        This function prints target counts and normalized proportions.
    """
    print(f"\n=== TARGET DISTRIBUTION: {target_col} ===")

    counts = df[target_col].value_counts(dropna=False)
    proportions = df[target_col].value_counts(dropna=False, normalize=True)

    print("\nCounts:")
    print(counts)

    print("\nProportions:")
    print(proportions)


def print_unique_values(
    df: pd.DataFrame,
    max_display: int = 10
) -> None:
    """
    Print the number of unique values for each column.

    If a column has more than `max_display` unique values, only a message
    is shown instead of printing all values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    max_display : int, default=10
        Maximum number of unique values to display for a column.

    Returns
    -------
    None
        This function prints the number of unique values for each column.
    """
    print("\n=== UNIQUE VALUES ===")

    for col in df.columns:
        unique_values = df[col].dropna().unique()
        num_unique = len(unique_values)

        if num_unique > max_display:
            values_to_print = "Too many unique values to display."
        else:
            values_to_print = unique_values

        print(f"Column: {col}")
        print(f"Number of unique values: {num_unique}")
        print(f"Values: {values_to_print}\n")


def print_feature_groups(
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    special_cols: Sequence[str]
) -> None:
    """
    Print grouped feature lists.

    Parameters
    ----------
    numeric_cols : Sequence[str]
        Names of numeric feature columns.

    categorical_cols : Sequence[str]
        Names of categorical feature columns.

    special_cols : Sequence[str]
        Names of special or coded feature columns.

    Returns
    -------
    None
        This function prints feature groups.
    """
    print("\n=== NUMERIC FEATURES ===")
    print(numeric_cols)

    print("\n=== CATEGORICAL FEATURES ===")
    print(categorical_cols)

    print("\n=== SPECIAL / CODED FEATURES ===")
    print(special_cols)


def print_summary(
    df: pd.DataFrame,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    special_cols: Sequence[str]
) -> None:
    """
    Print descriptive statistics for numeric, categorical, and special columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    numeric_cols : Sequence[str]
        Names of numeric columns.

    categorical_cols : Sequence[str]
        Names of categorical columns.

    special_cols : Sequence[str]
        Names of special or coded columns.

    Returns
    -------
    None
        This function prints summary statistics for the provided column groups.
    """
    print("\n=== NUMERIC COLUMNS SUMMARY ===")
    if numeric_cols:
        print(df[list(numeric_cols)].describe())
    else:
        print("No numeric columns provided.")

    print("\n=== CATEGORICAL COLUMNS SUMMARY ===")
    if categorical_cols:
        print(df[list(categorical_cols)].describe())
    else:
        print("No categorical columns provided.")

    print("\n=== SPECIAL COLUMNS SUMMARY ===")
    if special_cols:
        print(df[list(special_cols)].describe())
    else:
        print("No special columns provided.")


# =========================
# INTERNAL HELPERS
# =========================
def _split_target_groups(
    df: pd.DataFrame,
    target_col: str,
    positive_label: str = "bad",
    negative_label: str = "good"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into two target-based groups.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    target_col : str
        Name of the target column.

    positive_label : str, default="bad"
        Label representing the positive class.

    negative_label : str, default="good"
        Label representing the negative class.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        A tuple containing:
        - good_df : rows where target equals `negative_label`
        - bad_df : rows where target equals `positive_label`
    """
    good_df = df[df[target_col] == negative_label]
    bad_df = df[df[target_col] == positive_label]
    return good_df, bad_df


def _compute_numeric_test_results(
    df: pd.DataFrame,
    numeric_cols: Sequence[str],
    target_col: str
) -> List[Dict[str, Any]]:
    """
    Compute Welch's t-test results for numeric columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    numeric_cols : Sequence[str]
        Names of numeric columns to test.

    target_col : str
        Name of the target column.

    Returns
    -------
    list[dict[str, Any]]
        A list of dictionaries containing summary statistics and t-test results
        for each numeric feature.
    """
    good_df, bad_df = _split_target_groups(df, target_col)

    results: List[Dict[str, Any]] = []
    for col in numeric_cols:
        good_values = good_df[col].dropna()
        bad_values = bad_df[col].dropna()

        good_mean = good_values.mean()
        bad_mean = bad_values.mean()
        mean_diff = bad_mean - good_mean

        t_statistic, p_value = ttest_ind(
            bad_values,
            good_values,
            equal_var=False
        )

        results.append({
            "feature": col,
            "feature_type": "numeric",
            "test": "t-test",
            "good_mean": good_mean,
            "bad_mean": bad_mean,
            "mean_difference": mean_diff,
            "statistic": t_statistic,
            "abs_statistic": abs(t_statistic),
            "p_value": p_value
        })

    return results


def _compute_categorical_test_results(
    df: pd.DataFrame,
    categorical_cols: Sequence[str],
    target_col: str
) -> List[Dict[str, Any]]:
    """
    Compute chi-square test results for categorical columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    categorical_cols : Sequence[str]
        Names of categorical columns to test.

    target_col : str
        Name of the target column.

    Returns
    -------
    list[dict[str, Any]]
        A list of dictionaries containing chi-square test results
        for each categorical feature.
    """
    results: List[Dict[str, Any]] = []

    for col in categorical_cols:
        contingency_table = pd.crosstab(df[col], df[target_col], dropna=False)
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        results.append({
            "feature": col,
            "feature_type": "categorical",
            "test": "chi-square",
            "good_mean": None,
            "bad_mean": None,
            "mean_difference": None,
            "statistic": chi2,
            "abs_statistic": chi2,
            "p_value": p_value
        })

    return results


def _assign_importance_strength(
    results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Assign relative importance labels based on rank and statistical significance.

    Features in the top 20% with p < 0.05 are labeled 'strong'.
    Features in the next range up to 50% with p < 0.05 are labeled 'medium'.
    All others are labeled 'weak'.

    Parameters
    ----------
    results : list[dict[str, Any]]
        List of result dictionaries already sorted by importance.

    Returns
    -------
    list[dict[str, Any]]
        Updated list of result dictionaries with an added
        'importance_strength' key.
    """
    if not results:
        return results

    results = results.copy()

    top_k = max(1, round(len(results) * 0.2))
    mid_k = max(top_k + 1, round(len(results) * 0.5))

    for i, row in enumerate(results):
        if i < top_k and row["p_value"] < 0.05:
            row["importance_strength"] = "strong"
        elif i < mid_k and row["p_value"] < 0.05:
            row["importance_strength"] = "medium"
        else:
            row["importance_strength"] = "weak"

    return results


# =========================
# BY TARGET ANALYSIS
# =========================
def print_numeric_by_target(
    df: pd.DataFrame,
    numeric_cols: Sequence[str],
    target_col: str
) -> None:
    """
    Print descriptive statistics of numeric columns separately for each target class.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    numeric_cols : Sequence[str]
        Names of numeric columns.

    target_col : str
        Name of the target column.

    Returns
    -------
    None
        This function prints descriptive statistics by target class.
    """
    print(f"\n=== NUMERIC FEATURES BY {target_col} ===")

    labels = df[target_col].dropna().unique()

    for col in numeric_cols:
        print(f"\n--- {col} ---")

        for label in labels:
            subset = df[df[target_col] == label][col].dropna()

            print(f"\n{str(label).upper()}:")
            print(subset.describe())


def print_categorical_by_target(
    df: pd.DataFrame,
    categorical_cols: Sequence[str],
    target_col: str
) -> None:
    """
    Print count and proportion tables for categorical columns against the target.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    categorical_cols : Sequence[str]
        Names of categorical columns.

    target_col : str
        Name of the target column.

    Returns
    -------
    None
        This function prints count and proportion tables for each categorical
        feature versus the target.
    """
    print(f"\n=== CATEGORICAL FEATURES BY {target_col} ===")

    for col in categorical_cols:
        print(f"\n--- {col} vs {target_col} ---")

        counts_table = pd.crosstab(df[col], df[target_col], dropna=False)
        proportions_table = pd.crosstab(
            df[col],
            df[target_col],
            normalize="index",
            dropna=False
        )

        print("COUNTS:")
        print(counts_table)

        print("PROPORTIONS:")
        print(proportions_table)


def print_mean_difference(
    df: pd.DataFrame,
    numeric_cols: Sequence[str],
    target_col: str
) -> None:
    """
    Print class means and mean differences for numeric columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    numeric_cols : Sequence[str]
        Names of numeric columns.

    target_col : str
        Name of the target column.

    Returns
    -------
    None
        This function prints mean values for the good and bad classes and
        their difference.
    """
    print(f"\n=== MEAN DIFFERENCE BY {target_col} ===")

    numeric_results = _compute_numeric_test_results(df, numeric_cols, target_col)

    for row in numeric_results:
        print(f"\n-- {row['feature'].upper()} --")
        print(f"good mean: {row['good_mean']:.2f}")
        print(f"bad mean: {row['bad_mean']:.2f}")
        print(f"bad - good: {row['mean_difference']:.2f}")


# =========================
# STATISTICAL TESTS
# =========================
def t_test_numeric(
    df: pd.DataFrame,
    numeric_cols: Sequence[str],
    target_col: str
) -> None:
    """
    Perform Welch's t-test for numeric columns between target groups.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    numeric_cols : Sequence[str]
        Names of numeric columns to test.

    target_col : str
        Name of the target column.

    Returns
    -------
    None
        This function prints Welch's t-test results for each numeric feature.
    """
    print("\n=== NUMERICAL COLUMNS T-TEST ===")

    numeric_results = _compute_numeric_test_results(df, numeric_cols, target_col)

    for row in numeric_results:
        print(f"\n-- {row['feature'].upper()} --")
        print(f"t-statistic: {row['statistic']:.2f}")
        print(f"p-value: {row['p_value']:.5f}")

        if row["p_value"] < 0.05:
            print("→ Statistically significant difference")
        else:
            print("→ No significant difference")


def chi_square_test(
    df: pd.DataFrame,
    categorical_cols: Sequence[str],
    target_col: str
) -> None:
    """
    Perform chi-square tests for categorical columns against the target.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    categorical_cols : Sequence[str]
        Names of categorical columns to test.

    target_col : str
        Name of the target column.

    Returns
    -------
    None
        This function prints chi-square test results for each categorical feature.
    """
    print("\n=== CATEGORICAL COLUMNS CHI-SQUARE TEST ===")

    categorical_results = _compute_categorical_test_results(df, categorical_cols, target_col)

    for row in categorical_results:
        print(f"\n-- {row['feature'].upper()} --")
        print(f"chi2: {row['statistic']:.2f}")
        print(f"p-value: {row['p_value']:.5f}")

        if row["p_value"] < 0.05:
            print("→ Statistically significant association")
        else:
            print("→ No significant association")


# =========================
# FEATURE IMPORTANCE
# =========================
def build_feature_importance_table(
    df: pd.DataFrame,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    target_col: str
) -> pd.DataFrame:
    """
    Build a feature importance summary table using statistical tests.

    Numeric features are evaluated with Welch's t-test.
    Categorical features are evaluated with the chi-square test.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    numeric_cols : Sequence[str]
        Names of numeric feature columns.

    categorical_cols : Sequence[str]
        Names of categorical feature columns.

    target_col : str
        Name of the target column.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing feature name, feature type, test name,
        summary statistics, p-value, and assigned importance strength.
    """
    numeric_results = _compute_numeric_test_results(df, numeric_cols, target_col)
    categorical_results = _compute_categorical_test_results(df, categorical_cols, target_col)

    numeric_results = sorted(
        numeric_results,
        key=lambda row: row["abs_statistic"],
        reverse=True
    )
    numeric_results = _assign_importance_strength(numeric_results)

    categorical_results = sorted(
        categorical_results,
        key=lambda row: row["abs_statistic"],
        reverse=True
    )
    categorical_results = _assign_importance_strength(categorical_results)

    all_results = numeric_results + categorical_results
    feature_importance_table = pd.DataFrame(all_results)

    strength_order = {"strong": 0, "medium": 1, "weak": 2}
    type_order = {"numeric": 0, "categorical": 1}

    feature_importance_table["strength_rank"] = (
        feature_importance_table["importance_strength"].map(strength_order)
    )
    feature_importance_table["type_rank"] = (
        feature_importance_table["feature_type"].map(type_order)
    )

    feature_importance_table = (
        feature_importance_table
        .sort_values(
            by=["type_rank", "strength_rank", "abs_statistic"],
            ascending=[True, True, False]
        )
        .drop(columns=["strength_rank", "type_rank"])
        .reset_index(drop=True)
    )

    return feature_importance_table


def summarize_feature_importance(table: pd.DataFrame) -> None:
    """
    Print a human-readable summary of feature importance results.

    Parameters
    ----------
    table : pandas.DataFrame
        DataFrame returned by `build_feature_importance_table()`.

    Returns
    -------
    None
        This function prints a readable summary of numeric and categorical
        feature importance results.
    """
    print("\n=== FEATURE IMPORTANCE SUMMARY ===")

    print("\nNUMERIC FEATURES:")
    numeric_table = table[table["feature_type"] == "numeric"]

    for _, row in numeric_table.iterrows():
        name = row["feature"].replace("_", " ").title()
        print(
            f"{name} -> {row['importance_strength']} "
            f"(t={row['statistic']:.2f}, p={row['p_value']:.4g})"
        )

    print("\nCATEGORICAL FEATURES:")
    categorical_table = table[table["feature_type"] == "categorical"]

    for _, row in categorical_table.iterrows():
        name = row["feature"].replace("_", " ").title()
        print(
            f"{name} -> {row['importance_strength']} "
            f"(chi2={row['statistic']:.2f}, p={row['p_value']:.4g})"
        )


# =========================
# FULL EDA RUNNERS
# =========================
def run_full_eda(
    df: pd.DataFrame,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    special_cols: Sequence[str],
    target_col: str = "Risk"
) -> None:
    """
    Run the full EDA pipeline and print all outputs.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    numeric_cols : Sequence[str]
        Names of numeric feature columns.

    categorical_cols : Sequence[str]
        Names of categorical feature columns.

    special_cols : Sequence[str]
        Names of special or coded columns.

    target_col : str, default="Risk"
        Name of the target column.

    Returns
    -------
    None
        This function runs all EDA sections and prints the results.
    """
    print_shape(df)
    print_missing_values(df)
    print_target_distribution(df, target_col)
    print_unique_values(df)
    print_categorical_columns_unique_values(df, categorical_cols)
    print_feature_groups(numeric_cols, categorical_cols, special_cols)
    print_summary(df, numeric_cols, categorical_cols, special_cols)
    print_numeric_by_target(df, numeric_cols, target_col)
    print_categorical_by_target(df, categorical_cols, target_col)
    print_mean_difference(df, numeric_cols, target_col)
    t_test_numeric(df, numeric_cols, target_col)
    chi_square_test(df, categorical_cols, target_col)

    importance_table = build_feature_importance_table(
        df=df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        target_col=target_col
    )
    summarize_feature_importance(importance_table)


def run_selected_eda(
    df: pd.DataFrame,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    special_cols: Sequence[str],
    target_col: str = "Risk",
    shape: bool = True,
    missing: bool = True,
    target_distribution: bool = True,
    unique_values: bool = True,
    categorical_columns_unique_values: bool = False,
    feature_groups: bool = True,
    summary: bool = True,
    numeric_by_target: bool = False,
    categorical_by_target: bool = False,
    mean_difference: bool = True,
    t_test: bool = True,
    chi_square: bool = True,
    feature_importance: bool = True
) -> None:
    """
    Run only selected EDA sections.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    numeric_cols : Sequence[str]
        Names of numeric feature columns.

    categorical_cols : Sequence[str]
        Names of categorical feature columns.

    special_cols : Sequence[str]
        Names of special or coded columns.

    target_col : str, default="Risk"
        Name of the target column.

    shape : bool, default=True
        Whether to print the dataset shape.

    missing : bool, default=True
        Whether to print missing value counts.

    target_distribution : bool, default=True
        Whether to print target distribution.

    unique_values : bool, default=True
        Whether to print unique values by column.

    categorical_columns_unique_values : bool, default=False
        Whether to print unique values for categorical columns only.

    feature_groups : bool, default=True
        Whether to print grouped feature lists.

    summary : bool, default=True
        Whether to print summary statistics.

    numeric_by_target : bool, default=False
        Whether to print numeric summaries by target class.

    categorical_by_target : bool, default=False
        Whether to print categorical summaries by target class.

    mean_difference : bool, default=True
        Whether to print mean differences for numeric columns.

    t_test : bool, default=True
        Whether to run Welch's t-tests for numeric columns.

    chi_square : bool, default=True
        Whether to run chi-square tests for categorical columns.

    feature_importance : bool, default=True
        Whether to build and summarize the feature importance table.

    Returns
    -------
    None
        This function runs the selected EDA sections and prints the results.
    """
    if shape:
        print_shape(df)

    if missing:
        print_missing_values(df)

    if target_distribution:
        print_target_distribution(df, target_col)

    if unique_values:
        print_unique_values(df)

    if categorical_columns_unique_values:
        print_categorical_columns_unique_values(df, categorical_cols)

    if feature_groups:
        print_feature_groups(numeric_cols, categorical_cols, special_cols)

    if summary:
        print_summary(df, numeric_cols, categorical_cols, special_cols)

    if numeric_by_target:
        print_numeric_by_target(df, numeric_cols, target_col)

    if categorical_by_target:
        print_categorical_by_target(df, categorical_cols, target_col)

    if mean_difference:
        print_mean_difference(df, numeric_cols, target_col)

    if t_test:
        t_test_numeric(df, numeric_cols, target_col)

    if chi_square:
        chi_square_test(df, categorical_cols, target_col)

    if feature_importance:
        importance_table = build_feature_importance_table(
            df=df,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            target_col=target_col
        )
        summarize_feature_importance(importance_table)