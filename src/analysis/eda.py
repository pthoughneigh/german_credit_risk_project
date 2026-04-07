# Exploratory Data Analysis #
import pandas as pd
from numpy.ma.core import ceil
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
from unicodedata import normalize


def print_shape(df) -> None:
    print("\n=== SHAPE ===")
    rows, columns = df.shape
    print(f"rows: {rows}, columns: {columns}")

def print_missing_values(df) -> None:
    print("\n=== MISSING VALUES ===")
    print(df.isna().sum())

def print_target_distribution(df, target_col="Risk") -> None:
    print(f"\n=== TARGET DISTRIBUTION: {target_col} ===")
    counts = df[target_col].value_counts()
    proportions = df[target_col].value_counts(normalize=True)

    print("\nCounts:")
    print(counts)
    print("\nProportions:")
    print(proportions)

def print_unique_values(df):
    print("\n=== UNIQUE VALUES ===")
    for col in df.columns:
        num_unique = df[col].nunique()
        unique_values = df[col].dropna().unique()
        print(f"Column: {col}")
        print(f"Number of unique values: {num_unique}")
        print(f"Values: {'Too many unique values to display.' if num_unique > 10 
                else unique_values }\n")

def print_feature_groups(numeric_cols, categorical_cols, special_cols):
    print("\n=== NUMERIC FEATURES ===")
    print(numeric_cols)

    print("\n=== CATEGORICAL FEATURES ===")
    print(categorical_cols)

    print("\n=== SPECIAL / CODED FEATURES ===")
    print(special_cols)

def print_summary(df, numeric_cols, categorical_cols, special_cols):
    print("\n=== NUMERIC COLUMNS SUMMARY ===")
    print(df[numeric_cols].describe())
    print("\n=== CATEGORICAL COLUMNS SUMMARY ===")
    print(df[categorical_cols].describe())
    print("\n=== SPECIAL COLUMNS SUMMARY ===")
    print(df[special_cols].describe())

def print_numeric_by_target(df, numeric_cols, target_col):
    print(f"\n=== NUMERIC FEATURES BY {target_col} ===")

    for col in numeric_cols:
        print(f"\n--- {col} ---")

        for label in df[target_col].unique():
            subset = df[df[target_col] == label][col]

            print(f"\n{label.upper()}:")
            print(subset.describe())

def print_categorical_by_target(df, categorical_cols, target_col):
    for col in categorical_cols:
        print(f"\n--- {col} vs {target_col} ---")

        print("COUNTS: ")
        print(pd.crosstab(df[col], df[target_col], dropna=False))

        print("PROPORTIONS: ")
        print(pd.crosstab(df[col], df[target_col], normalize='index', dropna=False))

def print_mean_difference(df, numeric_cols, target_col):
    print(f"\n=== Mean difference by {target_col} ===")
    good_df = df[df[target_col] == 'good']
    bad_df = df[df[target_col] == 'bad']

    for col in numeric_cols:
        good_mean = good_df[col].mean()
        bad_mean = bad_df[col].mean()
        diff = bad_mean - good_mean

        print(f"--{col.upper()}--:")
        print(f"good mean: {good_mean:.2f}")
        print(f"bad mean: {bad_mean:.2f}")
        print(f"bad - good: {diff:.2f}")

def t_test_numeric(df, numeric_cols, target_col):
    print("\n======= NUMERICAL COLUMNS T-TEST ======== ")

    for col in numeric_cols:
        good_values = df[df[target_col] == 'good'][col].dropna()
        bad_values = df[df[target_col] == 'bad'][col].dropna()
        t_statistic, p_value = ttest_ind(bad_values, good_values, equal_var=False)

        print(f"--{col.upper()}--:")
        print(f"t-statistic: {t_statistic:.2f}")
        print(f"p-value: {p_value:.5f}")

        if p_value < 0.05:
            print("→ Statistically significant difference")
        else:
            print("→ No significant difference")

def chi_square_test(df, categorical_cols, target_col):
    print("\n======= CATEGORICAL COLUMNS CHI_SQUARE-TEST ======== ")
    for col in categorical_cols:
        print(f"--{col.upper()}--:")

        table = pd.crosstab(df[col], df[target_col], dropna=False)

        chi2, p, dof, expected = chi2_contingency(table)

        print(f"chi2: {chi2:.2f}")
        print(f"p-value: {p:.5f}")
        print(f"dof: {dof}") # degrees of freedom
        # print(f"expected: \n{expected}")

        if p < 0.05:
            print("→ Statistically significant association")
        else:
            print("→ No significant association")

def summarize_feature_importance(df, numeric_cols, categorical_cols, target_col):
    print("\n=== FEATURE IMPORTANCE SUMMARY ===")
    print("NUMERIC FEATURES:")
    good_df = df[df[target_col] == "good"]
    bad_df = df[df[target_col] == "bad"]
    num_cols_t_statistics = []
    cat_cols_chi2 = []
    for col in numeric_cols:
        good_values = good_df[col].dropna()
        bad_values = bad_df[col].dropna()
        t_statistic, p_value = ttest_ind(bad_values, good_values, equal_var=False)
        num_cols_t_statistics.append((col, t_statistic, p_value))
    sorted_num_cols_t_statistics = sorted(num_cols_t_statistics, key=lambda x: abs(x[1]), reverse=True)

    top_k = max(1, round(len(sorted_num_cols_t_statistics) * 0.2))
    mid_k = max(top_k + 1, round(len(sorted_num_cols_t_statistics) * 0.5)) # relative importance within this dataset
    for i, (col, t_stat, p_val) in enumerate(sorted_num_cols_t_statistics):
        if i < top_k and p_val < 0.05:
            strength = "strong"
        elif i < mid_k and p_val < 0.05:
            strength = "medium"
        else:
            strength = "weak"

        col = col.replace("_", " ").title()
        print(f"{col} -> {strength} (t={t_stat:.2f}, p={p_val:.4g})")

    print("\nCATEGORICAL FEATURES:")
    for col in categorical_cols:
        table = pd.crosstab(df[col], df[target_col], dropna=False)
        chi2, p_val, dof, expected = chi2_contingency(table)
        cat_cols_chi2.append((col, chi2, p_val))
    sorted_cat_cols_chi2 = sorted(cat_cols_chi2, key=lambda x: x[1], reverse=True)

    top_k = max(1, round(len(sorted_cat_cols_chi2) * 0.2))
    mid_k = max(top_k + 1, round(len(sorted_cat_cols_chi2) * 0.5)) # relative importance within this dataset
    for i, (col, chi2, p_val) in enumerate(sorted_cat_cols_chi2):
        if i < top_k and p_val < 0.05:
            strength = "strong"
        elif i < mid_k and p_val < 0.05:
            strength = "medium"
        else:
            strength = "weak"

        col = col.replace("_", " ").title()
        print(f"{col} -> {strength} (chi2={chi2:.2f}, p={p_val:.4g})")