from src.data.loader import load_raw_data
from src.data.cleaning import drop_useless_columns
from src.visualization.plots import (
    plot_target_distribution,
    plot_histogram,
    plot_categorical_counts,
    plot_numeric_by_target,
    plot_categorical_by_target
)
from src.config import (
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    SPECIAL_COLUMNS,
    TARGET_COLUMN
)
from src.analysis.eda import (
    print_shape,
    print_missing_values,
    print_target_distribution,
    print_unique_values,
    print_summary,
    print_feature_groups,
    print_numeric_by_target,
    print_categorical_by_target,
    print_mean_difference,
    t_test_numeric,
    chi_square_test
)

def main():
    df = load_raw_data()
    df = drop_useless_columns(df)

    print_shape(df)
    print_missing_values(df)
    print_target_distribution(df)
    print_unique_values(df)
    print_feature_groups(NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, SPECIAL_COLUMNS)
    print_summary(df, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, SPECIAL_COLUMNS)

    print_numeric_by_target(df, NUMERIC_COLUMNS, TARGET_COLUMN)
    print_categorical_by_target(df, CATEGORICAL_COLUMNS, TARGET_COLUMN)

    print_mean_difference(df, NUMERIC_COLUMNS, TARGET_COLUMN)
    t_test_numeric(df, NUMERIC_COLUMNS, TARGET_COLUMN)
    chi_square_test(df, CATEGORICAL_COLUMNS, TARGET_COLUMN)

    plot_target_distribution(df)
    for column in NUMERIC_COLUMNS:
        plot_histogram(df, column)
    for column in CATEGORICAL_COLUMNS:
        plot_categorical_counts(df, column)
    for column in NUMERIC_COLUMNS:
        plot_numeric_by_target(df, column, TARGET_COLUMN)
    for column in CATEGORICAL_COLUMNS + SPECIAL_COLUMNS:
        plot_categorical_by_target(df, column, TARGET_COLUMN)

if __name__ == "__main__":
    main()
