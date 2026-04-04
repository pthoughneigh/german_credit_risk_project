from src.data.loader import load_raw_data
from src.data.cleaning import drop_useless_columns
from src.visualization.plots import (
    plot_target_distribution,
    plot_histogram,
    plot_categorical_counts
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
    print_numeric_summary,
    print_feature_groups
)

def main():
    df = load_raw_data()
    df = drop_useless_columns(df)

    print_shape(df)
    print_missing_values(df)
    print_target_distribution(df)
    print_unique_values(df)
    print_feature_groups(NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, SPECIAL_COLUMNS)
    print_numeric_summary(df, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, SPECIAL_COLUMNS)

    plot_target_distribution(df)
    for column in NUMERIC_COLUMNS:
        plot_histogram(df, column)
    for column in CATEGORICAL_COLUMNS:
        plot_categorical_counts(df, column)

if __name__ == "__main__":
    main()
