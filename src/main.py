import pandas as pd
from src.data.loader import load_raw_data
from src.data.cleaning import drop_useless_columns
from outputs.reports.export import save_feature_importance_table
from src.features.engineering import create_basic_features
from src.features.preprocessing import prepare_modeling_dataset
from src.features.splitting import (
    split_features_target,
    train_test_split_stratified_custom
)
from src.visualization.plots import (
    plot_target_distribution,
    plot_histogram,
    plot_categorical_counts,
    plot_numeric_by_target,
    plot_categorical_by_target,
)
from src.config import (
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    SPECIAL_COLUMNS,
    TARGET_COLUMN,
    REPORTS_DIR,
    ORDINAL_COLUMNS,
    NOMINAL_COLUMNS
)
from src.analysis.eda import (
    run_selected_eda,
    build_feature_importance_table,
    summarize_feature_importance,
    print_missing_values,
    print_shape, print_unique_values,
)


def run_analysis(df):
    run_selected_eda(
        df=df,
        numeric_cols=NUMERIC_COLUMNS,
        categorical_cols=CATEGORICAL_COLUMNS,
        special_cols=SPECIAL_COLUMNS,
        target_col=TARGET_COLUMN,
        shape=True,
        missing=True,
        target_distribution=True,
        unique_values=True,
        categorical_columns_unique_values=True,
        feature_groups=True,
        summary=True,
        numeric_by_target=False,
        categorical_by_target=False,
        mean_difference=True,
        t_test=True,
        chi_square=True,
        feature_importance=False,
    )

    importance_table = build_feature_importance_table(
        df=df,
        numeric_cols=NUMERIC_COLUMNS,
        categorical_cols=CATEGORICAL_COLUMNS,
        target_col=TARGET_COLUMN,
    )

    print("\n=== FEATURE IMPORTANCE TABLE ===")
    print(importance_table)

    summarize_feature_importance(importance_table)

    return importance_table


def run_visualizations(df):
    plot_target_distribution(df, TARGET_COLUMN)

    for column in NUMERIC_COLUMNS:
        plot_histogram(df, column)
        plot_numeric_by_target(df, column, TARGET_COLUMN)

    for column in CATEGORICAL_COLUMNS:
        plot_categorical_counts(df, column)

    for column in CATEGORICAL_COLUMNS + SPECIAL_COLUMNS:
        plot_categorical_by_target(df, column, TARGET_COLUMN)


def main():
    df = load_raw_data()
    df = drop_useless_columns(df)
    df = create_basic_features(df)

    # importance_table = run_analysis(df)
    # basic_features_df = create_basic_features(df)

    # Processed data for model
    df_prepared = prepare_modeling_dataset(df, TARGET_COLUMN, ORDINAL_COLUMNS, NOMINAL_COLUMNS, NUMERIC_COLUMNS)
    # Splitting data for model
    X, y = split_features_target(df_prepared, TARGET_COLUMN)
    pd.set_option("display.max_rows", 1000)
    pd.set_option("display.max_columns", 1000)

    # Splitting data into train and test
    X_train, X_test, y_train, y_test = train_test_split_stratified_custom(X, y)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    print(y_train.value_counts(normalize=True))
    print(y_test.value_counts(normalize=True))

    # save_feature_importance_table(importance_table, REPORTS_DIR / "feature_importance.csv")
    # run_visualizations(df)


if __name__ == "__main__":
    main()