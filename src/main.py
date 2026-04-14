import pandas as pd
import numpy as np

from src.data.loader import load_raw_data
from src.data.cleaning import drop_useless_columns
from src.evaluation.roc_auc import compute_roc_points, compute_auc
from src.models.logistic_regression import (
    train_logistic_regression,
    predict, predict_proba,
)
from src.evaluation.metrics import (
    compute_accuracy,
    compute_confusion_matrix,
    compute_precision,
    compute_recall, compute_f1_score,
)
from outputs.reports.export import save_feature_importance_table
from src.features.engineering import create_basic_features
from src.features.preprocessing import (
    prepare_modeling_dataset,
    fit_numerical_scaler,
    transform_numerical_columns,
)
from src.features.splitting import (
    split_features_target,
    train_test_split_stratified_custom,
)
from src.visualization.plots import (
    plot_target_distribution,
    plot_histogram,
    plot_categorical_counts,
    plot_numeric_by_target,
    plot_categorical_by_target,
    plot_roc_curve,
)
from src.config import (
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    SPECIAL_COLUMNS,
    TARGET_COLUMN,
    REPORTS_DIR,
    ORDINAL_COLUMNS,
    NOMINAL_COLUMNS,
)
from src.analysis.eda import (
    run_selected_eda,
    build_feature_importance_table,
    summarize_feature_importance,
)


def run_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run exploratory data analysis and return a feature-importance summary table.
    """
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


def run_visualizations(df: pd.DataFrame) -> None:
    """
    Generate project visualizations for the raw dataset.
    """
    plot_target_distribution(df, TARGET_COLUMN)
    for column in NUMERIC_COLUMNS:
        plot_histogram(df, column)
        plot_numeric_by_target(df, column, TARGET_COLUMN)
    for column in CATEGORICAL_COLUMNS:
        plot_categorical_counts(df, column)
    for column in CATEGORICAL_COLUMNS + SPECIAL_COLUMNS:
        plot_categorical_by_target(df, column, TARGET_COLUMN)


def evaluate_model(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    threshold: float,
    dataset_name: str,
) -> None:
    """
    Evaluate logistic regression predictions on a dataset
    for a given classification threshold.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        True binary labels.
    weights : np.ndarray
        Learned model weights.
    bias : float
        Learned bias term.
    threshold : float
        Probability threshold for class prediction.
    dataset_name : str
        Name of the evaluated dataset (e.g. "train" or "test").
    """
    y_pred = predict(X, weights, bias, threshold)

    accuracy = compute_accuracy(y, y_pred)
    confusion = compute_confusion_matrix(y, y_pred)
    precision = compute_precision(y, y_pred)
    recall = compute_recall(y, y_pred)
    f1 = compute_f1_score(y, y_pred)

    print(f"\n=== {dataset_name.upper()} METRICS | threshold={threshold} ===")
    print("Predicted values:")
    print(y_pred[:10])
    print(f"Shape: {y_pred.shape}")

    print("\nAccuracy:")
    print(accuracy)

    print("\nConfusion matrix:")
    print(confusion)

    print("\nPrecision and recall:")
    print(f"Precision: {precision}")
    print(f"Recall:    {recall}")
    print(f"F1 score: {f1}")


def main() -> None:
    """
    Run the full workflow:
    data loading, preprocessing, splitting, scaling, training, and evaluation.
    """
    # Load and prepare raw data
    df = load_raw_data()
    df = drop_useless_columns(df)
    df = create_basic_features(df)

    # Optional EDA
    # importance_table = run_analysis(df)
    # save_feature_importance_table(importance_table, REPORTS_DIR / "feature_importance.csv")

    # Safe preprocessing (without numerical scaling)
    df_prepared = prepare_modeling_dataset(
        df=df,
        target_col=TARGET_COLUMN,
        ordinal_cols=ORDINAL_COLUMNS,
        nominal_cols=NOMINAL_COLUMNS,
    )

    # Split features and target
    X, y = split_features_target(df_prepared, TARGET_COLUMN)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split_stratified_custom(X, y)

    print("\nX_train sample:")
    print(X_train.head(10))

    # Fit scaler on training data only
    medians, means, stds = fit_numerical_scaler(X_train, NUMERIC_COLUMNS)

    # Apply scaling to both train and test using training statistics
    X_train = transform_numerical_columns(X_train, NUMERIC_COLUMNS, medians, means, stds)
    X_test = transform_numerical_columns(X_test, NUMERIC_COLUMNS, medians, means, stds)

    print("\nScaled X_train sample:")
    print(X_train.head(10))

    # Train logistic regression
    weights, bias, losses = train_logistic_regression(
        X_train.to_numpy(),
        y_train.to_numpy(),
        learning_rate=0.01,
        n_iterations=1000,
    )

    print("\nFirst 5 losses:")
    print(losses[:5])

    print("\nLast 5 losses:")
    print(losses[-5:])

    print("\nWeights:")
    print(weights)
    print(weights.shape)

    print("\nBias:")
    print(bias)
    print(type(bias))

    # Evaluate on train
    evaluate_model(
        X=X_train.to_numpy(),
        y=y_train.to_numpy(),
        weights=weights,
        bias=bias,
        threshold=0.5,
        dataset_name="train",
    )

    evaluate_model(
        X=X_train.to_numpy(),
        y=y_train.to_numpy(),
        weights=weights,
        bias=bias,
        threshold=0.3,
        dataset_name="train",
    )

    # Evaluate on test
    evaluate_model(
        X=X_test.to_numpy(),
        y=y_test.to_numpy(),
        weights=weights,
        bias=bias,
        threshold=0.5,
        dataset_name="test",
    )

    evaluate_model(
        X=X_test.to_numpy(),
        y=y_test.to_numpy(),
        weights=weights,
        bias=bias,
        threshold=0.3,
        dataset_name="test",
    )

    # ROC plot
    # y_scores_test = predict_proba(X_test.to_numpy(), weights, bias)

    # fpr, tpr = compute_roc_points(y_test.to_numpy(), y_scores_test)
    # auc = compute_auc(y_test.to_numpy(), y_scores_test)

    # plot_roc_curve(fpr, tpr, auc)

    # Optional visualizations
    # run_visualizations(df)


if __name__ == "__main__":
    main()