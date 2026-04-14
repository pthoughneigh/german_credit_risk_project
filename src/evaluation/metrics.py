import numpy as np


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True binary labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted binary labels.

    Returns
    -------
    float
        Proportion of correctly classified samples.
    """
    return float((y_true == y_pred).mean())


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute the confusion matrix for binary classification.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True binary labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted binary labels.

    Returns
    -------
    np.ndarray of shape (2, 2)
        Confusion matrix in the form:
        [[TN, FP],
         [FN, TP]]
    """
    TP = ((y_true == 1) & (y_pred == 1)).sum()
    TN = ((y_true == 0) & (y_pred == 0)).sum()
    FP = ((y_true == 0) & (y_pred == 1)).sum()
    FN = ((y_true == 1) & (y_pred == 0)).sum()

    return np.array([
        [TN, FP],
        [FN, TP]
    ])


def compute_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute precision for binary classification.

    Precision answers:
    Of all samples predicted as positive, how many were truly positive?

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True binary labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted binary labels.

    Returns
    -------
    float
        Precision score.

    Notes
    -----
    A small epsilon is added to the denominator to avoid division by zero.
    """
    TP = ((y_true == 1) & (y_pred == 1)).sum()
    FP = ((y_true == 0) & (y_pred == 1)).sum()

    return float(TP / (TP + FP + 1e-15))


def compute_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute recall for binary classification.

    Recall answers:
    Of all truly positive samples, how many did the model correctly identify?

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True binary labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted binary labels.

    Returns
    -------
    float
        Recall score.

    Notes
    -----
    A small epsilon is added to the denominator to avoid division by zero.
    """
    TP = ((y_true == 1) & (y_pred == 1)).sum()
    FN = ((y_true == 1) & (y_pred == 0)).sum()

    return float(TP / (TP + FN + 1e-15))

def compute_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute F1 score for binary classification.

    F1 score is the harmonic mean of precision and recall.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True binary labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted binary labels.

    Returns
    -------
    float
        F1 score.

    Notes
    -----
    - F1 score balances precision and recall
    - Useful when classes are imbalanced
    - Adds small epsilon to avoid division by zero
    """
    precision = compute_precision(y_true, y_pred)
    recall = compute_recall(y_true, y_pred)

    return 2 * (precision * recall) / (precision + recall + 1e-15)