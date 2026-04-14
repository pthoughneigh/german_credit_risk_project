import numpy as np
from typing import Tuple, List

from src.evaluation.metrics import compute_confusion_matrix


def compute_roc_points(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> Tuple[List[float], List[float]]:
    """
    Compute ROC curve points (FPR and TPR) for all unique thresholds.

    This function evaluates the model at different classification thresholds
    derived from predicted probabilities and computes:

    - True Positive Rate (TPR) = TP / (TP + FN)
    - False Positive Rate (FPR) = FP / (FP + TN)

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1), shape (n_samples,).

    y_scores : np.ndarray
        Predicted probabilities for the positive class (class 1),
        shape (n_samples,).

    Returns
    -------
    fpr_list : List[float]
        False Positive Rate values for each threshold.

    tpr_list : List[float]
        True Positive Rate values for each threshold.

    Notes
    -----
    - Thresholds are taken from unique predicted scores (sorted descending).
    - Each threshold defines a different decision boundary.
    - Used for constructing the ROC curve.
    """
    thresholds = np.unique(y_scores)[::-1]

    tpr_list = []
    fpr_list = []

    for t in thresholds:
        y_pred_binary = (y_scores >= t).astype(int)

        cm = compute_confusion_matrix(y_true, y_pred_binary)
        TN, FP = cm[0]
        FN, TP = cm[1]

        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return fpr_list, tpr_list


def compute_auc(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> float:
    """
    Compute Area Under the ROC Curve (AUC).

    This function calculates the AUC by:
    1. Computing ROC curve points (FPR, TPR)
    2. Sorting them by FPR
    3. Applying the trapezoidal rule for numerical integration

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1), shape (n_samples,).

    y_scores : np.ndarray
        Predicted probabilities for the positive class (class 1),
        shape (n_samples,).

    Returns
    -------
    auc : float
        Area under the ROC curve.

    Notes
    -----
    - AUC ranges from 0 to 1:
        - 0.5 → random model
        - 1.0 → perfect model
    - Higher AUC indicates better discrimination between classes.
    """
    fpr_list, tpr_list = compute_roc_points(y_true, y_scores)

    pairs = sorted(zip(fpr_list, tpr_list))
    fpr_sorted, tpr_sorted = zip(*pairs)

    auc = np.trapz(tpr_sorted, fpr_sorted)

    print("AUC:", auc)
    return float(auc)