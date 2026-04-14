import numpy as np
from typing import List, Tuple


def _linear_model(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """
    Compute the linear part of logistic regression.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input feature matrix.
    weights : np.ndarray of shape (n_features,)
        Model weight vector.
    bias : float
        Bias term.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Linear scores computed as X @ weights + bias.
    """
    return np.dot(X, weights) + bias


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Apply the sigmoid function in a numerically stable way.

    The sigmoid maps real-valued inputs to the interval (0, 1),
    making it suitable for probability estimation in logistic regression.

    Parameters
    ----------
    z : np.ndarray
        Input array of linear scores.

    Returns
    -------
    np.ndarray
        Transformed probabilities in the interval (0, 1).

    Notes
    -----
    This implementation is numerically more stable than the naive version
    because it avoids overflow for very large positive or negative values.
    """
    return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),
        np.exp(z) / (1 + np.exp(z))
    )


def predict_proba(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """
    Compute predicted probabilities for class 1.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input feature matrix.
    weights : np.ndarray of shape (n_features,)
        Model weight vector.
    bias : float
        Bias term.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Predicted probabilities for the positive class.
    """
    z = _linear_model(X, weights, bias)
    return sigmoid(z)


def compute_log_loss(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Compute binary cross-entropy loss (log loss).

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True binary labels (0 or 1).
    y_pred_proba : np.ndarray of shape (n_samples,)
        Predicted probabilities for class 1.

    Returns
    -------
    float
        Mean binary cross-entropy loss over all samples.

    Notes
    -----
    Predicted probabilities are clipped to avoid log(0), which would
    otherwise produce numerical issues.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred_proba)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return float(loss.mean())


def compute_gradients(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred_proba: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Compute gradients of the log-loss with respect to weights and bias.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input feature matrix.
    y_true : np.ndarray of shape (n_samples,)
        True binary labels.
    y_pred_proba : np.ndarray of shape (n_samples,)
        Predicted probabilities for class 1.

    Returns
    -------
    Tuple[np.ndarray, float]
        dw : np.ndarray of shape (n_features,)
            Gradient with respect to weights.
        db : float
            Gradient with respect to bias.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred_proba)

    error = y_pred - y_true
    n = error.shape[0]

    dw = (1 / n) * (X.T @ error)
    db = float((1 / n) * error.sum())

    return dw, db


def update_parameters(
    weights: np.ndarray,
    bias: float,
    dw: np.ndarray,
    db: float,
    learning_rate: float
) -> Tuple[np.ndarray, float]:
    """
    Update model parameters using gradient descent.

    Parameters
    ----------
    weights : np.ndarray of shape (n_features,)
        Current model weights.
    bias : float
        Current bias term.
    dw : np.ndarray of shape (n_features,)
        Gradient with respect to weights.
    db : float
        Gradient with respect to bias.
    learning_rate : float
        Step size used in gradient descent.

    Returns
    -------
    Tuple[np.ndarray, float]
        Updated weights and bias.
    """
    weights = weights - learning_rate * dw
    bias = bias - learning_rate * db

    return weights, bias


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    n_iterations: int = 1000
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Train logistic regression using batch gradient descent.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Training feature matrix.
    y : np.ndarray of shape (n_samples,)
        True binary labels.
    learning_rate : float, default=0.01
        Learning rate for gradient descent.
    n_iterations : int, default=1000
        Number of optimization steps.

    Returns
    -------
    Tuple[np.ndarray, float, List[float]]
        weights : np.ndarray of shape (n_features,)
            Learned model weights.
        bias : float
            Learned bias term.
        losses : List[float]
            Log-loss value at each iteration.
    """
    X = np.array(X)
    y = np.array(y)

    weights = np.zeros(X.shape[1], dtype=float)
    bias = 0.0
    losses: List[float] = []

    for _ in range(n_iterations):
        y_pred = predict_proba(X, weights, bias)
        loss = compute_log_loss(y, y_pred)
        dw, db = compute_gradients(X, y, y_pred)

        weights, bias = update_parameters(weights, bias, dw, db, learning_rate)
        losses.append(loss)

    return weights, bias, losses


def predict(
    X: np.ndarray,
    weights: np.ndarray,
    bias: float,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Convert predicted probabilities into binary class predictions.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input feature matrix.
    weights : np.ndarray of shape (n_features,)
        Learned model weights.
    bias : float
        Learned bias term.
    threshold : float, default=0.5
        Classification threshold. Probabilities greater than or equal
        to this value are classified as 1.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Binary predictions (0 or 1).
    """
    probabilities = predict_proba(X, weights, bias)
    return (probabilities >= threshold).astype(int)