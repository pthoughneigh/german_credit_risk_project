import numpy as np

def _linear_model(X, weights, bias):
    return np.dot(X, weights) + bias

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_proba(X, weights, bias):
    z = _linear_model(X, weights, bias)
    return sigmoid(z)

def compute_log_loss(y_true, y_pred_proba):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred_proba)
    y_pred = np.clip(y_pred, 1e-15, 1-1e-15)

    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss.mean()

def compute_gradients(X, y_true, y_pred_proba):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred_proba)

    error = y_pred - y_true
    n = error.shape[0]

    dw = 1/n * (X.T.dot(error))
    db = 1/n * error.sum()

    return dw, db

def update_parameters(weights, bias, dw, db, learning_rate):
    # w := w−α⋅dw
    # b := b−α⋅db

    weights = weights - learning_rate * dw
    bias = bias - learning_rate * db

    return weights, bias

def train_logistic_regression(X, y, learning_rate=0.01, n_iterations=1000):
    X = np.array(X)
    y = np.array(y)

    weights = np.zeros(X.shape[1])
    bias = 0
    losses = []

    for i in range(n_iterations):
        y_pred = predict_proba(X, weights, bias)
        loss = compute_log_loss(y, y_pred)
        dw, db = compute_gradients(X, y, y_pred)

        weights, bias = update_parameters(weights, bias, dw, db, learning_rate)
        losses.append(loss)

    return  weights, bias, losses