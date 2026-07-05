"""
Support Vector Machine from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np


def standardize_features(x):
    x = np.array(x)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    
    std[std == 0.0] = 1.0
    
    standardize = (x - mean)/std
    return standardize


def initialize_parameters(n_features):
    w = np.zeros(n_features)
    b = 0
    return {"w": w, "b": b}


def compute_scores(x, params):
    x = np.array(x)
    w = np.array(params["w"])
    b = np.array(params["b"])
    scores = x @ w + b
    return scores


def predict_from_scores(scores):
    class_pred = np.where(scores >= 0, 1, -1)
    return class_pred


def hinge_loss_example(score, y):
    loss = np.maximum(0, 1 - y * score)
    return loss


def svm_objective(x, y, params, reg_lambda):
    scores = compute_scores(x, params)
    loss = hinge_loss_example(scores, y)
    w = params["w"]
    obj = 1/len(x) * np.sum(loss + reg_lambda * (w @ w))
    return obj


def compute_gradients(x, y, params, reg_lambda):
    n_samples = x.shape[0]
    w = params["w"]
    margins = y * compute_scores(x, params)
    violation_mask = (margins < 1)
    
    dw = -(1/n_samples) * (x.T @ (y * violation_mask)) + 2 * reg_lambda * w
    db = -(1/n_samples) * np.sum(y * violation_mask)
    return {"dw": dw, "db": db}


def apply_update(params, grads, learning_rate):
    w_updated = params["w"] - learning_rate * grads["dw"]
    b_updated = params["b"] - learning_rate * grads["db"]
    return {"w": w_updated, "b": b_updated}


def train_svm(x, y, learning_rate, reg_lambda, n_epochs):
    params = initialize_parameters(x.shape[1])
    for _ in range(n_epochs):
        grads = compute_gradients(x, y, params, reg_lambda)
        params = apply_update(params, grads, learning_rate)
    return params


def predict_labels(x, params):
    scores = compute_scores(x, params)
    y_pred = predict_from_scores(scores)
    return y_pred


def accuracy_score(y_pred, y_true):
    accuracy = 1/y_true.shape[0] * np.sum(y_pred == y_true)
    return accuracy

