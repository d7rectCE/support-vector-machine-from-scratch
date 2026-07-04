"""
Support Vector Machine from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - standardize_features
def standardize_features(x):
    # TODO: rescale each column of x to have mean 0 and std 1 (leave zero-std columns alone).
    x = np.array(x)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    
    std[std == 0.0] = 1.0
    
    standardize = (x - mean)/std
    return standardize

# Step 2 - initialize_parameters
def initialize_parameters(n_features):
    # TODO: create starting weights and bias for a linear SVM
    w = np.zeros(n_features)
    b = 0
    return {"w": w, "b": b}

# Step 3 - compute_scores
def compute_scores(x, params):
    # TODO: score each example as a linear function of the current weights and bias.
    x = np.array(x)
    w = np.array(params["w"])
    b = np.array(params["b"])
    scores = x @ w + b
    return scores

# Step 4 - predict_from_scores
def predict_from_scores(scores):
    # TODO: convert a 1-D array of raw scores into +1 / -1 class predictions.
    class_pred = np.where(scores >= 0, 1, -1)
    return class_pred

# Step 5 - hinge_loss_example
def hinge_loss_example(score, y):
    # TODO: return the hinge loss for a single example with raw score `score` and label y in {-1, +1}.
    loss = np.maximum(0, 1 - y * score)
    return loss

# Step 6 - svm_objective
def svm_objective(x, y, params, reg_lambda):
    # TODO: return mean hinge loss over the dataset plus reg_lambda * (w dot w)
    scores = compute_scores(x, params)
    loss = hinge_loss_example(scores, y)
    w = params["w"]
    obj = 1/len(x) * np.sum(loss + reg_lambda * (w @ w))
    return obj

# Step 7 - compute_gradients
def compute_gradients(x, y, params, reg_lambda):
    """Return {'dw': ndarray shape (n_features,), 'db': float} = gradient of svm_objective."""
    n_samples = x.shape[0]
    w = params["w"]
    margins = y * compute_scores(x, params)
    violation_mask = (margins < 1)
    
    dw = -(1/n_samples) * (x.T @ (y * violation_mask)) + 2 * reg_lambda * w
    db = -(1/n_samples) * np.sum(y * violation_mask)
    return {"dw": dw, "db": db}

# Step 8 - apply_update
def apply_update(params, grads, learning_rate):
    w_updated = params["w"] - learning_rate * grads["dw"]
    b_updated = params["b"] - learning_rate * grads["db"]
    return {"w": w_updated, "b": b_updated}

# Step 9 - train_svm
def train_svm(x, y, learning_rate, reg_lambda, n_epochs):
    # TODO: fit a linear SVM by repeatedly updating parameters over n_epochs passes.
    params = initialize_parameters(np.shape(x[1]))
    for _ in range(n_epochs):
        grads = compute_gradients(x, y, params, reg_lambda)
        params = apply_update(params, grads, learning_rate)
    return params

# Step 10 - predict_labels
def predict_labels(x, params):
    # TODO: return an array of {-1, +1} labels, one per row of x, using params['w'] and params['b'].
    scores = compute_scores(x, params)
    y_pred = predict_from_scores(scores)
    return y_pred

# Step 11 - accuracy_score
def accuracy_score(y_pred, y_true):
    # TODO: return the fraction of positions where y_pred equals y_true.
    accuracy = 1/y_true.shape[0] * np.sum(y_pred == y_true)
    return accuracy

