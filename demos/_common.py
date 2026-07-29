"""Shared helpers for the demo scripts (lightweight, CPU friendly, no downloads)."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_blobs(n_per_class=200, dim=10, n_classes=3, sep=3.0, noise=1.0, seed=0):
    """Well separated Gaussian blobs -- a data set with a lot of redundancy."""
    rs = np.random.RandomState(seed)
    centers = rs.randn(n_classes, dim) * sep
    X = np.concatenate([rs.randn(n_per_class, dim) * noise + centers[c] for c in range(n_classes)])
    y = np.concatenate([np.full(n_per_class, c) for c in range(n_classes)])
    perm = rs.permutation(len(y))
    return torch.from_numpy(X[perm]).float(), torch.from_numpy(y[perm]).long()


def train_test_split(X, y, test_frac=0.3, seed=0):
    rs = np.random.RandomState(seed)
    n = X.shape[0]
    perm = rs.permutation(n)
    cut = int(n * (1 - test_frac))
    tr, te = perm[:cut], perm[cut:]
    return X[tr], y[tr], X[te], y[te]


def fit_logistic_regression(X, y, n_classes, weights=None, reg=1e-3, epochs=300, lr=0.1):
    """Reference training routine used to evaluate the summaries."""
    import models
    model = models.LogisticRegression(X.shape[1], n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
    weights = torch.ones(X.shape[0]) if weights is None else torch.as_tensor(weights).float()
    weights = weights / weights.mean()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.mean(loss_fn(model(X), y.long()) * weights)
        loss.backward()
        optimizer.step()
    return model


def accuracy(model, X, y):
    with torch.no_grad():
        return float((torch.argmax(model(X), dim=1) == y.long()).float().mean())


def evaluate_subset(X, y, X_test, y_test, inds, n_classes, weights=None, **kwargs):
    model = fit_logistic_regression(X[inds], y[inds], n_classes, weights=weights, **kwargs)
    return accuracy(model, X_test, y_test)
