"""Lightweight tests for reusable baseline models and metrics."""

import numpy as np
import torch

from graphtoolbox.models.baseline import (
    CovariateSeq2Seq,
    fit_tabular_baseline,
    mape,
    rmse,
    smape,
)


def test_baseline_metrics():
    target = np.array([1.0, 2.0, 4.0])
    prediction = np.array([1.0, 3.0, 2.0])
    assert np.isclose(rmse(prediction, target), np.sqrt(5.0 / 3.0))
    assert np.isclose(mape(prediction, target), 100.0 / 3.0)
    assert 0.0 < smape(prediction, target) < 200.0


def test_covariate_seq2seq_shape():
    model = CovariateSeq2Seq(n_features=3, hidden_channels=5, horizon=4)
    history = torch.randn(2, 7, 4)
    future = torch.randn(2, 4, 3)
    assert model(history, future).shape == (2, 4)


def test_fit_tabular_ridge_shapes():
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(30, 10))
    y_train = x_train[:, 0] - 0.5 * x_train[:, 1]
    x_validation = rng.normal(size=(5, 10))
    x_test = rng.normal(size=(7, 10))
    _, train_prediction, validation_prediction, test_prediction = fit_tabular_baseline(
        "ridge", x_train, y_train, x_test, x_validation=x_validation,
        y_validation=np.zeros(len(x_validation)))
    assert train_prediction.shape == (30,)
    assert validation_prediction.shape == (5,)
    assert test_prediction.shape == (7,)
