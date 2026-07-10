"""Tests for online expert aggregation (MLpol) on synthetic experts."""
import numpy as np

from graphtoolbox.aggregation import Aggregation


def _experts(seed=0, T=96):
    rng = np.random.default_rng(seed)
    y = 50.0 + 10.0 * np.sin(np.arange(T) * 0.2)
    # three experts of decreasing quality
    X = np.column_stack([y + rng.standard_normal(T) * s for s in (0.5, 2.0, 6.0)])
    return X, y


def _rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p) - y) ** 2)))


def test_mlpol_shape_and_convex_weights():
    X, y = _experts()
    agg = Aggregation(model="MLpol", loss="square").run(X, y, block_size=48)
    pred = np.asarray(agg.prediction_)
    assert pred.shape == (X.shape[0],)
    w = np.asarray(agg.coefficients_)
    assert w.shape == (X.shape[1],)
    assert np.all(w >= -1e-8) and abs(w.sum() - 1.0) < 1e-6


def test_mlpol_favours_best_expert_and_beats_worst():
    X, y = _experts()
    agg = Aggregation(model="MLpol", loss="square").run(X, y, block_size=48)
    w = np.asarray(agg.coefficients_)
    assert w.argmax() == 0                       # lowest-noise expert gets most weight
    assert _rmse(agg.prediction_, y) <= _rmse(X[:, -1], y)   # beats the worst expert
