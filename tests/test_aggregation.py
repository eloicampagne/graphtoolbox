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


def test_predict_observe_matches_run_by_blocks():
    X, y = _experts(T=480)
    h = 48
    ref = Aggregation(model="MLpol", loss="square").run(X, y, block_size=h)
    agg = Aggregation(model="MLpol", loss="square")
    preds = [X[:h].mean(axis=1)]                 # first block: uniform prior
    agg.observe(X[:h], y[:h])
    for s in range(h, X.shape[0], h):
        preds.append(agg.predict(X[s:s + h]))
        agg.observe(X[s:s + h], y[s:s + h])
    np.testing.assert_allclose(np.concatenate(preds), ref.prediction_, rtol=0, atol=1e-12)
    np.testing.assert_allclose(agg.coefficients_, ref.coefficients_, rtol=0, atol=1e-15)


def test_observe_continues_run_and_skips_missing_targets():
    X, y = _experts(T=200)
    ref = Aggregation(model="MLpol").run(X, y)
    agg = Aggregation(model="MLpol").run(X[:120], y[:120]).observe(X[120:], y[120:])
    np.testing.assert_allclose(agg.coefficients_, ref.coefficients_, rtol=0, atol=1e-15)

    y_nan = y.copy()
    y_nan[150:160] = np.nan
    X_nan = X.copy()
    X_nan[150:160] = np.nan                      # experts may be missing where y is
    keep = np.isfinite(y_nan)
    sub = Aggregation(model="MLpol").run(X[keep], y[keep])
    agg = Aggregation(model="MLpol").observe(X_nan, y_nan)
    np.testing.assert_allclose(agg.coefficients_, sub.coefficients_, rtol=0, atol=1e-15)


def test_observe_rejects_bad_inputs():
    import pytest
    X, y = _experts(T=48)
    agg = Aggregation(model="MLpol").observe(X, y)
    with pytest.raises(ValueError, match="Expected 3 experts"):
        agg.observe(X[:, :2], y)
    X_bad = X.copy()
    X_bad[5, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        agg.observe(X_bad, y)
    with pytest.raises(ValueError, match="offline oracle"):
        Aggregation(model="best").observe(X, y)
