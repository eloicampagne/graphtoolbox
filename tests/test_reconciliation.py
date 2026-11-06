import numpy as np
import pytest

from graphtoolbox.training.reconciliation import error_covariance, reconcile, shrinkage_intensity


def _errors(n=4, T=400, seed=0):
    rng = np.random.default_rng(seed)
    common = rng.normal(size=(1, T))
    base = rng.normal(size=(n, T)) + 2.0 * common
    top = base.sum(axis=0) + rng.normal(size=T)
    return base, top


@pytest.mark.parametrize("covariance", ["diag", "shrink"])
def test_reconciled_forecasts_are_coherent(covariance):
    base_err, top_err = _errors()
    rng = np.random.default_rng(1)
    base = rng.normal(size=(4, 10)) + 10.0
    top = base.sum(axis=0) + rng.normal(size=10)
    out = reconcile(base, top, base_err, top_err, covariance)
    assert out.shape == (5, 10)
    np.testing.assert_allclose(out[-1], out[:-1].sum(axis=0), rtol=1e-10, atol=1e-8)


def test_coherent_inputs_are_left_unchanged():
    base_err, top_err = _errors()
    base = np.arange(20, dtype=float).reshape(4, 5)
    for covariance in ("diag", "shrink"):
        out = reconcile(base, base.sum(axis=0), base_err, top_err, covariance)
        np.testing.assert_allclose(out[:-1], base, atol=1e-8)


def test_diag_weights_are_the_error_variances():
    base_err, top_err = _errors()
    E = np.vstack([base_err, top_err[None]])
    np.testing.assert_allclose(error_covariance(E, "diag"), np.diag(E.var(axis=1, ddof=1)))


def test_shrinkage_keeps_the_diagonal_and_lies_between_the_two_targets():
    base_err, top_err = _errors()
    E = np.vstack([base_err, top_err[None]])
    lam = shrinkage_intensity(E)
    W = error_covariance(E, "shrink")
    cov = E @ E.T / E.shape[1]
    assert 0.0 <= lam <= 1.0
    np.testing.assert_allclose(np.diag(W), np.diag(cov))
    off = ~np.eye(len(W), dtype=bool)
    np.testing.assert_allclose(W[off], (1 - lam) * cov[off])
    assert np.linalg.eigvalsh(W).min() > 0


def test_unknown_covariance_raises():
    base_err, top_err = _errors()
    with pytest.raises(ValueError, match="covariance"):
        error_covariance(np.vstack([base_err, top_err[None]]), "full")
