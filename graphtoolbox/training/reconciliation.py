"""MinT reconciliation of node forecasts with a forecast of their total.

The hierarchy has ``n`` nodes and one total, ``y_total = sum_i y_i``. Given base forecasts of the
nodes and of the total, MinT (Wickramasuriya et al., 2019) projects them onto the coherent subspace

    y_rec = S (S^T W^{-1} S)^{-1} S^T W^{-1} [y_nodes; y_total],    S = [I_n; 1_n^T],

where ``W`` estimates the covariance of the base forecast errors on held-out data. Two estimators of
``W`` are available:

``"diag"``
    The error variance of each series (WLS, variance scaling).
``"shrink"``
    The full error covariance, shrunk toward its diagonal with the intensity of Schafer and Strimmer
    (2005), as in the ``hts`` R package. The full sample covariance is near-singular when the node
    errors are strongly correlated, and the shrinkage keeps it invertible.

The functions work on NumPy arrays in float64, so that forecasts cached on disk can be reconciled
without a model. ``Trainer(reconcile=True, reconcile_covariance=...)`` uses the same estimators.
"""
from __future__ import annotations

import numpy as np

COVARIANCES = ("diag", "shrink")
# NumPy 2 linked to Apple Accelerate raises spurious floating-point flags in matmul on finite inputs;
# the products below are silenced, and the results are checked for finiteness where it matters.
_QUIET = dict(divide="ignore", over="ignore", invalid="ignore")


def summing_matrix(num_nodes: int) -> np.ndarray:
    """Structure matrix ``S = [I_n; 1_n^T]`` of ``num_nodes`` nodes and their total."""
    return np.vstack([np.eye(num_nodes), np.ones((1, num_nodes))])


def shrinkage_intensity(errors: np.ndarray) -> float:
    """Schafer-Strimmer shrinkage intensity of the correlation of ``errors`` [series, T] toward zero."""
    E = np.asarray(errors, dtype=float)
    T = E.shape[1]
    with np.errstate(**_QUIET):
        cov = E @ E.T / T
        xs = E / np.sqrt(np.diag(cov))[:, None]
        v = ((xs ** 2) @ (xs ** 2).T - (xs @ xs.T) ** 2 / T) / (T * (T - 1))
        cor = xs @ xs.T / T
    np.fill_diagonal(v, 0.0)
    np.fill_diagonal(cor, 0.0)
    return float(np.clip(v.sum() / (cor ** 2).sum(), 0.0, 1.0))


def error_covariance(errors: np.ndarray, covariance: str = "diag") -> np.ndarray:
    """Weight matrix ``W`` estimated from base forecast errors.

    Parameters
    ----------
    errors : array [series, T]
        Errors of the base forecasts (forecast minus observation) on held-out data, one row per series
        in the order of ``S``: the nodes, then the total.
    covariance : {"diag", "shrink"}
        ``"diag"`` keeps the error variances (unbiased); ``"shrink"`` shrinks the full covariance (second
        moments about zero) toward its diagonal with the intensity of :func:`shrinkage_intensity`.
    """
    E = np.asarray(errors, dtype=float)
    if covariance == "diag":
        return np.diag(E.var(axis=1, ddof=1))
    if covariance == "shrink":
        with np.errstate(**_QUIET):
            cov = E @ E.T / E.shape[1]
        lam = shrinkage_intensity(E)
        return lam * np.diag(np.diag(cov)) + (1 - lam) * cov
    raise ValueError(f"covariance must be one of {COVARIANCES}, got {covariance!r}")


def mint_projection(W: np.ndarray, S: np.ndarray) -> np.ndarray:
    """MinT mapping ``G = (S^T W^{-1} S)^{-1} S^T W^{-1}``; reconciled forecasts are ``S @ G @ y``."""
    Wi = np.linalg.inv(W)
    with np.errstate(**_QUIET):
        return np.linalg.solve(S.T @ Wi @ S, S.T @ Wi)


def reconcile(base: np.ndarray, top: np.ndarray, base_errors: np.ndarray, top_errors: np.ndarray,
              covariance: str = "diag") -> np.ndarray:
    """Reconcile node forecasts with a forecast of their total.

    Parameters
    ----------
    base : array [num_nodes, T]
        Base forecasts of the nodes.
    top : array [T]
        Base forecast of the total.
    base_errors, top_errors : arrays [num_nodes, T_val] and [T_val]
        Errors of the same models on held-out data, used to estimate ``W``.
    covariance : {"diag", "shrink"}
        Estimator of ``W`` (see :func:`error_covariance`).

    Returns
    -------
    array [num_nodes + 1, T]
        Coherent forecasts of the nodes, then of the total (the last row is the sum of the others).
    """
    base = np.asarray(base, dtype=float)
    S = summing_matrix(base.shape[0])
    W = error_covariance(np.vstack([base_errors, np.asarray(top_errors, dtype=float)[None]]), covariance)
    G = mint_projection(W, S)
    with np.errstate(**_QUIET):
        out = S @ G @ np.vstack([base, np.asarray(top, dtype=float)[None]])
    if not np.isfinite(out).all():
        raise FloatingPointError("reconciled forecasts are not finite; check the error matrix")
    return out
