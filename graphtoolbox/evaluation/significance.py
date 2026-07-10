"""Statistical significance tools for comparing cached forecasts.

Every function operates on prediction arrays, never on model weights, so a full
significance analysis runs on the cached forecasts without retraining anything.
Given a set of forecasts and the ground truth, the module provides pairwise
predictive-accuracy tests, bootstrap standard errors and confidence intervals
for error metrics, and the Model Confidence Set.

References
----------
Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy.
    Journal of Business & Economic Statistics, 13(3), 253-263.
Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of
    prediction mean squared errors. International Journal of Forecasting,
    13(2), 281-291.
Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite,
    heteroskedasticity and autocorrelation consistent covariance matrix.
    Econometrica, 55(3), 703-708.
Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set.
    Econometrica, 79(2), 453-497.
Kunsch, H. R. (1989). The jackknife and the bootstrap for general stationary
    observations. The Annals of Statistics, 17(3), 1217-1241.
"""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _np(x) -> np.ndarray:
    """Coerce a torch tensor or array-like to a 1-D float numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float).reshape(-1)


def _loss(pred: np.ndarray, target: np.ndarray, loss: str) -> np.ndarray:
    """Pointwise loss series used by the predictive-accuracy tests.

    ``squared`` gives an MSE/RMSE comparison, ``absolute`` an MAE comparison,
    and ``ape`` an (M)APE comparison restricted to strictly non-zero targets.
    """
    e = pred - target
    if loss == "squared":
        return e ** 2
    if loss == "absolute":
        return np.abs(e)
    if loss == "ape":
        return np.abs(e / target)
    raise ValueError(f"unknown loss '{loss}' (use 'squared', 'absolute' or 'ape')")


def _metric(pred: np.ndarray, target: np.ndarray, metric: str) -> float:
    """Scalar error metric on aligned prediction/target arrays."""
    e = pred - target
    if metric == "rmse":
        return float(np.sqrt(np.mean(e ** 2)))
    if metric == "mae":
        return float(np.mean(np.abs(e)))
    if metric == "mape":
        m = np.abs(target) > 0
        return float(np.mean(np.abs(e[m] / target[m])) * 100.0)
    if metric == "nmae":
        return float(np.mean(np.abs(e)) / np.mean(np.abs(target)))
    if metric == "bias":
        return float(np.mean(e))
    raise ValueError(f"unknown metric '{metric}'")


def _newey_west_lrv(d: np.ndarray, lag: int) -> float:
    """Newey-West (Bartlett-kernel) long-run variance of a mean-zero series.

    The Bartlett weights guarantee a non-negative estimate, which the original
    rectangular Diebold-Mariano truncation does not.
    """
    T = d.size
    d = d - d.mean()
    lrv = float(np.sum(d * d)) / T
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        gamma_k = float(np.sum(d[k:] * d[:-k])) / T
        lrv += 2.0 * w * gamma_k
    return lrv


def _holm(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjustment of a vector of p-values."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def _moving_block_indices(T: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Index vector of length T built from random overlapping blocks (Kunsch, 1989)."""
    if block_len < 1:
        raise ValueError("block_len must be >= 1")
    n_blocks = int(np.ceil(T / block_len))
    starts = rng.integers(0, T - block_len + 1, size=n_blocks) if T > block_len \
        else np.zeros(n_blocks, dtype=int)
    idx = (starts[:, None] + np.arange(block_len)[None, :]).reshape(-1)
    return idx[:T]


# --------------------------------------------------------------------------- #
# Pairwise predictive-accuracy test
# --------------------------------------------------------------------------- #
@dataclass
class DMResult:
    """Outcome of a Diebold-Mariano test between two forecasts.

    ``statistic`` is the HLN-corrected DM statistic with sign convention
    ``loss(a) - loss(b)``: a positive value means forecast ``a`` has the larger
    loss (is worse). ``mean_loss_diff`` is that same difference in level, and
    ``better`` names the lower-loss forecast.
    """
    statistic: float
    pvalue: float
    mean_loss_diff: float
    better: str
    horizon: int
    n: int


def diebold_mariano(pred_a, pred_b, target, *, loss: str = "squared",
                    h: int = 1, names=("a", "b")) -> DMResult:
    """Diebold-Mariano test of equal predictive accuracy between two forecasts.

    Uses a Newey-West long-run variance truncated at ``h - 1`` lags (the moving
    average order of ``h``-step forecast-error differentials) and the
    Harvey-Leybourne-Newbold small-sample correction, comparing the statistic to
    a Student-t distribution with ``n - 1`` degrees of freedom.

    Parameters
    ----------
    pred_a, pred_b : array-like
        The two forecast series (torch tensors or numpy arrays).
    target : array-like
        Ground-truth series.
    loss : {'squared', 'absolute', 'ape'}
        Loss under which accuracy is compared.
    h : int
        Forecast horizon; sets the autocovariance truncation lag to ``h - 1``.
    names : tuple[str, str]
        Labels for the two forecasts, used in the ``better`` field.

    Returns
    -------
    DMResult
    """
    a, b, y = _np(pred_a), _np(pred_b), _np(target)
    if not (a.size == b.size == y.size):
        raise ValueError("pred_a, pred_b and target must have the same length")
    d = _loss(a, y, loss) - _loss(b, y, loss)
    T = d.size
    dbar = float(d.mean())
    lrv = _newey_west_lrv(d, max(h - 1, 0))
    var_dbar = lrv / T
    if var_dbar <= 0 or not np.isfinite(var_dbar):
        stat, pval = float("nan"), float("nan")
    else:
        dm = dbar / np.sqrt(var_dbar)
        correction = np.sqrt(max(T + 1 - 2 * h + h * (h - 1) / T, 1e-12) / T)
        stat = dm * correction
        pval = float(2.0 * stats.t.sf(abs(stat), df=T - 1))
    better = names[1] if dbar > 0 else names[0]
    return DMResult(statistic=float(stat), pvalue=pval, mean_loss_diff=dbar,
                    better=better, horizon=h, n=T)


def pairwise_dm(preds: Dict[str, "np.ndarray"], target, *, loss: str = "squared",
                h: int = 1, correction: str = "holm"):
    """All pairwise Diebold-Mariano tests over a set of forecasts.

    Parameters
    ----------
    preds : dict[str, array-like]
        Mapping from model name to forecast series.
    target : array-like
        Ground-truth series.
    loss, h : see :func:`diebold_mariano`.
    correction : {'holm', 'none'}
        Multiple-testing adjustment applied to the collection of pairwise
        p-values (one per unordered pair).

    Returns
    -------
    dict
        ``statistic`` and ``pvalue`` as name-indexed square ``pandas.DataFrame``
        objects (statistic antisymmetric, p-values symmetric), ``pvalue_adjusted``
        with the correction applied, and ``ranking`` ordering models by mean loss.
    """
    import pandas as pd

    names = list(preds)
    y = _np(target)
    losses = {n: _loss(_np(preds[n]), y, loss) for n in names}
    m = len(names)
    stat = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    pval = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)

    pair_p = []
    pair_ij = []
    for i in range(m):
        for j in range(i + 1, m):
            ni, nj = names[i], names[j]
            r = diebold_mariano(preds[ni], preds[nj], target, loss=loss, h=h,
                                names=(ni, nj))
            stat.loc[ni, nj] = r.statistic
            stat.loc[nj, ni] = -r.statistic
            pval.loc[ni, nj] = pval.loc[nj, ni] = r.pvalue
            pair_p.append(r.pvalue)
            pair_ij.append((ni, nj))

    padj = pval.copy()
    if correction == "holm" and pair_p:
        finite = np.array([p if np.isfinite(p) else 1.0 for p in pair_p])
        adj = _holm(finite)
        for (ni, nj), a in zip(pair_ij, adj):
            padj.loc[ni, nj] = padj.loc[nj, ni] = a
    elif correction not in ("holm", "none"):
        raise ValueError("correction must be 'holm' or 'none'")

    mean_loss = {n: float(losses[n].mean()) for n in names}
    ranking = pd.Series(mean_loss).sort_values()
    return {"statistic": stat, "pvalue": pval, "pvalue_adjusted": padj,
            "ranking": ranking}


# --------------------------------------------------------------------------- #
# Bootstrap standard errors and confidence intervals for a metric
# --------------------------------------------------------------------------- #
@dataclass
class BootstrapResult:
    """Bootstrap summary for a scalar error metric."""
    metric: str
    point: float
    se: float
    ci_low: float
    ci_high: float
    level: float
    samples: np.ndarray


def bootstrap_metric(pred, target, *, metric: str = "rmse", n_boot: int = 2000,
                     block_len: int = 48, level: float = 0.95,
                     seed: Optional[int] = 0) -> BootstrapResult:
    """Moving-block bootstrap standard error and CI for an error metric.

    Resampling whole blocks of consecutive time steps preserves the temporal
    dependence of forecast errors, which an i.i.d. bootstrap would destroy. The
    default block length of 48 matches one day of half-hourly data.

    Parameters
    ----------
    pred, target : array-like
        Forecast and ground-truth series.
    metric : {'rmse', 'mae', 'mape', 'nmae', 'bias'}
        Metric whose sampling uncertainty is estimated.
    n_boot : int
        Number of bootstrap resamples.
    block_len : int
        Block length for the moving-block bootstrap.
    level : float
        Central confidence level for the percentile interval.
    seed : int or None
        Seed for reproducibility.

    Returns
    -------
    BootstrapResult
    """
    p, y = _np(pred), _np(target)
    if p.size != y.size:
        raise ValueError("pred and target must have the same length")
    T = p.size
    rng = np.random.default_rng(seed)
    point = _metric(p, y, metric)
    samples = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = _moving_block_indices(T, block_len, rng)
        samples[b] = _metric(p[idx], y[idx], metric)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(samples, [alpha, 1.0 - alpha])
    return BootstrapResult(metric=metric, point=point, se=float(samples.std(ddof=1)),
                           ci_low=float(lo), ci_high=float(hi), level=level,
                           samples=samples)


def bootstrap_table(preds: Dict[str, "np.ndarray"], target, *, metric: str = "rmse",
                    n_boot: int = 2000, block_len: int = 48, level: float = 0.95,
                    seed: Optional[int] = 0):
    """Bootstrap point estimate, standard error and CI for every forecast.

    Returns a ``pandas.DataFrame`` sorted by the metric, one row per model.
    """
    import pandas as pd

    rows = {}
    for i, (name, pred) in enumerate(preds.items()):
        r = bootstrap_metric(pred, target, metric=metric, n_boot=n_boot,
                             block_len=block_len, level=level,
                             seed=None if seed is None else seed + i)
        rows[name] = {metric: r.point, "se": r.se,
                      "ci_low": r.ci_low, "ci_high": r.ci_high}
    df = pd.DataFrame(rows).T
    return df.sort_values(metric)


# --------------------------------------------------------------------------- #
# Model Confidence Set (Hansen, Lunde & Nason, 2011)
# --------------------------------------------------------------------------- #
@dataclass
class MCSResult:
    """Outcome of the Model Confidence Set procedure."""
    included: list
    pvalues: "object"          # pandas.Series name -> MCS p-value
    eliminated_order: list
    alpha: float


def model_confidence_set(preds: Dict[str, "np.ndarray"], target, *,
                         loss: str = "squared", alpha: float = 0.10,
                         n_boot: int = 2000, block_len: int = 48,
                         seed: Optional[int] = 0) -> MCSResult:
    """Model Confidence Set with the range/T-max statistic and block bootstrap.

    Iteratively tests the hypothesis of equal predictive ability across the
    surviving models and eliminates the worst one until the hypothesis is no
    longer rejected. The surviving set contains, with probability at least
    ``1 - alpha``, the best model. Each model receives an MCS p-value; models
    with an MCS p-value above ``alpha`` form the confidence set.

    Parameters
    ----------
    preds : dict[str, array-like]
        Mapping from model name to forecast series.
    target : array-like
        Ground-truth series.
    loss : {'squared', 'absolute', 'ape'}
        Loss under which the models are compared.
    alpha : float
        Size of the test; the confidence set has level ``1 - alpha``.
    n_boot, block_len : int
        Moving-block bootstrap settings (shared across elimination steps).
    seed : int or None
        Seed for reproducibility.

    Returns
    -------
    MCSResult
    """
    import pandas as pd

    names = list(preds)
    y = _np(target)
    L = np.vstack([_loss(_np(preds[n]), y, loss) for n in names])   # [m, T]
    m0, T = L.shape
    rng = np.random.default_rng(seed)

    # Shared block-bootstrap index resamples and the per-timestep loss means.
    boot_idx = [_moving_block_indices(T, block_len, rng) for _ in range(n_boot)]
    Lbar = L.mean(axis=1)                                           # [m]
    Lbar_boot = np.vstack([L[:, idx].mean(axis=1) for idx in boot_idx])  # [B, m]

    alive = list(range(m0))
    eliminated_order = []
    mcs_p = {names[i]: 1.0 for i in range(m0)}
    running_p = 0.0

    while len(alive) > 1:
        a = np.array(alive)
        lbar = Lbar[a]                                              # [k]
        lbar_b = Lbar_boot[:, a]                                    # [B, k]
        # Deviation of each model's mean loss from the set average.
        d = lbar - lbar.mean()                                     # [k]
        d_b = lbar_b - lbar_b.mean(axis=1, keepdims=True)          # [B, k]
        var = ((d_b - d) ** 2).mean(axis=0)                        # [k]
        var = np.where(var > 0, var, np.nan)
        t = d / np.sqrt(var)                                       # standardized excess loss
        t_boot = (d_b - d) / np.sqrt(var)                          # [B, k]
        Tmax = np.nanmax(t)
        Tmax_boot = np.nanmax(t_boot, axis=1)
        pval = float(np.mean(Tmax_boot >= Tmax))

        running_p = max(running_p, pval)
        worst_local = int(np.nanargmax(t))
        worst = a[worst_local]
        mcs_p[names[worst]] = running_p

        if pval >= alpha:
            break
        eliminated_order.append(names[worst])
        alive.remove(worst)

    if len(alive) == 1:
        mcs_p[names[alive[0]]] = max(running_p, mcs_p[names[alive[0]]])

    included = [names[i] for i in alive]
    pvalues = pd.Series(mcs_p).sort_values(ascending=False)
    return MCSResult(included=included, pvalues=pvalues,
                     eliminated_order=eliminated_order, alpha=alpha)
