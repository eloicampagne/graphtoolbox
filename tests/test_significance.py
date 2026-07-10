"""Smoke tests for the significance module on synthetic forecasts with a known order."""
import numpy as np

from graphtoolbox.evaluation import (
    diebold_mariano, bootstrap_metric, model_confidence_set, pairwise_dm,
)


def _forecasts(seed=0):
    rng = np.random.default_rng(seed)
    T = 2000
    y = 500.0 + 100.0 * np.sin(np.arange(T) * 0.1)
    good = y + rng.standard_normal(T) * 2.0
    bad = y + rng.standard_normal(T) * 8.0
    good2 = y + np.random.default_rng(seed + 1).standard_normal(T) * 2.0
    return y, good, bad, good2


def test_dm_detects_clear_difference():
    y, good, bad, _ = _forecasts()
    r = diebold_mariano(good, bad, y, loss="squared", h=1, names=("good", "bad"))
    assert r.pvalue < 0.01
    assert r.better == "good"


def test_bootstrap_metric_interval_brackets_point():
    y, good, _, _ = _forecasts()
    r = bootstrap_metric(good, y, metric="rmse", n_boot=200, block_len=48, seed=0)
    assert r.se > 0
    assert r.ci_low <= r.point <= r.ci_high


def test_pairwise_dm_and_mcs_exclude_the_bad_model():
    y, good, bad, good2 = _forecasts()
    preds = {"good": good, "bad": bad, "good2": good2}

    pw = pairwise_dm(preds, y, loss="squared", h=1, correction="holm")
    assert pw["pvalue_adjusted"].loc["good", "bad"] < 0.05

    mcs = model_confidence_set(preds, y, loss="squared", alpha=0.10,
                               n_boot=300, block_len=48, seed=0)
    assert "bad" not in mcs.included
    assert "good" in mcs.included
