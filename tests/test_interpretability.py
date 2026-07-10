"""Tests for the ALE interpretability functions.

The array-level ALE computations and their plots are exercised directly on
synthetic tensors, and the group-level feature-importance path is run end to end
on a small synthetic panel with fake per-group contributions. The heavier
explanation-graph maps (which require Basemap) are out of scope here.
"""
import matplotlib
matplotlib.use("Agg")  # headless backend for CI

import numpy as np
import pandas as pd
import pytest
import torch

from graphtoolbox.interpretability import (
    compute_ALE, compute_ALE_avg_over_instants, compute_ALE_per_node,
    compute_ALE_per_node_avg_over_instants, ale_scalar_importance,
    plot_ALE, plot_ALE_avg, plot_ALE_nodes, plot_feature_importance_bar,
    get_group_feature_mats, compute_feature_importances_from_ALE,
)
from graphtoolbox.data import DataClass, GraphDataset


def _values_contrib(n=6, T=96, seed=0):
    rng = np.random.default_rng(seed)
    values = torch.tensor(rng.uniform(0, 10, size=(n, T)), dtype=torch.float32)
    contrib = torch.tanh(values) + torch.tensor(rng.standard_normal((n, T)) * 0.1, dtype=torch.float32)
    return values, contrib


def test_compute_ale_shapes_and_centering():
    values, contrib = _values_contrib()
    x, y, x_mid, ale = compute_ALE(values, contrib, n_bins=20)
    assert x.shape == y.shape == (values.numel(),)
    assert x_mid.shape == ale.shape
    assert abs(float(ale.mean())) < 1e-5          # ALE is centered


def test_compute_ale_variants_run():
    values, contrib = _values_contrib(T=96)
    xm, mean, std = compute_ALE_avg_over_instants(values, contrib, n_bins=10, period=48)
    assert xm.shape == mean.shape
    xmn, mat, counts = compute_ALE_per_node(values, contrib, n_bins=10)
    assert mat.shape[0] == values.shape[0]
    xmp, node_ale, counts2 = compute_ALE_per_node_avg_over_instants(
        values, contrib, n_bins=10, period=48)
    assert node_ale.shape[0] == values.shape[0]


@pytest.mark.parametrize("method", ["rms", "range", "tv"])
def test_ale_scalar_importance(method):
    _, _, x_mid, ale = compute_ALE(*_values_contrib(), n_bins=20)
    imp = ale_scalar_importance(ale, counts=None, method=method)
    assert np.isfinite(imp)


def test_plots_do_not_raise():
    values, contrib = _values_contrib()
    x, y, x_mid, ale = compute_ALE(values, contrib, n_bins=15)
    plot_ALE(x, y, x_mid, ale, label="temp")
    xm, mean, std = compute_ALE_avg_over_instants(values, contrib, n_bins=10, period=48)
    plot_ALE_avg(xm, mean, std, period=48, label="temp")
    xmn, mat, counts = compute_ALE_per_node(values, contrib, n_bins=10)
    plot_ALE_nodes(xmn, mat, counts=counts)
    df = pd.DataFrame({"feature": ["a", "b", "c"], "importance": [3.0, 1.0, 2.0]})
    plot_feature_importance_bar(df, top_k=2)
    matplotlib.pyplot.close("all")


# --------------------------------------------------------------------------- #
# End-to-end feature importance on a small synthetic panel
# --------------------------------------------------------------------------- #
def _panel(dates, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for j, node in enumerate(["A", "B", "C"]):
        temp = 15 + 5 * np.sin(np.arange(len(dates)) * 0.2) + rng.standard_normal(len(dates))
        load = 50 + 5 * j + 0.8 * temp + rng.standard_normal(len(dates))
        for i, d in enumerate(dates):
            rows.append({"date": str(d.date()), "node": node,
                         "temp": float(temp[i]), "load": float(load[i])})
    return pd.DataFrame(rows)


def test_feature_importances_from_ale_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _panel(pd.date_range("2020-01-01", periods=60, freq="D"), 0).to_csv("train.csv", index=False)
    _panel(pd.date_range("2020-03-05", periods=20, freq="D"), 1).to_csv("test.csv", index=False)

    data_kwargs = {"node_var": "node", "dummies": [], "features_to_lag": {"load": (1, 2)},
                   "day_inf_train": "2020-01-01", "day_sup_train": "2020-02-10",
                   "day_inf_val": "2020-02-11", "day_sup_val": "2020-02-29",
                   "day_inf_test": "2020-03-05", "day_sup_test": "2020-03-24"}
    feature_groups = {"weather": ["temp"], "lags": ["load_l1", "load_l2"]}
    dataset_kwargs = {"batch_size": 8, "adj_matrix": "eye", "target_base": "load",
                      "features_base": ["temp", "load_l1", "load_l2"],
                      "feature_groups": feature_groups}
    data = DataClass(path_train="train.csv", path_test="test.csv",
                     data_kwargs=data_kwargs, folder_config=".")
    common = dict(graph_folder="./no_graphs", dataset_kwargs=dataset_kwargs, out_channels=2)
    train = GraphDataset(data=data, period="train", **common)
    test = GraphDataset(data=data, period="test", scalers_feat=train.scalers_feat,
                        scalers_target=train.scalers_target, **common)

    # get_group_feature_mats resolves feature columns from data.df_test
    mats = get_group_feature_mats("weather", data, train, test)
    assert "temp" in mats

    n_nodes, T = test.num_nodes, mats["temp"].shape[1]
    group_outputs = {g: torch.randn(n_nodes, T) for g in feature_groups}
    df = compute_feature_importances_from_ALE(group_outputs, data, train, test,
                                              n_bins=10, mode="global")
    assert not df.empty
    assert set(df["group"]) <= set(feature_groups)
    assert (df["importance"] >= 0).all()
