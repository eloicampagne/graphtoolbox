"""A single-trial Optuna run over the synthetic pipeline (fast smoke test)."""
import numpy as np
import pandas as pd
from torch_geometric.nn.conv import GCNConv

from graphtoolbox.data import DataClass, GraphDataset
from graphtoolbox.models import myGNN
from graphtoolbox.optim import Optimizer
from graphtoolbox.training import set_device


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


def _datasets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    set_device("cpu")
    # The objective samples adj_matrix from the graph folder, so it must exist;
    # the empty 'eye' subdir triggers the identity-matrix fallback.
    (tmp_path / "graphs" / "eye").mkdir(parents=True)

    _panel(pd.date_range("2020-01-01", periods=60, freq="D"), 0).to_csv("train.csv", index=False)
    _panel(pd.date_range("2020-03-05", periods=15, freq="D"), 1).to_csv("test.csv", index=False)

    data_kwargs = {"node_var": "node", "dummies": [], "features_to_lag": {"load": (1, 2)},
                   "day_inf_train": "2020-01-01", "day_sup_train": "2020-02-10",
                   "day_inf_val": "2020-02-11", "day_sup_val": "2020-02-29",
                   "day_inf_test": "2020-03-05", "day_sup_test": "2020-03-19"}
    dataset_kwargs = {"batch_size": 8, "adj_matrix": "eye", "target_base": "load",
                      "features_base": ["temp", "load_l1", "load_l2"]}
    data = DataClass("train.csv", "test.csv", data_kwargs=data_kwargs, folder_config=".")
    common = dict(graph_folder="graphs", dataset_kwargs=dataset_kwargs, out_channels=2)
    train = GraphDataset(data=data, period="train", **common)
    val = GraphDataset(data=data, period="val", scalers_feat=train.scalers_feat,
                       scalers_target=train.scalers_target, **common)
    return train, val


def _optimizer(train, val, **kwargs):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return Optimizer(model=myGNN, dataset_train=train, dataset_val=val, out_channels=2,
                     conv_class=GCNConv,
                     optim_kwargs={"num_layers": (1, 2), "hidden_channels": (8, 16),
                                   "lr": (1e-3, 1e-2)}, **kwargs)


def test_optimizer_single_trial(tmp_path, monkeypatch):
    train, val = _datasets(tmp_path, monkeypatch)
    optimizer = _optimizer(train, val, num_epochs=1)
    optimizer.optimize(n_trials=1)
    assert optimizer.is_optimized
    assert np.isfinite(optimizer.study.best_value)
    assert "num_layers" in optimizer.study.best_trial.params


def test_epoch_runs_every_batch(tmp_path, monkeypatch):
    # The epoch loop used to return after its first batch.
    import torch
    from torch_geometric.loader import DataLoader
    train, val = _datasets(tmp_path, monkeypatch)
    opt = _optimizer(train, val)
    opt.model = myGNN(in_channels=train.num_node_features, num_layers=1, hidden_channels=8,
                      out_channels=2, conv_class=GCNConv)
    calls = []
    opt.model.register_forward_hook(lambda *_: calls.append(1))
    loader = DataLoader(train, batch_size=2, shuffle=False)
    opt._run_epoch(torch.optim.Adam(opt.model.parameters(), 1e-3), "train", loader)
    assert len(loader) > 1 and len(calls) == len(loader)


def test_patience_stops_trial(tmp_path, monkeypatch):
    # With min_delta larger than any change, only the first epoch improves:
    # the trial stops patience epochs later and saves that epoch's weights once.
    train, val = _datasets(tmp_path, monkeypatch)
    opt = _optimizer(train, val, num_epochs=50, patience=2, min_delta=1e9)
    opt.optimize(n_trials=1)
    trial = opt.study.trials[0]
    assert trial.user_attrs == {"epochs": 3, "best_epoch": 1}
    saved = list((tmp_path / "checkpoints_optim").rglob("*.params"))
    assert [p.name for p in saved] == ["epoch0.params"]


def test_pruner_is_passed(tmp_path, monkeypatch):
    import optuna
    train, val = _datasets(tmp_path, monkeypatch)
    opt = _optimizer(train, val, num_epochs=1)
    pruner = optuna.pruners.NopPruner()
    opt.optimize(n_trials=1, pruner=pruner)
    assert opt.study.pruner is pruner


def test_timeout_warns(tmp_path, monkeypatch):
    import pytest
    train, val = _datasets(tmp_path, monkeypatch)
    opt = _optimizer(train, val, num_epochs=1)
    # timeout=0 lets no trial start: the study must warn, not fail on best_trial.
    with pytest.warns(UserWarning) as record:
        opt.optimize(n_trials=3, timeout=0)
    messages = [str(w.message) for w in record]
    assert any("0 of 3 trials" in m for m in messages)
    assert any("No trial completed" in m for m in messages)
