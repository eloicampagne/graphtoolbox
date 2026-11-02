"""Seeded runs replay identically on CPU: the Trainer and the Optuna search."""
import numpy as np
import pandas as pd
import torch
from torch_geometric.nn.conv import GCNConv

from graphtoolbox.data import DataClass, GraphDataset
from graphtoolbox.models import myGNN
from graphtoolbox.optim import Optimizer
from graphtoolbox.training import Trainer, set_device
from graphtoolbox.utils import set_seed


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


def _datasets(tmp_path, monkeypatch, graph_folder):
    monkeypatch.chdir(tmp_path)
    set_device("cpu")
    _panel(pd.date_range("2020-01-01", periods=60, freq="D"), 0).to_csv("train.csv", index=False)
    _panel(pd.date_range("2020-03-05", periods=15, freq="D"), 1).to_csv("test.csv", index=False)
    data_kwargs = {"node_var": "node", "dummies": [], "features_to_lag": {"load": (1, 2)},
                   "day_inf_train": "2020-01-01", "day_sup_train": "2020-02-10",
                   "day_inf_val": "2020-02-11", "day_sup_val": "2020-02-29",
                   "day_inf_test": "2020-03-05", "day_sup_test": "2020-03-19"}
    dataset_kwargs = {"batch_size": 8, "adj_matrix": "eye", "target_base": "load",
                      "features_base": ["temp", "load_l1", "load_l2"]}
    data = DataClass("train.csv", "test.csv", data_kwargs=data_kwargs, folder_config=".")
    common = dict(graph_folder=graph_folder, dataset_kwargs=dataset_kwargs, out_channels=2)
    train = GraphDataset(data=data, period="train", **common)
    val = GraphDataset(data=data, period="val", scalers_feat=train.scalers_feat,
                       scalers_target=train.scalers_target, **common)
    test = GraphDataset(data=data, period="test", scalers_feat=train.scalers_feat,
                        scalers_target=train.scalers_target, **common)
    return train, val, test


def test_set_seed_replays_generators():
    set_seed(3)
    a = (torch.rand(4), np.random.rand(4))
    set_seed(3)
    b = (torch.rand(4), np.random.rand(4))
    assert torch.equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_seeded_trainer_replays(tmp_path, monkeypatch):
    train, val, test = _datasets(tmp_path, monkeypatch, "./no_graphs")

    def run():
        set_seed(7)
        model = myGNN(in_channels=train.num_node_features, num_layers=1,
                      hidden_channels=8, out_channels=2, conv_class=GCNConv)
        trainer = Trainer(model=model, dataset_train=train, dataset_val=val, dataset_test=test,
                          batch_size=4, model_kwargs={"lr": 1e-2, "num_epochs": 3},
                          reconcile=False, seed=7)
        pred, *_ = trainer.train(force_training=True, patience=3)
        return pred.detach().cpu()

    assert torch.equal(run(), run())


def test_seeded_optimizer_replays(tmp_path, monkeypatch):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    (tmp_path / "graphs" / "eye").mkdir(parents=True)
    train, val, _ = _datasets(tmp_path, monkeypatch, "graphs")

    def run():
        opt = Optimizer(model=myGNN, dataset_train=train, dataset_val=val, out_channels=2,
                        conv_class=GCNConv, num_epochs=2, seed=11,
                        optim_kwargs={"num_layers": (1, 2), "hidden_channels": (8, 16),
                                      "lr": (1e-3, 1e-2)})
        opt.optimize(n_trials=3)
        return [(t.params, t.value) for t in opt.study.trials]

    assert run() == run()


def test_seeded_rolling_trainer_replays(tmp_path, monkeypatch):
    from graphtoolbox.training import RollingTrainer
    train, val, test = _datasets(tmp_path, monkeypatch, "./no_graphs")

    def run():
        # No set_seed here: the rolling trainer builds the model and seeds it.
        torch.manual_seed(np.random.SeedSequence().entropy % 2**32)
        rolling = RollingTrainer(train, val, test, model_class=myGNN,
                                 model_kwargs=dict(in_channels=train.num_node_features,
                                                   num_layers=1, hidden_channels=8,
                                                   out_channels=2, conv_class=GCNConv),
                                 window_size=5, step_size=5, batch_size=4, reconcile=False,
                                 trainer_kwargs={"seed": 5, "force_training": True,
                                                 "patience": 2},
                                 num_epochs_initial=2, num_epochs_update=1)
        return [r["preds"].detach().cpu() for r in rolling.run()]

    a, b = run(), run()
    assert len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))
