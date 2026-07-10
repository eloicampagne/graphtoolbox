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


def test_optimizer_single_trial(tmp_path, monkeypatch):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
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

    optimizer = Optimizer(model=myGNN, dataset_train=train, dataset_val=val, out_channels=2,
                          conv_class=GCNConv, num_epochs=1,
                          optim_kwargs={"num_layers": (1, 2), "hidden_channels": (8, 16),
                                        "lr": (1e-3, 1e-2)})
    optimizer.optimize(n_trials=1)
    assert optimizer.is_optimized
    assert np.isfinite(optimizer.study.best_value)
    assert "num_layers" in optimizer.study.best_trial.params
