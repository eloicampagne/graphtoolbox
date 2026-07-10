"""End-to-end integration test on a synthetic panel.

Exercises the full pipeline with no external files: a synthetic CSV is written to a
temp directory, read by DataClass, turned into GraphDatasets (identity graph, since
no adjacency file exists), and trained through the Trainer, with and without MinT
reconciliation. This covers the data path, the training loop, metric evaluation,
and the reconciliation projection in one pass.
"""
import numpy as np
import pandas as pd
import pytest
from torch_geometric.nn.conv import GCNConv, GATConv
from torch_geometric import seed_everything

from graphtoolbox.data import DataClass, GraphDataset
from graphtoolbox.models import myGNN
from graphtoolbox.training import Trainer, RollingTrainer, set_device

NODES = ["A", "B", "C"]
OUT_CHANNELS = 2


def _panel(dates, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for j, node in enumerate(NODES):
        temp = 15 + 5 * np.sin(np.arange(len(dates)) * 0.2) + rng.standard_normal(len(dates))
        load = 50 + 5 * j + 0.8 * temp + rng.standard_normal(len(dates))
        for i, d in enumerate(dates):
            rows.append({"date": str(d.date()), "node": node,
                         "temp": float(temp[i]), "load": float(load[i])})
    return pd.DataFrame(rows)


@pytest.fixture
def datasets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)          # keep checkpoints / files out of the repo
    set_device("cpu")
    seed_everything(0)
    _panel(pd.date_range("2020-01-01", periods=60, freq="D"), 0).to_csv("train.csv", index=False)
    _panel(pd.date_range("2020-03-05", periods=15, freq="D"), 1).to_csv("test.csv", index=False)

    data_kwargs = {
        "node_var": "node", "dummies": [], "features_to_lag": {"load": (1, 2)},
        "day_inf_train": "2020-01-01", "day_sup_train": "2020-02-10",
        "day_inf_val": "2020-02-11", "day_sup_val": "2020-02-29",
        "day_inf_test": "2020-03-05", "day_sup_test": "2020-03-19",
    }
    dataset_kwargs = {
        "batch_size": 8, "adj_matrix": "eye",
        "features_base": ["temp", "load_l1", "load_l2"], "target_base": "load",
    }
    data = DataClass(path_train="train.csv", path_test="test.csv",
                     data_kwargs=data_kwargs, folder_config=".")
    common = dict(graph_folder="./no_graphs", dataset_kwargs=dataset_kwargs,
                  out_channels=OUT_CHANNELS)
    train = GraphDataset(data=data, period="train", **common)
    val = GraphDataset(data=data, period="val", scalers_feat=train.scalers_feat,
                       scalers_target=train.scalers_target, **common)
    test = GraphDataset(data=data, period="test", scalers_feat=train.scalers_feat,
                        scalers_target=train.scalers_target, **common)
    return train, val, test


def test_graphdataset_shapes(datasets):
    train, val, test = datasets
    assert train.num_nodes == len(NODES)
    assert train.num_node_features == 3          # temp, load_l1, load_l2
    assert len(train) > 0 and len(test) > 0
    sample = train[0]
    assert sample.x.shape == (len(NODES), 3)
    assert sample.y.shape == (len(NODES), OUT_CHANNELS)


def _train_once(datasets, reconcile, top_level_model=None):
    train, val, test = datasets
    seed_everything(0)
    model = myGNN(in_channels=train.num_node_features, num_layers=1,
                  hidden_channels=8, out_channels=OUT_CHANNELS, conv_class=GCNConv)
    trainer = Trainer(model=model, dataset_train=train, dataset_val=val, dataset_test=test,
                      batch_size=8, model_kwargs={"lr": 1e-2, "num_epochs": 1},
                      reconcile=reconcile, top_level_model=top_level_model)
    pred, target, *_ = trainer.train(force_training=True, patience=1)
    return pred, target


def test_pipeline_trains_without_reconciliation(datasets):
    pred, target = _train_once(datasets, reconcile=False)
    assert pred.shape == target.shape
    assert pred.shape[0] == len(NODES)
    assert bool(np.isfinite(pred.detach().cpu().numpy()).all())


def test_pipeline_trains_with_mint_reconciliation(datasets):
    pred, target = _train_once(datasets, reconcile=True, top_level_model="ridge")
    assert pred.shape[0] == len(NODES)
    assert bool(np.isfinite(pred.detach().cpu().numpy()).all())


def test_pipeline_collects_attention(datasets):
    train, val, test = datasets
    seed_everything(0)
    model = myGNN(in_channels=train.num_node_features, num_layers=1, hidden_channels=8,
                  out_channels=OUT_CHANNELS, conv_class=GATConv,
                  conv_kwargs={"heads": 2}, heads=2)
    trainer = Trainer(model=model, dataset_train=train, dataset_val=val, dataset_test=test,
                      batch_size=8, model_kwargs={"lr": 1e-2, "num_epochs": 1},
                      reconcile=False, return_attention=True)
    # with return_attention, train() returns (preds, targets, edge_index, attention_maps)
    pred, target, edge_index, attention = trainer.train(force_training=True, patience=1)
    assert pred.shape[0] == len(NODES)
    assert bool(np.isfinite(pred.detach().cpu().numpy()).all())


def test_rolling_trainer_rolls_over_windows(datasets):
    train, val, test = datasets
    set_device("cpu")
    seed_everything(0)
    model_kwargs = dict(in_channels=train.num_node_features, num_layers=1,
                        hidden_channels=8, out_channels=OUT_CHANNELS, conv_class=GCNConv)
    roller = RollingTrainer(
        dataset_train_0=train, dataset_val_0=val, dataset_test_full=test,
        model_class=myGNN, model_kwargs=model_kwargs,
        window_size=6, step_size=6, batch_size=8, reconcile=False,
        num_epochs_initial=1, num_epochs_update=1,
        trainer_kwargs={"lr": 1e-2, "force_training": True, "patience": 1},
    )
    results = roller.run()
    assert len(results) >= 2                     # expanding-window rolling forecast
    for r in results:
        assert r["preds"].shape[0] == len(NODES)
        assert bool(np.isfinite(r["preds"].detach().cpu().numpy()).all())
