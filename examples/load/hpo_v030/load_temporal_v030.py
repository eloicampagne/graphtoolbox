"""Recurrent graph cells on load under GraphToolbox 0.3.0, one-shot, unreconciled.

One-shot: the cell performs a single update on the full node-feature vector (a
sequence of one step carrying every feature as a channel), as a static convolution
does, with the configuration of the static sweep (hidden 64, batch 32, lr 1e-3,
300 epochs, patience 30, DTW graph, seed 42, CPU). Forecasts go to
./preds/temporal_oneshot_{cell}.pt (regional, [nodes, half-hours]) and the national
RMSE to ./load_temporal_v030.json. Resumable.
"""
import gc
import json
import os
import sys

import numpy as np
import torch
from torch_geometric import seed_everything

HERE = os.path.dirname(os.path.abspath(__file__))
LOAD = os.path.dirname(HERE)
os.chdir(LOAD)
import pandas as pd  # noqa: E402
from graphtoolbox.data import DataClass, GraphDataset  # noqa: E402

# Datasets of the static sweep (copied from sweep_row.py).
data_kwargs = {
    'node_var': 'Region',
    'features_to_lag': {'load': (1, 48), 'temp': (-47, -1)},
    'dummies': ['Instant', 'RegionInt', 'JourSemaine', 'DayType', 'offset'],
    'day_inf_train': '2015-01-01', 'day_sup_train': '2018-01-01',
    'day_inf_val': '2018-01-01', 'day_sup_val': '2018-12-31',
    'day_inf_test': '2019-01-01', 'day_sup_test': '2019-12-31',
    'computed_features': {
        'heat_deg':  lambda df: (15 - df['temp']).clip(lower=0),
        'cool_deg':  lambda df: (df['temp'] - 15).clip(lower=0),
        'temp_sq':   lambda df: df['temp'] ** 2,
        'is_summer': lambda df: (pd.to_datetime(df['date']).dt.month == 7).astype(float)
                                + (pd.to_datetime(df['date']).dt.month == 8).astype(float),
    },
}
dataset_kwargs = {
    'batch_size': 32, 'adj_matrix': 'dtw',
    'features_base': ['temp', 'nebu', 'wind', 'tempMax', 'tempMin', 'Posan', 'Instant', 'RegionInt',
                      'JourSemaine', 'JourFerie', 'offset', 'DayType', 'Weekend', 'temp_liss_fort',
                      'temp_liss_faible', 'heat_deg', 'cool_deg', 'temp_sq', 'is_summer']
                     + [f'load_l{t}' for t in range(1, 49)] + [f'temp_l{t}' for t in range(-47, 0)],
    'target_base': 'load',
}
data = DataClass(path_train="./train.csv", path_test="./test.csv", data_kwargs=data_kwargs, folder_config=".")
common = dict(graph_folder="../graph_representations", dataset_kwargs=dataset_kwargs, out_channels=48)
train = GraphDataset(data=data, period="train", **common)
val = GraphDataset(data=data, period="val", scalers_feat=train.scalers_feat, scalers_target=train.scalers_target, **common)
test = GraphDataset(data=data, period="test", scalers_feat=train.scalers_feat, scalers_target=train.scalers_target, **common)
from graphtoolbox.models import TemporalGNN  # noqa: E402
from graphtoolbox.training import Trainer, set_device  # noqa: E402
from torch_geometric_temporal.nn.recurrent import GConvGRU, DCRNN, TGCN, A3TGCN  # noqa: E402

set_device("cpu")
for ds in (train, val, test):
    ds._set_adj_matrix(adj_matrix="dtw")
F = train.num_node_features
CELLS = {"GConvGRU": (GConvGRU, {"K": 2}), "DCRNN": (DCRNN, {"K": 2}), "TGCN": (TGCN, {}), "A3TGCN": (A3TGCN, {})}
summary = {}
for name, (cell, ckw) in CELLS.items():
    path = os.path.join(HERE, "preds", f"temporal_oneshot_{name}.pt")
    if not os.path.exists(path):
        print(f"Training {name} (one-shot)", flush=True)
        seed_everything(42); gc.collect()
        model = TemporalGNN(in_channels=F, hidden_channels=64, out_channels=48, cell_class=cell,
                            seq_idx=list(range(F)), static_idx=[], seq_channels=F, cell_kwargs=ckw or None)
        trainer = Trainer(model=model, dataset_train=train, dataset_val=val, dataset_test=test, batch_size=32,
                          model_kwargs={"lr": 1e-3, "num_epochs": 300}, reconcile=False, lam_reg=0)
        pred, target = trainer.train(force_training=True, patience=30, plot_loss=False,
                                     saving_directory=os.path.join(HERE, "checkpoints", "temporal_oneshot", name))[:2]
        torch.save(pred.cpu(), path)
        torch.save(target.cpu(), os.path.join(HERE, "preds", "temporal_oneshot_target.pt"))
    p = torch.load(path, weights_only=False).numpy().sum(0).astype(float)
    y = torch.load(os.path.join(HERE, "preds", "temporal_oneshot_target.pt"), weights_only=False).numpy().sum(0).astype(float)
    summary[name] = {"rmse": float(np.sqrt(np.mean((p - y) ** 2))), "mape": float(100 * np.mean(np.abs(p - y) / y))}
    print(name, summary[name], flush=True)
json.dump(summary, open(os.path.join(HERE, "load_temporal_v030.json"), "w"), indent=1)
