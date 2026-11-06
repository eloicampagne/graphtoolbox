"""Recurrent graph cells on net-load under GraphToolbox 0.3.0, unreconciled.

Same cells and configuration as examples/netload/train_temporal_netload.py
(48 NetLoad lags unrolled as a sequence, hidden 128, batch 16, lr 1e-3, 100 epochs,
patience 30, seed 42, CPU), with reconcile=False so that the cells stand on the same
footing as the net-load table. Weights go to ./checkpoints_netload_v030/temporal,
forecasts to ./netload_v030/temporal/. Resumable. With --leads, the cells also receive
the wind and cloud-cover trajectories of the forecast day as static context
(see netload_data_v030.py); outputs then go to ./netload_v030/leads_temporal/ and
./netload_temporal_v030_leads.json.
"""
import gc
import json
import os
import sys

import numpy as np
import torch
from torch_geometric import seed_everything

HERE = os.path.dirname(os.path.abspath(__file__))
NETLOAD = os.path.join(os.path.dirname(os.path.dirname(HERE)), "netload")
sys.path.insert(0, NETLOAD)
sys.path.insert(0, HERE)
os.chdir(NETLOAD)
import netload_pipeline as P  # noqa: E402
from graphtoolbox.models import TemporalGNN  # noqa: E402
from graphtoolbox.training import Trainer, set_device  # noqa: E402
from train_temporal_netload import temporal_cells, TEMPORAL_HIDDEN, NUM_EPOCHS, PATIENCE, BATCH_SIZE  # noqa: E402

set_device("cpu")
from netload_data_v030 import direct_datasets  # noqa: E402

LEADS = "--leads" in sys.argv
# --only NAME[,NAME] restricts the run to some cells (the paper reports GConvGRU with --leads).
ONLY = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
TAG = "leads_temporal" if LEADS else "temporal"
OUT = os.path.join(HERE, "netload_v030", TAG)
os.makedirs(OUT, exist_ok=True)
train, val, test = direct_datasets(LEADS)
for ds in (train, val, test):
    ds._set_adj_matrix(adj_matrix=P.ADJ)

summary = {}
for name, (cell, ckw) in temporal_cells().items():
    if ONLY and name not in ONLY:
        continue
    path = os.path.join(OUT, f"{name}.pt")
    if not os.path.exists(path):
        print(f"Training {name}", flush=True)
        seed_everything(42); gc.collect()
        model = TemporalGNN.from_features(train.features, cell_class=cell, hidden_channels=TEMPORAL_HIDDEN,
                                          out_channels=P.OUT_CHANNELS, target="NetLoad", cell_kwargs=ckw or None)
        trainer = Trainer(model=model, dataset_train=train, dataset_val=val, dataset_test=test,
                          batch_size=BATCH_SIZE, model_kwargs={"lr": 1e-3, "num_epochs": NUM_EPOCHS},
                          reconcile=False, lam_reg=0)
        pred, target = trainer.train(plot_loss=False, force_training=True, patience=PATIENCE,
                                     saving_directory=os.path.join(HERE, "checkpoints_netload_v030", TAG, name))[:2]
        torch.save(pred.sum(dim=0).cpu(), path)
        torch.save(target.sum(dim=0).cpu(), os.path.join(OUT, "target.pt"))
    p = torch.load(path, weights_only=False).numpy().astype(float)
    y = torch.load(os.path.join(OUT, "target.pt"), weights_only=False).numpy().astype(float)
    summary[name] = {"rmse": float(np.sqrt(np.mean((p - y) ** 2))),
                     "smape": float(100 * np.mean(2 * np.abs(p - y) / (np.abs(p) + np.abs(y))))}
    print(name, summary[name], flush=True)
json.dump(summary, open(os.path.join(HERE, "netload_temporal_v030" + ("_leads" if LEADS else "") + ".json"), "w"), indent=1)
