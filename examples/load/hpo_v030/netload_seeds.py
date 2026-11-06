"""Five-seed retraining of the leading net-load models at the protocol of their table entries.

Direct GCN2Conv follows fair_sweep_netload.py and decomposed GatedGraphConv
follows decomp_sweep_netload.py (hidden 64, 1 layer, DTW graph, batch 16,
lr 1e-3, 300 epochs, patience 40, MinT with the gradient-boosting top level);
only the seed changes. Weights go to ./checkpoints_netload so that no table
checkpoint is overwritten. Writes ./netload_seeds.json.
"""
import gc
import json
import os
import sys

import numpy as np
import torch
from torch_geometric import seed_everything
from torch_geometric.nn.conv import GCN2Conv, GatedGraphConv

HERE = os.path.dirname(os.path.abspath(__file__))
NETLOAD = os.path.join(os.path.dirname(os.path.dirname(HERE)), "netload")
sys.path.insert(0, NETLOAD)
os.chdir(NETLOAD)
import netload_pipeline as P  # noqa: E402
import decomp_sweep_netload as D  # noqa: E402
from graphtoolbox.data.dataset import DataClass, GraphDataset  # noqa: E402
from graphtoolbox.models import myGNN  # noqa: E402
from graphtoolbox.training import Trainer  # noqa: E402

SEEDS = [42, 0, 1, 2, 3]
CACHE = os.path.join(HERE, "netload_seeds.json")
res = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
y = torch.load(os.path.join(NETLOAD, "results_all_convolutions_netload/target.pt"), weights_only=False).numpy()


def rmse(p):
    return float(np.sqrt(np.mean((p[:len(y)] - y) ** 2)))


def fit(cls, tr, va, te, seed, tag):
    seed_everything(seed); gc.collect()
    model = myGNN(in_channels=tr.num_node_features, num_layers=1, hidden_channels=64,
                  out_channels=P.OUT_CHANNELS, conv_class=cls, conv_kwargs={})
    trainer = Trainer(model=model, dataset_train=tr, dataset_val=va, dataset_test=te, batch_size=16,
                      model_kwargs={"lr": 1e-3, "num_epochs": 300}, reconcile=True,
                      top_level_model="xgb", lam_reg=0)
    out = trainer.train(force_training=True, save=False, patience=40,
                        saving_directory=os.path.join(HERE, "checkpoints_netload", tag))
    return out[0].sum(dim=0).cpu().numpy()


train, val, test, _ = P.build_datasets()
for ds in (train, val, test):
    ds._set_adj_matrix(adj_matrix="dtw")
for s in SEEDS:
    key = f"direct|GCN2Conv|s{s}"
    if key not in res:
        res[key] = rmse(fit(GCN2Conv, train, val, test, s, f"direct_GCN2Conv_s{s}"))
        json.dump(res, open(CACHE, "w"), indent=1)
    print(key, round(res[key], 1), flush=True)

data = DataClass(path_train=str(P.HERE / "train.csv"), path_test=str(P.HERE / "test.csv"),
                 data_kwargs=D.data_kwargs, folder_config=str(P.HERE))
comps = {}
for name, feats in D.COMPONENTS.items():
    dk = {"batch_size": 32, "adj_matrix": "dtw", "features_base": feats, "target_base": name}
    common = dict(graph_folder=P.GRAPH_FOLDER, dataset_kwargs=dk, out_channels=P.OUT_CHANNELS)
    tr = GraphDataset(data=data, period="train", **common)
    va = GraphDataset(data=data, period="val", scalers_feat=tr.scalers_feat, scalers_target=tr.scalers_target, **common)
    te = GraphDataset(data=data, period="test", scalers_feat=tr.scalers_feat, scalers_target=tr.scalers_target, **common)
    for ds in (tr, va, te):
        ds._set_adj_matrix(adj_matrix="dtw")
    comps[name] = (tr, va, te)
for s in SEEDS:
    key = f"decomposed|GatedGraphConv|s{s}"
    if key not in res:
        p = {c: fit(GatedGraphConv, *comps[c], s, f"decomp_GatedGraphConv_{c}_s{s}") for c in comps}
        res[key] = rmse(p["Load"] - p["Wind_power"] - p["Solar_power"])
        json.dump(res, open(CACHE, "w"), indent=1)
    print(key, round(res[key], 1), flush=True)
