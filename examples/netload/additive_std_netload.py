"""AdditiveGraphModel on direct net-load at the standardized config (h64/l1/300ep).

The framework's intrinsically-interpretable model: contiguous feature groups
(NetLoad lags / temperature lags / exogenous / calendar) each get their own
subnetwork and the prediction is their sum. Run at the same configuration as the
direct and decomposed sweeps, so it slots into the net-load comparison as an
additive GNN, the bridge between the plain GNN and the GAM.
"""
import os
import re
import gc
import numpy as np
import torch
from torch_geometric import seed_everything
from torch_geometric.nn.conv import GCNConv

from graphtoolbox.models import AdditiveGraphModel
from graphtoolbox.training import Trainer, set_device
import netload_pipeline as P

HIDDEN, LAYERS, EPOCHS, ADJ = 64, 1, 300, "dtw"
OUT = os.path.join(P.HERE, "results_std_additive_netload")
os.makedirs(OUT, exist_ok=True)


def agm_kind(f):
    if re.fullmatch(r"NetLoad_l\d+", f):        return "netload_lags"
    if re.fullmatch(r"temperature_l-?\d+", f):  return "temp_lags"
    return "exogenous"


def contiguous_groups(features):
    runs = []
    for f in features:
        k = agm_kind(f)
        if runs and runs[-1][0] == k:
            runs[-1][1].append(f)
        else:
            runs.append([k, [f]])
    n_exo = sum(1 for k, _ in runs if k == "exogenous")
    groups, e = {}, 0
    for k, fs in runs:
        if k == "exogenous":
            e += 1
            name = "exogenous" if n_exo == 1 else ("exogenous" if e == 1 else "calendar_dummies")
        else:
            name = k
        groups[name] = groups.get(name, 0) + len(fs)
    return groups


def smape(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return 100.0 * np.mean(2 * np.abs(p - y) / (np.abs(p) + np.abs(y) + 1e-8))


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


def main():
    set_device("cpu")
    train, val, test, nodes = P.build_datasets()
    for ds in (train, val, test):
        ds._set_adj_matrix(adj_matrix=ADJ)
    groups = contiguous_groups(train.features)
    assert sum(groups.values()) == train.num_node_features, "groups must tile all features"
    print("Feature groups:", groups)

    seed_everything(42); gc.collect()
    model = AdditiveGraphModel(feature_group_dims=groups, num_layers=LAYERS, hidden_channels=HIDDEN,
                               out_channels=P.OUT_CHANNELS, conv_class=GCNConv, conv_kwargs={})
    trainer = Trainer(model=model, dataset_train=train, dataset_val=val, dataset_test=test,
                      batch_size=16, model_kwargs={"lr": 1e-3, "num_epochs": EPOCHS},
                      reconcile=True, top_level_model="xgb", lam_reg=0)
    res = trainer.train(force_training=True, save=False, patience=30)
    pred, target = res[0], res[1]
    torch.save(pred, os.path.join(OUT, "AdditiveGraphModel.pt"))
    pn, y = pred.sum(dim=0).cpu().numpy(), target.sum(dim=0).cpu().numpy()
    print(f"\nAdditiveGraphModel (direct, h64/l1/300ep)  sMAPE {smape(pn, y):.2f}%  RMSE {rmse(pn, y):.0f} MW")
    print("References: best direct conv 1969 ; direct MLpol 1898 ; decomposed MLpol 1739 ; GAM nat 1763 ; GAM reg 1723")


if __name__ == "__main__":
    main()
