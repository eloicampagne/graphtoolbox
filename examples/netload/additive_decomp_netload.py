"""Decomposed AdditiveGraphModel for net-load (one AGM per component).

The additive-GNN analogue of the decomposed sweep and of the component GAM: one
AdditiveGraphModel per physical component (Load / Wind / Solar) at the
standardized config (h64/l1/300ep), recombined to national net-load.
"""
import os
import re
import gc
import numpy as np
import torch
from torch_geometric import seed_everything
from torch_geometric.nn.conv import GCNConv

from graphtoolbox.data.dataset import DataClass, GraphDataset
from graphtoolbox.models import AdditiveGraphModel
from graphtoolbox.training import Trainer, set_device
import netload_pipeline as P
import decomp_sweep_netload as D

HIDDEN, LAYERS, EPOCHS, ADJ = 64, 1, 300, "dtw"
OUT = os.path.join(P.HERE, "results_std_additive_decomp_netload")
os.makedirs(OUT, exist_ok=True)


def groups_for(features, comp):
    def kind(f):
        if re.fullmatch(rf"{comp}_l\d+", f):        return f"{comp}_lags"
        if re.fullmatch(r"temperature_l-?\d+", f):  return "temp_lags"
        return "exogenous"
    runs = []
    for f in features:
        k = kind(f)
        if runs and runs[-1][0] == k:
            runs[-1][1].append(f)
        else:
            runs.append([k, [f]])
    n_exo = sum(1 for k, _ in runs if k == "exogenous")
    g, e = {}, 0
    for k, fs in runs:
        nm = k if k != "exogenous" else ("exogenous" if n_exo == 1 else ("exogenous" if (e := e + 1) == 1 else "calendar"))
        g[nm] = g.get(nm, 0) + len(fs)
    return g


def smape(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return 100.0 * np.mean(2 * np.abs(p - y) / (np.abs(p) + np.abs(y) + 1e-8))


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


def main():
    set_device("cpu")
    data = DataClass(path_train=str(P.HERE / "train.csv"), path_test=str(P.HERE / "test.csv"),
                     data_kwargs=D.data_kwargs, folder_config=str(P.HERE))
    ynodes = torch.load(os.path.join(P.HERE, "results_all_convolutions_netload/target_nodes.pt"), weights_only=False)
    ynat = ynodes.sum(0).numpy(); Tt = len(ynat)
    net = np.zeros(Tt)
    for comp in ("Load", "Wind_power", "Solar_power"):
        dk = {"batch_size": 32, "adj_matrix": ADJ, "features_base": D.COMPONENTS[comp], "target_base": comp}
        common = dict(graph_folder=P.GRAPH_FOLDER, dataset_kwargs=dk, out_channels=P.OUT_CHANNELS)
        tr = GraphDataset(data=data, period="train", **common)
        va = GraphDataset(data=data, period="val", scalers_feat=tr.scalers_feat, scalers_target=tr.scalers_target, **common)
        te = GraphDataset(data=data, period="test", scalers_feat=tr.scalers_feat, scalers_target=tr.scalers_target, **common)
        for ds in (tr, va, te):
            ds._set_adj_matrix(adj_matrix=ADJ)
        g = groups_for(tr.features, comp)
        assert sum(g.values()) == tr.num_node_features, f"{comp}: groups must tile ({sum(g.values())} vs {tr.num_node_features})"
        print(f"{comp}: groups {g}")
        seed_everything(42); gc.collect()
        model = AdditiveGraphModel(feature_group_dims=g, num_layers=LAYERS, hidden_channels=HIDDEN,
                                   out_channels=P.OUT_CHANNELS, conv_class=GCNConv, conv_kwargs={})
        trainer = Trainer(model=model, dataset_train=tr, dataset_val=va, dataset_test=te, batch_size=16,
                          model_kwargs={"lr": 1e-3, "num_epochs": EPOCHS}, reconcile=True, top_level_model="xgb", lam_reg=0)
        res = trainer.train(force_training=True, save=False, patience=30)
        pred = res[0]
        torch.save(pred, os.path.join(OUT, f"{comp}.pt"))
        pn = pred.sum(0).cpu().numpy()
        net = net + pn * (1 if comp == "Load" else -1)
        print(f"  {comp:<12} national RMSE {rmse(pn, res[1].sum(0).cpu().numpy()):.0f} MW")

    net = net[:Tt]
    print("\n=== Decomposed AdditiveGraphModel, net-load, 2019 test ===")
    print(f"AGM decomposed (one per component, h64/l1/300ep)  sMAPE {smape(net, ynat):.2f}%  RMSE {rmse(net, ynat):.0f} MW")
    print("References: AGM direct 2353 ; decomposed GNN best 1762 / MLpol 1739 ; GAM nat 1763 / reg 1723")


if __name__ == "__main__":
    main()
