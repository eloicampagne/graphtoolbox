"""Seed variance of the best convolutions, at the paper configurations.

Retrains the strongest operators over five seeds to quantify seed-to-seed
variability of the national error and set it against the margins the paper
relies on: aggregation vs best-single on load, and direct vs decomposed on
net-load. This answers the single-run concern of the significance analysis.
"""
import os
import gc
import json
import numpy as np
import torch
from torch_geometric import seed_everything
from torch_geometric.nn.conv import LEConv, GatedGraphConv, GCN2Conv, FiLMConv

from graphtoolbox.data.dataset import DataClass, GraphDataset
from graphtoolbox.models import myGNN
from graphtoolbox.training import Trainer
import netload_pipeline as P
import decomp_sweep_netload as D   # component data_kwargs / feature lists

SEEDS = [42, 0, 1, 2, 3]
OUT = os.path.join(P.HERE, "results_seed_variance")
os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, "results.json")
res = json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


def build_load():
    import sys
    ld = str(P.HERE.parent / "load"); sys.path.insert(0, ld)
    import config as C
    data = DataClass(path_train=os.path.join(ld, "train.csv"), path_test=os.path.join(ld, "test.csv"),
                     data_kwargs=C.data_kwargs, folder_config=ld)
    common = dict(graph_folder=str(P.HERE.parent / "graph_representations"),
                  dataset_kwargs=C.dataset_kwargs, out_channels=48)
    tr = GraphDataset(data=data, period="train", **common)
    va = GraphDataset(data=data, period="val", scalers_feat=tr.scalers_feat, scalers_target=tr.scalers_target, **common)
    te = GraphDataset(data=data, period="test", scalers_feat=tr.scalers_feat, scalers_target=tr.scalers_target, **common)
    return tr, va, te


def build_component(comp):
    dk = {"batch_size": 32, "adj_matrix": "dtw", "features_base": D.COMPONENTS[comp], "target_base": comp}
    common = dict(graph_folder=P.GRAPH_FOLDER, dataset_kwargs=dk, out_channels=48)
    data = DataClass(path_train=str(P.HERE / "train.csv"), path_test=str(P.HERE / "test.csv"),
                     data_kwargs=D.data_kwargs, folder_config=str(P.HERE))
    tr = GraphDataset(data=data, period="train", **common)
    va = GraphDataset(data=data, period="val", scalers_feat=tr.scalers_feat, scalers_target=tr.scalers_target, **common)
    te = GraphDataset(data=data, period="test", scalers_feat=tr.scalers_feat, scalers_target=tr.scalers_target, **common)
    for ds in (tr, va, te):
        ds._set_adj_matrix(adj_matrix="dtw")
    return tr, va, te


def train_one(cls, kw, cfg, tr, va, te, seed):
    for ds in (tr, va, te):
        ds._set_adj_matrix(adj_matrix=cfg["adj"])
    seed_everything(seed); gc.collect()
    model = myGNN(in_channels=tr.num_node_features, num_layers=cfg["num_layers"],
                  hidden_channels=cfg["hidden"], out_channels=48, conv_class=cls, conv_kwargs=kw)
    trainer = Trainer(model=model, dataset_train=tr, dataset_val=va, dataset_test=te, batch_size=cfg["batch"],
                      model_kwargs={"lr": cfg["lr"], "num_epochs": cfg["epochs"]},
                      reconcile=True, top_level_model="xgb", lam_reg=0)
    r = trainer.train(force_training=True, save=False, patience=30)
    return r[0], r[1]  # pred [nodes,T], target [nodes,T]


STD = dict(num_layers=1, hidden=64, lr=1e-3, batch=16, adj="dtw", epochs=300)
LOAD_CFG = {
    "LEConv": (LEConv, {}, dict(num_layers=2, hidden=256, lr=1e-3, batch=16, adj="space", epochs=300)),
    "GatedGraphConv": (GatedGraphConv, {}, dict(num_layers=2, hidden=64, lr=1e-3, batch=32, adj="dtw", epochs=300)),
}
NET_DIRECT = {"GCN2Conv": (GCN2Conv, {}, STD), "FiLMConv": (FiLMConv, {}, STD)}
NET_DECOMP = {"FiLMConv": (FiLMConv, {}, STD)}


def run_direct(tag, convs, datasets):
    tr, va, te = datasets
    for cname, (cls, kw, cfg) in convs.items():
        for s in SEEDS:
            key = f"{tag}|{cname}|s{s}"
            if key in res:
                continue
            print(f"[{key}] {cfg['hidden']}h/{cfg['adj']}/{cfg['epochs']}ep")
            try:
                pred, tgt = train_one(cls, kw, cfg, tr, va, te, s)
                res[key] = rmse(pred.sum(0).cpu().numpy(), tgt.sum(0).cpu().numpy())
                print(f"  RMSE = {res[key]:.0f}"); json.dump(res, open(CACHE, "w"), indent=1)
            except Exception as e:
                print(f"  failed: {str(e).splitlines()[0]}")


def run_decomp(convs):
    comps = ["Load", "Wind_power", "Solar_power"]
    ds = {c: build_component(c) for c in comps}
    ynodes = torch.load(os.path.join(P.HERE, "results_all_convolutions_netload/target_nodes.pt"), weights_only=False)
    ynat = ynodes.sum(0).numpy()
    for cname, (cls, kw, cfg) in convs.items():
        for s in SEEDS:
            key = f"netdecomp|{cname}|s{s}"
            if key in res:
                continue
            print(f"[{key}] decomposed {cfg['hidden']}h/{cfg['epochs']}ep")
            try:
                net = np.zeros(len(ynat))
                for comp in comps:
                    pred, _ = train_one(cls, kw, cfg, *ds[comp], s)
                    net = net + pred.sum(0).cpu().numpy() * (1 if comp == "Load" else -1)
                res[key] = rmse(net[:len(ynat)], ynat)
                print(f"  net-load RMSE = {res[key]:.0f}"); json.dump(res, open(CACHE, "w"), indent=1)
            except Exception as e:
                print(f"  failed: {str(e).splitlines()[0]}")


def summary():
    print("\n===== SEED VARIANCE (national RMSE, MW, 5 seeds) =====")
    groups = [("load", LOAD_CFG), ("netdirect", NET_DIRECT), ("netdecomp", NET_DECOMP)]
    for tag, convs in groups:
        for cname in convs:
            vals = [res[f"{tag}|{cname}|s{s}"] for s in SEEDS if f"{tag}|{cname}|s{s}" in res]
            if vals:
                print(f"{tag:<10} {cname:<16} mean {np.mean(vals):.0f}  std {np.std(vals, ddof=1):.0f}  "
                      f"[{min(vals):.0f}, {max(vals):.0f}]  (n={len(vals)})")


def main():
    print("=== LOAD ==="); run_direct("load", LOAD_CFG, build_load())
    print("\n=== NET-LOAD DIRECT ==="); run_direct("netdirect", NET_DIRECT, P.build_datasets()[:3])
    print("\n=== NET-LOAD DECOMPOSED ==="); run_decomp(NET_DECOMP)
    summary()


if __name__ == "__main__":
    main()
