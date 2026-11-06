"""Net-load sweeps of the paper (direct and decomposed) under GraphToolbox 0.3.0, unreconciled.

Same ten convolutions and configuration as fair_sweep_netload.py and
decomp_sweep_netload.py (hidden 64, 1 layer, DTW graph, batch 16, lr 1e-3,
300 epochs, patience 40, seed 42), with reconcile=False so that the net-load
table reports forecasts on the same footing as the load table. Weights go to
./checkpoints_netload_v030, forecasts to ./netload_v030/. Resumable.

With --leads, every model also receives the wind and cloud-cover trajectories of the
forecast day (wind, wind_by_wind_power_weights, nebulosity,
nebulosity_by_solar_power_weights at the 48 half-hours, as the temperature already is),
the weather the per-half-hour GAM sees. Outputs then go to ./netload_v030/leads_seed{S}.

Usage: python netload_sweep_v030.py [--seed S] [--only direct|decomposed] [--convs A,B]
                                    [--comps Load,Wind_power,Solar_power] [--leads]
"""
import gc
import os
import sys

import numpy as np
import torch
from torch_geometric import seed_everything
from torch_geometric.nn.conv import (RGATConv, GatedGraphConv, SuperGATConv, FiLMConv, PANConv,
                                     GMMConv, GCN2Conv, GATConv, FastRGCNConv, ResGatedGraphConv)

HERE = os.path.dirname(os.path.abspath(__file__))
NETLOAD = os.path.join(os.path.dirname(os.path.dirname(HERE)), "netload")
sys.path.insert(0, NETLOAD)
sys.path.insert(0, HERE)
os.chdir(NETLOAD)
import netload_pipeline as P  # noqa: E402
import decomp_sweep_netload as D  # noqa: E402
from graphtoolbox.data.dataset import DataClass, GraphDataset  # noqa: E402
from graphtoolbox.models import myGNN  # noqa: E402
from graphtoolbox.training import Trainer  # noqa: E402

CONVS = {
    "RGATConv": (RGATConv, {"heads": 2}), "GatedGraphConv": (GatedGraphConv, {}),
    "SuperGATConv": (SuperGATConv, {}), "FiLMConv": (FiLMConv, {}), "PANConv": (PANConv, {}),
    "GMMConv": (GMMConv, {}), "GCN2Conv": (GCN2Conv, {}), "GATConv": (GATConv, {"heads": 2}),
    "FastRGCNConv": (FastRGCNConv, {}), "ResGatedGraphConv": (ResGatedGraphConv, {}),
}
arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
SEED = int(arg("--seed", 42))
ONLY = arg("--only")
if arg("--convs"):
    CONVS = {k: v for k, v in CONVS.items() if k in arg("--convs").split(",")}
LEADS = "--leads" in sys.argv
RUN = ("leads_" if LEADS else "") + f"seed{SEED}"
OUT = os.path.join(HERE, "netload_v030", RUN)
import netload_data_v030 as ND  # noqa: E402
from netload_data_v030 import direct_datasets  # noqa: E402

# --lead-vars restricts the weather trajectories, e.g. --lead-vars wind,nebulosity (unweighted only).
if arg("--lead-vars"):
    ND.LEAD_VARS = {k: v for k, v in ND.LEAD_VARS.items() if k in arg("--lead-vars").split(",")}
    RUN = "leads" + "-".join(sorted(ND.LEAD_VARS)) + f"_seed{SEED}"
    OUT = os.path.join(HERE, "netload_v030", RUN)
LEAD_VARS, lead_cols = ND.LEAD_VARS, ND.lead_cols
os.makedirs(OUT, exist_ok=True)


def fit(name, cls, kw, tr, va, te, tag):
    path = os.path.join(OUT, f"{tag}.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    print(f"Training {tag} (seed {SEED})", flush=True)
    seed_everything(SEED); gc.collect()
    model = myGNN(in_channels=tr.num_node_features, num_layers=1, hidden_channels=64,
                  out_channels=P.OUT_CHANNELS, conv_class=cls, conv_kwargs=kw)
    trainer = Trainer(model=model, dataset_train=tr, dataset_val=va, dataset_test=te, batch_size=16,
                      model_kwargs={"lr": 1e-3, "num_epochs": 300}, reconcile=False, lam_reg=0)
    out = trainer.train(force_training=True, save=False, patience=40, plot_loss=False,
                        saving_directory=os.path.join(HERE, "checkpoints_netload_v030", RUN, tag))
    pred = out[0].detach().cpu()
    torch.save(pred, path)
    tgt_path = os.path.join(OUT, f"target__{tag.split('__')[-1] if '__' in tag else 'NetLoad'}.pt")
    if not os.path.exists(tgt_path):
        torch.save(out[1].detach().cpu(), tgt_path)
    return pred


if ONLY in (None, "direct"):
    train, val, test = direct_datasets(LEADS)
    for ds in (train, val, test):
        ds._set_adj_matrix(adj_matrix="dtw")
    for name, (cls, kw) in CONVS.items():
        fit(name, cls, kw, train, val, test, name)

if ONLY in (None, "decomposed"):
    dkw = dict(D.data_kwargs)
    if LEADS:
        dkw["features_to_lag"] = {**D.data_kwargs["features_to_lag"], **{v: (-47, -1) for v in LEAD_VARS}}
    data = DataClass(path_train=str(P.HERE / "train.csv"), path_test=str(P.HERE / "test.csv"),
                     data_kwargs=dkw, folder_config=str(P.HERE))
    extra = {"Load": lead_cols({"wind", "cloud"}), "Wind_power": lead_cols({"wind"}),
             "Solar_power": lead_cols({"cloud"})} if LEADS else {}
    wanted = arg("--comps").split(",") if arg("--comps") else list(D.COMPONENTS)
    comps = {}
    for comp, feats in D.COMPONENTS.items():
        if comp not in wanted:
            continue
        feats = feats + extra.get(comp, [])
        dk = {"batch_size": 32, "adj_matrix": "dtw", "features_base": feats, "target_base": comp}
        common = dict(graph_folder=P.GRAPH_FOLDER, dataset_kwargs=dk, out_channels=P.OUT_CHANNELS)
        tr = GraphDataset(data=data, period="train", **common)
        va = GraphDataset(data=data, period="val", scalers_feat=tr.scalers_feat, scalers_target=tr.scalers_target, **common)
        te = GraphDataset(data=data, period="test", scalers_feat=tr.scalers_feat, scalers_target=tr.scalers_target, **common)
        for ds in (tr, va, te):
            ds._set_adj_matrix(adj_matrix="dtw")
        comps[comp] = (tr, va, te)
    for name, (cls, kw) in CONVS.items():
        for comp, dsets in comps.items():
            fit(name, cls, kw, *dsets, f"{name}__{comp}")
print("done", flush=True)
