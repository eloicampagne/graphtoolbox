"""Additive graph model on net-load under 0.3.0, unreconciled, with first-order ALE
(graphtoolbox.interpretability.additive_ale) importances and curves.

Same model and groups as netload/train_additive_ale_netload.py (AdditiveGraphModel,
GCNConv, 2 layers of width 64, DTW graph, batch 32, lr 1e-3, 300 epochs,
patience 30, seed 42), trained into ./checkpoints_additive_v030 with
reconcile=False. Writes ./additive_v030/{ale_importance.csv, ale_group_importance.csv,
ale_curves.npz, group_contributions.npz, summary.json}. An existing checkpoint is reused
unless --retrain is given. With --leads, the model also receives the wind and
cloud-cover trajectories of the forecast day (netload_data_v030.py), which form two
further feature groups, and outputs go to ./additive_v030_leads/.
"""
import gc
import json
import os
import sys

import numpy as np
import torch
from torch_geometric import seed_everything
from torch_geometric.nn.conv import GCNConv

HERE = os.path.dirname(os.path.abspath(__file__))
NETLOAD = os.path.join(os.path.dirname(os.path.dirname(HERE)), "netload")
sys.path.insert(0, NETLOAD)
os.chdir(NETLOAD)
import netload_pipeline as P  # noqa: E402
import re  # noqa: E402
import train_additive_ale_netload as A  # noqa: E402
from train_additive_ale_netload import DUMMIES  # noqa: E402
from graphtoolbox.models import AdditiveGraphModel  # noqa: E402
from graphtoolbox.training import Trainer, set_device  # noqa: E402
from graphtoolbox.interpretability import aggregate_ale_importance, additive_ale_from_dataset  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, HERE)
from netload_data_v030 import direct_datasets  # noqa: E402

LEADS = "--leads" in sys.argv


def kind(f):
    if re.fullmatch(r"(wind|wind_by_wind_power_weights)_l-\d+", f):
        return "wind_leads"
    if re.fullmatch(r"(nebulosity|nebulosity_by_solar_power_weights)_l-\d+", f):
        return "cloud_leads"
    return A._agm_kind(f)


def contiguous_groups(features):
    runs = []
    for f in features:
        k = kind(f)
        if runs and runs[-1][0] == k:
            runs[-1][1].append(f)
        else:
            runs.append([k, [f]])
    return runs


OUT = os.path.join(HERE, "additive_v030" + ("_leads" if LEADS else ""))
os.makedirs(OUT, exist_ok=True)
set_device("cpu")
train, val, test = direct_datasets(LEADS)
for ds in (train, val, test):
    ds._set_adj_matrix(adj_matrix="dtw")

runs = contiguous_groups(train.features)
n_exo = sum(1 for k, _ in runs if k == "exogenous")
dims, feature_groups, e = {}, {}, 0
for k, fs in runs:
    name = k
    if k == "exogenous":
        e += 1
        name = "exogenous" if n_exo == 1 or e == 1 else "calendar_dummies"
    dims[name] = dims.get(name, 0) + len(fs)
    feature_groups[name] = DUMMIES if name == "calendar_dummies" else fs
assert sum(dims.values()) == train.num_node_features
assert len({k for k, _ in runs if k != "exogenous"}) == len([k for k, _ in runs if k != "exogenous"]), runs
print({k: len(v) for k, v in feature_groups.items()}, flush=True)

seed_everything(42); gc.collect()
model = AdditiveGraphModel(feature_group_dims=dims, num_layers=2, hidden_channels=64,
                           out_channels=P.OUT_CHANNELS, conv_class=GCNConv, conv_kwargs={})
trainer = Trainer(model=model, dataset_train=train, dataset_val=val, dataset_test=test, batch_size=32,
                  model_kwargs={"lr": 1e-3, "num_epochs": 300}, reconcile=False, lam_reg=0)
CKPT = os.path.join(HERE, "checkpoints_additive_v030" + ("_leads" if LEADS else ""), "AdditiveGraphModel")
# An existing checkpoint is reused unless --retrain is given.
retrain = "--retrain" in sys.argv or not (os.path.isdir(CKPT) and os.listdir(CKPT))
res = trainer.train(force_training=retrain, patience=30, save=True, plot_loss=False, saving_directory=CKPT)
pred, target, group_outputs = res[0].detach().cpu(), res[1].detach().cpu(), res[4]
p, y = pred.sum(dim=0).numpy(), target.sum(dim=0).numpy()
summary = {"rmse": float(np.sqrt(np.mean((p - y) ** 2))),
           "smape": float(100 * np.mean(2 * np.abs(p - y) / (np.abs(p) + np.abs(y) + 1e-8))),
           "n_groups": len(group_outputs)}
torch.save(pred, os.path.join(OUT, "AdditiveGraphModel.pt"))
# National contributions of the groups; with the intercept they sum to the national forecast.
Tp = p.shape[0]
contrib = {g: c.cpu().numpy().sum(axis=0)[:Tp] for g, c in group_outputs.items()}
intercept = float(trainer.group_intercept_test.sum())
gap = np.abs(intercept + sum(contrib.values()) - p).max()
assert gap < 1.0, gap
np.savez(os.path.join(OUT, "group_contributions.npz"), target=y, prediction=p, intercept=intercept,
         groups=np.array(list(contrib)), **{f"contrib:{g}": v for g, v in contrib.items()})
summary["reconstruction_max_abs_gap_mw"] = float(gap)
# Exact importance of each group in an additive model: the spread of its contribution.
summary["contribution_std_mw"] = {g: float(v.std()) for g, v in contrib.items()}

# First-order ALE (Apley and Zhu) of every input feature, through the model: the feature of
# one node is moved across each quantile bin and the change of the national forecast is
# averaged per bin. Curves are kept per horizon (48 half-hours).
model = trainer.model.cpu().eval()
rows, curves = [], {}
for g, sl in model.group_index_map.items():
    for f in list(train.features)[sl]:
        try:
            r = additive_ale_from_dataset(model, test, f, n_bins=20)
        except ValueError:  # constant at the forecast origin (e.g. the time of day)
            print("constant at the origin:", f, flush=True)
            continue
        rows.append({"feature": f"{g}:{f}", "group": g, "importance": r["importance"]})
        curves[f"{g}:{f}:x"], curves[f"{g}:{f}:ale"], curves[f"{g}:{f}:counts"] = r["edges"], r["ale"], r["counts"]
imp = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
imp.to_csv(os.path.join(OUT, "ale_importance.csv"), index=False)
grouped = aggregate_ale_importance(imp)
grouped.to_csv(os.path.join(OUT, "ale_group_importance.csv"), index=False)
summary["group_share"] = {r.group: float(r.share) for r in grouped.itertuples(index=False)}
np.savez(os.path.join(OUT, "ale_curves.npz"), **curves)
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1)); print(imp.head(12).to_string(index=False))
